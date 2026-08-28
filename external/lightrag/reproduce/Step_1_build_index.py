"""Step 1 — 用多跳 QA 数据集的 corpus 构建 LightRAG 索引（graph + 向量库）。

把 corpus 读成 list[str] 逐条 insert，每个 "{title}\n{text}" 是一个独立 doc。
端点/分块/embedding 截断配置统一来自 lightrag_config.build_rag_kwargs()。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import asyncio
import datetime
import json
import time
import traceback
from collections import Counter

from lightrag_config import INDEX_BATCH_SIZE, build_rag_kwargs

# lightrag_config 必须先于 lightrag 内核 import：内核 import 时 load_dotenv(override=True)
# 会覆盖命令行注入的 env（--env-file），先读 config 才能保住注入值。
# E402 由仓库根 ruff 配置全局忽略；isort: skip 保持该行在 config 之后
from lightrag import LightRAG  # isort: skip


def _log(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _readable_err(e: Exception) -> str:
    resp = getattr(e, "response", None)
    if resp is None:
        cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
        return f"{type(e).__name__}: {cause or e}"
    status = getattr(resp, "status_code", "?")
    try:
        body = resp.text
    except Exception:
        body = str(e)
    return f"{type(e).__name__} [HTTP {status}] {body[:1500]}"


async def _log_progress(doc_status_path: Path, total: int, interval: float) -> None:
    """后台轮询 doc_status 文件，定期打印 processed/total 进度。

    LightRAG 的 ainsert() 是阻塞式一次性调用，内部 pipeline 逐篇处理；
    doc_status 逐条落盘，读文件即可拿到最新进度。图/向量库要等全部处理
    完才统一落盘（_insert_done），期间只在内存，进程中断会丢。
    """
    last_processed = -1
    while True:
        try:
            if doc_status_path.exists():
                data = json.loads(doc_status_path.read_text(encoding="utf-8"))
                c = Counter(
                    v.get("status", "?") if isinstance(v, dict) else "?" for v in data.values()
                )
                processed = c.get("processed", 0)
                if processed != last_processed:
                    _log(
                        f"进度: 已完成 {processed}/{total} "
                        f"(processing {c.get('processing', 0)}, "
                        f"pending {c.get('pending', 0)}, "
                        f"failed {c.get('failed', 0)})"
                    )
                    last_processed = processed
        except Exception:
            # 读文件偶发 IO 错误不影响主流程，跳过本轮
            pass
        await asyncio.sleep(interval)


async def ainsert_docs(
    rag, docs, work_dir: Path, progress_interval: float = 60.0, batch_size: int = INDEX_BATCH_SIZE
):
    """分批 async insert。

    每批 ainsert 返回时图与向量库真正落盘，形成断点：中断后重跑，内核按文档
    内容哈希去重跳过已完成的批，最多丢一批。批大小见 lightrag_config.INDEX_BATCH_SIZE。
    """
    total = len(docs)
    n_batches = (total + batch_size - 1) // batch_size
    doc_status_path = work_dir / "kv_store_doc_status.json"
    _log(
        f"=== ainsert 开始（共 {total} 条 doc，分 {n_batches} 批 × {batch_size}，"
        f"进度每 {progress_interval:.0f}s 打印）==="
    )
    monitor = asyncio.create_task(_log_progress(doc_status_path, total, progress_interval))
    t0 = time.time()
    try:
        for bi in range(n_batches):
            batch = docs[bi * batch_size : (bi + 1) * batch_size]
            tb = time.time()
            _log(f"--- 批次 {bi + 1}/{n_batches}（{len(batch)} docs）---")
            try:
                await rag.ainsert(batch)
            except Exception as e:
                _log(
                    f"!!! ainsert 失败于批次 {bi + 1}/{n_batches}"
                    f"（累计 {time.time() - t0:.0f}s），真 bug，直接抛出："
                )
                _log(f"    异常类型: {type(e).__name__}")
                _log(f"    可读错误: {_readable_err(e)[:1500]}")
                _log("    完整 traceback:")
                traceback.print_exc()
                _log(f"    已完成 {bi} 批已落盘，重跑会自动跳过。")
                raise
            _log(
                f"--- 批次 {bi + 1}/{n_batches} 完成，耗时 {time.time() - tb:.0f}s"
                f"（累计 {time.time() - t0:.0f}s）---"
            )
    finally:
        monitor.cancel()
    _log(f"=== ainsert 成功，耗时 {time.time() - t0:.0f}s ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="为多跳 QA corpus 构建 LightRAG 索引")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument(
        "--max_docs", type=int, default=None, help="冒烟测试用：只建前 N 条 doc 的索引"
    )
    parser.add_argument(
        "--progress_interval", type=float, default=60.0, help="进度日志打印间隔（秒）"
    )
    args = parser.parse_args()

    data_name = args.data_name
    work_dir = Path("caches") / data_name
    docs_path = work_dir / "contexts" / f"{data_name}_corpus_docs.json"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"找不到 {docs_path}，请先运行 "
            f"`python reproduce/Step_0_load_dataset.py --data_name {data_name}`"
        )

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    if args.max_docs:
        docs = docs[: args.max_docs]
    print(f"[index] {data_name}: 准备 insert {len(docs)} 条 doc")

    rag = LightRAG(working_dir=str(work_dir), **build_rag_kwargs())
    # LightRAG auto-initializes storages in __init__ (auto_manage_storages_states=True by default)

    loop = asyncio.get_event_loop()
    ok = loop.run_until_complete(
        ainsert_docs(rag, docs, work_dir, progress_interval=args.progress_interval)
    )
    if ok:
        print(f"[index] {data_name}: insert 完成。索引位于 {work_dir}")
    else:
        print(f"[index] {data_name}: insert 失败，可重跑（已 insert 的 doc 会自动去重跳过）")


if __name__ == "__main__":
    main()
