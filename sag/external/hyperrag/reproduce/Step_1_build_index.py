"""Step 1 — 用多跳 QA 数据集的 corpus 构建知识超图 + 实体/关系向量库。

正确用法（回归 insert 的原生设计）：把 corpus 读成 list[str] 逐条 insert，
每个 "{title}\\n{text}" 是一个独立 doc，doc 内部再按 token 切块。
原 Step_1.py 的 f.read() 整文件成一个大串是误用，此处修正。
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import time
import traceback

from hyperrag import HyperRAG
from hyperrag.utils import EmbeddingFunc
from hyperrag_config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_FUNC_MAX_ASYNC,
    EMBEDDING_MAX_TOKEN_SIZE,
    EMBEDDING_MODEL_NAME,
    INDEX_BATCH_SIZE,
    LLM_FUNC_MAX_ASYNC,
    embedding_func,
    llm_model_func,
)


def _log(msg: str):
    """带时间戳的打印（无缓冲，立刻可见）。"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _readable_err(e: Exception) -> str:
    """把异常转成可读字符串——重点解决 302ai 的 BadRequestError.message 是 brotli
    压缩字节（乱码）的问题：从异常的 response 里抠出响应体并解压。"""
    resp = getattr(e, "response", None)
    if resp is None:
        # tenacity RetryError 等无 response，展开 __cause__
        cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
        return f"{type(e).__name__}: {cause or e}"
    enc = resp.headers.get("content-encoding") if hasattr(resp, "headers") else None
    raw = resp.content if hasattr(resp, "content") else None
    status = getattr(resp, "status_code", "?")
    try:
        if raw is not None:
            # 复用 hyperrag.llm 的解压（gzip/deflate/br，带嗅探兜底）
            from hyperrag.llm import _decompress_body

            body = _decompress_body(raw, enc).decode("utf-8", errors="replace")
        else:
            body = resp.text
    except Exception:
        body = str(e)
    return f"{type(e).__name__} [HTTP {status}] {body}"


def insert_docs(rag, docs, batch_size: int = INDEX_BATCH_SIZE):
    """分批 insert，失败即抛（不整批重试）。

    每批 insert 返回时图与向量库真正落盘，形成断点：中断后重跑，内核按文档
    内容哈希去重跳过已完成的批，最多丢一批。批大小见
    hyperrag_config.INDEX_BATCH_SIZE。

    单 chunk 的偶发错误已由 tenacity 在 LLM 调用层重试 5 次 + 容错跳过兜住；
    能冒泡到这里的都是真 bug（如 KeyError/AttributeError），应立即暴露，
    打印完整 traceback 后让进程崩溃退出，不浪费时间整批重试。
    """
    total = len(docs)
    n_batches = (total + batch_size - 1) // batch_size
    _log(
        f"=== insert 开始（共 {total} 条 doc，分 {n_batches} 批 × {batch_size}，真 bug 直接抛）==="
    )
    t0 = time.time()
    for bi in range(n_batches):
        batch = docs[bi * batch_size : (bi + 1) * batch_size]
        tb = time.time()
        _log(f"--- 批次 {bi + 1}/{n_batches}（{len(batch)} docs）---")
        try:
            rag.insert(batch)
        except Exception as e:
            _log(
                f"!!! insert 失败于批次 {bi + 1}/{n_batches}（累计 {time.time() - t0:.0f}s），真 bug，不重试，直接抛出："
            )
            _log(f"    异常类型: {type(e).__name__}")
            _log(f"    可读错误: {_readable_err(e)[:1500]}")
            _log("    完整 traceback:")
            traceback.print_exc()
            _log(f"    已完成 {bi} 批已落盘，重跑会自动跳过。")
            raise  # 直接抛，让进程崩溃退出，暴露 bug 第一现场
        _log(
            f"--- 批次 {bi + 1}/{n_batches} 完成，耗时 {time.time() - tb:.0f}s（累计 {time.time() - t0:.0f}s）---"
        )
    _log(f"=== insert 成功，耗时 {time.time() - t0:.0f}s ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="为多跳 QA corpus 构建超图索引")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument(
        "--max_docs", type=int, default=None, help="冒烟测试用：只建前 N 条 doc 的索引"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="展开细粒度日志（进入 chunk 抽取、LLM 输入等都打印）"
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

    rag = HyperRAG(
        working_dir=str(work_dir),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        tiktoken_model_name="gpt-4o-mini",
        verbose=args.verbose,
        embedding_func_max_async=EMBEDDING_FUNC_MAX_ASYNC,
        llm_model_max_async=LLM_FUNC_MAX_ASYNC,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
            tokenize_url=EMBEDDING_BASE_URL,
            tokenize_model=EMBEDDING_MODEL_NAME,
            tokenize_api_key=EMBEDDING_API_KEY,
        ),
    )

    ok = insert_docs(rag, docs)
    if ok:
        print(f"[index] {data_name}: insert 完成。索引位于 {work_dir}")
    else:
        print(f"[index] {data_name}: insert 失败，可重跑（已 insert 的 doc 会自动去重跳过）")


if __name__ == "__main__":
    main()
