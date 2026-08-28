"""Step 1 — 用 corpus docs 构建 HippoRAG 2 离线索引（igraph 图 + Parquet 向量库）。

端点/分块/embedding 截断配置来自环境变量。
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils import cost_tracker
from src.hipporag.utils.config_utils import BaseConfig

# ── 配置默认值（全部从 .env 读取，无硬编码）──
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:32768/v1")
LLM_NAME = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "1")
EMBEDDING_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-large-en-v1.5")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9990/v1")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
EMBEDDING_MAX_SEQ_LEN = int(
    os.getenv("EMBEDDING_MAX_SEQ_LEN", os.getenv("EMBED_MAX_TOKENS", "512"))
)
EMBEDDING_TOKENIZER_PATH = os.getenv("EMBEDDING_TOKENIZER_PATH", "")


def atomic_json_dump(payload, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
os.environ.setdefault("EMBEDDING_API_KEY", os.getenv("EMBEDDING_API_KEY", "1"))
os.environ.setdefault("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)
if EMBEDDING_TOKENIZER_PATH:
    os.environ.setdefault("EMBEDDING_TOKENIZER_PATH", EMBEDDING_TOKENIZER_PATH)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_source_id(
    data_name: str, explicit: str | None, repo_root: Path, cache_root: Path | None = None
) -> str:
    """解析批次目录（两级："数据集/批次"）。
    优先级：显式 --source-id > 最新 upload 批次 > 复用已有缓存 > 数据集名。"""
    if explicit:
        prefix = f"{data_name}-"
        batch = explicit[len(prefix) :] if explicit.startswith(prefix) else explicit
        return f"{data_name}/{batch}"
    model = (os.getenv("LLM_MODEL", "") or "").split("/")[-1]
    upload_root = repo_root / "pipeline" / "evaluation" / "source" / "SAG" / model / data_name
    if upload_root.is_dir():
        timestamps = sorted(d for d in upload_root.iterdir() if d.is_dir())
        if timestamps:
            return f"{data_name}/{timestamps[-1].name}"
    # 无 upload 记录：复用已有缓存（caches/<data_name>/ 下最新批次子目录；或旧版扁平 caches/<data_name>）
    cache_root = cache_root or Path("caches")
    ds_dir = cache_root / data_name
    if ds_dir.is_dir():
        # 只认 upload 时间戳格式的批次子目录（YYYYMMDD_HHMMSS）；contexts/questions 等固定目录不算批次
        subs = sorted(
            d
            for d in ds_dir.iterdir()
            if d.is_dir()
            and len(d.name) == 15
            and d.name[8] == "_"
            and d.name[:8].isdigit()
            and d.name[9:].isdigit()
        )
        if subs:
            latest = max(subs, key=lambda d: d.stat().st_mtime)
            print(
                f"[source-id] 无 upload 记录，复用已有缓存批次: {data_name}/{latest.name}",
                file=sys.stderr,
            )
            return f"{data_name}/{latest.name}"
        print(f"[source-id] 无 upload 记录，复用已有扁平缓存: {data_name}", file=sys.stderr)
        return data_name
    print(
        f"[source-id] 未找到 upload 记录或已有缓存，按数据集名新建：caches/{data_name}/。",
        file=sys.stderr,
    )
    return data_name


def main():
    parser = argparse.ArgumentParser(description="为多跳 QA corpus 构建 HippoRAG 2 索引")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--force_reindex", action="store_true", help="强制重建索引")
    parser.add_argument(
        "--force_openie", action="store_true", help="强制重新抽取 OpenIE（忽略已有缓存）"
    )
    parser.add_argument(
        "--openie_mode",
        default="online",
        choices=["online", "offline"],
        help="兼容参数：external 集成仅支持 online",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="绑定 run_upload.py 的 source_config_id；默认自动发现最新上传批次",
    )
    parser.add_argument("--max_docs", type=int, default=None, help="只建前 N 条 doc（冒烟测试用）")
    args = parser.parse_args()

    if args.openie_mode != "online":
        raise SystemExit("external 集成仅支持 openie_mode=online（离线 OpenIE 未随包分发）")
    data_name = args.data_name
    source_id = resolve_source_id(data_name, args.source_id, REPOSITORY_ROOT, Path("caches"))
    work_dir = Path("caches") / source_id
    docs_path = work_dir / "contexts" / f"{data_name}_corpus_docs.json"
    if not docs_path.exists():
        raise FileNotFoundError(
            f"找不到 {docs_path}，请先运行 Step_0_load_dataset.py --data_name {data_name}"
        )

    with open(docs_path, encoding="utf-8") as f:
        docs = json.load(f)
    if args.max_docs:
        docs = docs[: args.max_docs]

    config = BaseConfig(
        llm_base_url=LLM_BASE_URL,
        llm_name=LLM_NAME,
        dataset=data_name,
        embedding_model_name=EMBEDDING_NAME,
        embedding_base_url=EMBEDDING_BASE_URL,
        embedding_batch_size=EMBEDDING_BATCH_SIZE,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        max_new_tokens=2048,
        qa_max_new_tokens=4096,
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        force_index_from_scratch=args.force_reindex,
        force_openie_from_scratch=args.force_openie,
        openie_mode=args.openie_mode,
        corpus_len=len(docs),
    )
    if EMBEDDING_MAX_SEQ_LEN is not None:
        config.embedding_max_seq_len = EMBEDDING_MAX_SEQ_LEN

    print(f"[index] {data_name}: 准备 insert {len(docs)} 条 doc")
    t0 = time.time()

    hipporag = HippoRAG(global_config=config, save_dir=str(work_dir))
    cost_tracker.set_output_path(os.path.join(hipporag.working_dir, "cost.json"))
    try:
        hipporag.index(docs)
    except Exception:
        traceback.print_exc()
        raise

    elapsed = time.time() - t0
    cost_tracker.dump()
    usage = {}
    cost_path = os.path.join(hipporag.working_dir, "cost.json")
    if os.path.exists(cost_path):
        with open(cost_path, encoding="utf-8") as f:
            usage = json.load(f)
    metrics_path = work_dir / "index_metrics_latest.json"
    atomic_json_dump(
        {
            "status": "completed",
            "dataset": data_name,
            "source_config_id": source_id,
            "started_at": datetime.fromtimestamp(t0, tz=UTC).isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "runtime_seconds": round(elapsed, 6),
            "llm_usage": usage,
        },
        metrics_path,
    )
    print(f"[index] {data_name}: 完成，耗时 {elapsed:.0f}s。索引位于 caches/{source_id}/")
    print(f"[metrics] index metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
