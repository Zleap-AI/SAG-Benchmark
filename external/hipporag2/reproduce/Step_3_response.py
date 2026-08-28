"""Step 3 — HippoRAG 2 SAG2 风格 QA：检索一次 + Thought:/Answer: 出答案。

HippoRAG 2 的 QA prompt（src/hipporag/prompts/templates/rag_qa_musique.py）本身就是 SAG2 标准结构
（system=rag_qa_system + few-shot + user=Wikipedia Title:...+Question:+Thought:），
不需要额外改写。本文件只做检索 + 调用内置 QA + 保存结果。

默认行为（复现一致，等价源 main_qwen.py 完整流程）：
  每题只检索一次 + 调用内置 QA。index 本身幂等（mdhash 去重 +
  graph.graphml 加载 + OpenIE 跳过已有 chunk），重复调用边际成本低，不重复检索。
  --force 强制从零重建索引/图；--qa_only 复用已有检索结果只跑 QA。
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 直接指向 utils 目录而非仓库根：只加载 sample_identity.py 单文件，
# 避免 pipeline 包 __init__ 引入本项目精简 venv 中不存在的重型依赖。
EVALUATION_UTILS = REPOSITORY_ROOT / "pipeline" / "evaluation" / "utils"
if str(EVALUATION_UTILS) not in sys.path:
    sys.path.insert(0, str(EVALUATION_UTILS))
from sample_identity import (
    index_unique_records,
    validate_identity_coverage,
    validate_question_identity,
)
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils import cost_tracker
from src.hipporag.utils.config_utils import BaseConfig
from src.hipporag.utils.misc_utils import QuerySolution

# ── 配置默认值 ──
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:32768/v1")
LLM_NAME = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY", "1")
EMBEDDING_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-large-en-v1.5")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9990/v1")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_MAX_SEQ_LEN = int(
    os.getenv("EMBEDDING_MAX_SEQ_LEN", os.getenv("EMBED_MAX_TOKENS", "512"))
)

os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
os.environ.setdefault("EMBEDDING_API_KEY", os.getenv("EMBEDDING_API_KEY", "1"))
os.environ.setdefault("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def atomic_json_dump(payload, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# 与 HippoRAG.rag_qa 内部的检索评测口径一致
RECALL_K_LIST = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]


def scores_to_list(scores) -> list[float | None] | None:
    """把 doc_scores（ndarray/list/None）转成可 JSON 序列化的 float 列表。

    非有限值写成 null：json.dump 会把 NaN/Infinity 写成非法 JSON 字面量。
    """
    if scores is None:
        return None
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    out: list[float | None] = []
    for v in scores:
        f = float(v)
        out.append(f if math.isfinite(f) else None)
    return out


def usage_from_metadata(metadata) -> dict:
    metadata = metadata or {}
    prompt = int(metadata.get("prompt_tokens", 0) or 0)
    completion = int(metadata.get("completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "cache_hit": bool(metadata.get("cache_hit", False)),
    }


def sum_usage(usages: list[dict]) -> dict:
    return {
        "prompt_tokens": sum(u["prompt_tokens"] for u in usages),
        "completion_tokens": sum(u["completion_tokens"] for u in usages),
        "total_tokens": sum(u["total_tokens"] for u in usages),
        "calls": len(usages),
        "cache_hits": sum(1 for u in usages if u.get("cache_hit")),
    }


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
    parser = argparse.ArgumentParser(description="HippoRAG 2 retrieve + SAG2 QA")
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--qa_only", action="store_true", help="只用已有检索结果跑 QA")
    parser.add_argument("--retrieve_only", action="store_true", help="只检索不跑 QA")
    parser.add_argument(
        "--force", action="store_true", help="强制从零重建索引/图（force_index_from_scratch）"
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="绑定 run_upload.py 的 source_config_id；默认自动发现最新上传批次",
    )
    args = parser.parse_args()

    run_started_perf = time.perf_counter()
    run_started_wall = datetime.now(UTC)
    index_seconds = retrieval_seconds = qa_seconds = 0.0
    index_usage = {}
    retrieval_metrics = {}

    data_name = args.data_name
    source_id = resolve_source_id(data_name, args.source_id, REPOSITORY_ROOT, Path("caches"))
    work_dir = Path("caches") / source_id
    questions_path = work_dir / "questions" / f"{data_name}_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(
            f"找不到 {questions_path}，请先运行 Step_0_load_dataset.py --data_name {data_name}"
        )

    questions = json.load(open(questions_path))
    all_queries = [q["question"] for q in questions]
    question_by_id = index_unique_records(questions, "id", "HippoRAG2 questions")
    gold_docs_path = work_dir / "contexts" / f"{data_name}_corpus_docs.json"

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
        corpus_len=len(json.load(open(gold_docs_path))) if gold_docs_path.exists() else 0,
        force_index_from_scratch=args.force,
    )
    if EMBEDDING_MAX_SEQ_LEN is not None:
        config.embedding_max_seq_len = EMBEDDING_MAX_SEQ_LEN

    # 最终结果统一输出到 outputs/（检索 + QA），索引等中间产物留在 caches/
    sub_dir = f"{LLM_NAME}_{EMBEDDING_NAME}".replace("/", "_")
    index_work = work_dir / sub_dir
    qa_dir = Path("outputs") / source_id / sub_dir / "qa_result"
    qa_dir.mkdir(parents=True, exist_ok=True)

    hipporag = HippoRAG(global_config=config, save_dir=str(work_dir))
    cost_tracker.set_output_path(str(qa_dir / "cost.json"))
    ret_path = qa_dir / "retrieval_results_latest.json"
    qa_path = qa_dir / "qa_results_latest.json"
    if args.qa_only:
        index_metrics_path = work_dir / "index_metrics_latest.json"
        if index_metrics_path.exists():
            with index_metrics_path.open(encoding="utf-8") as f:
                previous_index = json.load(f)
            index_seconds = float(previous_index.get("runtime_seconds", 0.0) or 0.0)
            index_usage = previous_index.get("llm_usage", {}) or {}

    if not args.qa_only:
        # 默认跳过 index：索引由 Step_1_build_index.py 负责构建（run.sh 中 Step_1 已跑过）。
        # 这里 HippoRAG.__init__ 已自动加载 caches/<source_id>/<llm>_<emb>/graph.graphml
        # 与三个 parquet 向量库；重复 index() 会因幂等逻辑不完善导致
        # graph 与 embedding store 不同步（AssertionError: entity+passage != vcount）。
        # 需要从零重建时显式传 --force（等价旧行为）。
        if args.force:
            if gold_docs_path.exists():
                docs = json.load(open(gold_docs_path))
            else:
                docs = []  # retrieve_only without build will fail

            index_t0 = time.perf_counter()
            hipporag.index(docs)
            index_seconds = time.perf_counter() - index_t0
            index_usage = getattr(getattr(hipporag, "openie", None), "last_usage", {}) or {}
        else:
            # 索引存在性检查：跳过 index 的前提是 Step_1 已建好完整索引。
            # 缺任一产物时直接报错，避免静默用空图检索后在 prepare_retrieval_objects 崩掉。
            required = [
                index_work / "graph.graphml",
                index_work / "chunk_embeddings" / "vdb_chunk.parquet",
                index_work / "entity_embeddings" / "vdb_entity.parquet",
                index_work / "fact_embeddings" / "vdb_fact.parquet",
            ]
            missing = [str(p) for p in required if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"索引产物缺失，无法直接检索：{missing}\n"
                    f"请先运行 Step_1_build_index.py --data_name {data_name} 构建索引，\n"
                    f"或在 Step_3 传 --force 在本步重建（耗时且会调 LLM）。"
                )
            index_seconds = 0.0
            index_usage = {}
        gold_docs_list = [q.get("gold_docs", []) for q in questions]
        retrieve_t0 = time.perf_counter()
        retrieve_result = hipporag.retrieve(
            queries=all_queries,
            gold_docs=gold_docs_list,
        )
        retrieval_seconds = time.perf_counter() - retrieve_t0
        if isinstance(retrieve_result, tuple):
            retrieval_metrics = retrieve_result[1] if len(retrieve_result) > 1 else {}
        query_solutions = (
            retrieve_result[0] if isinstance(retrieve_result, tuple) else retrieve_result
        )

        # save retrieval results
        if len(query_solutions) != len(questions):
            raise ValueError("HippoRAG2 retrieval count mismatch")
        retrieval_records = []
        for idx, qs in enumerate(query_solutions):
            sample_id = questions[idx]["id"]
            validate_question_identity(
                questions[idx]["question"],
                qs.question,
                sample_id,
                "HippoRAG2 retrieval",
            )
            retrieval_records.append(
                {
                    "id": sample_id,
                    "question": qs.question,
                    "docs": qs.docs,
                    # --qa_only 需要它来重建 QuerySolution.doc_scores
                    "doc_scores": scores_to_list(qs.doc_scores),
                }
            )
        index_unique_records(retrieval_records, "id", "HippoRAG2 retrieval results")
        atomic_json_dump(retrieval_records, ret_path)
        print(f"[retrieve] {len(retrieval_records)} 条 -> {ret_path}")

        if args.retrieve_only:
            runtime_path = qa_dir / "retrieval_metrics_latest.json"
            atomic_json_dump(
                {
                    "status": "completed",
                    "dataset": data_name,
                    "runtime_seconds": round(time.perf_counter() - run_started_perf, 6),
                    "index_seconds": round(index_seconds, 6),
                    "retrieval_seconds": round(retrieval_seconds, 6),
                    "retrieval_metrics": retrieval_metrics,
                    "llm_usage": index_usage,
                },
                runtime_path,
            )
            print(f"[metrics] retrieval metrics -> {runtime_path}")
            return

    else:
        # qa_only: 复用已有检索结果（HippoRAG 无 load_retrieval_results，直接解析 json）
        records = json.load(open(ret_path))
        retrieval_by_id = index_unique_records(records, "id", "HippoRAG2 retrieval results")
        validate_identity_coverage(
            question_by_id.keys(), retrieval_by_id.keys(), "HippoRAG2 qa_only"
        )
        query_solutions = []
        for question in questions:
            sample_id = question["id"]
            record = retrieval_by_id[sample_id]
            validate_question_identity(
                question["question"],
                record.get("question"),
                sample_id,
                "HippoRAG2 qa_only",
            )
            # doc_scores 需还原为 ndarray：QuerySolution.to_dict() 会调它的 .tolist()。
            # dtype=float 保证 null 变成 nan 而非 object 数组（后者会让 round() 抛错）。
            saved_scores = record.get("doc_scores")
            query_solutions.append(
                QuerySolution(
                    question=record["question"],
                    docs=record["docs"],
                    doc_scores=(
                        np.array(saved_scores, dtype=float) if saved_scores is not None else None
                    ),
                )
            )
        if any(r.get("doc_scores") is None for r in records):
            print(
                "[warning] 检索产物缺少 doc_scores（由旧版本脚本生成）；"
                "结果里的 doc_scores 将为 null，重跑一次完整 Step_3 可补全"
            )

        # rag_qa 收到 QuerySolution 即跳过检索，拿不到检索指标；这里用已保存的
        # 检索结果 + gold_docs 重算，口径与内核一致。无 gold_docs 时退回读落盘文件。
        qa_only_gold_docs = [q.get("gold_docs", []) for q in questions]
        if any(qa_only_gold_docs):
            recall_evaluator = RetrievalRecall(global_config=config)
            retrieval_metrics, _ = recall_evaluator.calculate_metric_scores(
                gold_docs=qa_only_gold_docs,
                retrieved_docs=[qs.docs for qs in query_solutions],
                k_list=RECALL_K_LIST,
            )
        else:
            saved_metrics_path = qa_dir / "retrieval_metrics_latest.json"
            if saved_metrics_path.exists():
                with saved_metrics_path.open(encoding="utf-8") as f:
                    retrieval_metrics = json.load(f).get("retrieval_metrics", {}) or {}
            else:
                print(f"[warning] 无 gold_docs 且无 {saved_metrics_path.name}，检索指标将为空")

    # QA (SAG2 prompt built into HippoRAG.rag_qa)
    gold_answers_list = [q["gold_answers"] for q in questions]
    gold_docs_list = [q.get("gold_docs", []) for q in questions]
    if not any(gold_docs_list):
        print(
            "[warning] questions 缺少 gold_docs；请重新运行 Step_0_load_dataset.py 以启用逐题 Recall"
        )
    qa_t0 = time.perf_counter()
    result = hipporag.rag_qa(
        query_solutions, gold_docs=gold_docs_list, gold_answers=gold_answers_list
    )
    qa_seconds = time.perf_counter() - qa_t0

    if isinstance(result, tuple):
        qa_solutions = result[0]
        qa_metadata = result[2] if len(result) > 2 else []
        if len(result) > 3 and result[3] is not None:
            retrieval_metrics = result[3]
        overall = result[4] if len(result) >= 5 else {}
    else:
        qa_solutions = result.get("results", query_solutions)
        qa_metadata = result.get("metadata", [])
        overall = result.get("overall_metrics", {})

    if len(qa_solutions) != len(questions):
        raise ValueError("HippoRAG2 QA result count mismatch")
    qa_usages = [usage_from_metadata(m) for m in qa_metadata]
    records = []
    for idx, qs in enumerate(qa_solutions):
        sample_id = questions[idx]["id"]
        row = qs.to_dict()
        validate_question_identity(
            questions[idx]["question"],
            row.get("question"),
            sample_id,
            "HippoRAG2 QA",
        )
        row["id"] = sample_id
        row["llm_usage"] = qa_usages[idx] if idx < len(qa_usages) else usage_from_metadata({})
        records.append(row)
    qa_usage = sum_usage(qa_usages)
    index_prompt = int((index_usage or {}).get("prompt_tokens", 0) or 0)
    index_completion = int((index_usage or {}).get("completion_tokens", 0) or 0)
    index_total = int((index_usage or {}).get("total_tokens", index_prompt + index_completion) or 0)
    index_calls = int((index_usage or {}).get("calls", 0) or 0)
    index_cache_hits = int((index_usage or {}).get("cache_hits", 0) or 0)
    payload = {
        "status": "completed",
        "dataset": data_name,
        "source_config_id": source_id,
        "started_at": run_started_wall.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "runtime_metrics": {
            "total_seconds": round(time.perf_counter() - run_started_perf, 6),
            "index_seconds": round(index_seconds, 6),
            "retrieval_seconds": round(retrieval_seconds, 6),
            "qa_seconds": round(qa_seconds, 6),
        },
        "llm_usage": {
            "index": index_usage,
            "qa": qa_usage,
            "total": {
                "prompt_tokens": index_prompt + qa_usage["prompt_tokens"],
                "completion_tokens": index_completion + qa_usage["completion_tokens"],
                "total_tokens": index_total + qa_usage["total_tokens"],
                "calls": index_calls + qa_usage["calls"],
                "cache_hits": index_cache_hits + qa_usage["cache_hits"],
            },
        },
        "retrieval_metrics": retrieval_metrics,
        "overall_metrics": overall,
        "results": records,
    }
    atomic_json_dump(payload, qa_path)
    print(f"[metrics] qa runtime/token metrics -> {qa_path}")
    # ── 结果摘要：任何方式运行 Step_3 都会打印关键指标 ──
    print("==> [摘要] 指标")
    print("  [Recall] " + "  ".join(f"{k}={v}" for k, v in payload["retrieval_metrics"].items()))
    print("  [QA]     " + "  ".join(f"{k}={v}" for k, v in payload["overall_metrics"].items()))
    print("  [Tokens] " + json.dumps(payload["llm_usage"]["total"], ensure_ascii=False))
    cost_tracker.dump()


if __name__ == "__main__":
    main()
