#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aiohttp>=3.9", "jieba==0.42.1", "regex>=2024.0", "tiktoken>=0.12",
#   "python-dotenv>=1.0", "pyuca==1.2",
# ]
# ///
"""Run SAG2 or Agentic SAG2 on the three paper benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_data import DatasetLoader
from agentic_sag2 import run_agentic
from model_api import RecordedJev, RequestJournal
from sag2 import (
    CHUNK_POLICY,
    DEFAULT_CHUNK_CHARS,
    EXPANSION_METHOD,
    FINAL_METHOD,
    RETRIEVAL_METHOD,
    CorpusIndex,
    JevClient,
    compact_json,
    corpus_fingerprint,
    search_sag2,
)
from sample_identity import index_unique_records, validate_question_identity

DATASETS = ("musique", "2wikimultihopqa", "hotpotqa")
DATA_RELEASE = "5ec05b38deecc3318bb432c69865959c56058990"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sag2", "agentic", "local"], default="sag2")
    parser.add_argument(
        "--model-path", type=Path, help="Local reranker directory; no downloads at runtime"
    )
    parser.add_argument("--reranker", choices=["jina", "qwen"], default="qwen")
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4, help="Local Qwen inference batch size")
    parser.add_argument(
        "--rerank-protocol",
        choices=["sag2", "standard"],
        default="sag2",
        help="standard: original query, no expansion or final DPP; local mode only",
    )
    parser.add_argument(
        "--datasets",
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", *DATASETS],
    )
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing source files from the pinned HippoRAG 2 release",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run the first N questions of each dataset; corpus remains complete",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--index-workers",
        type=int,
        default=8,
        help="Parallel indexing processes; independent of query concurrency",
    )
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--seed-k", type=int, default=10)
    parser.add_argument(
        "--max-hops",
        "--expansions",
        type=int,
        help="Expansion rounds per SAG2 call (0-2): default 2 in sag2 mode, 1 in agentic mode",
    )
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument(
        "--return-k", type=int, default=20, help="Passages returned per agent tool call"
    )
    parser.add_argument("--llm-base-url", help="OpenAI-compatible API base URL, including /v1")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-api", choices=["responses", "chat"])
    parser.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    parser.add_argument(
        "--reasoning-effort", choices=["default", "none", "low", "medium", "high", "xhigh", "max"]
    )
    parser.add_argument("--expansion-term-limit", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-list", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache" / "sag2")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing suite output directory using identical settings",
    )
    parser.add_argument(
        "--env-file", type=Path, default=Path(os.getenv("SAG_ENV_FILE", ROOT / ".env"))
    )
    parser.add_argument("--jev-base-url", help="Jev base URL or the complete /decisions endpoint")
    parser.add_argument("--jev-model")
    parser.add_argument(
        "--jev-question-type",
        choices=["noul"],
        default="noul",
        help="SAG2 paper protocol: noul",
    )
    parser.add_argument(
        "--local-address", help="Bind model connections to a local IP, e.g. a VPN interface"
    )
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Prepare and validate every dataset without calling Jev",
    )
    return parser


async def download_missing(args: argparse.Namespace, names: list[str]) -> None:
    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        for name in names:
            upstream = name
            for suffix in (".json", "_corpus.json"):
                path = args.dataset_dir / f"{name}{suffix}"
                if path.is_file():
                    continue
                url = f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{DATA_RELEASE}/{upstream}{suffix}"
                print(f"下载 {path.name}", flush=True)
                async with session.get(url) as response:
                    response.raise_for_status()
                    with path.with_suffix(".part").open("wb") as stream:
                        async for block in response.content.iter_chunked(1024 * 1024):
                            stream.write(block)
                path.with_suffix(".part").replace(path)


def load_dataset(name: str, directory: Path) -> tuple[list[dict], list[dict], str]:
    loader = DatasetLoader(name, str(directory))
    loader.validate_source_pair()
    corpus = loader.load_corpus()
    records = loader.get_question_records()
    for record in records:
        if not record["gold_docs"]:
            raise ValueError(f"Question {record['id']} has no gold retrieval documents")
    docs = set(loader.get_docs())
    missing = sum(doc not in docs for record in records for doc in record["gold_docs"])
    if missing:
        logging.warning(
            "%s: %s gold passages do not exactly match the corpus; Benchmark counts them as misses",
            name,
            missing,
        )
    return corpus, records, "passage"


def recall_metrics(
    gold: list[str], retrieved: list[str], k_values: list[int], prefix: str = "Recall"
) -> dict:
    gold_set = set(gold)
    return {f"{prefix}@{k}": len(gold_set & set(retrieved[:k])) / len(gold_set) for k in k_values}


def unique_documents(candidates: list[dict], top_k: int) -> list[dict]:
    documents = {}
    for candidate in candidates:
        documents.setdefault(candidate["document_id"], candidate)
    return list(documents.values())[:top_k]


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(compact_json(value), encoding="utf-8")
    temporary.replace(path)


async def run_dataset(
    name: str, args: argparse.Namespace, client: JevClient | None, suite: Path
) -> dict:
    started = time.perf_counter()
    corpus, records, unit = load_dataset(name, args.dataset_dir)
    selected = records[args.offset : None if args.limit is None else args.offset + args.limit]
    if not selected:
        raise ValueError(f"{name}: requested question range is empty")
    fingerprint = corpus_fingerprint(args.dataset_dir / f"{name}_corpus.json")
    index = CorpusIndex(
        args.cache_dir / f"{name}.sqlite",
        corpus,
        fingerprint,
        args.chunk_chars,
        index_workers=args.index_workers,
    )
    output = suite / name
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": name,
        "algorithm": {"sag2": "SAG2", "agentic": "Agentic SAG2", "local": "Local SAG2"}[args.mode],
        "metric_unit": unit,
        "corpus_sha256": fingerprint,
        "questions_sha256": corpus_fingerprint(args.dataset_dir / f"{name}.json"),
        "algorithm_sha256": corpus_fingerprint(ROOT / "scripts" / "sag2.py"),
        "chunking_sha256": corpus_fingerprint(ROOT / "scripts" / "chunking.py"),
        "candidate_k": args.candidate_k,
        "seed_k": args.seed_k,
        "max_hops": args.max_hops
        if args.max_hops is not None
        else (1 if args.mode == "agentic" else 2),
        "expansion_term_limit": args.expansion_term_limit,
        "chunk_chars": args.chunk_chars,
        "chunk_policy": CHUNK_POLICY,
        "top_k": args.top_k,
        "k_list": args.k_list,
        "offset": args.offset,
        "limit": args.limit,
        "jev_base_url": args.jev_base_url,
        "jev_model": args.jev_model,
        "jev_question_type": args.jev_question_type,
        "expansion_term_method": EXPANSION_METHOD,
        "expansion_retrieval_method": RETRIEVAL_METHOD,
        "final_ranking": FINAL_METHOD,
        "carried_bm25_top_k": 5,
        "extra_final_rerank": False,
        "local_final_ranking": "dpp_top10_no_model_request",
        "jev_filtering": "none",
        "tokenizer": "python-jieba-search/FTS5-unicode61; zero-hit title repair",
    }
    if args.mode == "agentic":
        config.update(
            llm=args.llm_config,
            max_tool_calls=args.max_tool_calls,
            return_k=args.return_k,
            final_ranking="llm_native_tool_calling",
            agent_code_sha256=corpus_fingerprint(ROOT / "scripts/agentic_sag2.py"),
            api_code_sha256=corpus_fingerprint(ROOT / "scripts/model_api.py"),
            agent_prompt_sha256=corpus_fingerprint(ROOT / "prompts/agentic_sag2.txt"),
            tool_schema_sha256=corpus_fingerprint(ROOT / "prompts/sag2_tool.json"),
            final_schema_sha256=corpus_fingerprint(ROOT / "prompts/agentic_result.json"),
        )
    if args.mode == "local":
        config.update(
            local_model=client.config if client is not None else None,
            rerank_protocol=args.rerank_protocol,
            scorer_code_sha256=corpus_fingerprint(ROOT / "scripts/local_reranker.py"),
            jev_base_url=None,
            jev_model=None,
            jev_question_type=None,
            final_ranking="raw_reranker" if args.rerank_protocol == "standard" else FINAL_METHOD,
            local_final_ranking="none"
            if args.rerank_protocol == "standard"
            else "dpp_top10_no_model_request",
        )
    config_path = output / "config.json"
    checkpoint_path = output / "details.jsonl"
    try:
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise ValueError(
                    f"{name}: resume settings/data/code differ from {config_path}; use identical arguments or a new output directory"
                )
        elif checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint_path} has no configuration")
        else:
            write_json(config_path, config)
        print(
            f"{name}: {index.document_count} 段原文 / {index.total} 个索引片段 / {len(selected)} 题",
            flush=True,
        )
        if args.index_only:
            return {
                "dataset": name,
                "question_count": len(selected),
                "document_count": index.document_count,
                "chunk_count": index.total,
                "status": "indexed",
                "metric_unit": unit,
            }
        completed = []
        if checkpoint_path.exists():
            completed = [
                json.loads(line)
                for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
            ]
        by_id = index_unique_records(completed, "dataset_sample_id", f"{name} checkpoint")
        selected_ids = {record["id"] for record in selected}
        if set(by_id) - selected_ids:
            raise ValueError(f"{name}: checkpoint contains questions outside the requested range")
        for record in selected:
            if record["id"] in by_id:
                validate_question_identity(
                    record["question"], by_id[record["id"]]["question"], record["id"], "checkpoint"
                )
        pending = iter(
            (args.offset + i, record)
            for i, record in enumerate(selected)
            if record["id"] not in by_id
        )
        prefix = "Recall"

        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:

            async def worker() -> None:
                for question_index, record in pending:
                    if args.mode == "agentic":
                        journal = RequestJournal(
                            client.session, output / "requests" / str(question_index)
                        )
                        agent_jev = RecordedJev(
                            journal, args.jev_base_url, args.jev_model, args.jev_api_key
                        )
                        result = await run_agentic(
                            index,
                            agent_jev,
                            record["question"],
                            journal,
                            args.llm_config,
                            args.llm_api_key,
                            {
                                key: config[key]
                                for key in (
                                    "candidate_k",
                                    "seed_k",
                                    "max_hops",
                                    "expansion_term_limit",
                                    "top_k",
                                    "return_k",
                                    "max_tool_calls",
                                )
                            },
                        )
                    else:
                        scorer = client
                        if args.mode == "local":
                            from local_reranker import LocalScorer

                            scorer = LocalScorer(
                                client,
                                output / "requests" / str(question_index),
                                args.rerank_protocol,
                            )
                        result = await search_sag2(
                            index,
                            scorer,
                            record["question"],
                            candidate_k=args.candidate_k,
                            seed_k=args.seed_k,
                            max_hops=config["max_hops"],
                            expansion_term_limit=args.expansion_term_limit,
                        )
                        if args.mode == "local":
                            result["algorithm"] = "Local SAG2"
                            result["reranker_usage"] = result.pop("jev_usage")
                            result["jev_requests"] = 0
                            for step in result["trace"]:
                                if step["stage"] == "jev":
                                    step["stage"] = "reranker"
                            if args.rerank_protocol == "standard":
                                result["candidates"].sort(
                                    key=lambda c: (-c["jev_score"], -c["fusion_score"], c["id"])
                                )
                    retrieved = unique_documents(result.pop("candidates"), args.top_k)
                    baseline = unique_documents(result.pop("baseline"), args.top_k)
                    gold = record["gold_docs"]
                    field = "full_doc"
                    row = {
                        "question_index": question_index,
                        "dataset_sample_id": record["id"],
                        "question": record["question"],
                        "retrieved_docs": [c["full_doc"] for c in retrieved],
                        "bm25_docs": [c["full_doc"] for c in baseline],
                        "retrieved_source_ids": [c["source_id"] for c in retrieved],
                        "metrics": recall_metrics(
                            gold, [c[field] for c in retrieved], args.k_list, prefix
                        ),
                        "all_evidence_metrics": {
                            f"AllEvidence@{k}": int(set(gold) <= {c[field] for c in retrieved[:k]})
                            for k in args.k_list
                        },
                        "bm25_metrics": recall_metrics(
                            gold, [c[field] for c in baseline], args.k_list, prefix
                        ),
                        **result,
                    }
                    checkpoint.write(compact_json(row) + "\n")
                    checkpoint.flush()
                    by_id[record["id"]] = row
                    if len(by_id) % 10 == 0 or len(by_id) == len(selected):
                        print(
                            f"{name}: 已完成 {len(by_id)}/{len(selected)}，累计 {time.perf_counter() - started:.1f} 秒",
                            flush=True,
                        )

            async with asyncio.TaskGroup() as group:
                for _ in range(args.concurrency):
                    group.create_task(worker())
        ordered = [by_id[record["id"]] for record in selected]
        search_rows = [
            {
                key: row[key]
                for key in ("question_index", "dataset_sample_id", "question", "retrieved_docs")
            }
            for row in ordered
        ]
        write_json(output / "search_results.json", search_rows)
        write_json(
            output / "bm25_results.json",
            [
                {**row, "retrieved_docs": detail["bm25_docs"]}
                for row, detail in zip(search_rows, ordered)
            ],
        )
        usage_field = "reranker_usage" if args.mode == "local" else "jev_usage"
        summary = {
            "dataset": name,
            "status": "completed",
            "question_count": len(ordered),
            "document_count": index.document_count,
            "chunk_count": index.total,
            "metric_unit": unit,
            "metrics": {
                key: sum(row["metrics"][key] for row in ordered) / len(ordered)
                for key in ordered[0]["metrics"]
            },
            "bm25_metrics": {
                key: round(sum(row["bm25_metrics"][key] for row in ordered) / len(ordered), 4)
                for key in ordered[0]["bm25_metrics"]
            },
            "mean_query_ms": round(sum(row["elapsed_ms"] for row in ordered) / len(ordered)),
            "wall_seconds_this_invocation": round(time.perf_counter() - started, 3),
            "requests": sum(row[usage_field]["requests"] for row in ordered),
            "input_tokens": sum(row[usage_field]["input_tokens"] for row in ordered),
            "output_tokens": sum(row[usage_field]["output_tokens"] for row in ordered),
            ("actual_reranker_models" if args.mode == "local" else "actual_jev_models"): sorted(
                {
                    model
                    for row in ordered
                    for model in (
                        row[usage_field]["actual_models"]
                        if args.mode == "agentic"
                        else [batch["model"] for batch in row[usage_field]["batches"]]
                    )
                }
            ),
            "search_results": str(output / "search_results.json"),
        }
        summary["algorithm"] = config["algorithm"]
        if args.mode == "local":
            summary.update(
                jev_requests=0,
                model_api_requests=0,
                model_api_cost_usd=0,
                rerank_protocol=args.rerank_protocol,
            )
        summary["all_evidence_metrics"] = {
            f"AllEvidence@{k}": sum(
                row["all_evidence_metrics"][f"AllEvidence@{k}"] for row in ordered
            )
            / len(ordered)
            for k in args.k_list
        }
        if args.mode == "agentic":
            summary["search_calls"] = sum(row["search_calls"] for row in ordered)
            summary["llm_usage"] = {
                field: sum(row["llm_usage"][field] for row in ordered)
                for field in (
                    "requests",
                    "reused_requests",
                    "input_tokens",
                    "cached_input_tokens",
                    "cache_write_input_tokens",
                    "output_tokens",
                    "reasoning_output_tokens",
                )
            }
            for service in ("llm", "jev"):
                costs = [row[f"{service}_usage"]["provider_cost_usd"] for row in ordered]
                summary[f"{service}_provider_cost_usd"] = (
                    sum(costs) if all(c is not None for c in costs) else None
                )
        write_json(output / "metrics.json", summary)
        print(f"{name}: {summary['metrics']}", flush=True)
        return summary
    finally:
        index.close()


async def run(args: argparse.Namespace) -> None:
    names = list(DATASETS) if args.datasets == ["all"] else list(dict.fromkeys(args.datasets))
    if "all" in names:
        raise ValueError("--datasets all must be used alone")
    if args.download:
        await download_missing(args, names)
    # Validate all file pairs before any billable model calls.
    for name in names:
        DatasetLoader(name, str(args.dataset_dir)).validate_source_pair()
    suite = (
        args.resume
        or args.output_dir
        or ROOT
        / "output"
        / "sag2"
        / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    )
    if not args.resume and suite.exists():
        raise ValueError(f"Output directory already exists: {suite}; use --resume to continue")
    if args.resume and not suite.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {suite}")
    suite.mkdir(parents=True, exist_ok=True)
    load_dotenv(args.env_file)
    args.jev_base_url = args.jev_base_url or os.getenv(
        "JEV_BASE_URL", "https://openrouter.ai/api/alpha/decisions"
    )
    args.jev_model = args.jev_model or os.getenv("JEV_MODEL", "typesafe/jev-1.13")
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("JEV_API_KEY", "")
    args.jev_api_key = api_key
    if args.mode == "agentic":
        args.llm_config = {
            "base_url": args.llm_base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            "model": args.llm_model or os.getenv("LLM_MODEL", "gpt-6.1-sol"),
            "api": args.llm_api or os.getenv("LLM_API", "responses"),
            "reasoning_effort": args.reasoning_effort or os.getenv("LLM_REASONING_EFFORT", "high"),
        }
        args.llm_api_key = os.getenv(args.llm_api_key_env) or os.getenv("OPENAI_API_KEY", "")
        if args.llm_config["api"] not in ("responses", "chat"):
            raise ValueError("LLM_API must be responses or chat")
        if not args.index_only and not args.llm_api_key:
            raise ValueError(f"Set {args.llm_api_key_env} or OPENAI_API_KEY for agentic mode")
    if not args.index_only and args.mode != "local" and not api_key:
        raise ValueError("请设置 JEV_API_KEY（项目 .env、--env-file 或环境变量）")
    local_model = None
    if args.mode == "local" and not args.index_only:
        from local_reranker import LocalReranker

        local_model = LocalReranker(
            args.model_path, args.reranker, args.backend, args.device, args.batch_size
        )
        print(
            f"本地推理：{args.reranker} / {args.backend} / {local_model.device}，模型API请求为0",
            flush=True,
        )
    summaries = []
    connector = aiohttp.TCPConnector(
        limit=args.concurrency,
        local_addr=(args.local_address, 0) if args.local_address else None,
    )
    async with aiohttp.ClientSession(
        connector=connector, timeout=aiohttp.ClientTimeout(total=args.timeout)
    ) as session:
        client = (
            local_model
            if args.mode == "local"
            else JevClient(
                session,
                args.jev_base_url,
                args.jev_model,
                api_key,
                question_type=args.jev_question_type,
            )
        )
        for name in names:
            summaries.append(await run_dataset(name, args, client, suite))
            write_json(
                suite / "summary.json",
                {
                    "algorithm": {"sag2": "SAG2", "agentic": "Agentic SAG2", "local": "Local SAG2"}[
                        args.mode
                    ],
                    "datasets": summaries,
                },
            )
    print(f"结果目录：{suite}", flush=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_hops is None:
        args.max_hops = 1 if args.mode == "agentic" else 2
    for name in (
        "concurrency",
        "index_workers",
        "candidate_k",
        "seed_k",
        "expansion_term_limit",
        "top_k",
        "chunk_chars",
        "timeout",
        "max_tool_calls",
        "return_k",
        "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.chunk_chars < 2:
        parser.error("--chunk-chars must be at least 2")
    if (
        not 0 <= args.max_hops <= 2
        or args.offset < 0
        or (args.limit is not None and args.limit <= 0)
    ):
        parser.error("max-hops must be 0-2; offset must be nonnegative; limit must be positive")
    if args.mode == "agentic" and not 5 <= args.top_k <= args.return_k <= args.candidate_k:
        parser.error("agentic mode requires 5 <= top-k <= return-k <= candidate-k")
    if args.mode == "local" and not args.index_only and args.model_path is None:
        parser.error("local mode requires --model-path")
    if args.mode == "local" and args.backend == "mlx" and args.reranker != "jina":
        parser.error("--backend mlx requires --reranker jina")
    if args.rerank_protocol == "standard" and (args.mode != "local" or args.max_hops != 0):
        parser.error("standard reranking requires --mode local --expansions 0")
    if args.seed_k > args.candidate_k or args.top_k > args.candidate_k:
        parser.error("seed-k and top-k cannot exceed candidate-k")
    if not args.k_list or min(args.k_list) <= 0 or max(args.k_list) > args.top_k:
        parser.error("k-list values must be positive and cannot exceed top-k")
    if args.resume and args.output_dir:
        parser.error("--resume and --output-dir cannot be used together")
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
