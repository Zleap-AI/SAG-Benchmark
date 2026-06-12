import argparse
import asyncio
import json
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单独测试 pipeline.modules.search.multi_vector.MultiSearcher"
    )
    parser.add_argument(
        "--source-config-id",
        default="musique-20260512_213908",
        help="信息源 ID；多个用英文逗号分隔",
    )
    parser.add_argument(
        "--query",
        default="When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?",
        help="查询文本",
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "precise"),
        default="fast",
        help="multi_vector 模式：fast=BM25实体召回+小规模扩展，precise=BM25实体召回+LLM过滤",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=300,
        help="搜索超时时间；<=0 表示不设置超时",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="控制台日志级别",
    )
    parser.add_argument(
        "--log-format",
        default="text",
        choices=("text", "json"),
        help="控制台日志格式",
    )
    parser.add_argument(
        "--output",
        default="summary",
        choices=("summary", "json"),
        help="summary 输出固定摘要；json 输出完整原始结果",
    )
    parser.add_argument(
        "--sort-by",
        default="none",
        choices=("score", "rank", "none"),
        help="summary 输出排序方式；默认按返回顺序",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [to_jsonable(v) for v in value]
        return str(value)


def print_summary(result: dict, sort_by: str) -> None:
    sections = list(result.get("sections", []))
    if sort_by == "score":
        sections.sort(key=lambda item: item.get("score") or 0, reverse=True)
    elif sort_by == "rank":
        sections.sort(key=lambda item: item.get("rank") if item.get("rank") is not None else 10**9)

    print()
    for index, section in enumerate(sections, start=1):
        content = (section.get("content") or "").strip()
        heading = (section.get("heading") or "")[:28]
        print(f"[{index}] {heading}")
        print(content)
        print()

    timings = result.get("_timings") or {}
    if timings:
        timing_order = [
            "step1_entity_bm25",
            "step2_embedding",
            "step3_dual_recall",
            "step3_seed_recall",
            "step3_fast_recall",
            "step4_event_entities",
            "step4_seed_event_entities",
            "step4_fast_event_entities",
            "step5_expand",
            "step5_fast_expand",
            "step6_coarse_rank",
            "step6_expand_rank",
            "step6_candidate_rank",
            "step6_fast_expand_rank",
            "step6_fast_final_rank",
            "step7_llm_filter",
            "step8_chunks",
            "native_chunk_total",
        ]
        print("耗时统计:")
        printed = set()
        for key in timing_order:
            if key in timings:
                print(f"  {key}: {float(timings[key]):.3f}s")
                printed.add(key)
                if key == "step1_entity_bm25":
                    for child_key in (
                        "step1_entity_bm25_es_search",
                    ):
                        if child_key in timings:
                            print(f"    |-- {child_key}: {float(timings[child_key]):.3f}s")
                            printed.add(child_key)
                if key == "step5_expand":
                    for child_key in (
                        "step5_get_new_entities",
                        "step5_update_entities",
                        "step5_entity_to_event",
                        "step5_fetch_event_entities",
                        "step5_update_state",
                    ):
                        if child_key in timings:
                            print(f"    |-- {child_key}: {float(timings[child_key]):.3f}s")
                            printed.add(child_key)
                if key in ("step3_seed_recall", "step3_fast_recall"):
                    for child_key in (
                        "step3_seed_entity_to_event_candidates",
                        "step3_seed_event1_rank",
                        "step3_seed_event2_query",
                        "step3_seed_merge",
                        "step3_seed_fetch_candidate_entities",
                        "step3_seed_score",
                    ):
                        if child_key in timings:
                            print(f"    |-- {child_key}: {float(timings[child_key]):.3f}s")
                            printed.add(child_key)
                if key == "step5_fast_expand":
                    for child_key in (
                        "step5_get_new_entities",
                        "step5_update_entities",
                        "step5_entity_to_event",
                        "step5_fetch_event_entities",
                        "step5_update_state",
                    ):
                        if child_key in timings:
                            print(f"    |-- {child_key}: {float(timings[child_key]):.3f}s")
                            printed.add(child_key)
                if key == "step7_llm_filter":
                    for child_key in (
                        "step7_fetch_candidate_contents",
                        "step7_llm_call",
                    ):
                        if child_key in timings:
                            print(f"    |-- {child_key}: {float(timings[child_key]):.3f}s")
                            printed.add(child_key)
        for key, value in timings.items():
            if key not in printed and key != "total":
                print(f"  {key}: {float(value):.3f}s")
        if "total" in timings:
            print(f"  total: {float(timings['total']):.3f}s")


async def main() -> int:
    args = parse_args()
    from pipeline.utils.logger import setup_logging

    setup_logging(level=args.log_level, format_type=args.log_format)
    print("脚本已启动，正在准备参数...", flush=True)

    source_config_ids = [
        item.strip()
        for item in args.source_config_id.split(",")
        if item.strip()
    ]
    if not source_config_ids:
        raise ValueError("--source-config-id 不能为空")

    print("正在导入搜索模块...", flush=True)
    from pipeline.core.storage.elasticsearch import close_es_client
    from pipeline.db import close_database
    from pipeline.modules.search.config import MultiConfig
    from pipeline.modules.search.multi_vector import MultiSearcher

    config = MultiConfig(
        strategy="multi",
        mode=args.mode,
    )

    print("正在初始化 MultiSearcher...", flush=True)
    searcher = MultiSearcher(config=config)
    try:
        print("正在预热搜索依赖...", flush=True)
        await searcher.warmup(config)
        print(
            "开始检索，后续会调用 LLM/Embedding/ES，可能需要等待: "
            f"source_config_ids={source_config_ids}, query={args.query!r}, "
            f"mode={args.mode}, "
            f"timeout={args.timeout_seconds}s",
            flush=True,
        )
        search_task = searcher.search_for_sections(
            query=args.query,
            source_config_ids=source_config_ids,
            config=config,
        )
        if args.timeout_seconds and args.timeout_seconds > 0:
            result = await asyncio.wait_for(search_task, timeout=args.timeout_seconds)
        else:
            result = await search_task
    finally:
        await close_es_client()
        await close_database()

    if args.output == "json":
        print(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2))
    else:
        print_summary(result, args.sort_by)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
