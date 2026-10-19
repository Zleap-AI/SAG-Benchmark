#!/usr/bin/env python3
"""
Pipeline 脚本 - 完全复现 benchmark.py 的 upload 阶段

使用示例（与 benchmark.py 完全一致）：
    # 上传数据集（自动从 pipeline/evaluation/markdown_datasets 目录读取）
    python scripts/run_pipeline.py \
        --foundation upload \
        --dataset hotpotqa \
        --chunk-mode heading_strict
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.utils import (
    RunStatsTracker,
    get_logger,
    llm_tracking_scope,
    llm_tracking_stage,
)

logger = get_logger("scripts.run_pipeline")


# ============================================================
# Upload Dataset -upload_corpus
# ============================================================


async def upload_dataset(args, run_stats: RunStatsTracker | None = None):
    """上传数据集的 markdown 文件到系统"""
    run_stats = run_stats or RunStatsTracker()
    logger.info("=" * 60)
    logger.info("开始上传 corpus 到系统")
    logger.info("=" * 60)

    # 获取当前模型名称
    try:
        from pipeline.core.ai.factory import create_llm_client

        llm_client = await create_llm_client(scenario="extract")
        model_name = (
            llm_client.client.config.model
            if hasattr(llm_client, "client")
            else llm_client.config.model
        )

        # 过滤模型名称，只保留最后一层（例如：Qwen/qwen3 -> qwen3）
        if "/" in model_name:
            filtered_model_name = model_name.split("/")[-1]
        else:
            filtered_model_name = model_name

        logger.info(f"当前使用模型: {model_name} -> 过滤后: {filtered_model_name}")
    except Exception as e:
        logger.warning(f"获取模型名称失败: {e}，使用默认模型名称 'default'")
        filtered_model_name = "default"
        model_name = "default"

    # 使用固定路径
    dataset_name = args.dataset
    md_dir = (
        Path(__file__).parent.parent
        / "pipeline"
        / "evaluation"
        / "markdown_datasets"
        / dataset_name
    )

    if not md_dir.exists():
        error_msg = f"错误：markdown 目录不存在: {md_dir}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

    # 获取所有 md 文件
    md_files = sorted(md_dir.glob("*.md"))

    if not md_files:
        error_msg = f"错误：在 {md_dir} 中未找到 .md 文件"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

    logger.info(f"找到 {len(md_files)} 个 markdown 文件")

    # 1. 生成信息源 ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_config_id = f"{dataset_name}-{timestamp}"

    # 设置默认名称和描述
    source_name = f"{dataset_name}-{timestamp}"
    source_description = f"Evaluation corpus for {dataset_name} dataset"

    logger.info(f"信息源 ID: {source_config_id}")
    logger.info(f"信息源名称: {source_name}")
    logger.info(f"描述: {source_description}\n")

    # ── 解析 embedding 维度并设为进程级活跃维度 ───────────────────
    # Extract 路由预检必须早于 embedding probe 和数据写入。
    atomic_mode = getattr(args, "atomic", False)
    requested_extract_strategy = getattr(args, "extract_prompt_strategy", None)
    extract_strategy = requested_extract_strategy or ("original" if atomic_mode else "compact")
    from pipeline.modules.extract.config import ExtractPromptStrategy
    from pipeline.modules.extract.prompt_router import resolve_extract_prompt_route

    extract_route = resolve_extract_prompt_route(
        ExtractPromptStrategy(extract_strategy),
        test_mode=atomic_mode,
    )

    from pipeline.core.ai.embedding_dim import resolve_embedding_dim
    from pipeline.storage.indexing import (
        assert_indices_ready,
        index_suffix,
        resolve_all_index_names,
        set_active_embedding_dim,
    )

    dim_info = await resolve_embedding_dim(
        force_probe=getattr(args, "refresh_embedding_dim", False)
    )
    embedding_dim = dim_info["dim"]
    set_active_embedding_dim(embedding_dim)
    es_indices = resolve_all_index_names(embedding_dim)
    logger.info(
        f"🧭 embedding 维度: {embedding_dim} (来源={dim_info['dim_source']}, "
        f"model={dim_info['model']})"
    )
    logger.info(f"   ES 索引映射: {es_indices}")

    # ── ES 索引预检：必须已存在且 dims 匹配，防止 auto_create 生成野生索引 ──
    from pipeline.core.config.settings import get_settings
    from pipeline.exceptions import StorageError

    if get_settings().effective_vector_backend == "elasticsearch":
        try:
            await assert_indices_ready(embedding_dim)
        except StorageError as e:
            logger.error(str(e))
            return {"status": "error", "message": str(e)}

    # 提取模式提示：--atomic 打开原子事项（test_extract.yaml，每事项2实体），否则融合大事项
    try:
        _lang = get_settings().llm_language
    except Exception:
        _lang = "unknown"
    logger.info(
        f"提取模式: {'原子事项 (test_extract)' if atomic_mode else '融合大事项 (extract)'} "
        f"| test_mode={atomic_mode} | llm_language={_lang}"
    )
    logger.info(
        f"Extract prompt strategy: {extract_strategy} (template={extract_route.template_name})"
    )

    # 2. 创建 PipelineEngine
    from pipeline import ExtractBaseConfig, PipelineEngine
    from pipeline.engine.config import TaskConfig
    from pipeline.modules.load.config import DocumentLoadConfig

    task_config = TaskConfig(
        task_name=f"Upload {dataset_name} Corpus",
        source_config_id=source_config_id,
        source_name=source_name,
    )

    # 3. 创建 PipelineEngine
    engine = PipelineEngine(task_config=task_config)

    # 4. 初始化统计
    file_results = []
    total_sections = 0
    total_events = 0
    total_files_processed = 0

    enable_extraction = getattr(args, "enable_extraction", True)

    # 用于记录每个文档的LLM Token统计
    file_token_stats = []

    # 获取 chunk_mode
    chunk_mode = getattr(args, "chunk_mode", "heading_strict")

    for idx, md_file in enumerate(md_files, 1):
        file_token_before = run_stats.token_snapshot()
        logger.info(f"[{idx}/{len(md_files)}] 处理文件: {md_file.name}")
        file_size_mb = md_file.stat().st_size / 1024 / 1024
        logger.info(f"  文件大小: {file_size_mb:.2f} MB")

        # Load 阶段 - 加载文档
        try:
            with run_stats.measure("load") as load_measure:
                with llm_tracking_stage("LOAD"):
                    await engine.load_async(
                        DocumentLoadConfig(
                            path=str(md_file),
                            recursive=False,
                            source_config_id=source_config_id,
                            chunk_mode=chunk_mode,
                        )
                    )
            load_time = load_measure.elapsed_seconds
            logger.info(f"  ✓ 文档加载完成，耗时: {load_time:.1f} 秒")
        except Exception as e:
            error_msg = f"文档加载失败 ({md_file.name}): {e}"
            logger.error(error_msg, exc_info=True)
            file_results.append({"file": md_file.name, "status": "error", "message": str(e)})
            continue

        # 获取 Load 结果
        engine_result = engine.get_result()
        if not engine_result or not engine_result.load_result:
            error_msg = f"Load 阶段失败：无法获取加载结果 ({md_file.name})"
            logger.error(error_msg)
            file_results.append({"file": md_file.name, "status": "error", "message": error_msg})
            continue

        # 从 engine_result 获取数据
        try:
            article_id = engine_result.article_id
            load_result = engine_result.load_result
            sections_count = load_result.stats.get("chunk_count", 0) if load_result.stats else 0
            total_sections += sections_count

            logger.info(f"  Article ID: {article_id}")
            logger.info(f"  文档片段数: {sections_count}")
        except Exception as e:
            error_msg = f"读取 Load 结果失败 ({md_file.name}): {e}"
            logger.error(error_msg, exc_info=True)
            file_results.append({"file": md_file.name, "status": "error", "message": str(e)})
            continue

        events_count = 0
        extract_time = 0.0

        # Extract 阶段 - 提取事项（可选）
        if enable_extraction:
            logger.info("  开始提取事项...")

            try:
                with run_stats.measure("extract") as extract_measure:
                    with llm_tracking_stage("EXTRACT"):
                        await engine.extract_async(
                            ExtractBaseConfig(
                                parallel=True,
                                enable_entity_vector_sync=True,
                                enable_event_entity_vector_sync=True,
                                test_mode=getattr(args, "atomic", False),
                                extract_prompt_strategy=extract_strategy,
                            )
                        )
                extract_time = extract_measure.elapsed_seconds
                logger.info(f"  ✓ 事项提取完成，耗时: {extract_time:.1f} 秒")

                # 获取 Extract 结果
                engine_result = engine.get_result()
                if engine_result and engine_result.extract_result:
                    extract_result = engine_result.extract_result
                    events_count = len(extract_result.data_ids) if extract_result.data_ids else 0
                    total_events += events_count
                    logger.info(f"  生成事项数: {events_count}")
                else:
                    logger.warning("  ⚠️  Extract 结果为空")
            except Exception as e:
                error_msg = f"事项提取失败 ({md_file.name}): {e}"
                logger.error(error_msg, exc_info=True)
                # 提取失败不返回错误，因为 Load 已经成功
        else:
            logger.info("  跳过提取阶段（enable_extraction=False）")

        # 记录文件处理结果
        file_results.append(
            {
                "file": md_file.name,
                "article_id": article_id,
                "sections_count": sections_count,
                "events_count": events_count,
                "status": "completed",
            }
        )

        # 如果是提取模式，记录该文件的token统计
        if enable_extraction:
            file_token_after = run_stats.token_snapshot()
            token_delta = run_stats.token_delta(file_token_before, file_token_after)
            file_tokens = {
                "file": md_file.name,
                "input_tokens": token_delta["input_tokens"],
                "output_tokens": token_delta["output_tokens"],
                "total_tokens": token_delta["total_tokens"],
                "calls": token_delta["calls"],
                # Backward-compatible aliases.
                "prompt_tokens": token_delta["input_tokens"],
                "completion_tokens": token_delta["output_tokens"],
                "processing_time": {
                    "load_time": round(load_time, 3),
                    "extract_time": round(extract_time, 3),
                },
            }
            file_token_stats.append(file_tokens)

            # 仅在有token消耗时显示
            if token_delta["total_tokens"] > 0:
                logger.info(
                    f"  📊 LLM Tokens: 输入={token_delta['input_tokens']:,}, "
                    f"输出={token_delta['output_tokens']:,}, "
                    f"总计={token_delta['total_tokens']:,}"
                )

        total_files_processed += 1
        logger.info("  ✓ 文件处理完成\n")

    # 5. 保存结果到 pipeline/evaluation/source/SAG/{filtered_model_name}/{dataset_name}/{timestamp}/
    source_dir = (
        Path("pipeline/evaluation/source/SAG") / filtered_model_name / dataset_name / timestamp
    )
    source_dir.mkdir(parents=True, exist_ok=True)

    # 获取统一的 upload 时间与 token 统计，同时保留旧字段兼容已有消费者。
    upload_stats = run_stats.get_summary()
    token_summary = run_stats.token_tracker.get_summary()
    timing_stages = upload_stats["timing"]["stages"]
    total_time_load = timing_stages.get("load", {}).get("seconds", 0.0)
    total_time_extract = timing_stages.get("extract", {}).get("seconds", 0.0)

    result = {
        "source_config_id": source_config_id,
        "source_name": source_name,
        "source_description": source_description,
        "dataset_name": dataset_name,
        "model_name": model_name,  # 添加原始模型名称
        "filtered_model_name": filtered_model_name,  # 添加过滤后的模型名称
        "file_count": len(md_files),
        "successful_files": total_files_processed,
        "failed_files": len([r for r in file_results if r["status"] == "error"]),
        "total_sections_count": total_sections,
        "total_events_count": total_events,
        "processing_time": {
            "total_load_time": round(total_time_load, 3),
            "total_extract_time": round(total_time_extract, 3),
            "total_time": round(total_time_load + total_time_extract, 3),
        },
        "file_results": file_results,
        "timestamp": timestamp,
        "status": "completed",
        "extraction_enabled": enable_extraction,
        "embedding": {
            "model": dim_info["model"],
            "base_url": dim_info["base_url"],
            "dim": embedding_dim,
            "requested_dimensions": dim_info["requested_dimensions"],
            "expected_dim": dim_info["expected_dim"],
            "dim_source": dim_info["dim_source"],
            "probed_at": dim_info["probed_at"],
            "es_index_suffix": index_suffix(embedding_dim),
            "es_indices": es_indices,
        },
        "llm_token_usage": token_summary,  # 添加LLM token使用统计
        "upload_stats": upload_stats,
        "file_token_stats": file_token_stats,  # 每个文件的token统计
    }

    # 保存到 source_info.json
    source_info_path = source_dir / "source_info.json"
    with open(source_info_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 信息源结果已保存: {source_info_path}")

    # 返回结果
    logger.info("=" * 60)
    logger.info("✅ Corpus 上传完成")
    logger.info(f"  总文件数: {len(md_files)}")
    logger.info(f"  成功: {result['successful_files']}")
    logger.info(f"  失败: {result['failed_files']}")
    logger.info(f"  总片段数: {total_sections}")
    logger.info(f"  总事项数: {total_events}")
    logger.info(
        f"  结果保存位置: pipeline/evaluation/source/SAG/{filtered_model_name}/{dataset_name}/{timestamp}/"
    )
    logger.info("=" * 60)

    # 主动关闭数据库连接和AI客户端，避免 "Event loop is closed" 警告
    try:
        logger.info("关闭数据库连接和AI客户端...")
        # 关闭数据库连接
        from pipeline.db import close_database
        from pipeline.storage import close_storage_facade

        await close_storage_facade()
        await close_database()

        logger.info("✓ 所有连接已关闭")
    except Exception as e:
        logger.warning(f"关闭连接时出现警告: {e}")

    return result


# ============================================================
# Main
# ============================================================


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Pipeline 脚本 - 完全复现 benchmark.py 的 upload 阶段"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "musique",
            "hotpotqa",
            "2wikimultihopqa",
            "sample",
            "test_hotpotqa",
            "narrativeqa",
        ],
        help="数据集名称",
    )

    parser.add_argument(
        "--foundation",
        type=str,
        choices=["upload"],
        default="upload",
        help="Foundation mode: 'upload' to load and upload corpus",
    )

    parser.add_argument(
        "--chunks-per-file",
        type=int,
        default=500,
        help="每个 markdown 文件包含的文档数量（默认：500）",
    )

    parser.add_argument(
        "--chunk-mode",
        type=str,
        default="heading_strict",
        choices=["standard", "heading_strict", "overlap"],
        help="分块模式（默认：heading_strict）",
    )

    parser.add_argument(
        "--enable-extraction", action="store_true", default=True, help="启用事项提取（默认：True）"
    )

    parser.add_argument(
        "--no-extraction", action="store_false", dest="enable_extraction", help="禁用事项提取"
    )

    parser.add_argument(
        "--atomic",
        action="store_true",
        default=False,
        help="打开原子事项模式（读取 test_extract.yaml，每个事项恰好2个实体）。不加则默认融合大事项",
    )

    parser.add_argument(
        "--extract-prompt-strategy",
        choices=["original", "compact"],
        default=None,
        # Keep the old override parseable for reproducibility, but do not expose
        # the retired prompt strategy in the public upload help. The supported
        # user-facing modes are: default compact, --atomic, and --no-extraction.
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--refresh-embedding-dim",
        action="store_true",
        help="忽略维度缓存，强制重新 probe embedding 服务端维度",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="启用 verbose 模式：在日志中打印每次 LLM 调用的完整输入 prompt 和输出 response",
    )

    args = parser.parse_args()
    run_stats = RunStatsTracker()

    # 启用 verbose LLM 日志
    if args.verbose:
        from pipeline.utils import enable_llm_verbose

        enable_llm_verbose()
        logger.info("🔍 Verbose 模式已启用：将打印每次 LLM 调用的完整输入和输出")

    try:
        # 步骤1：生成 markdown 文件
        logger.info("=" * 70)
        logger.info("步骤 1/2: 生成 Markdown 文件")
        logger.info("=" * 70)

        from pipeline.evaluation.utils import DatasetLoader

        loader = DatasetLoader(args.dataset)
        with run_stats.measure("markdown") as markdown_measure:
            save_result = loader.save_as_markdown(
                chunks_per_file=args.chunks_per_file, force_regenerate=True
            )
        load_time = markdown_measure.elapsed_seconds

        logger.info("✓ Markdown 文件生成完成")
        logger.info(f"  输出目录: {save_result['output_dir']}")
        logger.info(f"  总文档数: {save_result['stats']['total_chunks']:,}")
        logger.info(f"  文件数: {save_result['stats']['num_files']} 个")
        logger.info(f"  每文件文档数: {args.chunks_per_file}")
        if save_result["stats"].get("last_file_chunks"):
            logger.info(f"  最后一个文件文档数: {save_result['stats']['last_file_chunks']} 个")
        logger.info(f"  耗时: {load_time:.1f} 秒")
        logger.info("=" * 70 + "\n")

        # 步骤2：上传到系统
        logger.info("=" * 70)
        logger.info("步骤 2/2: 上传 Corpus 到系统")
        logger.info("=" * 70)

        with llm_tracking_scope(run_stats.token_tracker):
            logger.info("✅ LLM token 追踪已启用（当前运行上下文）")
            result = await upload_dataset(args, run_stats=run_stats)

        if result["status"] == "completed":
            # 显示完整的上传结果统计
            logger.info("\n" + "=" * 70)
            logger.info("✅ 完整流程执行完成")
            logger.info("=" * 70)
            logger.info(f"数据集: {result['dataset_name']}")
            logger.info(f"Source Config ID: {result['source_config_id']}")
            logger.info(f"文件数: {result['file_count']}")
            logger.info(f"成功: {result['successful_files']}")
            logger.info(f"失败: {result['failed_files']}")
            logger.info(f"总片段数: {result['total_sections_count']:,}")
            logger.info(f"总事项数: {result['total_events_count']:,}")

            # 显示处理时间
            if "processing_time" in result:
                time_info = result["processing_time"]
                logger.info("\n处理时间:")
                logger.info(f"  Markdown 生成: {load_time:.1f} 秒")
                logger.info(f"  Load 阶段: {time_info['total_load_time']:.1f} 秒")
                logger.info(f"  Extract 阶段: {time_info['total_extract_time']:.1f} 秒")
                logger.info(f"  总计: {load_time + time_info['total_time']:.1f} 秒")

            # 显示 LLM Token 消耗统计
            if "llm_token_usage" in result and result["llm_token_usage"]["total_tokens"] > 0:
                token_summary = result["llm_token_usage"]
                logger.info("\n" + "=" * 70)
                logger.info("LLM Token 消耗统计")
                logger.info("=" * 70)
                logger.info(f"总调用次数: {token_summary['total_calls']:,}")
                logger.info(f"总输入 Tokens: {token_summary['total_prompt']:,}")
                logger.info(f"总输出 Tokens: {token_summary['total_completion']:,}")
                logger.info(f"总 Tokens: {token_summary['total_tokens']:,}")

                # 按阶段显示
                if token_summary.get("stages"):
                    logger.info("\n按阶段统计:")
                    for stage, stats in token_summary["stages"].items():
                        logger.info(f"\n  {stage}:")
                        logger.info(f"    调用次数: {stats['calls']:,}")
                        logger.info(f"    输入 Tokens: {stats['prompt']:,}")
                        logger.info(f"    输出 Tokens: {stats['completion']:,}")
                        logger.info(f"    总计 Tokens: {stats['total']:,}")

            logger.info("=" * 70 + "\n")
            return 0
        else:
            logger.error(f"上传失败: {result.get('message', 'Unknown error')}")
            return 1

    finally:
        # 确保关闭所有资源
        from pipeline.core.ai.factory import close_all_clients
        from pipeline.db import close_database
        from pipeline.storage import close_storage_facade

        # 等待所有待处理的任务完成
        await asyncio.sleep(0.1)

        try:
            await close_all_clients()
        except Exception as e:
            logger.warning(f"关闭AI客户端时出错: {e}")

        try:
            await close_storage_facade()
        except Exception as e:
            logger.warning(f"关闭存储后端时出错: {e}")

        try:
            await close_database()
        except Exception as e:
            logger.warning(f"关闭数据库时出错: {e}")

        # 再次等待，确保所有清理完成
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
