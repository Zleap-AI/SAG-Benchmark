"""
Elasticsearch 索引初始化脚本

创建所有 ES 索引并验证
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.storage.backends.elasticsearch.client import ElasticsearchClient
from pipeline.storage.backends.elasticsearch.documents import REGISTERED_DOCUMENTS
from pipeline.storage.backends.elasticsearch.index_naming import (
    IndexDimMismatchError,
    assert_index_dims,
    resolve_index_name,
    with_dense_vector_dims,
)
from pipeline.utils import get_logger

logger = get_logger("scripts.init_elasticsearch")


# 输出辅助函数
def print_header(text: str) -> None:
    """打印标题"""
    logger.info("\n" + "=" * 70)
    logger.info(f"  {text}")
    logger.info("=" * 70)


def print_success(text: str) -> None:
    """打印成功信息"""
    logger.info(f"  ✓ {text}")


def print_info(text: str) -> None:
    """打印普通信息"""
    logger.info(f"  • {text}")


def print_warning(text: str) -> None:
    """打印警告信息"""
    logger.warning(f"  ⚠️  {text}")


def print_error(text: str) -> None:
    """打印错误信息"""
    logger.error(f"  ✗ {text}")


async def create_indices(es_client: ElasticsearchClient, dim: int) -> dict[str, str]:
    """
    创建所有 ES 索引（按运行时 embedding 维度）。

    - 索引名：base + 维度后缀（1024 且开启 legacy 兼容时无后缀）
    - mapping：deepcopy 后改写所有 dense_vector 的 dims
    - 已存在：校验 dims 一致；不一致 fail-fast（ES 不允许改 dims）

    Returns:
        dict: 索引名 -> 状态 ("created", "skipped", "failed", "dim_mismatch")
    """
    print_header(f"创建索引 (embedding_dim={dim})")

    results = {}

    for document_cls in REGISTERED_DOCUMENTS:
        try:
            # 从 Document 类获取索引配置
            base_name = getattr(document_cls, "BASE_INDEX_NAME", document_cls.Index.name)
            index_name = resolve_index_name(base_name, dim)
            mapping = with_dense_vector_dims(document_cls._doc_type.mapping.to_dict(), dim)
            settings = getattr(document_cls.Index, "settings", {})
        except AttributeError as e:
            print_error(f"Document 类 {document_cls.__name__} 配置获取失败: {e}")
            results[document_cls.__name__] = "failed"
            continue

        # 检查索引是否已存在
        exists = await es_client.index_exists(index_name)

        if exists:
            try:
                await assert_index_dims(es_client, index_name, dim)
            except IndexDimMismatchError as e:
                print_error(str(e))
                results[index_name] = "dim_mismatch"
                continue
            print_info(f"{index_name}: 已存在且维度匹配 (dims={dim})，跳过创建")
            results[index_name] = "skipped"
            continue

        # 创建索引
        try:
            print_info(f"{index_name}: 开始创建 (dims={dim})...")
            await es_client.create_index(index=index_name, mappings=mapping, settings=settings)
            print_success(f"{index_name}: 创建成功")
            results[index_name] = "created"
        except Exception as e:
            print_error(f"{index_name}: 创建失败 - {e}")
            results[index_name] = "failed"

    return results


async def verify_indices(es_client: ElasticsearchClient, dim: int) -> bool:
    """
    验证所有索引是否创建成功

    Returns:
        bool: 是否所有索引都验证通过
    """
    print_header("验证索引")

    all_success = True

    for document_cls in REGISTERED_DOCUMENTS:
        try:
            base_name = getattr(document_cls, "BASE_INDEX_NAME", document_cls.Index.name)
            index_name = resolve_index_name(base_name, dim)
        except AttributeError as e:
            print_error(f"Document 类 {document_cls.__name__} 索引名称获取失败: {e}")
            all_success = False
            continue

        exists = await es_client.index_exists(index_name)

        if exists:
            # 额外校验维度一致性
            try:
                await assert_index_dims(es_client, index_name, dim)
                print_success(f"{index_name}: 验证通过 (dims={dim})")
            except IndexDimMismatchError as e:
                print_error(str(e))
                all_success = False
                continue
        else:
            print_error(f"{index_name}: 验证失败，索引不存在")
            all_success = False

    return all_success


async def main() -> None:
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="初始化 Elasticsearch 索引")
    parser.add_argument("--dim", type=int, default=None, help="强制指定维度（跳过 probe）")
    parser.add_argument("--refresh-dim", action="store_true", help="忽略缓存，强制重新 probe")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    es_client = None

    try:
        print_header("New SAG Elasticsearch 索引初始化")

        # 1. 创建 ES 客户端
        print_info("正在连接 Elasticsearch...")
        es_client = ElasticsearchClient()

        # 2. 检查连接
        if not await es_client.check_connection():
            print_error("Elasticsearch 连接失败，请检查配置")
            raise Exception("ES连接失败，请检查配置")

        print_success("Elasticsearch 连接成功")

        # 2.5 解析 embedding 维度
        if args.dim:
            dim = args.dim
            print_info(f"使用命令行指定维度: dim={dim}（跳过 probe）")
        else:
            from pipeline.core.ai.embedding_dim import resolve_embedding_dim

            dim_info = await resolve_embedding_dim(force_probe=args.refresh_dim)
            dim = dim_info["dim"]
            print_success(
                f"embedding 维度: {dim} (来源={dim_info['dim_source']}, model={dim_info['model']})"
            )

        # 3. 创建索引
        create_results = await create_indices(es_client, dim)

        # 4. 验证索引
        verify_success = await verify_indices(es_client, dim)

        # 5. 总结
        print_header("操作总结")

        created_count = sum(1 for status in create_results.values() if status == "created")
        skipped_count = sum(1 for status in create_results.values() if status == "skipped")
        failed_count = sum(1 for status in create_results.values() if status == "failed")
        mismatch_count = sum(1 for status in create_results.values() if status == "dim_mismatch")

        if created_count > 0:
            print_success(f"新创建索引: {created_count} 个")
        if skipped_count > 0:
            print_info(f"跳过索引: {skipped_count} 个（已存在）")
        if mismatch_count > 0:
            print_error(f"维度不匹配索引: {mismatch_count} 个")
        if failed_count > 0:
            print_error(f"失败索引: {failed_count} 个")

        if verify_success and failed_count == 0 and mismatch_count == 0:
            print_success("所有索引初始化成功！")
        else:
            print_error("部分索引初始化失败，请查看详细信息")
            raise Exception("索引初始化未完全成功")

        logger.info("=" * 70 + "\n")

    except Exception as e:
        print_error(f"索引初始化失败: {e}")
        sys.exit(1)

    finally:
        # 关闭连接
        if es_client:
            await es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
