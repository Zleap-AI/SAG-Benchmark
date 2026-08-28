#!/usr/bin/env python3
"""
计算 SAG2 知识图谱的 Indexing 指标。

从 MySQL 数据库中读取指定 source_config 的 event 和 entity 数据，
构建无向图（event 和 entity 均为节点，event_entity 关联为无向边），
然后计算与 GraphRAG-Benchmark/Evaluation/indexing_eval.py 一致的结构化指标。

用法:
    uv run python scripts/calculate_graph_metrics.py \
        --source-config <source_config_id>

数据库连接默认读取项目 Settings/.env，也可通过 CLI 参数显式覆盖。
"""

import argparse
import json
import time
from pathlib import Path

import igraph as ig
import pymysql

from pipeline.core.config.settings import get_settings
from pipeline.evaluation.judge.indexing import analyze_graph as analyze_reference_graph

# ── 图指标计算 ──


def analyze_graph(g: ig.Graph) -> dict:
    """计算基准 20 项指标，并附加 SAG2 专属节点类型/度分布字段。"""
    metrics = analyze_reference_graph(g)
    degrees = g.degree(mode="all")
    degree_values = sorted(set(degrees))
    degree_distribution = {int(d): sum(1 for dd in degrees if dd == d) for d in degree_values}
    node_types = g.vs["node_type"] if "node_type" in g.vs.attributes() else []
    metrics.update(
        {
            "num_events": sum(1 for node_type in node_types if node_type == "event"),
            "num_entities": sum(1 for node_type in node_types if node_type == "entity"),
            "max_degree": max(degrees) if degrees else 0,
            "degree_distribution_top20": dict(
                sorted(degree_distribution.items(), key=lambda item: item[0])[:20]
            ),
        }
    )
    return metrics


# ── 数据加载 ──


def load_graph_from_mysql(
    conn: pymysql.Connection,
    source_config_id: str,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    从 MySQL 加载指定 source_config 的 events 和 entities，
    返回节点列表和边列表。

    Args:
        conn: pymysql 连接
        source_config_id: 数据源配置 ID

    Returns:
        (nodes, edges) 元组
        - nodes: [(node_id, node_type, name), ...] node_type: "event" 或 "entity"
        - edges: [(event_id, entity_id), ...]
    """
    cursor = conn.cursor()

    # 加载 events
    print(f"[1/3] 加载 events for source_config '{source_config_id}' ...")
    t0 = time.time()
    cursor.execute(
        "SELECT id, title FROM source_event WHERE source_config_id = %s",
        (source_config_id,),
    )
    events = cursor.fetchall()
    print(f"      加载 {len(events)} 个 events ({time.time() - t0:.1f}s)")

    # 加载 entities
    print(f"[2/3] 加载 entities for source_config '{source_config_id}' ...")
    t0 = time.time()
    cursor.execute(
        "SELECT id, name FROM entity WHERE source_config_id = %s",
        (source_config_id,),
    )
    entities = cursor.fetchall()
    print(f"      加载 {len(entities)} 个 entities ({time.time() - t0:.1f}s)")

    # 构建节点列表
    nodes = []
    # event nodes
    for eid, title in events:
        label = title if title else eid[:20]
        nodes.append((eid, "event", label))
    # entity nodes
    for eid, name in entities:
        label = name if name else eid[:20]
        nodes.append((eid, "entity", label))

    # 加载 event_entity 关联
    print("[3/3] 加载 event_entity 关联 ...")
    t0 = time.time()

    # 分批加载，避免一次性加载所有数据导致内存压力
    edges = []
    batch_size = 50000
    offset = 0
    while True:
        cursor.execute(
            "SELECT event_id, entity_id FROM event_entity "
            "WHERE event_id IN (SELECT id FROM source_event WHERE source_config_id = %s) "
            "AND entity_id IN (SELECT id FROM entity WHERE source_config_id = %s) "
            "LIMIT %s OFFSET %s",
            (source_config_id, source_config_id, batch_size, offset),
        )
        batch = cursor.fetchall()
        if not batch:
            break
        edges.extend(batch)
        offset += batch_size
        print(f"      已加载 {offset} 条边 ...", end="\r")

    print(f"\n      加载 {len(edges)} 条关联边 ({time.time() - t0:.1f}s)")

    cursor.close()
    return nodes, edges


def build_graph(nodes: list[tuple[str, str, str]], edges: list[tuple[str, str]]) -> ig.Graph:
    """
    根据节点和边列表构建 igraph 无向图。

    Args:
        nodes: [(node_id, node_type, label), ...]
        edges: [(event_id, entity_id), ...]

    Returns:
        igraph.Graph 对象（无向图）
    """
    print("\n构建 igraph 图 ...")
    t0 = time.time()

    # 建立 node_id → index 的映射
    id_to_idx = {nid: i for i, (nid, _, _) in enumerate(nodes)}

    g = ig.Graph()
    g.add_vertices(len(nodes))

    # 设置节点属性
    g.vs["name"] = [n[0] for n in nodes]
    g.vs["node_type"] = [n[1] for n in nodes]
    g.vs["label"] = [n[2] for n in nodes]

    # 添加无向边
    # 只添加两端节点都存在的边
    valid_edges = []
    missing = 0
    for event_id, entity_id in edges:
        if event_id in id_to_idx and entity_id in id_to_idx:
            valid_edges.append((id_to_idx[event_id], id_to_idx[entity_id]))
        else:
            missing += 1

    if missing > 0:
        print(f"      警告: {missing} 条边的端点不在节点列表中，已跳过")

    g.add_edges(valid_edges)

    print(f"      构建完成: {g.vcount()} 个节点, {g.ecount()} 条边 ({time.time() - t0:.1f}s)")
    return g


# ── 输出格式化 ──


def print_metrics(metrics: dict, title: str = "Graph Indexing Metrics"):
    """打印指标到控制台"""
    width = 80
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)

    # 分组打印
    groups = [
        ("基本规模", ["num_nodes", "num_events", "num_entities", "num_edges"]),
        (
            "度数分析",
            [
                "average_degree",
                "max_degree",
                "num_isolated_nodes",
                "num_nodes_excluding_isolated",
                "num_nodes_degree_above_1",
                "num_nodes_degree_above_2",
                "num_nodes_degree_above_3",
            ],
        ),
        ("密度与聚类", ["density", "average_clustering_coefficient", "diameter"]),
        (
            "连通分量",
            [
                "num_components",
                "largest_component_size",
                "num_components_excluding_isolated",
                "num_components_above_average",
                "num_nodes_excluding_isolated_comp",
            ],
        ),
        (
            "分量大小分布",
            [
                "average_component_size",
                "median_component_size",
                "trimmed_mean_component_size",
                "geometric_mean_component_size",
                "harmonic_mean_component_size",
            ],
        ),
    ]

    for group_name, keys in groups:
        print(f"\n  ── {group_name} ──")
        for key in keys:
            if key in metrics:
                value = metrics[key]
                print(f"  {key:40s}: {value}")

    # 度分布 top 20
    if "degree_distribution_top20" in metrics:
        dd = metrics["degree_distribution_top20"]
        print("\n  ── 度分布 Top-20 ──")
        print(f"  {'degree':>8s}  {'count':>8s}")
        print(f"  {'-' * 8}  {'-' * 8}")
        for deg, cnt in list(dd.items())[:20]:
            print(f"  {int(deg):8d}  {int(cnt):8d}")

    print()
    print("=" * width)


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="计算 SAG2 知识图谱的 Indexing 指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 指定 source_config
  uv run python scripts/calculate_graph_metrics.py --source-config <source_config_id>

  # 保存 JSON 结果
  uv run python scripts/calculate_graph_metrics.py \
      --source-config <source_config_id> --output <output.json>

数据库连接默认读取项目 Settings/.env，也可通过对应 CLI 参数显式覆盖。
        """,
    )

    parser.add_argument(
        "--source-config",
        type=str,
        required=True,
        help="要计算的 source_config_id",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.mysql_host,
        help="MySQL 主机（默认读取 MYSQL_HOST）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.mysql_port,
        help="MySQL 端口（默认读取 MYSQL_PORT）",
    )
    parser.add_argument(
        "--user",
        type=str,
        default=settings.mysql_user,
        help="MySQL 用户名（默认读取 MYSQL_USER）",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=settings.mysql_password,
        help="MySQL 密码（默认读取 MYSQL_PASSWORD）",
    )
    parser.add_argument(
        "--database",
        type=str,
        default=settings.mysql_database,
        help="MySQL 数据库名（默认读取 MYSQL_DATABASE）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出 JSON 文件路径（可选）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("  SAG2 知识图谱 Indexing 指标计算")
    print("=" * 80)
    print(f"  Source Config: {args.source_config}")
    print(f"  MySQL: {args.user}@{args.host}:{args.port}/{args.database}")
    print()

    # 连接 MySQL
    print("连接 MySQL ...")
    t_start = time.time()
    conn = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        database=args.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
    )
    print(f"  已连接 ({time.time() - t_start:.1f}s)")

    try:
        # 加载数据
        nodes, edges = load_graph_from_mysql(conn, args.source_config)

        # 构建图
        g = build_graph(nodes, edges)

        # 计算指标
        print("\n计算图指标 ...")
        t0 = time.time()
        metrics = analyze_graph(g)
        print(f"  计算完成 ({time.time() - t0:.1f}s)")

        # 添加元数据
        metrics["metadata"] = {
            "source_config_id": args.source_config,
            "graph_type": "undirected bipartite (event-entity)",
            "total_nodes": g.vcount(),
            "total_edges": g.ecount(),
        }

        # 打印结果
        print_metrics(metrics, f"Indexing Metrics — {args.source_config}")

        # 保存 JSON
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"✅ 结果已保存: {output_path}")

        total_time = time.time() - t_start
        print(f"\n总耗时: {total_time:.1f}s")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
