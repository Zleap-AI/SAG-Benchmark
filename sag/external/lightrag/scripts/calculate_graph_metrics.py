#!/usr/bin/env python3
"""计算 LightRAG 知识图谱的 Indexing 指标。

从 caches/ 读取 graph_chunk_entity_relation.graphml，计算 20 个图结构指标。
指标定义与 GraphRAG-Benchmark/Evaluation/indexing_eval.py 完全一致。

用法:
    cd external/hypergraphrag
    uv run python scripts/calculate_graph_metrics.py --data_name test_hotpotqa

    # 指定图文件根目录
    uv run python scripts/calculate_graph_metrics.py --data_name musique --dataset_root caches

    # 保存结果到文件
    uv run python scripts/calculate_graph_metrics.py --data_name test_hotpotqa --output results.json
"""

import argparse
import json
import os
import warnings

import igraph as ig
import numpy as np

# ── 图指标计算（与 GraphRAG-Benchmark/Evaluation/indexing_eval.py 一致）──


def analyze_graph(g: ig.Graph) -> dict:
    num_nodes = g.vcount()
    num_edges = g.ecount()
    average_degree = sum(g.degree()) / num_nodes if num_nodes > 0 else 0
    density = g.density()
    components = g.components()
    num_components = len(components)
    largest_component_size = components.giant().vcount()
    average_clustering_coefficient = g.transitivity_avglocal_undirected()
    diameter = g.diameter() if g.is_connected() else float("inf")

    component_sizes = [len(c) for c in components if len(c) > 1]
    if component_sizes:
        average_component_size = sum(component_sizes) / len(component_sizes)
        median_component_size = np.median(component_sizes)
        num_components_excluding_isolated = len(component_sizes)
        num_components_above_average = sum(1 for s in component_sizes if s > average_component_size)
        num_nodes_excluding_isolated = sum(component_sizes)
        sorted_sizes = sorted(component_sizes)
        trimmed_mean_component_size = (
            sum(sorted_sizes[1:-1]) / (len(sorted_sizes) - 2)
            if len(sorted_sizes) > 2
            else average_component_size
        )
        geometric_mean_component_size = (
            np.exp(np.mean(np.log(component_sizes))) if len(component_sizes) > 0 else 0
        )
        harmonic_mean_component_size = (
            len(component_sizes) / sum(1.0 / s for s in component_sizes)
            if len(component_sizes) > 0
            else 0
        )
    else:
        average_component_size = 0
        median_component_size = 0
        num_components_excluding_isolated = 0
        num_components_above_average = 0
        num_nodes_excluding_isolated = 0
        trimmed_mean_component_size = 0
        geometric_mean_component_size = 0
        harmonic_mean_component_size = 0

    degrees = g.degree(mode="all")
    num_isolated_nodes = sum(1 for d in degrees if d == 0)
    num_nodes_excluding_isolated = sum(1 for d in degrees if d > 0)
    num_nodes_degree_above_1 = sum(1 for d in degrees if d > 1)
    num_nodes_degree_above_2 = sum(1 for d in degrees if d > 2)
    num_nodes_degree_above_3 = sum(1 for d in degrees if d > 3)

    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "average_degree": float(average_degree),
        "density": float(density),
        "num_components": float(num_components),
        "largest_component_size": float(largest_component_size),
        "average_clustering_coefficient": float(average_clustering_coefficient),
        "diameter": float(diameter),
        "average_component_size": float(average_component_size),
        "median_component_size": float(median_component_size),
        "trimmed_mean_component_size": float(trimmed_mean_component_size),
        "geometric_mean_component_size": float(geometric_mean_component_size),
        "harmonic_mean_component_size": float(harmonic_mean_component_size),
        "num_components_excluding_isolated": float(num_components_excluding_isolated),
        "num_components_above_average": float(num_components_above_average),
        "num_nodes_excluding_isolated": float(num_nodes_excluding_isolated),
        "num_isolated_nodes": float(num_isolated_nodes),
        "num_nodes_degree_above_1": float(num_nodes_degree_above_1),
        "num_nodes_degree_above_2": float(num_nodes_degree_above_2),
        "num_nodes_degree_above_3": float(num_nodes_degree_above_3),
    }


# ── 图加载 ──


def load_graph(data_name: str, dataset_root: str) -> ig.Graph:
    path = os.path.join(dataset_root, data_name, "graph_chunk_entity_relation.graphml")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"图文件不存在: {path}\n请先运行 reproduce/Step_1_build_index.py 构建索引"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ig.Graph.Read_GraphML(path)
    return g


# ── 主流程 ──


def main() -> int:
    parser = argparse.ArgumentParser(description="LightRAG 图指标计算（对标 judge indexing_eval）")
    parser.add_argument("--data_name", type=str, required=True, help="数据集名")
    parser.add_argument("--dataset_root", type=str, default="caches", help="图文件根目录")
    parser.add_argument("--output", type=str, default=None, help="JSON 输出路径")
    parser.add_argument("--verbose", action="store_true", help="详细打印")
    args = parser.parse_args()

    g = load_graph(args.data_name, args.dataset_root)
    if args.verbose:
        print(f"[{args.data_name}] nodes={g.vcount()} edges={g.ecount()}")

    metrics = analyze_graph(g)
    metrics["dataset"] = args.data_name

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"已保存 → {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
