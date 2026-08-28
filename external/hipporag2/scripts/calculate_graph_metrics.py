#!/usr/bin/env python3
"""计算 HippoRAG 2 知识图谱的 Indexing 指标。

从 caches/ 读取 graph.graphml，计算 20 个图结构指标。
指标定义与 GraphRAG-Benchmark/Evaluation/indexing_eval.py 完全一致。

用法:
    cd external/hipporag2
    uv run python scripts/calculate_graph_metrics.py \
        --data_name test_hotpotqa \
        --llm_name Qwen3.6-35B-A3B-FP8 \
        --emb_name text-embedding-bge-large-en-v1.5
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


def _graph_in(parent: str) -> str | None:
    """在 parent 的直接子目录里找 graph.graphml，多个时取 mtime 最新的。"""
    if not os.path.isdir(parent):
        return None
    found = [
        os.path.join(parent, d, "graph.graphml")
        for d in os.listdir(parent)
        if os.path.isfile(os.path.join(parent, d, "graph.graphml"))
    ]
    return max(found, key=os.path.getmtime) if found else None


def _is_batch_name(name: str) -> bool:
    """判断目录名是否为 upload 批次时间戳格式 YYYYMMDD_HHMMSS。

    只认这一格式，才能把批次目录与 contexts/questions/<llm>_<emb>/ 等固定目录区分开。
    """
    return len(name) == 15 and name[8] == "_" and name[:8].isdigit() and name[9:].isdigit()


def discover_graph(base: str) -> str | None:
    """自动发现 graph.graphml。

    支持两种布局：带批次层 <base>/<batch>/<llm>_<emb>/（批次目录名形如
    YYYYMMDD_HHMMSS）与扁平 <base>/<llm>_<emb>/。批次布局优先于扁平布局，
    存在多个批次时取 mtime 最新的批次，与 Step 脚本 resolve_source_id 口径一致。
    """
    if not os.path.isdir(base):
        return None
    batches = [
        os.path.join(base, d)
        for d in os.listdir(base)
        if _is_batch_name(d) and os.path.isdir(os.path.join(base, d))
    ]
    flat = _graph_in(base)
    for batch in sorted(batches, key=os.path.getmtime, reverse=True):
        nested = _graph_in(batch)
        if nested:
            if flat:
                print(f"[auto] 已忽略扁平布局图文件: {flat}", flush=True)
            return nested
    return flat


def load_graph(data_name: str, dataset_root: str, llm_name: str, emb_name: str) -> ig.Graph:
    path = os.path.join(dataset_root, data_name, f"{llm_name}_{emb_name}", "graph.graphml")
    if not os.path.isfile(path):
        base = os.path.join(dataset_root, data_name)
        discovered = discover_graph(base)
        if discovered:
            path = discovered
            print(f"[auto] 自动发现图文件: {discovered}", flush=True)
        else:
            raise FileNotFoundError(
                f"图文件不存在: {path}\n"
                f"已在 {base}/ 下按 <batch>/<llm>_<emb>/ 与 <llm>_<emb>/ 两种布局自动查找，均未命中\n"
                f"请先运行 reproduce/Step_1_build_index.py --data_name <dataset> 构建索引\n"
                f"可手动指定 --llm_name 和 --emb_name 参数精确匹配子目录名"
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = ig.Graph.Read_GraphML(path)
    return g


# ── 主流程 ──


def main() -> int:
    parser = argparse.ArgumentParser(
        description="HippoRAG 2 图指标计算（对标 judge indexing_eval）"
    )
    parser.add_argument("--data_name", type=str, required=True, help="数据集名")
    parser.add_argument("--dataset_root", type=str, default="caches", help="图文件根目录")
    parser.add_argument(
        "--llm_name",
        type=str,
        default=None,
        help="LLM 名称（变量目录，如 Qwen3.6-35B-A3B-FP8；不填则自动发现）",
    )
    parser.add_argument(
        "--emb_name",
        type=str,
        default=None,
        help="Embedding 名称（如 text-embedding-bge-large-en-v1.5）",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON 输出路径")
    parser.add_argument("--verbose", action="store_true", help="详细打印")
    args = parser.parse_args()

    # 构造 <llm>_<emb>（若未指定则传占位，由 load_graph 自动发现）
    llm = args.llm_name or ""
    emb = args.emb_name or ""
    if not llm and not emb:
        llm, emb = "_", "_"  # 哨兵：load_graph 会走自动发现

    g = load_graph(args.data_name, args.dataset_root, llm, emb)
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
