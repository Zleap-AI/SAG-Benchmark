r"""Graph indexing statistics aligned with GraphRAG-Benchmark/Evaluation/indexing_eval.py.

Computes graph metrics (nodes, edges, degree, density, components, clustering, diameter)
from GraphML, Parquet, pickle, or picklez files.

WARNING: pickle input is inherently unsafe. Only load trusted local files.
"""

import os
import pickle
import warnings

import igraph as ig
import numpy as np
import pandas as pd


def analyze_graph(g: ig.Graph) -> dict[str, float]:
    """Compute comprehensive graph metrics."""
    num_nodes = g.vcount()
    num_edges = g.ecount()
    average_degree = sum(g.degree()) / num_nodes if num_nodes > 0 else 0
    density = g.density()
    components = g.components()
    num_components = len(components)
    largest_component_size = components.giant().vcount()
    average_clustering_coefficient = g.transitivity_avglocal_undirected()
    diameter = g.diameter() if g.is_connected() else float("inf")

    component_sizes = [len(component) for component in components if len(component) > 1]
    if component_sizes:
        average_component_size = sum(component_sizes) / len(component_sizes)
        median_component_size = float(np.median(component_sizes))
        num_components_excluding_isolated = len(component_sizes)
        num_components_above_average = sum(
            1 for size in component_sizes if size > average_component_size
        )
        num_nodes_excluding_isolated = sum(component_sizes)
        component_sizes_sorted = sorted(component_sizes)
        trimmed_mean_component_size = (
            sum(component_sizes_sorted[1:-1]) / (len(component_sizes_sorted) - 2)
            if len(component_sizes_sorted) > 2
            else average_component_size
        )
        geometric_mean_component_size = float(
            np.exp(np.mean(np.log(component_sizes))) if component_sizes else 0
        )
        harmonic_mean_component_size = (
            len(component_sizes) / sum(1.0 / size for size in component_sizes)
            if component_sizes
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


def load_graph_from_parquet(entities_path: str, relationships_path: str) -> ig.Graph:
    """Load graph from entities.parquet and relationships.parquet."""
    entities_df = pd.read_parquet(entities_path)
    relationships_df = pd.read_parquet(relationships_path)
    g = ig.Graph()

    for _, row in entities_df.iterrows():
        entity_id = row["id"]
        g.add_vertex(name=entity_id)

    for _, row in relationships_df.iterrows():
        source_id = row["source"]
        target_id = row["target"]
        if source_id not in g.vs["name"]:
            g.add_vertex(name=source_id)
        if target_id not in g.vs["name"]:
            g.add_vertex(name=target_id)
        weight = row.get("weight", 1)
        g.add_edge(source_id, target_id, weight=weight)

    return g


def load_graph_from_pickle(pickle_path: str) -> ig.Graph:
    """Load graph from pickle file. Only use with trusted local files."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def load_graph_from_picklez(picklez_path: str) -> ig.Graph:
    """Load graph from picklez (igraph native) file."""
    return ig.Graph.Read_Picklez(picklez_path)


def load_graph_from_graphml(graphml_path: str) -> ig.Graph:
    """Load graph from GraphML file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ig.Graph.Read_GraphML(graphml_path)


def process_graphs_microsoft_graphrag(
    base_path: str, folder_name: str = ""
) -> list[dict[str, float]]:
    """Process GraphRAG parquet graphs."""
    results: list[dict[str, float]] = []
    for subdir, _dirs, _files in os.walk(base_path):
        entities_path = os.path.join(subdir, "entities.parquet")
        relationships_path = os.path.join(subdir, "relationships.parquet")
        if os.path.exists(entities_path) and os.path.exists(relationships_path):
            try:
                g = load_graph_from_parquet(entities_path, relationships_path)
                results.append(analyze_graph(g))
            except Exception as e:
                print(f"Error processing {subdir}: {e}")
    return results


def process_graphs_lightrag_fastgraphrag(
    base_path: str, folder_name: str = ""
) -> list[dict[str, float]]:
    """Process LightRAG (GraphML) and Fast-GraphRAG (picklez) graphs."""
    results: list[dict[str, float]] = []
    for subdir, _dirs, _files in os.walk(base_path):
        lightrag_path = os.path.join(subdir, "graph_chunk_entity_relation.graphml")
        fastgraphrag_path = os.path.join(subdir, "graph_igraph_data.pklz")
        if os.path.exists(lightrag_path):
            try:
                g = ig.Graph.Read_GraphML(lightrag_path)
                results.append(analyze_graph(g))
            except Exception as e:
                print(f"Error loading LightRAG graph from {lightrag_path}: {e}")
        elif os.path.exists(fastgraphrag_path):
            try:
                g = load_graph_from_picklez(fastgraphrag_path)
                results.append(analyze_graph(g))
            except Exception as e:
                print(f"Error loading Fast-GraphRAG graph from {fastgraphrag_path}: {e}")
    return results


def process_graphs_hipporag2(base_path: str, folder_name: str) -> list[dict[str, float]]:
    """Process HippoRAG2 pickle graphs."""
    results: list[dict[str, float]] = []
    for subdir, _dirs, _files in os.walk(base_path):
        target_folder = os.path.join(subdir, folder_name)
        if os.path.exists(target_folder):
            graph_path = os.path.join(target_folder, "graph.pickle")
            if os.path.exists(graph_path):
                try:
                    g = load_graph_from_pickle(graph_path)
                    results.append(analyze_graph(g))
                except Exception as e:
                    print(f"Error processing {subdir}: {e}")
    return results


def process_graphs_graphml(base_path: str, pattern: str = "*.graphml") -> list[dict[str, float]]:
    """Process generic GraphML files."""
    results: list[dict[str, float]] = []
    for subdir, _dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".graphml"):
                graph_path = os.path.join(subdir, file)
                try:
                    g = load_graph_from_graphml(graph_path)
                    results.append(analyze_graph(g))
                except Exception as e:
                    print(f"Error processing {graph_path}: {e}")
    return results


def calculate_average(results: list[dict[str, float]]) -> dict[str, float]:
    """Compute per-metric averages across all graphs."""
    if not results:
        return {}
    avg_results: dict[str, float] = dict.fromkeys(results[0], 0.0)
    for result in results:
        for key, value in result.items():
            avg_results[key] += value
    num_graphs = len(results)
    for key in avg_results:
        avg_results[key] /= num_graphs
    return avg_results


def calculate_indexing_metrics(
    framework: str, base_path: str, folder_name: str | None = None
) -> dict[str, float]:
    """Entry point: compute indexing metrics for a given framework.

    Args:
        framework: 'microsoft_graphrag', 'lightrag', 'fast_graphrag', 'hipporag2', 'graphml'
        base_path: Root path containing graph data.
        folder_name: Subdirectory name (required for hipporag2).
    """
    if framework == "microsoft_graphrag":
        results = process_graphs_microsoft_graphrag(base_path, folder_name or "")
    elif framework in ("lightrag", "fast_graphrag"):
        results = process_graphs_lightrag_fastgraphrag(base_path, folder_name or "")
    elif framework == "hipporag2":
        if not folder_name:
            raise ValueError("HippoRAG2 requires folder_name parameter")
        results = process_graphs_hipporag2(base_path, folder_name)
    elif framework == "graphml":
        results = process_graphs_graphml(base_path)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    if not results:
        print(f"Warning: No graph data found for {framework} in {base_path}")
        return {}
    return calculate_average(results)
