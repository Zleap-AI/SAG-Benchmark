"""Contract test for the 20 GraphRAG-Benchmark indexing metrics."""

import math

import igraph as ig
import pytest

from pipeline.evaluation.judge.indexing import analyze_graph
from scripts.calculate_graph_metrics import analyze_graph as analyze_sag2_graph


def test_indexing_metrics_match_reference_formula_on_disconnected_graph():
    graph = ig.Graph(n=5, edges=[(0, 1), (1, 2), (3, 4)], directed=False)

    metrics = analyze_graph(graph)

    assert set(metrics) == {
        "num_nodes",
        "num_edges",
        "average_degree",
        "density",
        "num_components",
        "largest_component_size",
        "average_clustering_coefficient",
        "diameter",
        "average_component_size",
        "median_component_size",
        "trimmed_mean_component_size",
        "geometric_mean_component_size",
        "harmonic_mean_component_size",
        "num_components_excluding_isolated",
        "num_components_above_average",
        "num_nodes_excluding_isolated",
        "num_isolated_nodes",
        "num_nodes_degree_above_1",
        "num_nodes_degree_above_2",
        "num_nodes_degree_above_3",
    }
    assert metrics["num_nodes"] == 5.0
    assert metrics["num_edges"] == 3.0
    assert metrics["average_degree"] == pytest.approx(1.2)
    assert metrics["density"] == pytest.approx(0.3)
    assert metrics["num_components"] == 2.0
    assert metrics["largest_component_size"] == 3.0
    assert metrics["average_component_size"] == 2.5
    assert metrics["median_component_size"] == 2.5
    assert metrics["geometric_mean_component_size"] == pytest.approx(math.sqrt(6))
    assert metrics["harmonic_mean_component_size"] == pytest.approx(2.4)
    assert math.isinf(metrics["diameter"])


def test_sag2_graph_metrics_preserve_reference_values_and_only_add_extensions():
    graph = ig.Graph(n=3, edges=[(0, 1)], directed=False)
    graph.vs["node_type"] = ["event", "entity", "entity"]

    reference = analyze_graph(graph)
    sag2 = analyze_sag2_graph(graph)

    for key, expected in reference.items():
        actual = sag2[key]
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert actual == expected
    assert sag2["num_events"] == 1
    assert sag2["num_entities"] == 2
    assert sag2["max_degree"] == 1
