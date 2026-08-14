import pytest
from pydantic import ValidationError

from pipeline.modules.search.benchmark_utils import build_sag2_config
from pipeline.modules.search.config import SAG2RerankConfig


def test_sag2_rerank_quantity_defaults_are_separate():
    config = SAG2RerankConfig()

    assert config.rerank_top_k == 10
    assert config.max_results == 10


def test_benchmark_builder_applies_rerank_top_k_and_final_max_results():
    config = build_sag2_config(
        max_sections=10,
        rerank_top_k=3,
        max_results=8,
    )

    assert config.sag2_rerank.rerank_top_k == 3
    assert config.sag2_rerank.max_results == 8
    assert config.max_sections == 10


def test_sag2_builder_applies_candidate_scope_overrides():
    config = build_sag2_config(
        max_sections=7,
        scope_enabled=True,
        event_top_k=321,
        bootstrap_entity_limit=12,
        include_event_content=False,
    )

    assert config.sag2_scope.enabled is True
    assert config.sag2_scope.event_top_k == 321
    assert config.sag2_scope.bootstrap_entity_limit == 12
    assert config.sag2_scope.include_event_content is False


def test_benchmark_builder_rejects_rerank_top_k_above_max_results():
    with pytest.raises(ValueError, match="不能大于"):
        build_sag2_config(
            max_sections=10,
            rerank_top_k=6,
            max_results=5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("rerank_top_k", 0), ("max_results", 0)],
)
def test_benchmark_builder_rejects_nonpositive_rerank_limits(field, value):
    kwargs = {field: value}

    with pytest.raises(ValidationError):
        build_sag2_config(max_sections=10, **kwargs)
