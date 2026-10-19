"""Canonical public naming stays importable; retired aliases stay retired."""

import pytest


def test_canonical_public_names_importable():
    from pipeline import PipelineEngine, PipelineError
    from pipeline.models import PipelineBaseModel
    from pipeline.storage.providers.database import MySQLDatabaseStore

    assert PipelineEngine is not None
    assert PipelineError is not None
    assert PipelineBaseModel is not None
    assert MySQLDatabaseStore is not None


@pytest.mark.parametrize(
    "module_name, attr",
    [
        ("pipeline", "pipelineError"),
        ("pipeline", "pipelineEngine"),
        ("pipeline.models", "pipelineBaseModel"),
        ("pipeline.storage.providers.database", "MySqlDatabaseStore"),
    ],
)
def test_retired_camelcase_aliases_no_longer_exist(module_name, attr):
    import importlib

    module = importlib.import_module(module_name)
    assert not hasattr(module, attr)


def test_sag_graph_public_names_and_default_strategy():
    from pipeline.modules.search import SAGSearcher
    from pipeline.modules.search.benchmark_utils import SUPPORTED_STRATEGIES, build_strategy_config
    from pipeline.modules.search.config import RerankConfig, RerankStrategy, SAGConfig
    from pipeline.modules.search.sag_graph import SAGGraphSearcher

    strategy, config = build_strategy_config("sag", top_k=10)

    assert SAGSearcher is not SAGGraphSearcher
    assert strategy is RerankStrategy.SAG
    assert RerankConfig().strategy is RerankStrategy.SAG
    assert isinstance(config, SAGConfig)
    assert config.strategy == "sag"
    assert "sag" in SUPPORTED_STRATEGIES
    assert "sag2" not in SUPPORTED_STRATEGIES


def test_retired_sag2_strategy_and_module_are_unavailable():
    import importlib.util

    from pydantic import ValidationError

    from pipeline.modules.search.config import RerankStrategy, SAGConfig

    with pytest.raises(ValueError):
        RerankStrategy("sag2")
    with pytest.raises(ValidationError):
        SAGConfig(strategy="sag2")
    assert importlib.util.find_spec("pipeline.modules.search.sag2") is None
