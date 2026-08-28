"""Tests for SAGConfig — independent SAG2 configuration decoupled from MultiConfig."""

import pytest
from pydantic import ValidationError

from pipeline.modules.search.config import (
    MultiConfig,
    SAGConfig,
)


class TestSAGConfigIndependence:
    """SAGConfig must not be a subclass of MultiConfig."""

    def test_sagconfig_not_subclass_of_multiconfig(self):
        assert not issubclass(SAGConfig, MultiConfig), "SAGConfig must not inherit from MultiConfig"

    def test_multiconfig_not_subclass_of_sagconfig(self):
        assert not issubclass(MultiConfig, SAGConfig), "MultiConfig must not inherit from SAGConfig"


class TestSAGConfigDefaults:
    """SAGConfig default values must match pre-split MultiConfig(strategy="sag2") SAG2 fields."""

    def test_default_values_match_multiconfig_sag2_equivalent(self):
        sag = SAGConfig()
        multi = MultiConfig(strategy="sag2")

        # Top-level SAG2 fields
        assert sag.max_sections == multi.max_sections
        assert sag.sag2_rewrite_query_enabled == multi.sag2_rewrite_query_enabled
        assert sag.sag2_enable_entity_extraction == multi.sag2_enable_entity_extraction
        assert sag.sag2_use_fast_mode == multi.sag2_use_fast_mode
        assert sag.use_mlflow_prompts == multi.use_mlflow_prompts
        assert sag.mlflow_prompt_alias == multi.mlflow_prompt_alias
        assert sag.mlflow_tracking_uri == multi.mlflow_tracking_uri

    def test_nested_config_defaults_match(self):
        sag = SAGConfig()
        multi = MultiConfig(strategy="sag2")

        assert sag.sag2_recall.model_dump() == multi.sag2_recall.model_dump()
        assert sag.sag2_scope.model_dump() == multi.sag2_scope.model_dump()
        assert sag.sag2_expand.model_dump() == multi.sag2_expand.model_dump()
        assert sag.sag2_rerank.model_dump() == multi.sag2_rerank.model_dump()

    def test_full_model_dump_sag2_subset_equivalent(self):
        """SAGConfig model_dump must equal the SAG2-relevant subset of MultiConfig."""
        sag = SAGConfig()
        multi = MultiConfig(strategy="sag2")

        sag_dump = sag.model_dump()
        multi_dump = multi.model_dump()
        for key in sag_dump:
            assert key in multi_dump, f"SAGConfig field {key!r} missing from MultiConfig"
            sag_val = sag_dump[key]
            multi_val = multi_dump[key]
            # Normalize nested models to dicts for comparison
            if hasattr(multi_val, "model_dump"):
                multi_val = multi_val.model_dump()
            assert sag_val == multi_val, (
                f"Field {key!r} differs: SAGConfig={sag_val!r}, MultiConfig={multi_val!r}"
            )


class TestSAGConfigDictOverride:
    """SAGConfig must accept dict overrides for nested configs."""

    def test_dict_override_sag2_recall(self):
        cfg = SAGConfig(sag2_recall={"max_entities": 30, "score_threshold": 0.5})
        assert cfg.sag2_recall.max_entities == 30
        assert cfg.sag2_recall.score_threshold == 0.5
        # Unspecified fields retain defaults
        assert cfg.sag2_recall.query_recall_event_max == 20

    def test_dict_override_sag2_scope(self):
        cfg = SAGConfig(
            sag2_scope={"enabled": True, "event_top_k": 500, "include_event_content": False}
        )
        assert cfg.sag2_scope.enabled is True
        assert cfg.sag2_scope.event_top_k == 500
        assert cfg.sag2_scope.include_event_content is False

    def test_dict_override_sag2_expand(self):
        cfg = SAGConfig(sag2_expand={"max_hops": 2, "enabled": False})
        assert cfg.sag2_expand.max_hops == 2
        assert cfg.sag2_expand.enabled is False

    def test_dict_override_sag2_rerank(self):
        cfg = SAGConfig(sag2_rerank={"strategy": "rrf", "max_results": 7})
        assert cfg.sag2_rerank.strategy == "rrf"
        assert cfg.sag2_rerank.max_results == 7


class TestSAGConfigNoMultiESFields:
    """SAGConfig must not accept or expose MultiES-specific fields."""

    def test_no_mode_field(self):
        assert "mode" not in SAGConfig.model_fields

    def test_no_fast_fields(self):
        fast_fields = [
            "fast_entity_k",
            "fast_entity_event_candidate_k",
            "fast_entity_event_k",
            "fast_query_event_k",
            "fast_answer_k",
            "fast_expand_answer_k",
            "fast_vector_weight",
            "fast_entity_weight",
            "fast_channel_weight",
        ]
        for name in fast_fields:
            assert name not in SAGConfig.model_fields, (
                f"SAGConfig must not have MultiES field {name!r}"
            )

    def test_no_entity_top_k(self):
        assert "entity_top_k" not in SAGConfig.model_fields
        assert "multi_top_k" not in SAGConfig.model_fields

    def test_no_multi_hop_fields(self):
        assert "max_hops" not in SAGConfig.model_fields
        assert "max_events" not in SAGConfig.model_fields
        assert "max_expand_events_per_hop" not in SAGConfig.model_fields


class TestSAGConfigCompatConversion:
    """_get_sag2_config must convert legacy MultiConfig to SAGConfig preserving SAG2 params."""

    def test_compat_from_multiconfig_preserves_sag2_fields(self):
        from pipeline.modules.search.config import SearchConfig
        from pipeline.modules.search.searcher import SAGSearcher

        searcher = SAGSearcher(prompt_manager=None)
        multi = MultiConfig(
            strategy="sag2",
            max_sections=7,
            sag2_recall={"max_entities": 25},
            sag2_scope={"enabled": True, "event_top_k": 300},
            sag2_expand={"max_hops": 2},
            sag2_rerank={"strategy": "rrf", "max_results": 8},
            sag2_rewrite_query_enabled=True,
            sag2_use_fast_mode=True,
            use_mlflow_prompts=True,
            mlflow_prompt_alias="production",
            mlflow_tracking_uri="http://example.com",
        )
        search_config = SearchConfig(
            query="test",
            source_config_id="src1",
            strategy_config=multi,
        )

        with pytest.warns(DeprecationWarning, match="Passing MultiConfig for SAG2 is deprecated"):
            result = searcher._get_sag2_config(search_config)

        assert isinstance(result, SAGConfig)
        assert result.max_sections == 7
        assert result.sag2_recall.max_entities == 25
        assert result.sag2_scope.enabled is True
        assert result.sag2_scope.event_top_k == 300
        assert result.sag2_expand.max_hops == 2
        assert result.sag2_rerank.strategy == "rrf"
        assert result.sag2_rerank.max_results == 8
        assert result.sag2_rewrite_query_enabled is True
        assert result.sag2_use_fast_mode is True
        assert result.use_mlflow_prompts is True
        assert result.mlflow_prompt_alias == "production"
        assert result.mlflow_tracking_uri == "http://example.com"

    def test_compat_from_sagconfig_passthrough(self):
        from pipeline.modules.search.config import SearchConfig
        from pipeline.modules.search.searcher import SAGSearcher

        searcher = SAGSearcher(prompt_manager=None)
        sag = SAGConfig(max_sections=5, sag2_use_fast_mode=True)
        search_config = SearchConfig(
            query="test",
            source_config_id="src1",
            strategy_config=sag,
        )

        result = searcher._get_sag2_config(search_config)
        assert result is sag  # Same instance passthrough

    def test_compat_from_dict(self):
        from pipeline.modules.search.config import SearchConfig
        from pipeline.modules.search.searcher import SAGSearcher

        searcher = SAGSearcher(prompt_manager=None)
        search_config = SearchConfig(
            query="test",
            source_config_id="src1",
            strategy_config={"max_sections": 12, "sag2_use_fast_mode": True},
        )

        result = searcher._get_sag2_config(search_config)
        assert isinstance(result, SAGConfig)
        assert result.max_sections == 12
        assert result.sag2_use_fast_mode is True

    def test_compat_from_none_returns_default(self):
        from pipeline.modules.search.config import SearchConfig
        from pipeline.modules.search.searcher import SAGSearcher

        searcher = SAGSearcher(prompt_manager=None)
        search_config = SearchConfig(
            query="test",
            source_config_id="src1",
            strategy_config=None,
        )

        result = searcher._get_sag2_config(search_config)
        assert isinstance(result, SAGConfig)
        assert result.max_sections == 10  # default


class TestSAGConfigValidation:
    """SAGConfig field validation."""

    def test_max_sections_range(self):
        SAGConfig(max_sections=1)
        SAGConfig(max_sections=50)
        with pytest.raises(ValidationError):
            SAGConfig(max_sections=0)
        with pytest.raises(ValidationError):
            SAGConfig(max_sections=51)

    def test_strategy_default(self):
        cfg = SAGConfig()
        assert cfg.strategy == "sag2"
