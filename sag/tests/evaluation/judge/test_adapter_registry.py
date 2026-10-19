"""Tests for AdapterRegistry and build_default_registry."""

import pytest

from pipeline.evaluation.judge.adapters.registry import (
    AdapterRegistry,
    build_default_registry,
)
from pipeline.evaluation.judge.errors import AdapterNotFoundError


class _DummyAdapter:
    name = "dummy"

    def locate_source(self, request):
        pass

    def convert(self, source):
        pass


class TestAdapterRegistry:
    def test_register_and_get(self):
        reg = AdapterRegistry()
        reg.register(_DummyAdapter())
        assert reg.get("dummy").name == "dummy"

    def test_duplicate_raises(self):
        reg = AdapterRegistry()
        reg.register(_DummyAdapter())
        with pytest.raises(ValueError, match="Duplicate"):
            reg.register(_DummyAdapter())

    def test_empty_name_raises(self):
        reg = AdapterRegistry()

        class _Bad:
            name = ""

            def locate_source(self, request):
                pass

            def convert(self, source):
                pass

        with pytest.raises(ValueError, match="empty"):
            reg.register(_Bad())

    def test_unknown_project_raises(self):
        reg = AdapterRegistry()
        with pytest.raises(AdapterNotFoundError):
            reg.get("nonexistent")

    def test_names(self):
        reg = AdapterRegistry()
        reg.register(_DummyAdapter())
        names = reg.names()
        assert "dummy" in names

    def test_build_default_registry_has_five_adapters(self):
        reg = build_default_registry()
        names = reg.names()
        assert len(names) == 6
        for expected in ("graphrag", "hipporag2", "hypergraphrag", "hyperrag", "lightrag"):
            assert expected in names
