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
