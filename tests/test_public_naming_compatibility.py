"""Public naming migrations keep the historical import names working."""

from pipeline import PipelineEngine, PipelineError, pipelineEngine, pipelineError
from pipeline.models import PipelineBaseModel, pipelineBaseModel
from pipeline.storage.providers.database import MySQLDatabaseStore, MySqlDatabaseStore


def test_canonical_public_names_and_compatibility_aliases():
    assert pipelineEngine is PipelineEngine
    assert pipelineError is PipelineError
    assert pipelineBaseModel is PipelineBaseModel
    assert MySqlDatabaseStore is MySQLDatabaseStore
