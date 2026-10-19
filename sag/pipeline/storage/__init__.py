"""
Unified storage facade.

This package keeps application code away from concrete MySQL, Elasticsearch,
and OceanBase clients. The current implementation provides the factory and
capability layer used by the first migration phase.
"""

from pipeline.storage.capabilities import StorageCapabilities, StorageHealth
from pipeline.storage.facade import (
    StorageFacade,
    close_storage_facade,
    get_storage_facade,
    reset_storage_facade,
)
from pipeline.storage.factory import create_storage_facade

__all__ = [
    "StorageCapabilities",
    "StorageFacade",
    "StorageHealth",
    "close_storage_facade",
    "create_storage_facade",
    "get_storage_facade",
    "reset_storage_facade",
]
