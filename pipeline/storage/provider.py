"""Factory for composed storage backends."""

from pipeline.core.config.settings import Settings
from pipeline.storage.facade import StorageFacade
from pipeline.storage.providers.database import MySqlDatabaseStore, OceanBaseDatabaseStore
from pipeline.storage.providers.search import ElasticsearchSearchStore, OceanBaseSearchStore
from pipeline.storage.providers.vector import ElasticsearchVectorStore, OceanBaseVectorStore


def create_storage_facade(settings: Settings) -> StorageFacade:
    """Create a storage facade from DATABASE_BACKEND and VECTOR_BACKEND."""
    database_backend = settings.effective_database_backend
    vector_backend = settings.effective_vector_backend

    if database_backend == "mysql":
        database = MySqlDatabaseStore()
    elif database_backend == "oceanbase":
        database = OceanBaseDatabaseStore()
    else:
        raise ValueError(f"Unsupported DATABASE_BACKEND: {database_backend}")

    if vector_backend == "elasticsearch":
        vector = ElasticsearchVectorStore()
        search = ElasticsearchSearchStore(vector)
    elif vector_backend == "oceanbase":
        vector = OceanBaseVectorStore()
        search = OceanBaseSearchStore(vector)
    else:
        raise ValueError(f"Unsupported VECTOR_BACKEND: {vector_backend}")

    return StorageFacade(database=database, vector=vector, search=search)
