"""Storage backend capability models."""

from typing import Optional

from pydantic import BaseModel


class StorageCapabilities(BaseModel):
    """Capabilities exposed by the active storage composition."""

    database_backend: str
    vector_backend: str
    supports_vector_search: bool
    supports_fulltext_search: bool
    supports_hybrid_search: bool
    supports_transactional_vector_write: bool
    supports_json: bool = True
    supports_foreign_keys: bool = True
    max_vector_dims: Optional[int] = None


class StorageHealth(BaseModel):
    """Health check result for the composed storage facade."""

    ok: bool
    database_ok: bool
    vector_ok: bool
    capabilities: StorageCapabilities
    message: str = ""
