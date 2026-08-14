"""Public storage index-routing API.

Upload and benchmark entry points use this module to coordinate the active
embedding dimension without depending on Elasticsearch backend internals.
"""

from pipeline.storage.backends.elasticsearch.active_dim import (
    get_active_embedding_dim,
    reset_active_embedding_dim,
    set_active_embedding_dim,
)
from pipeline.storage.backends.elasticsearch.index_naming import (
    ALL_BASE_INDICES,
    BASE_INDEX_ENTITY_VECTORS,
    BASE_INDEX_EVENT_ENTITY_VECTORS,
    BASE_INDEX_EVENT_VECTORS,
    BASE_INDEX_SOURCE_CHUNKS,
    LEGACY_DIM,
    IndexDimMismatchError,
    assert_index_dims,
    extract_dense_vector_dims,
    index_suffix,
    resolve_all_index_names,
    resolve_index_name,
    with_dense_vector_dims,
)

__all__ = [
    "ALL_BASE_INDICES",
    "BASE_INDEX_ENTITY_VECTORS",
    "BASE_INDEX_EVENT_ENTITY_VECTORS",
    "BASE_INDEX_EVENT_VECTORS",
    "BASE_INDEX_SOURCE_CHUNKS",
    "LEGACY_DIM",
    "IndexDimMismatchError",
    "assert_index_dims",
    "extract_dense_vector_dims",
    "get_active_embedding_dim",
    "index_suffix",
    "reset_active_embedding_dim",
    "resolve_all_index_names",
    "resolve_index_name",
    "set_active_embedding_dim",
    "with_dense_vector_dims",
]
