"""Storage facade errors."""

from pipeline.exceptions import StorageError


class StorageBackendError(StorageError):
    """Raised when a configured storage backend cannot be created or used."""


class UnsupportedStorageOperation(StorageBackendError):
    """Raised when a backend does not implement a requested operation yet."""
