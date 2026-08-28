"""Judge domain exceptions — library code raises these, CLI maps to exit codes.

JudgeError hierarchy replaces ad-hoc print()/SystemExit in library code.
"""

from __future__ import annotations


class JudgeError(Exception):
    """Base for all Judge domain errors."""


class JudgeConfigurationError(JudgeError):
    """Invalid configuration (missing env vars, bad parameter combinations)."""


class ArtifactPathError(JudgeError):
    """Path validation failure (escape attempt, outside artifact root)."""


class SourceRunNotFoundError(JudgeError):
    """Adapter could not locate a valid source run."""


class SourceRunAmbiguousError(JudgeError):
    """Adapter found multiple candidate source runs — requires explicit selection."""


class AdapterNotFoundError(JudgeError):
    """No adapter registered for the given project name."""


class AdapterConversionError(JudgeError):
    """Adapter failed to convert native results (e.g. missing question, duplicate)."""


class PredictionValidationError(JudgeError):
    """Predictions file failed schema or consistency validation."""


class JudgeResumeConflictError(JudgeError):
    """Resume conflict: hash mismatch, run-id collision, or incompatible params."""


class JudgeExecutionError(JudgeError):
    """Unrecoverable Judge runtime failure (programming error, not sample-level)."""


class MetricResultError(JudgeError):
    """A metric produced a non-finite or otherwise invalid result."""


class DatasetError(JudgeError):
    """Base class for raw benchmark dataset failures."""


class UnsupportedDatasetError(DatasetError):
    """No explicit dataset adapter is registered for the requested dataset."""


class DatasetSchemaError(DatasetError):
    """A raw dataset row does not satisfy its declared adapter schema."""


class DatasetCapabilityError(DatasetError):
    """A requested evaluation capability is not provided by the dataset."""


class GroundTruthMatchError(DatasetError):
    """A prediction cannot be matched to one canonical ground-truth row."""


class AmbiguousGroundTruthMatchError(GroundTruthMatchError):
    """A prediction matches multiple canonical ground-truth rows."""


class DatasetFileNotFoundError(DatasetError):
    """The requested dataset file could not be found."""
