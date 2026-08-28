"""Pipeline Judge — LLM-based evaluation for Generation and Retrieval."""

from pipeline.evaluation.judge.artifacts import (
    ArtifactLayoutResolver,
    JudgeArtifactStore,
    PredictionArtifactLayout,
    sanitize_path_component,
    sha256_file,
)
from pipeline.evaluation.judge.conversion import PredictionConversionService
from pipeline.evaluation.judge.errors import (
    AdapterConversionError,
    AdapterNotFoundError,
    AmbiguousGroundTruthMatchError,
    ArtifactPathError,
    DatasetCapabilityError,
    DatasetError,
    DatasetFileNotFoundError,
    DatasetSchemaError,
    GroundTruthMatchError,
    JudgeConfigurationError,
    JudgeError,
    JudgeExecutionError,
    JudgeResumeConflictError,
    MetricResultError,
    PredictionValidationError,
    SourceRunAmbiguousError,
    SourceRunNotFoundError,
    UnsupportedDatasetError,
)
from pipeline.evaluation.judge.evaluation_service import JudgeEvaluationService
from pipeline.evaluation.judge.models import (
    AdapterConversion,
    ConversionManifest,
    ConversionRequest,
    ConversionResult,
    EvaluationKind,
    FileDigest,
    JudgeDetailedResult,
    JudgeMetricResult,
    JudgeRunManifest,
    JudgeRunParameters,
    JudgeRunStatus,
    JudgeRunSummary,
    JudgeSample,
    LatestRunPointer,
    SampleEvaluationStatus,
    SourceRun,
)
from pipeline.evaluation.judge.runner import (
    JudgeEvaluationRunner,
    SampleEvaluator,
)

__all__ = [
    # Models
    "JudgeSample",
    "JudgeMetricResult",
    "JudgeDetailedResult",
    "JudgeRunSummary",
    "JudgeRunManifest",
    "JudgeRunParameters",
    "JudgeRunStatus",
    "EvaluationKind",
    "SampleEvaluationStatus",
    "ConversionRequest",
    "ConversionResult",
    "ConversionManifest",
    "SourceRun",
    "AdapterConversion",
    "FileDigest",
    "LatestRunPointer",
    # Errors
    "JudgeError",
    "DatasetError",
    "UnsupportedDatasetError",
    "DatasetSchemaError",
    "DatasetCapabilityError",
    "GroundTruthMatchError",
    "AmbiguousGroundTruthMatchError",
    "DatasetFileNotFoundError",
    "JudgeConfigurationError",
    "ArtifactPathError",
    "SourceRunNotFoundError",
    "SourceRunAmbiguousError",
    "AdapterNotFoundError",
    "AdapterConversionError",
    "PredictionValidationError",
    "JudgeResumeConflictError",
    "JudgeExecutionError",
    "MetricResultError",
    # Artifacts
    "ArtifactLayoutResolver",
    "JudgeArtifactStore",
    "PredictionArtifactLayout",
    "sanitize_path_component",
    "sha256_file",
    # Services
    "PredictionConversionService",
    "JudgeEvaluationService",
    # Runner
    "JudgeEvaluationRunner",
    "SampleEvaluator",
]
