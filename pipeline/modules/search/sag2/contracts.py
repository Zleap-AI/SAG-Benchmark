"""Typed request and result contracts for the staged SAG2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from pipeline.modules.search.config import SAGConfig

if TYPE_CHECKING:
    from pipeline.modules.search.sag2.candidate_scope import SAG2CandidateSubgraph


SAG2_TIMING_STAGE_ORDER = (
    "rewrite_query",
    "query_embedding",
    "candidate_pool",
    "parallel_recall",
    "event_similarity_filter",
    "merge_routes",
    "expand",
    "score_sort",
    "rerank",
    "answer_graph",
    "step8_chunks",
    "finalize",
    "other_overhead",
)

SAG2_EVENT_STATS_KEYS = (
    "query_event_count",
    "entity_event_count",
    "expand_event_count",
    "merged_event_count",
    "score_top_event_count",
    "rank_event_count",
    "seen_event_count",
    "candidate_scope_event_count",
    "candidate_scope_entity_count",
    "candidate_scope_edge_count",
    "query_entity_count",
    "new_entity_count",
)


class SAG2Entity(BaseModel):
    """Entity extracted during SAG2 query rewriting."""

    name: str
    weight: float


class SAG2RewriteOutput(BaseModel):
    """Structured output for SAG2 query rewriting and entity extraction."""

    rewritten_query: str = Field(..., description="Rewritten query")
    entities: list[SAG2Entity] = Field(default_factory=list, description="Extracted entities")


class SAG2RerankOutput(BaseModel):
    """Structured output for the SAG2 rerank prompt."""

    thought_process: str = Field(default="", description="Reasoning trace")
    useful_relations: list[str] = Field(
        default_factory=list,
        description='Relation indices such as "[id]", ordered by usefulness',
    )


@dataclass
class SAG2SearchState:
    """Per-request mutable state used for route and evidence diagnostics."""

    entity_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)
    seen_event_ids: set[str] = field(default_factory=set)
    all_clues: list[dict[str, Any]] = field(default_factory=list)
    all_nodes: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class SAG2Request:
    query: str
    source_config_ids: list[str]
    config: SAGConfig
    gold_evidences: list[str]


@dataclass(slots=True)
class SAG2RecallResult:
    rewritten_query: str | None
    rewritten_entities: list[SAG2Entity]
    effective_query: str
    query_vector: list[float]
    scope: SAG2CandidateSubgraph | None
    query_events: list[dict[str, Any]]
    entity_events: dict[str, list[str]]
    entity_event_docs: list[dict[str, Any]]
    query_event_ids: list[str]
    entity_event_ids: list[str]
    query_entity_ids: list[str]
    event_ids: list[str]
    event_scores: dict[str, float]
    initial_route_stats: dict[str, int]


@dataclass(slots=True)
class SAG2ExpandResult:
    event_ids: list[str]
    event_scores: dict[str, float]
    expand_event_ids: list[str]
    expand_event_scores: dict[str, float]
    # event_id -> entity_ids from graph expansion
    expand_entity_events: dict[str, list[str]]
    # entity_id -> event_ids discovered from seed events
    seed_new_entity_events: dict[str, list[str]]
    new_entity_ids: list[str]
    expand_new_entity_ids: list[str]
    expand_hops: list[dict[str, Any]]
    # Route counters produced by the expand stage. The orchestrator merges
    # these into Recall's initial counters before emitting diagnostics.
    route_stats: dict[str, int]


@dataclass(slots=True)
class SAG2RerankResult:
    rank_events: list[dict[str, Any]]
    rank_event_ids: list[str]
    rank_scores: dict[str, float]
    rank_failed: bool
    rank_method: str
    rank_event_map: dict[str, dict[str, Any]]
    display_event_ids: list[str]
    failure_reason: str | None
    retry_count: int


__all__ = [
    "SAG2Entity",
    "SAG2RewriteOutput",
    "SAG2RerankOutput",
    "SAG2SearchState",
    "SAG2Request",
    "SAG2RecallResult",
    "SAG2ExpandResult",
    "SAG2RerankResult",
    "SAG2_TIMING_STAGE_ORDER",
    "SAG2_EVENT_STATS_KEYS",
]
