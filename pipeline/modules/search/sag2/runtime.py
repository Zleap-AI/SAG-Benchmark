"""SAG2 runtime dependencies shared by the SAG2 search pipeline.

This module owns SAG2 reusable infrastructure without importing another
search strategy. SAG2 keeps its graph recall, expansion, ranking, and evidence
logic in ``sag2.py`` while this runtime owns clients, storage ports, prompts,
and chunk hydration.
"""

import inspect
from typing import Any

from pydantic import BaseModel, Field

from pipeline.core.ai.factory import create_llm_client
from pipeline.modules.load.processor import DocumentProcessor
from pipeline.modules.search.config import SAGConfig
from pipeline.modules.search.prompts import PromptProvider
from pipeline.storage import get_storage_facade
from pipeline.utils.token_counter import TokenCounter


class SAG2NEROutput(BaseModel):
    """Structured output for the shared SAG2 entity extraction step."""

    named_entities: list[str] = Field(..., description="Named entities extracted from the query")


class SAG2Runtime:
    """Own SAG2's infrastructure dependencies and reusable search steps.

    The class deliberately contains no strategy dispatch.  It is a dependency
    bundle for SAG2, not another public search strategy.
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        *,
        rerank_client=None,
        settings=None,
    ):
        self._llm_client = None
        self._processor: DocumentProcessor | None = None
        self._prompts: PromptProvider | None = None
        self._rerank_client = rerank_client
        self._settings = settings
        storage = get_storage_facade()
        self._search_store = storage.search
        self._vector_store = storage.vector
        self._event_universe_store = storage.event_universe
        self.token_counter = token_counter or TokenCounter()

    @property
    def llm_client(self):
        """Return the initialized client for timing/token accounting."""

        return self._llm_client

    @property
    def event_universe_store(self):
        """Return the canonical relational event-universe port."""

        return self._event_universe_store

    def get_rerank_client(self):
        if self._rerank_client is None:
            from pipeline.core.ai.rerank import get_rerank_client

            self._rerank_client = get_rerank_client()
        return self._rerank_client

    def get_settings(self):
        if self._settings is None:
            from pipeline.core.config import get_settings

            self._settings = get_settings()
        return self._settings

    async def get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = await create_llm_client(scenario="search")
        return self._llm_client

    async def aclose(self) -> None:
        """Release Runtime-owned clients without closing the process store.

        StorageFacade is process-scoped and is closed by the benchmark/script
        lifecycle. SAG2Runtime only closes the LLM client it created, which is
        safe when multiple search strategies share the storage facade.
        """
        client = self._llm_client
        self._llm_client = None
        self._processor = None
        self._prompts = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is None:
            return
        result = close()
        if inspect.isawaitable(result):
            await result

    async def get_processor(self) -> DocumentProcessor:
        if self._processor is None:
            self._processor = DocumentProcessor(llm_client=await self.get_llm_client())
        return self._processor

    def get_prompts(self, config: SAGConfig | None = None) -> PromptProvider:
        if self._prompts is None:
            config = config or SAGConfig()
            self._prompts = PromptProvider(
                use_mlflow=getattr(config, "use_mlflow_prompts", False),
                alias=getattr(config, "mlflow_prompt_alias", "latest"),
                tracking_uri=getattr(config, "mlflow_tracking_uri", None),
            )
            self._prompts.load_all()
        return self._prompts

    async def search_entities_by_text(
        self,
        query: str,
        source_config_ids: list[str],
        size: int,
        allowed_entity_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._search_store.search_entities_by_text(
            query=query,
            source_config_ids=source_config_ids,
            size=size,
            entity_ids=allowed_entity_ids,
        )

    async def search_events_by_vector(
        self,
        *,
        query_vector: list[float],
        k: int,
        source_config_ids: list[str] | None = None,
        event_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return await self._search_store.search_events_by_vector(
            query_vector=query_vector,
            k=k,
            source_config_ids=source_config_ids,
            event_ids=event_ids,
            vector_field="content_vector",
        )

    async def search_event_entities_by_vector(
        self,
        *,
        query_vector: list[float],
        k: int,
        event_ids: list[str],
        source_config_ids: list[str],
    ) -> list[dict[str, Any]]:
        return await self._vector_store.search_event_entities_by_vector(
            query_vector=query_vector,
            k=k,
            event_ids=event_ids,
            source_config_ids=source_config_ids,
        )

    async def get_events_by_ids(
        self, event_ids: list[str], source_includes: list[str]
    ) -> list[dict[str, Any]]:
        return await self._search_store.get_events_by_ids(
            event_ids,
            source_includes=source_includes,
        )

    async def search_events_by_text(
        self,
        *,
        query: str,
        event_ids: list[str],
        k: int,
        source_config_ids: list[str],
    ) -> list[dict[str, Any]]:
        return await self._search_store.search_events_by_text(
            query=query,
            event_ids=event_ids,
            k=k,
            source_config_ids=source_config_ids,
        )

    async def get_entities_by_ids(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        return await self._search_store.get_entities_by_ids(entity_ids)

    async def retrieve_entity_event_pairs(
        self,
        entity_ids: list[str],
        source_config_ids: list[str],
        *,
        max_events_per_entity: int,
        exclude_event_ids: set[str] | None = None,
    ) -> tuple[dict[str, list[str]], list[str]]:
        """Resolve entity-to-event relations through the canonical store.

        This is a reusable storage capability shared by recall and expansion;
        it intentionally contains no stage-specific orchestration.
        """

        ordered_entities: list[str] = []
        seen_entities: set[str] = set()
        for entity_id in entity_ids:
            if entity_id and entity_id not in seen_entities:
                seen_entities.add(entity_id)
                ordered_entities.append(entity_id)
        if not ordered_entities:
            return {}, []

        excluded = exclude_event_ids or set()
        event_to_entities: dict[str, list[str]] = {}
        per_entity_count: dict[str, int] = {}
        rows = await self._event_universe_store.get_event_entity_pairs_by_entities(
            ordered_entities,
            source_config_ids=source_config_ids,
        )
        for event_id, entity_id in rows:
            if (
                not event_id
                or not entity_id
                or entity_id not in seen_entities
                or event_id in excluded
                or per_entity_count.get(entity_id, 0) >= max_events_per_entity
            ):
                continue
            per_entity_count[entity_id] = per_entity_count.get(entity_id, 0) + 1
            bucket = event_to_entities.setdefault(event_id, [])
            if entity_id not in bucket:
                bucket.append(entity_id)
        return event_to_entities, list(event_to_entities)

    async def extract_entities(self, query: str, config: SAGConfig | None = None) -> list[str]:
        """Extract query entities using SAG2's shared PromptProvider path."""

        prompts = self.get_prompts(config)
        messages = prompts.get_messages("ner", query=query)
        parsed: SAG2NEROutput = await (await self.get_llm_client()).chat_parsed(
            messages, SAG2NEROutput
        )
        return [str(entity).strip() for entity in parsed.named_entities if str(entity).strip()]

    async def retrieve_entity_candidates(
        self,
        query_entities: list[str],
        source_config_ids: list[str],
        *,
        entity_top_k: int,
        key_similarity_threshold: float,
        allowed_entity_ids: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[float]]:
        """Embed query entities and retrieve matching entity IDs from ES."""

        if not query_entities:
            return [], [], []

        embeddings = await (await self.get_processor()).generate_embeddings_batch(query_entities)
        entity_ids: list[str] = []
        entity_names: list[str] = []
        scores: list[float] = []
        seen: set[str] = set()
        for vector in embeddings:
            hits = await self._vector_store.search_entities_by_vector(
                query_vector=vector,
                k=entity_top_k,
                source_config_ids=source_config_ids,
                entity_ids=allowed_entity_ids,
            )
            for hit in hits:
                score = float(hit.get("_score", 0.0) or 0.0)
                entity_id = str(hit.get("entity_id") or "")
                if score < key_similarity_threshold or not entity_id or entity_id in seen:
                    continue
                seen.add(entity_id)
                entity_ids.append(entity_id)
                entity_names.append(str(hit.get("name") or ""))
                scores.append(score)
        return entity_ids, entity_names, scores

    async def coarse_rank_events(
        self,
        event_ids: list[str],
        source_config_ids: list[str],
        *,
        max_events: int,
        query_vector: list[float],
    ) -> list[dict[str, Any]]:
        """Score a known event set directly through the event vector repository."""

        if not event_ids:
            return []
        results = await self.search_events_by_vector(
            query_vector=query_vector,
            k=max_events,
            source_config_ids=source_config_ids,
            event_ids=event_ids,
        )
        return [
            {"event_id": str(item.get("event_id") or ""), "score": float(item.get("_score") or 0.0)}
            for item in results
            if item.get("event_id")
        ]

    async def fetch_event_chunks(self, event_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Hydrate event chunks through the canonical event-universe port."""

        return await self._event_universe_store.get_chunks_by_event_ids(event_ids)
