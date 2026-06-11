"""
多元事项检索器 (ES-First 版本)

与 multi.py 的区别：
- Step3 通道1: ES event_entity_vectors 替换 MySQL EventEntity JOIN SourceEvent
- Step4:       ES event_vectors 替换 MySQL SourceEvent + EventEntity 两张表查询
- Step5:       ES event_entity_vectors + event_vectors 替换 MySQL 多表 JOIN
- Step7:       LLM 过滤 prompt 去掉 thought_process，只返回 ID 列表（减少输出 token）
- Step8:       保留 MySQL（event_vectors 索引不含 chunk_id 字段）

其他 Step1/2/6 与 multi.py 完全一致。

使用示例：
    from pipeline.modules.search.multi_vector import MultiSearcher, MultiConfig

    config = MultiConfig(multi_top_k=20, similarity_threshold=0.4)
    searcher = MultiSearcherES()
    results = await searcher.search(
        query="海尔集团人单合一模式",
        source_config_ids=["source_1", "source_2"],
        config=config,
    )
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from pipeline.core.ai.factory import create_llm_client, get_embedding_client
from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.core.storage.elasticsearch import get_es_client
from pipeline.core.storage.repositories.entity_repository import EntityVectorRepository
from pipeline.core.storage.repositories.event_repository import EventVectorRepository
from pipeline.core.storage.repositories.event_entity_repository import EventEntityRepository
from pipeline.core.storage.repositories.source_chunk_repository import SourceChunkRepository
from pipeline.db import SourceChunk, SourceEvent, get_session_factory
from pipeline.modules.search.config import MultiConfig
from pipeline.modules.search.multi import (
    _NER_ONE_SHOT_INPUT,
    _NER_ONE_SHOT_OUTPUT,
    _NER_SYSTEM_PROMPT,
    _NER_TEMPLATE,
)
from pipeline.utils import get_logger

logger = get_logger("search.multi_es")

# ── LLM 过滤提示词（本地版本：去掉 thought_process，只要求返回 ID 列表）──

_RERANK_SYSTEM_PROMPT_LOCAL = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly {top_k} relationships most useful for answering this multi-hop question.

Return JSON with "useful_relations" (list of {top_k} index numbers, most useful first)."""

_RERANK_EXAMPLE_1_INPUT_LOCAL = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "useful_relations" (list of 5 index numbers, most useful first).

Question:
When did Lothair Ii's mother die?

Relationship descriptions:
[53] bertha married to theobald of arles
[54] bertha married to adalbert ii of tuscany
[42] lothair ii son of ermengarde of tours
[43] lothair ii married to teutberga
[41] lothair ii son of emperor lothair i
[60] lothair ii husband of waldrada
[67] waldrada was mistress of lothair ii
"""

_RERANK_EXAMPLE_1_OUTPUT_LOCAL = """{"useful_relations": ["42", "41", "43", "60", "67"]}"""

_RERANK_EXAMPLE_2_INPUT_LOCAL = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
What country is the composer of "Erta Eterna" from?

Relationship descriptions:
[12] terra eterna composed by paulo flores
[15] paulo flores born in angola
[18] paulo flores genre is semba
[22] angola located in africa
[25] semba originated in angola
[30] paulo flores nationality angolan
"""
_RERANK_EXAMPLE_2_OUTPUT_LOCAL = """{"useful_relations": ["12", "15", "30", "22", "25"]}"""

_RERANK_EXAMPLE_3_INPUT_LOCAL = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
Who is the director of the film that won the award also won by "The Hurt Locker"?

Relationship descriptions:
[5] the hurt locker won academy award best picture
[8] the hurt locker directed by kathryn bigelow
[12] moonlight won academy award best picture
[15] moonlight directed by barry jenkins
[20] la la land won golden globe best musical
[25] barry jenkins born in miami
"""  
_RERANK_EXAMPLE_3_OUTPUT_LOCAL = """{"useful_relations": ["5", "12", "15", "8", "25"]}"""

_RERANK_TEMPLATE_LOCAL = """Question:
{question}

Relationship descriptions:
{relations}
"""


@dataclass
class MultiSearchState:
    entity_ids: set[str] = field(default_factory=set)
    relation_ids: set[str] = field(default_factory=set)


class MultiSearcherES:
    """
    多元事项检索器 (ES-First 版本)

    检索多元事项（每个事项包含 >= 3 个实体）。

    与 MultiSearcher 的区别：
    - 数据访问层全面使用 ES，仅 Step8 保留 MySQL
    - 减少 MySQL 连接和 JOIN 查询
    - event_vectors.entity_ids 字段直接提供 event→entity 映射
    - event_entity_vectors 索引提供 entity→event 反向查询
    """

    def __init__(self, config: Optional[MultiConfig] = None):
        self._llm_client = None
        self._ner_llm_client = None
        self._embedding_client = None
        es_client = get_es_client()
        self._entity_repo = EntityVectorRepository(es_client)
        self._event_repo = EventVectorRepository(es_client)
        self._event_entity_repo = EventEntityRepository(es_client)
        self._chunk_repo = SourceChunkRepository(es_client)
        self._spacy_nlp = None
        self._spacy_model_name = None

        if config and self._resolve_search_mode(config) == "fast":
            self._get_spacy_nlp(self._resolve_spacy_model(config))

    # ── 延迟初始化 ──────────────────────────────────────────────

    def _resolve_search_mode(self, config: Optional[Any]) -> str:
        """
        解析 multi_vector 的搜索模式。

        fast:   快速模式，使用 spaCy NER + 小规模扩展
        precise: 精准模式，使用 spaCy NER + LLM 过滤

        不指定时默认 fast。
        """
        default_mode = MultiConfig.model_fields["mode"].default
        raw_mode = getattr(config, "mode", default_mode) if config is not None else default_mode
        mode = str(raw_mode).strip().lower()
        if mode in ("precise", "fast"):
            return mode
        raise ValueError("multi_vector mode 仅支持 fast 或 precise")

    def _resolve_ner_backend(self, config: Optional[Any]) -> str:
        return "spacy"

    def _resolve_spacy_model(self, config: Optional[Any]) -> str:
        default_model = MultiConfig.model_fields["spacy_model"].default
        if config is None:
            return default_model
        return getattr(config, "spacy_model", None) or default_model

    def _resolve_ranking_strategy(self, config: Optional[Any]) -> str:
        mode = self._resolve_search_mode(config)
        if mode == "fast":
            return "fast"
        return "coarse_llm"

    async def _get_llm_client(self):
        if self._llm_client is None:
            self._llm_client = await create_llm_client(scenario="search")
        return self._llm_client

    async def _get_ner_llm_client(self):
        """Step1 专用：小模型做实体提取，通过 LLM_NER_MODEL 环境变量指定"""
        if self._ner_llm_client is None:
            from pathlib import Path

            from dotenv import dotenv_values

            # Pydantic Settings extra="ignore" 不会加载未知字段到 os.environ
            # 所以需要直接解析 .env 文件
            env_file = Path(__file__).parent.parent.parent.parent / ".env"
            env_vars = dotenv_values(str(env_file))
            ner_model = env_vars.get("LLM_NER_MODEL", "") or os.environ.get("LLM_NER_MODEL", "")

            if ner_model:
                self._ner_llm_client = await create_llm_client(
                    scenario="search",
                    model_config={"model": ner_model},
                )
                logger.info(f"[NER] 使用独立模型: {ner_model}")
            else:
                self._ner_llm_client = await self._get_llm_client()
                logger.info("[NER] 未设置 LLM_NER_MODEL，复用主 LLM 客户端")
        return self._ner_llm_client

    def _get_entity_repo(self) -> EntityVectorRepository:
        return self._entity_repo

    def _get_event_repo(self) -> EventVectorRepository:
        return self._event_repo

    def _get_event_entity_repo(self) -> EventEntityRepository:
        return self._event_entity_repo

    async def _get_embedding_client(self):
        if self._embedding_client is None:
            self._embedding_client = await get_embedding_client(scenario="general")
        return self._embedding_client

    def _get_spacy_nlp(self, model_name: str):
        if self._spacy_nlp is None or self._spacy_model_name != model_name:
            try:
                import spacy
            except ImportError as exc:
                raise RuntimeError(
                    "multi_vector fast 模式需要安装 spaCy：pip install spacy"
                ) from exc

            try:
                self._spacy_nlp = spacy.load(model_name)
            except OSError as exc:
                raise RuntimeError(
                    f"无法加载 spaCy 模型 {model_name!r}。请先安装，例如："
                    f"python -m spacy download {model_name}"
                ) from exc
            self._spacy_model_name = model_name
            logger.info(f"[NER] 使用 spaCy 模型: {model_name}")
        return self._spacy_nlp

    async def warmup(self, config: Optional[MultiConfig] = None) -> None:
        config = config or MultiConfig()
        await self._get_embedding_client()
        mode = self._resolve_search_mode(config)
        ner_backend = self._resolve_ner_backend(config)
        ranking_strategy = self._resolve_ranking_strategy(config)
        if ranking_strategy == "coarse_llm":
            await self._get_llm_client()
        if ner_backend == "llm":
            await self._get_ner_llm_client()
        elif ner_backend == "spacy":
            self._get_spacy_nlp(self._resolve_spacy_model(config))
        else:
            raise ValueError(f"不支持的 NER 后端: {ner_backend}")
        logger.info(f"[multi_vector] mode={mode}, ner={ner_backend}, ranking={ranking_strategy}")

    # ── Step1: 实体提取 ─────────────────────────────────────────

    async def step1_extract_entities_llm(self, query: str) -> List[str]:
        llm_client = self._ner_llm_client
        if llm_client is None:
            raise RuntimeError("NER LLM client 未初始化，请先调用 await searcher.warmup(config)")

        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=_NER_SYSTEM_PROMPT),
            LLMMessage(role=LLMRole.USER, content=_NER_ONE_SHOT_INPUT),
            LLMMessage(role=LLMRole.ASSISTANT, content=_NER_ONE_SHOT_OUTPUT),
            LLMMessage(role=LLMRole.USER, content=_NER_TEMPLATE.format(query)),
        ]

        response = await llm_client.chat_with_schema(
            messages,
            response_schema={
                "type": "object",
                "properties": {
                    "named_entities": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["named_entities"],
            },
        )

        entities = response.get("named_entities", response.get("entities", []))
        entities = [str(e).strip() for e in entities if e]

        logger.info(f"[entity.extract.llm] entities={entities}")
        return entities

    async def step1_extract_entities_spacy(
        self,
        query: str,
        model_name: str = "en_core_web_sm",
    ) -> List[str]:
        nlp = self._get_spacy_nlp(model_name)
        doc = nlp(query)
        entities: List[str] = []
        seen: set = set()
        for ent in doc.ents:
            text = ent.text.strip()
            if text and text not in seen:
                seen.add(text)
                entities.append(text)

        logger.info(
            f"[entity.extract.spacy] model={model_name}, entities={entities}"
        )
        return entities

    async def step1_extract_entities_by_mode(
        self,
        query: str,
        mode: str = "llm",
        spacy_model: Optional[str] = None,
    ) -> List[str]:
        if mode == "llm":
            return await self.step1_extract_entities_llm(query)
        if mode == "spacy":
            spacy_model = spacy_model or MultiConfig.model_fields["spacy_model"].default
            return await self.step1_extract_entities_spacy(query, spacy_model)
        raise ValueError(f"不支持的实体提取模式: {mode}，支持: llm, spacy")

    # ── Step2: 实体向量检索 ─────────────────────────────────────

    async def step2_retrieve_entities(
        self,
        query_entities: List[str],
        source_config_ids: List[str],
        *,
        query_vectors: List[List[float]],
        entity_top_k: int,
        key_similarity_threshold: float,
        state: MultiSearchState,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Step2: 根据 query 提取的实体名称，从 ES 向量检索相似实体

        每个查询实体最多找到 entity_top_k 个，相似度分数必须 >= key_similarity_threshold。

        Args:
            query_entities: Step1 提取的实体名称列表
            source_config_ids: 信息源 ID 列表
            entity_top_k: 每个查询实体检索的最大数量，必须由 search() 显式传入
            key_similarity_threshold: 实体最低相似度阈值，必须由 search() 显式传入

        Returns:
            entity_ids: 去重后的实体 ID 列表
        """
        if not query_entities:
            return []

        top_k = entity_top_k
        threshold = key_similarity_threshold

        repo = self._entity_repo

        # query_entities 已在 Step1 后统一清洗，这里只做数量校验。
        if not query_entities:
            return []

        if query_vectors is None:
            raise RuntimeError("Step2 需要传入 query_vectors，请在 search() 中批量生成后复用")
        if len(query_vectors) != len(query_entities):
            raise RuntimeError(
                f"Step2 query_vectors 数量不匹配: vectors={len(query_vectors)}, entities={len(query_entities)}"
            )

        t_es = time.perf_counter()
        search_tasks = [
            repo.search_similar(
                query_vector=query_vector,
                k=top_k,
                source_config_ids=source_config_ids,
            )
            for query_vector in query_vectors
        ]
        batch_results = await asyncio.gather(*search_tasks)
        if timings is not None:
            timings["step2_entity_es_search"] = time.perf_counter() - t_es

        entity_ids: List[str] = []
        scores_for_log: List[float] = []
        seen: set = set()

        for results in batch_results:
            for hit in results:
                score = hit.get("_score", 0.0)
                if score < threshold:
                    continue
                eid = hit.get("entity_id", "")
                if eid and eid not in seen:
                    seen.add(eid)
                    entity_ids.append(eid)
                    scores_for_log.append(score)
                    state.entity_ids.add(eid)

        logger.info(
            f"[entity.search] query_entities={query_entities} -> "
            f"batch_queries={len(query_entities)}, "
            f"retrieved {len(entity_ids)} entities, "
            f"top_scores={scores_for_log[:5] if scores_for_log else []}"
        )
        return entity_ids

    # ── Step3: 双通道召回 (ES-First) ────────────────────────────

    async def step3_retrieve_events(
        self,
        query: str,
        source_config_ids: List[str],
        *,
        query_vector: List[float],
        multi_top_k: int,
        similarity_threshold: float,
        entity_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step3: 双通道召回 + 去重合并

        通道1 (entity→event): ES event_entity_vectors 批量查询
        通道2 (query→event): ES event_vectors title_vector kNN

        两个通道结果按 event_id 去重合并，仅返回 event_id 和 score。

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            entity_ids: Step2 检索到的实体 ID（可选，用于通道1）
            multi_top_k: 通道2 query→event 最大数量
            similarity_threshold: 通道2 向量最低相似度阈值

        Returns:
            [{"event_id": str, "score": float}, ...]
        """
        threshold = similarity_threshold

        merged: Dict[str, float] = {}

        if query_vector is None:
            raise RuntimeError("Step3 需要传入 query_vector，请在 search() 中批量生成后复用")

        event_repo = self._event_repo
        channel_tasks = []
        entity_event_task_index: Optional[int] = None

        # ── 通道1: entity → event（ES event_entity_vectors）──
        if entity_ids:
            ee_repo = self._event_entity_repo
            entity_event_task_index = len(channel_tasks)
            channel_tasks.append(
                ee_repo.get_event_ids_by_entity_ids(
                    entity_ids=entity_ids,
                    source_config_ids=source_config_ids,
                )
            )

        # ── 通道2: query → event（ES event_vectors title_vector kNN）──
        query_event_task_index = len(channel_tasks)
        channel_tasks.append(
            event_repo.search_similar_by_content(
                query_vector=query_vector,
                k=multi_top_k * 3,
                source_config_ids=source_config_ids,
            )
        )

        channel_results = await asyncio.gather(*channel_tasks)

        if entity_event_task_index is not None:
            event_ids_from_entities = channel_results[entity_event_task_index]
            db_count = 0
            for eid in event_ids_from_entities:
                merged[eid] = 0.0
                db_count += 1
        else:
            db_count = 0

        es_results = channel_results[query_event_task_index]

        es_count = 0
        es_new_count = 0

        for hit in es_results:
            if es_count >= multi_top_k:
                break

            score = hit.get("_score", 0.0)
            if score < threshold:
                continue

            eid = hit.get("event_id", "")
            if not eid:
                continue

            if eid not in merged:
                es_new_count += 1
            merged[eid] = score
            es_count += 1

        items = [{"event_id": eid, "score": score} for eid, score in merged.items()]

        logger.info(
            f"[event.recall] entity_to_event={db_count}, query_to_event={es_new_count}, "
            f"merged={len(items)}"
        )
        return items

    @staticmethod
    def _merge_fast_event_channels(
        event1: List[Dict[str, Any]],
        event2: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """合并 event1/entity 通道和 event2/query 通道，保留最高向量分和通道来源。"""
        merged: Dict[str, Dict[str, Any]] = {}

        def add_events(events: List[Dict[str, Any]], channel: str) -> None:
            for item in events:
                eid = item.get("event_id")
                if not eid:
                    continue
                score = float(item.get("score", 0.0) or 0.0)
                if eid not in merged:
                    merged[eid] = {
                        "event_id": eid,
                        "score": score,
                        "vector_score": score,
                        "channels": [channel],
                        "entity_channel_score": None,
                        "query_channel_score": None,
                    }
                elif score > float(merged[eid].get("vector_score", 0.0) or 0.0):
                    merged[eid]["score"] = score
                    merged[eid]["vector_score"] = score

                if channel not in merged[eid]["channels"]:
                    merged[eid]["channels"].append(channel)
                score_key = f"{channel}_channel_score"
                prev_score = merged[eid].get(score_key)
                if prev_score is None or score > float(prev_score):
                    merged[eid][score_key] = score

        add_events(event1, "entity")
        add_events(event2, "query")

        results = list(merged.values())
        results.sort(key=lambda item: item.get("vector_score", 0.0), reverse=True)
        return results

    @staticmethod
    def _score_fast_events_with_entity_boost(
        events: List[Dict[str, Any]],
        seed_entity_ids: List[str],
        event_entities: Dict[str, List[str]],
        config: MultiConfig,
    ) -> List[Dict[str, Any]]:
        """只用于 expand 前 seed 事项选择：向量分 + 是否命中 seed entity + 双通道奖励。"""
        if not events:
            return []

        vector_scores = [float(item.get("vector_score", item.get("score", 0.0)) or 0.0) for item in events]
        min_score = min(vector_scores)
        max_score = max(vector_scores)
        score_range = max_score - min_score
        seed_entity_set = set(seed_entity_ids)

        scored: List[Dict[str, Any]] = []
        for item in events:
            eid = item["event_id"]
            raw_vector_score = float(item.get("vector_score", item.get("score", 0.0)) or 0.0)
            if score_range > 1e-9:
                vector_score_norm = (raw_vector_score - min_score) / score_range
            else:
                vector_score_norm = 1.0

            matched_entity_ids = sorted(set(event_entities.get(eid) or []) & seed_entity_set)
            entity_hit_score = 1.0 if matched_entity_ids else 0.0
            channel_score = 1.0 if len(item.get("channels") or []) > 1 else 0.0
            final_score = (
                config.fast_vector_weight * vector_score_norm
                + config.fast_entity_weight * entity_hit_score
                + config.fast_channel_weight * channel_score
            )

            scored_item = dict(item)
            scored_item["score"] = final_score
            scored_item["seed_score"] = final_score
            scored_item["vector_score_norm"] = vector_score_norm
            scored_item["entity_hit_score"] = entity_hit_score
            scored_item["entity_hit_count"] = len(matched_entity_ids)
            scored_item["matched_entity_ids"] = matched_entity_ids
            scored_item["channel_score"] = channel_score
            scored.append(scored_item)

        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored

    @staticmethod
    def _merge_fast_and_expanded_for_final_rank(
        seed_items: List[Dict[str, Any]],
        expanded_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        合并第一跳 seed 事项和扩展事项，并按 query-content 相似度统一排序。

        fast 第一跳的 seed_score 只负责挑选第一跳事项；最终展示顺序只看
        final_similarity_score，避免实体增强分影响最后的 chunk 顺序。
        """
        final_items: List[Dict[str, Any]] = []
        seen_event_ids: set = set()
        for source_name, source_items in (
            ("seed", seed_items),
            ("expanded", expanded_items),
        ):
            for item in source_items:
                eid = item.get("event_id")
                if not eid or eid in seen_event_ids:
                    continue
                seen_event_ids.add(eid)
                final_item = dict(item)
                final_similarity_score = float(
                    final_item.get("vector_score", final_item.get("score", 0.0)) or 0.0
                )
                final_item["final_similarity_score"] = final_similarity_score
                final_item["score"] = final_similarity_score
                final_item["fast_stage"] = source_name
                final_items.append(final_item)

        final_items.sort(
            key=lambda item: item.get("final_similarity_score", 0.0),
            reverse=True,
        )
        return final_items

    async def step3_fast_recall(
        self,
        query: str,
        source_config_ids: List[str],
        *,
        entity_ids: List[str],
        query_vector: List[float],
        config: MultiConfig,
        timings: Dict[str, float],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        fast Step3:
        1. key -> entity topN
        2. entity -> event candidates -> query-content similarity top20 (event1)
        3. query -> event top20 (event2)
        4. union(event1,event2) -> 用 entity 命中二值增强分选 seed top5
        """
        seed_entity_ids = entity_ids[: config.fast_entity_k]

        t0 = time.perf_counter()
        candidate_event_ids = []
        if seed_entity_ids:
            candidate_event_ids = await self._event_entity_repo.get_event_ids_by_entity_ids(
                entity_ids=seed_entity_ids,
                source_config_ids=source_config_ids,
                size=config.fast_entity_event_candidate_k,
            )
        timings["step3_seed_entity_to_event_candidates"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        event1 = await self.step6_coarse_rank(
            query=query,
            event_ids=candidate_event_ids,
            source_config_ids=source_config_ids,
            query_vector=query_vector,
            max_events=config.fast_entity_event_k,
        )
        timings["step3_seed_event1_rank"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        event2_hits = await self._event_repo.search_similar_by_content(
            query_vector=query_vector,
            k=config.fast_query_event_k * 3,
            source_config_ids=source_config_ids,
        )
        event2: List[Dict[str, Any]] = []
        for hit in event2_hits:
            if len(event2) >= config.fast_query_event_k:
                break
            score = float(hit.get("_score", 0.0) or 0.0)
            if score < config.similarity_threshold:
                continue
            eid = hit.get("event_id", "")
            if eid:
                event2.append({"event_id": eid, "score": score})
        timings["step3_seed_event2_query"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        merged = self._merge_fast_event_channels(event1, event2)
        timings["step3_seed_merge"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        merged_event_ids = [item["event_id"] for item in merged]
        merged_event_fields = await self.step4_fetch_events(
            merged_event_ids,
            source_includes=["entity_ids"],
            log_label="seed候选事项实体",
        )
        merged_event_entities = {
            eid: fields.get("entity_ids") or []
            for eid, fields in merged_event_fields.items()
            if fields.get("entity_ids")
        }
        timings["step3_seed_fetch_candidate_entities"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        scored = self._score_fast_events_with_entity_boost(
            events=merged,
            seed_entity_ids=seed_entity_ids,
            event_entities=merged_event_entities,
            config=config,
        )
        seed_items = scored[: config.fast_answer_k]
        timings["step3_seed_score"] = time.perf_counter() - t0

        logger.info(
            f"[event.seed] entities={len(seed_entity_ids)}/{len(entity_ids)}, "
            f"entity_candidates={len(candidate_event_ids)}, "
            f"event1={len(event1)}, event2={len(event2)}, "
            f"merged={len(merged)}, seed={len(seed_items)}, "
            f"entity_boosted={sum(1 for item in scored if item.get('entity_hit_score'))}"
        )
        return seed_items, {"event1": event1, "event2": event2, "merged": scored}

    # ── Step4: 事项实体 / 内容查询 (ES-First) ───────────────────

    async def step4_fetch_events(
        self,
        event_ids: List[str],
        source_includes: List[str],
        log_label: str = "事项",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Step4: 按需从 ES event_vectors 查询事项字段。

        通过 source_includes 控制返回字段：
        - Step5 扩展阶段只查 entity_ids
        - Step7 LLM 精选前只查 content

        Args:
            event_ids: 事项 ID 列表
            source_includes: 需要从 ES 返回的字段列表，不需要包含 event_id
            log_label: 日志里的阶段标签

        Returns:
            {event_id: {field: value, ...}}
        """
        if not event_ids:
            return {}

        includes = ["event_id"]
        for field in source_includes:
            if field != "event_id" and field not in includes:
                includes.append(field)

        event_repo = self._event_repo
        events = await event_repo.get_events_by_ids(
            event_ids,
            source_includes=includes,
        )

        event_map: Dict[str, Dict[str, Any]] = {}
        event_entity_relations = 0

        for event in events:
            eid = event.get("event_id", "")
            if not eid:
                continue
            item = {
                field: event.get(field)
                for field in includes
                if field != "event_id"
            }
            entity_ids = item.get("entity_ids")
            if isinstance(entity_ids, list):
                event_entity_relations += len(entity_ids)
            event_map[eid] = item

        extra = ""
        if "entity_ids" in includes:
            extra = f", event_entity_relations={event_entity_relations}"

        logger.info(
            f"[event.fetch] label={log_label}, input_event_ids={len(event_ids)}, "
            f"found_events={len(event_map)}{extra}"
        )
        return event_map

    # ── 去重辅助 ────────────────────────────────────────────────

    def get_new_entity_ids(
        self,
        event_entities: Dict[str, List[str]],
        state: MultiSearchState,
    ) -> List[str]:
        """
        从 event_entities 中找出当前搜索尚未访问过的实体 ID

        用于扩展时发现新实体，决定是否需要进一步检索。

        Args:
            event_entities: {event_id: [entity_id, ...]}

        Returns:
            新的实体 ID 列表（去重）
        """
        new_ids: List[str] = []
        seen_ids = set()
        total = 0
        for entity_ids in event_entities.values():
            for eid in entity_ids:
                if not eid or eid in seen_ids:
                    continue
                seen_ids.add(eid)
                total += 1
                if eid not in state.entity_ids:
                    new_ids.append(eid)
        logger.info(
            f"[entity.dedupe] total={total}, "
            f"already_tracked={total - len(new_ids)}, "
            f"new={len(new_ids)}"
        )
        return new_ids

    # ── Step5: 多跳扩展 (ES-First) ──────────────────────────────

    async def step5_expand(
        self,
        event_entities: Dict[str, List[str]],
        *,
        max_hops: int,
        max_expand_events_per_hop: int,
        state: MultiSearchState,
        source_config_ids: Optional[List[str]] = None,
        timings: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Step5: 多跳扩展

        逻辑：
          hop=0: entity_set = step2 实体, relation_set = step3 合并事件
          hop=N: prev_hop_entities → 新 entity_ids (不在 entity_set)
                 新 entity_ids → 新 event_ids (不在 relation_set)
                 更新两个 set, prev_hop_entities = 本跳新事件的 entities

        ES 版本：
        - entity → event: event_entity_vectors.get_event_ids_by_entity_ids()
        - event → detail + entities: event_vectors.get_events_by_ids()

        Args:
            event_entities: Step4 返回的 {event_id: [entity_id, ...]}
            source_config_ids: 信息源 ID 列表（可选）
            max_hops: 最大跳数，必须由 search() 显式传入
            max_expand_events_per_hop: 每跳 entity->event 最多召回事项数量，必须由 search() 显式传入

        Returns:
            expanded_event_ids: 所有扩展轮次召回的新事项 ID
        """
        expanded_event_ids: List[str] = []

        # hop=0: 初始化 relation_set（entity_set 已由 step2 填充）
        state.relation_ids.update(event_entities.keys())

        if max_hops == 0:
            return expanded_event_ids

        # 上一跳的 event_entities，用于每轮发现新 entity_ids
        prev_hop_entities = event_entities

        ee_repo = self._event_entity_repo
        for hop in range(max_hops):
            pre_events = len(state.relation_ids)
            pre_entities = len(state.entity_ids)

            # 1. 从上一跳 events 找新 entity_ids（不在 entity_set 中）
            t_step = time.perf_counter()
            new_entity_ids = self.get_new_entity_ids(prev_hop_entities, state)
            if timings is not None:
                timings["step5_get_new_entities"] = (
                    timings.get("step5_get_new_entities", 0.0)
                    + time.perf_counter() - t_step
                )

            if not new_entity_ids:
                logger.info(
                    f"[event.expand] hop={hop+1}/{max_hops} "
                    f"no_new_entities tracked_entities={len(state.entity_ids)}"
                )
                break

            # 2. 新 entity_ids 加入 entity_set
            t_step = time.perf_counter()
            state.entity_ids.update(new_entity_ids)
            if timings is not None:
                timings["step5_update_entities"] = (
                    timings.get("step5_update_entities", 0.0)
                    + time.perf_counter() - t_step
                )

            logger.info(
                f"[event.expand] hop={hop+1}/{max_hops} "
                f"entities: {pre_entities} -> +{len(new_entity_ids)} new, "
                f"total={len(state.entity_ids)}"
            )

            # 3. 新 entity_ids → ES event_entity_vectors 查新 event_ids
            #    event_entity_vectors 自带 source_config_id，无需 JOIN
            t_step = time.perf_counter()
            all_new_event_ids = await ee_repo.get_event_ids_by_entity_ids(
                entity_ids=new_entity_ids,
                source_config_ids=source_config_ids,
                exclude_event_ids=list(state.relation_ids),
                size=max_expand_events_per_hop,
            )
            if timings is not None:
                timings["step5_entity_to_event"] = (
                    timings.get("step5_entity_to_event", 0.0)
                    + time.perf_counter() - t_step
                )

            new_event_ids = all_new_event_ids

            if not new_event_ids:
                logger.info(
                    f"[event.expand] hop={hop+1}/{max_hops} "
                    f"no_new_events tracked_events={len(state.relation_ids)}"
                )
                break

            expanded_event_ids.extend(new_event_ids)
            is_last_hop = hop == max_hops - 1

            if is_last_hop:
                # 最后一跳的 event_ids 只需要进入 Step6 粗排，不再反查 entity_ids。
                t_step = time.perf_counter()
                state.relation_ids.update(new_event_ids)
                if timings is not None:
                    timings["step5_update_state"] = (
                        timings.get("step5_update_state", 0.0)
                        + time.perf_counter() - t_step
                    )

                logger.info(
                    f"[event.expand] hop={hop+1}/{max_hops} done: "
                    f"events {pre_events} -> {len(state.relation_ids)} (+{len(new_event_ids)}), "
                    f"entities {pre_entities} -> {len(state.entity_ids)}, "
                    f"limit={max_expand_events_per_hop}, "
                    f"last_hop_skip_event_entities=True"
                )
                break

            # 4. 非最后一跳才查新事项实体，供下一跳继续扩展
            t_step = time.perf_counter()
            hop_event_fields = await self.step4_fetch_events(
                new_event_ids,
                source_includes=["entity_ids"],
                log_label="事项实体",
            )
            hop_entities = {
                eid: fields.get("entity_ids") or []
                for eid, fields in hop_event_fields.items()
                if fields.get("entity_ids")
            }
            if timings is not None:
                timings["step5_fetch_event_entities"] = (
                    timings.get("step5_fetch_event_entities", 0.0)
                    + time.perf_counter() - t_step
                )

            # 5. 新 event_ids 加入 relation_set
            t_step = time.perf_counter()
            state.relation_ids.update(new_event_ids)

            # 6. 保存本跳结果，供下一跳使用
            prev_hop_entities = hop_entities
            if timings is not None:
                timings["step5_update_state"] = (
                    timings.get("step5_update_state", 0.0)
                    + time.perf_counter() - t_step
                )

            logger.info(
                f"[event.expand] hop={hop+1}/{max_hops} done: "
                f"events {pre_events} -> {len(state.relation_ids)} (+{len(new_event_ids)}), "
                f"entities {pre_entities} -> {len(state.entity_ids)}, "
                f"limit={max_expand_events_per_hop}"
            )

        return expanded_event_ids

    # ── Step6: 粗排序 (ES) ──────────────────────────────────────

    async def step6_coarse_rank(
        self,
        query: str,
        event_ids: List[str],
        *,
        query_vector: List[float],
        max_events: int,
        source_config_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Step6: 粗排序

        用 query 向量在 ES event_vectors 中做 kNN 搜索，通过 event_ids 过滤，
        返回最多 max_events 条按相似度降序的结果。

        Args:
            query: 查询文本
            event_ids: 需要排序的事项 ID 列表
            source_config_ids: 信息源 ID 列表（可选）
            max_events: 最大返回数量，必须由 search() 显式传入

        Returns:
            [{"event_id": str, "score": float}, ...] 按相似度降序
        """
        if not event_ids:
            return []

        event_repo = self._event_repo
        results = await event_repo.search_similar_by_content(
            query_vector=query_vector,
            k=max_events,
            source_config_ids=source_config_ids,
            event_ids=event_ids,
        )

        scored = []
        for hit in results:
            eid = hit.get("event_id", "")
            score = hit.get("_score", 0.0)
            if eid:
                scored.append({"event_id": eid, "score": score})

        top_score_str = f"{scored[0]['score']:.4f}" if scored else "0"
        logger.info(
            f"[event.rank.coarse] input={len(event_ids)}, "
            f"returned={len(scored)}, "
            f"top_score={top_score_str}"
        )
        return scored

    # ── Step7: LLM 精选 ─────────────────────────────────────────
    # （与 multi.py 完全一致）

    def _parse_llm_filter_response(
        self,
        useful_relations: List[str],
        valid_ids: set,
    ) -> List[str]:
        """解析 LLM 返回的 useful_relations（纯 index 字符串列表），去重 + 校验"""
        selected: List[str] = []
        for rel_id in useful_relations:
            rel_id = str(rel_id).strip()
            if rel_id in valid_ids and rel_id not in selected:
                selected.append(rel_id)
        return selected

    async def step7_llm_filter(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Step7: LLM 精选最相关的多元事项

        将候选事项格式化为 [id] content，通过 few-shot prompt 让 LLM
        挑选 top_k 条最相关的事项，解析响应并映射回原始数据。

        Args:
            query: 查询文本
            items: 候选事项 [{event_id, content, score}]
            top_k: 精选返回数量

        Returns:
            筛选后的事项列表，保持 LLM 选择的顺序
        """
        if not items:
            return []

        top_k = min(top_k, len(items))

        # 1. 构建 idx → event_id 映射 + 格式化 relation 文本
        idx_to_event_id: Dict[str, str] = {}
        relation_lines: List[str] = []

        for i, item in enumerate(items):
            idx = str(i)
            idx_to_event_id[idx] = item["event_id"]
            text = item.get("content", "").strip()
            relation_lines.append(f"[{i}] {text}")

        relations_str = "\n".join(relation_lines)
        valid_ids = set(idx_to_event_id.keys())

        # 2. 构建 messages：SYSTEM + 3 组 few-shot + 最终 prompt
        system_prompt = _RERANK_SYSTEM_PROMPT_LOCAL.format(top_k=top_k)
        messages = [
            LLMMessage(role=LLMRole.SYSTEM, content=system_prompt),
            # few-shot 1
            LLMMessage(role=LLMRole.USER, content=_RERANK_EXAMPLE_1_INPUT_LOCAL),
            LLMMessage(role=LLMRole.ASSISTANT, content=_RERANK_EXAMPLE_1_OUTPUT_LOCAL),
            # few-shot 2
            LLMMessage(role=LLMRole.USER, content=_RERANK_EXAMPLE_2_INPUT_LOCAL),
            LLMMessage(role=LLMRole.ASSISTANT, content=_RERANK_EXAMPLE_2_OUTPUT_LOCAL),
            # few-shot 3
            LLMMessage(role=LLMRole.USER, content=_RERANK_EXAMPLE_3_INPUT_LOCAL),
            LLMMessage(role=LLMRole.ASSISTANT, content=_RERANK_EXAMPLE_3_OUTPUT_LOCAL),
            # 实际查询
            LLMMessage(
                role=LLMRole.USER,
                content=_RERANK_TEMPLATE_LOCAL.format(question=query, relations=relations_str),
            ),
        ]

        # 3. 调用 LLM
        llm_client = self._llm_client
        if llm_client is None:
            raise RuntimeError("LLM client 未初始化，请先调用 await searcher.warmup(config)")

        response = await llm_client.chat_with_schema(
            messages,
            response_schema={
                "type": "object",
                "properties": {
                    "useful_relations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["useful_relations"],
            },
        )

        # 打印完整输出
        logger.info(f"[event.filter.llm.raw] raw_response={response}")

        # 4. 解析 + 去重校验
        useful_relations = response.get("useful_relations", [])
        selected_indices = self._parse_llm_filter_response(
            useful_relations,
            valid_ids,
        )

        # 5. 映射回原始数据
        results = []
        event_id_to_item = {item["event_id"]: item for item in items}
        for idx in selected_indices[:top_k]:
            event_id = idx_to_event_id.get(idx)
            if event_id and event_id in event_id_to_item:
                results.append(event_id_to_item[event_id])

        return results

    # ── Step8: Chunk 查找 (MySQL) ────────────────────────────────

    async def step8_fetch_chunks(
        self,
        event_ids: List[str],
    ) -> Dict[str, Dict[str, str]]:
        """
        Step8: 根据 event_id 查找关联的 chunk (MySQL)

        event_vectors 索引不含 chunk_id 字段，因此此步骤保留 MySQL。

        source_event.chunk_id → source_chunk 查详情

        Args:
            event_ids: 事项 ID 列表

        Returns:
            {event_id: {"chunk_id": str, "heading": str, "content": str}}
        """
        if not event_ids:
            return {}

        session_factory = get_session_factory()
        event_chunk_map: Dict[str, str] = {}
        result_map: Dict[str, Dict[str, str]] = {}

        async with session_factory() as session:
            # 1. 查 event → chunk_id
            stmt = select(SourceEvent.id, SourceEvent.chunk_id).where(
                SourceEvent.id.in_(event_ids)
            )
            result = await session.execute(stmt)
            chunk_ids: set = set()
            for row in result.fetchall():
                eid, chunk_id = row[0], row[1]
                if chunk_id:
                    event_chunk_map[eid] = chunk_id
                    chunk_ids.add(chunk_id)

            if not chunk_ids:
                return {}

            # 2. 查 chunk 详情
            chunk_stmt = select(SourceChunk).where(SourceChunk.id.in_(chunk_ids))
            result = await session.execute(chunk_stmt)
            chunk_map: Dict[str, Dict[str, str]] = {}
            for chunk in result.scalars().all():
                chunk_map[chunk.id] = {
                    "chunk_id": chunk.id,
                    "source_id": chunk.source_id or "",
                    "source_config_id": chunk.source_config_id or "",
                    "heading": chunk.heading or "",
                    "content": chunk.content or "",
                    "rank": chunk.rank,
                }

            # 3. 按 event_id 映射
            for eid, chunk_id in event_chunk_map.items():
                if chunk_id in chunk_map:
                    result_map[eid] = chunk_map[chunk_id]

        logger.info(
            f"[chunk.fetch] events={len(event_ids)} -> "
            f"chunk_ids={len(chunk_ids)}, matched={len(result_map)}"
        )
        return result_map

    async def search_fast(
        self,
        query: str,
        source_config_ids: List[str],
        *,
        entity_ids: List[str],
        query_vector: List[float],
        config: MultiConfig,
        state: MultiSearchState,
        timings: Dict[str, float],
        t_total: float,
    ) -> Dict[str, Any]:
        """
        fast 专用流程：
        key->entity(5) -> entity-filtered event1(20 by query-content score)
        query->event2(20) -> seed top5 -> expand -> expanded top5
        -> seed/expanded 按 query-content 相似度统一排序 -> chunk。
        """
        state.entity_ids = set(entity_ids[: config.fast_entity_k])

        t0 = time.perf_counter()
        seed_items, _ = await self.step3_fast_recall(
            query=query,
            source_config_ids=source_config_ids,
            entity_ids=entity_ids,
            query_vector=query_vector,
            config=config,
            timings=timings,
        )
        timings["step3_fast_recall"] = time.perf_counter() - t0

        seed_event_ids = [item["event_id"] for item in seed_items]
        if not seed_event_ids:
            timings["total"] = time.perf_counter() - t_total
            return {
                "items": [],
                "_timings": timings,
                "_query_vector": query_vector,
            }

        t0 = time.perf_counter()
        seed_event_fields = await self.step4_fetch_events(
            seed_event_ids,
            source_includes=["entity_ids"],
            log_label="seed事项实体",
        )
        seed_event_entities = {
            eid: fields.get("entity_ids") or []
            for eid, fields in seed_event_fields.items()
            if fields.get("entity_ids")
        }
        timings["step4_fast_event_entities"] = time.perf_counter() - t0

        state.relation_ids.update(seed_event_ids)

        t0 = time.perf_counter()
        expanded_event_ids = await self.step5_expand(
            event_entities=seed_event_entities,
            source_config_ids=source_config_ids,
            max_hops=config.max_hops,
            max_expand_events_per_hop=config.max_expand_events_per_hop,
            state=state,
            timings=timings,
        )
        timings["step5_fast_expand"] = time.perf_counter() - t0

        expanded_items: List[Dict[str, Any]] = []
        if expanded_event_ids and config.fast_expand_answer_k > 0:
            t0 = time.perf_counter()
            expanded_items = await self.step6_coarse_rank(
                query=query,
                event_ids=expanded_event_ids,
                source_config_ids=source_config_ids,
                query_vector=query_vector,
                max_events=config.fast_expand_answer_k,
            )
            timings["step6_fast_expand_rank"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        final_items = self._merge_fast_and_expanded_for_final_rank(
            seed_items,
            expanded_items,
        )
        timings["step6_fast_final_rank"] = time.perf_counter() - t0
        rank_input_count = len(seed_items) + len(expanded_items)
        output_scores = [
            float(item.get("final_similarity_score", 0.0) or 0.0)
            for item in final_items
        ]
        min_output_score = min(output_scores) if output_scores else 0.0
        max_output_score = max(output_scores) if output_scores else 0.0
        logger.info(
            f"[fast.rank] input={rank_input_count}, output={len(final_items)}, "
            f"score_min={min_output_score:.4f}, score_max={max_output_score:.4f}"
        )

        t0 = time.perf_counter()
        filtered_event_ids = [item["event_id"] for item in final_items]
        chunk_map = await self.step8_fetch_chunks(filtered_event_ids)
        timings["step8_chunks"] = time.perf_counter() - t0

        deduped: List[Dict[str, Any]] = []
        seen_chunk_ids: set = set()
        for item in final_items:
            item["chunk"] = chunk_map.get(item["event_id"])
            chunk = item.get("chunk")
            if not chunk:
                deduped.append(item)
                continue
            cid = chunk.get("chunk_id")
            if cid and cid in seen_chunk_ids:
                continue
            if cid:
                seen_chunk_ids.add(cid)
            deduped.append(item)

        timings["total"] = time.perf_counter() - t_total
        logger.info(
            f"[fast.done] seed={len(seed_items)}, expanded_selected={len(expanded_items)}, "
            f"final_events={len(final_items)}, final_chunks={len(deduped)}"
        )
        return {
            "items": deduped,
            "_timings": timings,
            "_query_vector": query_vector,
        }

    # ── 主搜索接口 ──────────────────────────────────────────────

    async def search(
        self,
        query: str,
        source_config_ids: List[str],
        config: Optional[MultiConfig] = None,
    ) -> Dict[str, Any]:
        """
        搜索多元事项

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            config: MultiConfig 配置

        Returns:
            {
                "items": [
                    {
                        "event_id": str,
                        "content": str,
                        "score": float,
                        "chunk": {"chunk_id": str, "heading": str, "content": str} or None,
                    }
                ],
                "_timings": {"total": float}
            }
        """
        config = config or MultiConfig()
        state = MultiSearchState()
        timings: Dict[str, float] = {}
        t_total = time.perf_counter()

        logger.info(
            f"[multi_es.start] mode={config.mode}, ranking={self._resolve_ranking_strategy(config)}, "
            f"multi_top_k={config.multi_top_k}"
        )

        # Step1: 提取实体
        t0 = time.perf_counter()
        query_entities = await self.step1_extract_entities_by_mode(
            query=query,
            mode=self._resolve_ner_backend(config),
            spacy_model=self._resolve_spacy_model(config),
        )
        query_entities = [
            name.strip()
            for name in query_entities
            if name and name.strip()
        ]
        timings["step1_ner"] = time.perf_counter() - t0

        # Generate one embedding batch for the full query and extracted entities.
        t0 = time.perf_counter()
        embedding_client = self._embedding_client
        if embedding_client is None:
            raise RuntimeError("Embedding client not initialized; call await searcher.warmup(config) first")
        embedding_texts = [query] + query_entities
        embedding_vectors = await embedding_client.batch_generate(embedding_texts)
        query_vector = embedding_vectors[0]
        entity_query_vectors = embedding_vectors[1:]
        timings["step2_embedding"] = time.perf_counter() - t0

        # Step2: ES 向量检索实体
        entity_ids = await self.step2_retrieve_entities(
            query_entities=query_entities,
            source_config_ids=source_config_ids,
            query_vectors=entity_query_vectors,
            entity_top_k=config.entity_top_k,
            key_similarity_threshold=config.key_similarity_threshold,
            state=state,
            timings=timings,
        )
        timings["step2_entity_vector"] = time.perf_counter() - t0

        if self._resolve_ranking_strategy(config) == "fast":
            return await self.search_fast(
                query=query,
                source_config_ids=source_config_ids,
                entity_ids=entity_ids,
                query_vector=query_vector,
                config=config,
                state=state,
                timings=timings,
                t_total=t_total,
            )

        t0 = time.perf_counter()
        event_items = await self.step3_retrieve_events(
            query=query,
            source_config_ids=source_config_ids,
            entity_ids=entity_ids,
            query_vector=query_vector,
            multi_top_k=config.multi_top_k,
            similarity_threshold=config.similarity_threshold,
        )
        timings["step3_dual_recall"] = time.perf_counter() - t0

        event_ids = [item["event_id"] for item in event_items]
        if not event_ids:
            timings["total"] = time.perf_counter() - t_total
            return {
                "items": [],
                "_timings": timings,
                "_query_vector": query_vector,
            }

        t0 = time.perf_counter()
        event_fields = await self.step4_fetch_events(
            event_ids,
            source_includes=["entity_ids"],
            log_label="事项实体",
        )
        event_entities = {
            eid: fields.get("entity_ids") or []
            for eid, fields in event_fields.items()
            if fields.get("entity_ids")
        }
        timings["step4_event_entities"] = time.perf_counter() - t0

        # 初始召回的 event 进入黑名单，避免扩展阶段重复占位。
        state.relation_ids.update(event_ids)

        t0 = time.perf_counter()
        expanded_event_ids = await self.step5_expand(
            event_entities=event_entities,
            source_config_ids=source_config_ids,
            max_hops=config.max_hops,
            max_expand_events_per_hop=config.max_expand_events_per_hop,
            state=state,
            timings=timings,
        )
        timings["step5_expand"] = time.perf_counter() - t0

        all_event_ids = set(event_ids)
        all_event_ids.update(expanded_event_ids)

        t0 = time.perf_counter()
        candidate_items = await self.step6_coarse_rank(
            query=query,
            event_ids=list(all_event_ids),
            source_config_ids=source_config_ids,
            query_vector=query_vector,
            max_events=config.max_events,
        )
        timings["step6_coarse_rank"] = time.perf_counter() - t0
        candidate_scores = [
            float(item.get("score", 0.0) or 0.0)
            for item in candidate_items
        ]
        min_candidate_score = min(candidate_scores) if candidate_scores else 0.0
        max_candidate_score = max(candidate_scores) if candidate_scores else 0.0
        logger.info(
            f"[event.candidates] initial={len(event_ids)}, expanded={len(expanded_event_ids)}, "
            f"input={len(all_event_ids)}, output={len(candidate_items)}, "
            f"score_min={min_candidate_score:.4f}, score_max={max_candidate_score:.4f}"
        )

        t_step7 = time.perf_counter()
        t0 = time.perf_counter()
        candidate_event_ids = [item["event_id"] for item in candidate_items]
        event_contents = await self.step4_fetch_events(
            candidate_event_ids,
            source_includes=["content"],
            log_label="候选事项内容",
        )
        timings["step7_fetch_candidate_contents"] = time.perf_counter() - t0

        candidates = []
        for item in candidate_items:
            eid = item["event_id"]
            detail = event_contents.get(eid, {})
            candidates.append({
                "event_id": eid,
                "content": detail.get("content") or "",
                "score": item["score"],
            })

        t0 = time.perf_counter()
        items = await self.step7_llm_filter(
            query=query,
            items=candidates,
            top_k=config.max_sections,
        )
        timings["step7_llm_call"] = time.perf_counter() - t0
        timings["step7_llm_filter"] = time.perf_counter() - t_step7

        t0 = time.perf_counter()
        selected_event_ids = [item["event_id"] for item in items]
        chunk_map = await self.step8_fetch_chunks(selected_event_ids)
        timings["step8_chunks"] = time.perf_counter() - t0

        for item in items:
            item["chunk"] = chunk_map.get(item["event_id"])

        deduped: List[Dict[str, Any]] = []
        seen_chunk_ids: set = set()
        for item in items:
            chunk = item.get("chunk")
            if not chunk:
                deduped.append(item)
                continue
            cid = chunk.get("chunk_id")
            if cid and cid in seen_chunk_ids:
                continue
            if cid:
                seen_chunk_ids.add(cid)
            deduped.append(item)
        items = deduped

        logger.info(
            f"[precise.done] events={len(candidate_items)}, selected={len(items)}, "
            f"chunks={sum(1 for item in items if item.get('chunk'))}"
        )

        timings["total"] = time.perf_counter() - t_total
        return {
            "items": items,
            "_timings": timings,
            "_query_vector": query_vector,
        }

    # ── 段落返回兼容接口 ───────────────────────────────────────

    async def search_for_sections(
        self,
        query: str,
        source_config_ids: List[str],
        query_vector: Optional[List[float]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        多元事项检索，返回段落列表。

        保持旧搜索引擎需要的 {"sections": [...], "_timings": {...}} 返回格式。

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            query_vector: 可选的预计算向量（暂未使用）
            config: MultiConfig 或 SearchConfig 对象

        Returns:
            {"sections": [...], "_timings": {...}}
        """
        t_total = time.perf_counter()
        multi_config = config if isinstance(config, MultiConfig) else MultiConfig()

        result = await self.search(query, source_config_ids, multi_config)
        timings = result.get("_timings", {}).copy()
        if "total" in timings:
            timings.pop("total")

        seen_chunk_ids: set = set()
        sections = []
        for i, item in enumerate(result.get("items", [])):
            chunk = item.get("chunk")
            if not chunk:
                continue
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            sections.append({
                "chunk_id": chunk_id,
                "source_id": chunk["source_id"],
                "source_config_id": chunk["source_config_id"],
                "heading": chunk["heading"],
                "content": chunk["content"],
                "rank": chunk.get("rank", i),
                "score": item["score"],
                "weight": item["score"],
            })

        # Native 补充：去重后不足 max_sections 时，用 query→chunk 填充。
        target = multi_config.max_sections
        ranking_strategy = self._resolve_ranking_strategy(multi_config)
        if ranking_strategy == "coarse_llm" and len(sections) < target:
            multi_count = len(sections)
            supplement = await self.search_chunks(
                query=query,
                source_config_ids=source_config_ids,
                config=multi_config,
                query_vector=query_vector or result.get("_query_vector"),
            )
            supplement_timings = supplement.get("_timings", {})
            if "total" in supplement_timings:
                timings["native_chunk_total"] = supplement_timings["total"]
            native_added = 0
            for sec in supplement.get("sections", []):
                if sec["chunk_id"] in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(sec["chunk_id"])
                sections.append(sec)
                native_added += 1
                if len(sections) >= target:
                    break
            logger.info(
                f"[native.fill] multi={multi_count}, native=+{native_added}, "
                f"total={len(sections)}"
            )

        timings["total"] = time.perf_counter() - t_total

        return {
            "sections": sections[:target],
            "_timings": timings,
        }

    # 兼容旧 pipelineEngine 接口名；内部不执行模型 rerank。
    search_for_rerank = search_for_sections

    async def search_chunks(
        self,
        query: str,
        source_config_ids: List[str],
        config: Optional[MultiConfig] = None,
        query_vector: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Query→Chunk 直接向量检索

        跳过实体提取和多跳扩展，直接用 query 向量检索 chunk。
        用于简单场景或作为 Multi 管线的补充通道。

        Args:
            query: 查询文本
            source_config_ids: 信息源 ID 列表
            config: MultiConfig 配置

        Returns:
            {"sections": [...], "_timings": {"total": float}}
        """
        config = config or MultiConfig()
        start_time = time.perf_counter()

        if query_vector is None:
            raise RuntimeError("search_chunks 需要传入 query_vector，请复用 search() 中批量生成的 query 向量")

        es_results = await self._chunk_repo.search_similar_by_content(
            query_vector=query_vector,
            k=config.max_sections * 2,
            source_config_ids=source_config_ids,
        )

        sections = []
        for result in es_results:
            score = result.get("_score", 0.0)
            sections.append({
                "chunk_id": result.get("chunk_id"),
                "source_id": result.get("source_id"),
                "source_config_id": result.get("source_config_id"),
                "heading": result.get("heading"),
                "content": result.get("content"),
                "rank": result.get("rank"),
                "score": score,
                "weight": score,
            })

        sections = sorted(sections, key=lambda x: x["score"], reverse=True)[
            : config.max_sections
        ]
        total_time = time.perf_counter() - start_time

        logger.info(
            f"[query.chunk] returned={len(sections)}, total_time={total_time:.3f}s"
        )

        return {
            "sections": sections,
            "_timings": {"total": total_time},
        }


MultiSearcher = MultiSearcherES
ESFirstMultiSearcher = MultiSearcherES

__all__ = ["MultiSearcherES", "MultiSearcher", "ESFirstMultiSearcher"]
