"""各阶段证据覆盖率追踪器（EvidenceTracker）。

Shared by SAG2Searcher and evaluation flows.
自包含：仅依赖 logger + stdlib。
"""

from typing import Any

from pipeline.utils import get_logger

logger = get_logger("search.evidence")


class EvidenceTracker:
    """各阶段证据覆盖率追踪器。

    gold_evidences 为 gold paragraph text（"title\\ncontent"），每个 event 唯一对应一个 chunk。
    中间阶段只记录 event_id 列表（不查 chunk），Step8 后调用 resolve(chunk_map) 做逐阶段文本匹配。

    用法：
        tracker = EvidenceTracker(gold_evidences)
        ... 在 search() 各阶段调用 tracker.check(stage, event_ids) ...
        # Step8 后：用 chunk_map 反查每阶段 event 对应的段落文本，匹配 gold
        tracker.resolve(chunk_map)
        print(tracker.summary())  → 每阶段各自的 hit/total
    """

    def __init__(self, gold_evidences: set[str]):
        self._gold = {self._normalize(g) for g in (gold_evidences or set())}
        self._total = len(self._gold)
        # (stage, candidate_count, event_ids) 列表，event_ids 去重但保序
        self._stages: list[tuple[str, int, list[str]]] = []
        self._records: list[dict[str, Any]] = []

    @staticmethod
    def _normalize(text: str) -> str:
        t = (text or "").strip()
        while "\n\n\n" in t:
            t = t.replace("\n\n\n", "\n\n")
        lines = t.split("\n")
        return "\n".join(" ".join(line.split()) for line in lines).lower()

    @staticmethod
    def _chunk_to_text(c: dict[str, str]) -> str:
        heading = c.get("heading", "") or c.get("title", "") or ""
        content = c.get("content", "") or ""
        # 去掉 content 开头的 markdown 标题行（与 normalize_section 一致）
        lines = content.split("\n")
        if lines and lines[0].strip().lstrip("#").strip() == heading.strip():
            content = "\n".join(lines[1:]).lstrip("\n")
        return f"{heading}\n{content}"

    @property
    def total_evidence(self) -> int:
        return self._total

    def all_event_ids(self) -> list[str]:
        """收集所有阶段的全量 event_id（去重保序），供 Step8 批量查 chunk。"""
        all_ids: list[str] = []
        seen: set[str] = set()
        for _, _, event_ids in self._stages:
            for eid in event_ids:
                if eid not in seen:
                    seen.add(eid)
                    all_ids.append(eid)
        return all_ids

    def check(self, stage: str, candidate_ids: list[str]) -> None:
        """记录当前阶段的候选 event_id（不去重，留到 resolve 阶段做 chunk 文本匹配）。"""
        # 输入去重保序
        deduped: list[str] = []
        seen: set[str] = set()
        for eid in candidate_ids:
            if eid and eid not in seen:
                seen.add(eid)
                deduped.append(eid)
        self._stages.append((stage, len(deduped), deduped))
        logger.info(
            "[evidence] %s: candidates=%d",
            stage,
            len(deduped),
        )

    def resolve(self, event_id_to_chunk: dict[str, dict[str, str]]) -> None:
        """用 chunk_map 反查每阶段 event_id 对应的段落文本，匹配 gold paragraph。

        每个 event 唯一对应一个 chunk，因此每个 stage 的 hit 独立计算。
        """
        if not self._gold:
            return

        results: list[dict[str, Any]] = []
        for stage, cnt, event_ids in self._stages:
            chunk_texts: set[str] = set()
            for eid in event_ids:
                c = event_id_to_chunk.get(eid, {})
                if c:
                    chunk_texts.add(self._normalize(self._chunk_to_text(c)))
            hit_texts = chunk_texts & self._gold
            hit = len(hit_texts)
            cov = round(hit / self._total, 4) if self._total > 0 else 0.0
            # 反查命中的 event_id
            hit_event_ids = [
                eid
                for eid in event_ids
                if self._normalize(self._chunk_to_text(event_id_to_chunk.get(eid, {}))) in hit_texts
            ]
            results.append(
                {
                    "stage": stage,
                    "candidates": cnt,
                    "hit": hit,
                    "hit_event_ids": hit_event_ids,
                    "total_evidence": self._total,
                    "coverage": cov,
                }
            )
            logger.info(
                "[evidence] %s: hit=%d/%d (%.2f%%), candidates=%d",
                stage,
                hit,
                self._total,
                cov * 100,
                cnt,
            )

        self._records = results

    def summary(self) -> dict[str, Any]:
        return {
            "total_evidence": self._total,
            "stages": self._records,
        }
