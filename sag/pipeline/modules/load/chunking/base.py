"""
RAG 切片四层基类定义
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pipeline.modules.load.chunking.types import (
    ChunkDraft,
    InputDocument,
    SectionDraft,
    StructuredBlock,
)


class BaseInputNormalizer(ABC):
    """输入层：统一输入格式"""

    @abstractmethod
    def normalize(self, content: str, source_path: Path | None = None) -> InputDocument:
        pass


class BaseBlockParser(ABC):
    """结构识别层：识别块结构"""

    @abstractmethod
    def parse_blocks(self, doc: InputDocument) -> list[StructuredBlock]:
        pass


class BaseArticleSectionBuilder(ABC):
    """ArticleSection 生成层：生产最小可引用单元"""

    @abstractmethod
    async def build_sections(
        self,
        doc: InputDocument,
        blocks: list[StructuredBlock],
    ) -> list[SectionDraft]:
        pass


class BaseSourceChunkAssembler(ABC):
    """SourceChunk 组装层：面向 embedding/检索"""

    @abstractmethod
    def assemble_chunks(
        self,
        doc: InputDocument,
        sections: list[SectionDraft],
    ) -> list[ChunkDraft]:
        pass
