"""
RAG 切片框架核心类型定义
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class BlockType(str, Enum):
    """结构块类型"""

    TEXT = "TEXT"
    FORMULA = "FORMULA"
    TABLE = "TABLE"
    CODE = "CODE"
    IMAGE = "IMAGE"


@dataclass
class InputDocument:
    """输入层标准化后的文档"""

    content: str
    source_path: Path | None = None
    is_markdown: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass
class StructuredBlock:
    """结构识别层产物"""

    block_id: str
    block_type: BlockType
    raw_content: str
    heading: str = ""
    start_index: int = 0
    end_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class SectionDraft:
    """ArticleSection 草稿"""

    order_index: int
    render_group_index: int
    heading: str
    content: str
    raw_content: str
    section_type: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkDraft:
    """SourceChunk 草稿"""

    rank: int
    heading: str
    content: str
    raw_content: str
    chunk_type: str
    section_order_indices: list[int]
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkingResult:
    """整条切片链路结果"""

    input_doc: InputDocument
    blocks: list[StructuredBlock]
    article_sections: list[SectionDraft]
    source_chunks: list[ChunkDraft]
