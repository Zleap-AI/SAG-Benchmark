"""
来源片段 Document 模型

对应 Elasticsearch 索引：source_chunks
"""

from elasticsearch_dsl import Boolean, Date, DenseVector, Document, Integer, Keyword, Text

from pipeline.storage.backends.elasticsearch.index_naming import (
    BASE_INDEX_SOURCE_CHUNKS,
    LEGACY_DIM,
)


class SourceChunkDocument(Document):
    """来源片段文档模型

    注意：DenseVector(dims=...) 在类定义期求值，这里的 LEGACY_DIM 只是"声明默认值"。
    实际建索引时由 scripts/init_elasticsearch.py 通过
    index_naming.with_dense_vector_dims() 按 probe 到的运行时维度改写。
    """

    # 核心字段
    chunk_id = Keyword(required=True)  # 对应 SourceChunk.id
    source_id = Keyword(required=True)  # 对应 Article.id 或 Conversation.id
    source_config_id = Keyword(required=True)
    rank = Integer()  # 片段排序

    # 文本字段（使用默认分词器）
    heading = Text(fields={"keyword": Keyword()})
    content = Text()

    # 向量字段
    heading_vector = DenseVector(dims=LEGACY_DIM, index=True, similarity="cosine")
    content_vector = DenseVector(dims=LEGACY_DIM, index=True, similarity="cosine")

    # 元数据
    chunk_type = Keyword()  # 片段类型：TEXT, CODE, TABLE等
    content_length = Integer()  # 内容长度

    # 关联字段
    references = Keyword(multi=True)  # 关联的 ArticleSection ID 列表

    # 时间字段
    created_time = Date()
    updated_time = Date()
    is_delete = Boolean()  # 软删除标记

    # 供 index_naming.resolve_index_name() 使用的 base 名（不含维度后缀）
    BASE_INDEX_NAME = BASE_INDEX_SOURCE_CHUNKS

    class Index:
        """索引配置"""

        name = BASE_INDEX_SOURCE_CHUNKS
        settings = {"number_of_shards": 3, "number_of_replicas": 1}

    def save(self, **kwargs):
        """保存文档"""
        return super().save(**kwargs)
