"""
事件向量 Document 模型

对应 Elasticsearch 索引：event_vectors
"""

from elasticsearch_dsl import Boolean, Date, DenseVector, Document, Keyword, Text

from pipeline.storage.backends.elasticsearch.index_naming import (
    BASE_INDEX_EVENT_VECTORS,
    LEGACY_DIM,
)


class EventVectorDocument(Document):
    """事件向量文档模型

    注意：DenseVector(dims=...) 在类定义期求值，这里的 LEGACY_DIM 只是"声明默认值"。
    实际建索引时由 scripts/init_elasticsearch.py 通过
    index_naming.with_dense_vector_dims() 按 probe 到的运行时维度改写。
    """

    # 字段定义
    event_id = Keyword(required=True)
    source_config_id = Keyword(required=True)
    source_type = Keyword(required=True)
    source_id = Keyword(required=True)

    # 文本字段（使用默认分词器）
    title = Text(fields={"keyword": Keyword()})
    summary = Text()
    content = Text()

    # 向量字段
    title_vector = DenseVector(dims=LEGACY_DIM, index=True, similarity="cosine")
    content_vector = DenseVector(dims=LEGACY_DIM, index=True, similarity="cosine")

    # 分类和标签
    category = Keyword()
    tags = Keyword(multi=True)  # 数组字段
    entity_ids = Keyword(multi=True)  # 关联的实体ID列表

    # 时间字段
    start_time = Date()
    end_time = Date()
    created_time = Date()
    is_delete = Boolean()  # 软删除标记

    # 供 index_naming.resolve_index_name() 使用的 base 名（不含维度后缀）
    BASE_INDEX_NAME = BASE_INDEX_EVENT_VECTORS

    class Index:
        """索引配置"""

        name = BASE_INDEX_EVENT_VECTORS
        settings = {"number_of_shards": 12, "number_of_replicas": 1}

    def save(self, **kwargs):
        """保存文档"""
        return super().save(**kwargs)
