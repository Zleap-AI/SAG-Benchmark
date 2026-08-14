"""
实体向量 Document 模型

对应 Elasticsearch 索引：entity_vectors
"""

from elasticsearch_dsl import Boolean, Date, DenseVector, Document, Keyword, Text

from pipeline.storage.backends.elasticsearch.index_naming import (
    BASE_INDEX_ENTITY_VECTORS,
    LEGACY_DIM,
)


class EntityVectorDocument(Document):
    """实体向量文档模型

    注意：DenseVector(dims=...) 在类定义期求值，这里的 LEGACY_DIM 只是"声明默认值"。
    实际建索引时由 scripts/init_elasticsearch.py 通过
    index_naming.with_dense_vector_dims() 按 probe 到的运行时维度改写。
    """

    # 字段定义
    entity_id = Keyword(required=True)
    source_config_id = Keyword(required=True)
    type = Keyword(required=True)  # 实体类型：PERSON, ORGANIZATION, TOPIC等
    name = Text(fields={"keyword": Keyword()})
    vector = DenseVector(dims=LEGACY_DIM, index=True, similarity="cosine")
    created_time = Date()
    is_delete = Boolean()  # 软删除标记

    # 供 index_naming.resolve_index_name() 使用的 base 名（不含维度后缀）
    BASE_INDEX_NAME = BASE_INDEX_ENTITY_VECTORS

    class Index:
        """索引配置"""

        name = BASE_INDEX_ENTITY_VECTORS
        settings = {"number_of_shards": 24, "number_of_replicas": 1}

    def save(self, **kwargs):
        """保存文档"""
        return super().save(**kwargs)
