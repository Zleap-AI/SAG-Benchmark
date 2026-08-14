"""
配置管理模块

使用pydantic-settings管理配置，支持从环境变量和.env文件读取
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def resolve_env_file() -> Path:
    """Resolve dotenv deterministically, with an explicit process-level override."""
    override = os.getenv("SAG_ENV_FILE")
    if not override:
        return DEFAULT_ENV_FILE

    path = Path(override).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SAG_ENV_FILE does not point to a file: {resolved}")
    return resolved


# LLM 可靠性常量（统一默认值，避免多处硬编码）
DEFAULT_LLM_MAX_RETRIES = 2


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ======================
    # 存储后端配置
    # ======================
    storage_profile: str = Field(
        default="mysql_es",
        description="存储方案预设：mysql_es / oceanbase_es / oceanbase_full",
    )
    database_backend: Optional[str] = Field(
        default=None,
        description="结构化数据库后端：mysql / oceanbase；留空时由 storage_profile 推导",
    )
    vector_backend: Optional[str] = Field(
        default=None,
        description="向量/检索后端：elasticsearch / oceanbase；留空时由 storage_profile 推导",
    )

    # ======================
    # 数据库配置
    # ======================
    mysql_host: str = Field(default="localhost", description="MySQL主机")
    mysql_port: int = Field(default=3306, description="MySQL端口")
    mysql_user: str = Field(default="sag2", description="MySQL用户")
    mysql_password: str = Field(default="sag2", description="MySQL密码")
    mysql_database: str = Field(default="sag2", description="MySQL数据库名")

    # ======================
    # OceanBase配置（MySQL兼容模式）
    # ======================
    oceanbase_host: str = Field(default="localhost", description="OceanBase主机")
    oceanbase_port: int = Field(default=2881, description="OceanBase SQL端口")
    oceanbase_user: str = Field(default="", description="OceanBase用户")
    oceanbase_password: str = Field(default="", description="OceanBase密码")
    oceanbase_database: str = Field(default="sag2", description="OceanBase数据库名")
    oceanbase_compat_mode: str = Field(default="mysql", description="OceanBase租户兼容模式")
    oceanbase_vector_index_type: str = Field(default="HNSW", description="OceanBase向量索引类型")
    oceanbase_vector_index_lib: str = Field(default="VSAG", description="OceanBase向量索引库")
    oceanbase_vector_index_m: int = Field(default=16, description="OceanBase HNSW M参数")
    oceanbase_vector_index_ef_construction: int = Field(default=200, description="OceanBase HNSW构建参数")
    oceanbase_vector_search_ef_search: int = Field(
        default=64,
        ge=1,
        le=10000,
        description="OceanBase HNSW查询阶段ef_search参数",
        validation_alias=AliasChoices(
            "OCEANBASE_VECTOR_SEARCH_EF_SEARCH",
            "OCEANBASE_VECTOR_INDEX_EF_SEARCH",
        ),
    )

    # ======================
    # Elasticsearch配置
    # ======================
    es_host: str = Field(default="localhost", description="ES主机")
    es_port: int = Field(default=9201, description="ES端口")
    es_scheme: str = Field(default="http", description="ES协议(http/https)")
    es_username: str | None = Field(default="elastic", description="ES用户名")
    es_password: str | None = Field(
        default=None, description="ES密码", validation_alias="ELASTIC_PASSWORD"
    )

    # ======================
    # LLM配置（使用中转API或OpenAI官方）
    # ======================
    llm_api_key: str = Field(default="", description="LLM API密钥")
    llm_model: str = Field(default="sophnet/Qwen3-30B-A3B-Thinking-2507", description="LLM模型")
    llm_base_url: str | None = Field(
        default=None, description="LLM API基础URL（留空使用OpenAI官方）"
    )
    llm_data_inspection: bool = Field(
        default=False, description="是否启用LLM内容过滤（绿网），默认关闭"
    )

    # 是否启用模型的思考模式（enable_thinking），默认关闭，需在.env中显式启用
    llm_enable_think: bool = Field(
        default=False, description="是否启用模型的思考模式（enable_thinking）"
    )

    # #baseline
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM温度参数")
    llm_max_tokens: int = Field(default=30000, ge=1, description="LLM最大输出token数")
    llm_top_p: float = Field(default=1.0, gt=0.0, le=1.0, description="LLM top_p参数（vLLM要求>0）")
    llm_frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="频率惩罚")
    llm_presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="存在惩罚")
    llm_top_k: int = Field(default=-1, ge=-1, description="LLM top_k参数，-1表示禁用")
    llm_min_p: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM min_p参数")
    llm_repetition_penalty: float = Field(
        default=1.0, ge=0.0, le=2.0, description="重复惩罚，1.0表示关闭"
    )


    # LLM 可靠性参数
    llm_timeout: int = Field(default=300, ge=1, description="LLM超时时间(秒)")
    llm_max_retries: int = Field(
        default=DEFAULT_LLM_MAX_RETRIES, ge=0, description="LLM最大重试次数"
    )

    # ======================
    # Judge LLM 配置（独立于主 LLM，避免评测模型漂移）
    # ======================
    judge_llm_api_key: str = Field(default="", description="Judge LLM API密钥")
    judge_llm_model: str = Field(default="", description="Judge LLM模型")
    judge_llm_base_url: str | None = Field(default=None, description="Judge LLM API基础URL")
    judge_llm_max_async: int = Field(default=4, ge=1, description="Judge LLM最大并发数")
    judge_llm_timeout: int = Field(default=300, ge=1, description="Judge LLM超时时间(秒)")
    judge_llm_max_retries: int = Field(
        default=DEFAULT_LLM_MAX_RETRIES, ge=0, description="Judge LLM最大重试次数"
    )
    judge_llm_enable_thinking: bool = Field(default=False, description="Judge LLM思考模式开关")
    judge_llm_max_tokens: int = Field(default=8000, ge=1, description="Judge LLM最大输出token数")
    judge_allow_fallback: bool = Field(
        default=False,
        description="Judge LLM 配置缺失时是否允许 fallback 到主 LLM",
    )

    # 数据库配置开关
    use_db_config: bool = Field(default=True, description="是否使用数据库配置")

    # ======================
    # Embedding配置（使用中转API或OpenAI官方）
    # ======================
    embedding_api_key: str = Field(
        default="", description="Embedding API密钥（留空使用llm_api_key）"
    )
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B", description="Embedding模型"
    )
    embedding_dimensions: int | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "EMBEDDING_DIM",  # .env 现行写法（优先）
            "EMBEDDING_DIMENSIONS",  # 历史/OpenAI 风格写法，向后兼容
        ),
        description=(
            "Embedding 期望维度。仅作为 probe 结果的校验基准与离线 fallback；"
            "真实维度以 EmbeddingDimResolver 的 probe 结果为准。"
        ),
    )
    embedding_request_dimensions: int | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_REQUEST_DIMENSIONS"),
        description=(
            "透传给 embeddings.create(dimensions=…) 的值（matryoshka 模型专用）。"
            "留空则不传该参数。注意 bge-large-en-v1.5 不支持该参数，传了会 400。"
        ),
    )
    embedding_dim_strict: bool = Field(
        default=False,
        validation_alias=AliasChoices("EMBEDDING_DIM_STRICT"),
        description="True 时 probe 结果与 EMBEDDING_DIM 不一致直接报错，False 时以 probe 为准并告警",
    )
    embedding_dim_probe_timeout: int = Field(
        default=30,
        ge=1,
        validation_alias=AliasChoices("EMBEDDING_DIM_PROBE_TIMEOUT"),
        description="维度 probe 请求超时（秒）",
    )
    es_index_legacy_unsuffixed: bool = Field(
        default=True,
        validation_alias=AliasChoices("ES_INDEX_LEGACY_UNSUFFIXED"),
        description=(
            "True（默认）时 1024 维继续使用无后缀索引名（兼容现存数据）；"
            "False 时所有维度统一带 _<dim> 后缀"
        ),
    )
    embedding_base_url: str | None = Field(
        default=None, description="Embedding API基础URL（留空使用llm_base_url）"
    )

    # ======================
    # LLM 语言配置
    # ======================
    # ======================
    # Rerank Configuration
    # ======================
    rerank_api_key: str | None = Field(
        default=None,
        description="Rerank API key; fallback to embedding_api_key/llm_api_key when empty",
    )
    rerank_model_name: str | None = Field(default=None, description="Rerank model name")
    rerank_base_url: str | None = Field(
        default=None, description="Rerank API base URL; fallback to embedding_base_url when empty"
    )
    rerank_endpoint: Optional[str] = Field(
        default="/rerank",
        description="Rerank 请求端点路径，拼接到 rerank_base_url 之后（默认 /rerank；"
                    "若平台路由为 /v1/reranks 等可改为 /reranks；"
                    "若 rerank_base_url 已含完整端点路径则直接使用、不再拼接）",
    )

    llm_language: str = Field(
        default="en", description="LLM输出语言(zh/en)，决定加载哪个语言版本的提示词"
    )

    # ======================
    # 应用配置
    # ======================
    server_type: str = Field(default="LOCAL", description="服务环境类型（SAAS/LOCAL）")
    benchmark: bool = Field(default=False, description="Benchmark模式，跳过LLM调用")
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: str = Field(default="json", description="日志格式")

    # ======================
    # MLflow 配置
    # ======================
    mlflow_port: int = Field(default=5000, description="MLflow Docker容器端口")
    mlflow_url: str | None = Field(
        default="http://localhost:5000", description="MLflow Tracking Server地址"
    )

    # ======================
    # 性能配置
    # ======================
    db_pool_size: int = Field(default=100, description="数据库连接池大小")
    db_max_overflow: int = Field(default=200, description="数据库连接池最大溢出")
    db_pool_recycle: int = Field(default=3600, description="数据库连接回收时间(秒)")

    # 缓存TTL
    cache_entity_ttl: int = Field(default=86400, description="实体缓存TTL(秒)")
    cache_llm_ttl: int = Field(default=604800, description="LLM缓存TTL(秒)")
    cache_search_ttl: int = Field(default=3600, description="搜索缓存TTL(秒)")

    @property
    def mysql_url(self) -> str:
        """MySQL连接URL"""
        from urllib.parse import quote_plus

        # URL编码用户名和密码，避免特殊字符问题
        encoded_user = quote_plus(self.mysql_user)
        encoded_password = quote_plus(self.mysql_password)
        return (
            f"mysql+aiomysql://{encoded_user}:{encoded_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    @property
    def oceanbase_url(self) -> str:
        """OceanBase连接URL（MySQL兼容模式）"""
        from urllib.parse import quote_plus

        encoded_user = quote_plus(self.oceanbase_user)
        encoded_password = quote_plus(self.oceanbase_password)
        return (
            f"mysql+aiomysql://{encoded_user}:{encoded_password}"
            f"@{self.oceanbase_host}:{self.oceanbase_port}/{self.oceanbase_database}"
            f"?charset=utf8mb4"
        )

    @property
    def effective_database_backend(self) -> str:
        """实际结构化数据库后端。"""
        if self.database_backend:
            return self.database_backend.lower()
        profile = self.storage_profile.lower()
        if profile in {"oceanbase_es", "oceanbase_full"}:
            return "oceanbase"
        return "mysql"

    @property
    def effective_vector_backend(self) -> str:
        """实际向量/检索后端。"""
        if self.vector_backend:
            return self.vector_backend.lower()
        profile = self.storage_profile.lower()
        if profile == "oceanbase_full":
            return "oceanbase"
        return "elasticsearch"

    @property
    def database_url(self) -> str:
        """当前结构化数据库连接URL。"""
        backend = self.effective_database_backend
        if backend == "mysql":
            return self.mysql_url
        if backend == "oceanbase":
            return self.oceanbase_url
        raise ValueError(f"不支持的 DATABASE_BACKEND: {backend}")

    @property
    def elasticsearch_url(self) -> str:
        """Elasticsearch连接URL"""
        return f"{self.es_scheme}://{self.es_host}:{self.es_port}"

    @property
    def es_url(self) -> str:
        """Elasticsearch连接URL（兼容旧版本）"""
        return self.elasticsearch_url

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"日志级别必须是: {', '.join(allowed)}")
        return v.upper()

    @field_validator("storage_profile")
    @classmethod
    def validate_storage_profile(cls, v: str) -> str:
        """验证存储方案预设"""
        normalized = v.lower()
        allowed = ["mysql_es", "oceanbase_es", "oceanbase_full"]
        if normalized not in allowed:
            raise ValueError(f"STORAGE_PROFILE 必须是: {', '.join(allowed)}")
        return normalized

    @field_validator("database_backend")
    @classmethod
    def validate_database_backend(cls, v: Optional[str]) -> Optional[str]:
        """验证结构化数据库后端"""
        if v is None or v == "":
            return None
        normalized = v.lower()
        allowed = ["mysql", "oceanbase"]
        if normalized not in allowed:
            raise ValueError(f"DATABASE_BACKEND 必须是: {', '.join(allowed)}")
        return normalized

    @field_validator("vector_backend")
    @classmethod
    def validate_vector_backend(cls, v: Optional[str]) -> Optional[str]:
        """验证向量/检索后端"""
        if v is None or v == "":
            return None
        normalized = v.lower()
        allowed = ["elasticsearch", "oceanbase"]
        if normalized not in allowed:
            raise ValueError(f"VECTOR_BACKEND 必须是: {', '.join(allowed)}")
        return normalized

    @field_validator("oceanbase_vector_index_type")
    @classmethod
    def validate_oceanbase_vector_index_type(cls, v: str) -> str:
        """验证 OceanBase 向量索引类型"""
        normalized = v.upper()
        allowed = ["HNSW", "HNSW_SQ", "HNSW_BQ", "IVF_FLAT", "IVF_SQ8", "IVF_PQ"]
        if normalized not in allowed:
            raise ValueError(f"OCEANBASE_VECTOR_INDEX_TYPE 必须是: {', '.join(allowed)}")
        return normalized

    @field_validator("oceanbase_vector_index_lib")
    @classmethod
    def validate_oceanbase_vector_index_lib(cls, v: str) -> str:
        """验证 OceanBase 向量索引库"""
        normalized = v.upper()
        allowed = ["VSAG", "OB"]
        if normalized not in allowed:
            raise ValueError(f"OCEANBASE_VECTOR_INDEX_LIB 必须是: {', '.join(allowed)}")
        return normalized

    @model_validator(mode="after")
    def validate_storage_backend_combination(self) -> "Settings":
        """验证存储组合，只开放生产支持的三种方案。"""
        database_backend = self.effective_database_backend
        vector_backend = self.effective_vector_backend
        allowed = {
            ("mysql", "elasticsearch"),
            ("oceanbase", "elasticsearch"),
            ("oceanbase", "oceanbase"),
        }
        if (database_backend, vector_backend) not in allowed:
            raise ValueError(
                "不支持的存储组合: "
                f"DATABASE_BACKEND={database_backend}, VECTOR_BACKEND={vector_backend}"
            )
        if vector_backend == "oceanbase":
            index_type = self.oceanbase_vector_index_type
            index_lib = self.oceanbase_vector_index_lib
            if index_type.startswith("HNSW") and index_lib != "VSAG":
                raise ValueError("OceanBase HNSW 向量索引要求 OCEANBASE_VECTOR_INDEX_LIB=VSAG")
            if index_type.startswith("IVF") and index_lib != "OB":
                raise ValueError("OceanBase IVF 向量索引要求 OCEANBASE_VECTOR_INDEX_LIB=OB")
        return self

    @field_validator("llm_language")
    @classmethod
    def validate_llm_language(cls, v: str) -> str:
        """验证LLM语言配置"""
        allowed = ["zh", "en"]
        if v.lower() not in allowed:
            raise ValueError(f"LLM语言必须是: {', '.join(allowed)}")
        return v.lower()

    @field_validator("server_type")
    @classmethod
    def validate_server_type(cls, v: str) -> str:
        """验证服务环境类型"""
        normalized = v.upper()
        allowed = ["SAAS", "LOCAL"]
        if normalized not in allowed:
            raise ValueError(f"SERVER_TYPE 必须是: {', '.join(allowed)}")
        return normalized


@lru_cache
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings(_env_file=resolve_env_file())  # type: ignore[call-arg]
