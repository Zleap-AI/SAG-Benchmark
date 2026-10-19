"""
GraphRAG 端点与旋钮配置 —— 从仓库根 .env 读取，回填 os.environ 供内核展开。

与 hyperrag 的 hyperrag_config.py 同职责：读 env + 给全默认值。区别在最后一步——
graphrag 内核 `load_config()` 会对 settings.yaml 执行 `Template.substitute(os.environ)`
（严格模式，缺失占位符直接 KeyError），所以这里用 `os.environ.setdefault()` 把
每一个 `${VAR}` 占位符都回填，而不是把值传给某个函数。

import 本模块**不产生副作用**；调用方必须在 `load_config()` 之前显式调用
`apply_env()` 完成回填（幂等，setdefault 不覆盖已存在的值）。

LLM 与 embedding 跑在 *不同* 端点、用 *不同* key，settings.yaml 的
default_chat_model / default_embedding_model 各自独立持有 api_key/api_base/model，
双端点适配在此完整保留。
"""

import os

# ---- LLM (Qwen3.6-35B-A3B-FP8, self-hosted vLLM) ---------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", "1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:32768/v1")

# ---- Embedding (bge-large-en-v1.5) -----------------------------------------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9990/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# ---- GraphRAG 专用旋钮 -----------------------------------------------------
# encoding_model 必须显式设 cl100k_base：留空会触发 tiktoken.encoding_name_for_model()
# 而 Qwen 模型名不在 tiktoken 表里 → KeyError，配置加载直接炸。
ENCODING_MODEL = os.getenv("GRAPHRAG_ENCODING_MODEL", "cl100k_base")
# 并发：回退 LLM_FUNC_MAX_ASYNC → MAX_ASYNC → 16（对齐 hyperrag_config 的约定）
CONCURRENT_REQUESTS = int(
    os.getenv(
        "GRAPHRAG_CONCURRENT_REQUESTS",
        os.getenv("LLM_FUNC_MAX_ASYNC", os.getenv("MAX_ASYNC", "32")),
    )
)
# 上游 -1 是「动态无限重试」，端点错时表现为静默空转；固定 5 让端点错误尽早暴露。
MAX_RETRIES = int(os.getenv("GRAPHRAG_MAX_RETRIES", "5"))
# chunk 口径与 hyperrag/lightrag 一致
CHUNK_SIZE = int(os.getenv("GRAPHRAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("GRAPHRAG_CHUNK_OVERLAP", "100"))
# embedding 截断上限（bge 硬上限 512，留余量）
EMBED_MAX_TOKENS = int(os.getenv("GRAPHRAG_EMBED_MAX_TOKENS", "400"))
# 关思考链：vLLM 部署的 Qwen3 系列必须通过 chat_template_kwargs 透传，否则 content 返回 None
DISABLE_THINKING = os.getenv("GRAPHRAG_DISABLE_THINKING", "1")
# false → LOOSE 解析，graphrag 自己从文本解析 JSON，对自建端点更稳
MODEL_SUPPORTS_JSON = os.getenv("GRAPHRAG_MODEL_SUPPORTS_JSON", "false")

# GRAPHRAG_EMBED_TOKENIZER: 可选，bge tokenizer 本地路径，供 cost_meter.truncate_texts 精确截断。
# 未设置时退回 cl100k_base 估算。不加入 _DEFAULTS（纯可选，无默认路径）。

# ---- 回填 os.environ 的键（settings.yaml 的 ${VAR} 占位符展开依赖这些键）-------
_DEFAULTS = {
    # chat 端点
    "LLM_API_KEY": LLM_API_KEY,
    "LLM_MODEL": LLM_MODEL,
    "LLM_BASE_URL": LLM_BASE_URL,
    # embedding 端点
    "EMBEDDING_API_KEY": EMBEDDING_API_KEY,
    "EMBEDDING_MODEL_NAME": EMBEDDING_MODEL_NAME,
    "EMBEDDING_BASE_URL": EMBEDDING_BASE_URL,
    "EMBEDDING_DIM": str(EMBEDDING_DIM),
    # 旋钮
    "GRAPHRAG_ENCODING_MODEL": ENCODING_MODEL,
    "GRAPHRAG_CONCURRENT_REQUESTS": str(CONCURRENT_REQUESTS),
    "GRAPHRAG_MAX_RETRIES": str(MAX_RETRIES),
    "GRAPHRAG_CHUNK_SIZE": str(CHUNK_SIZE),
    "GRAPHRAG_CHUNK_OVERLAP": str(CHUNK_OVERLAP),
    "GRAPHRAG_EMBED_MAX_TOKENS": str(EMBED_MAX_TOKENS),
    "GRAPHRAG_DISABLE_THINKING": DISABLE_THINKING,
    "GRAPHRAG_MODEL_SUPPORTS_JSON": MODEL_SUPPORTS_JSON,
}


def apply_env() -> None:
    """把 _DEFAULTS 回填到 os.environ（settings.yaml 的 ${VAR} 展开依赖这些键）。

    必须在 load_config() 之前调用。幂等：setdefault 不覆盖 shell 已 export 的值。
    """
    for key, value in _DEFAULTS.items():
        os.environ.setdefault(key, value)
