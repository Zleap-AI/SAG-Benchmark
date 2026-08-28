"""
Endpoint configuration for Hyper-RAG —— 全部从仓库根 .env 读取。

LLM 与 embedding 跑在 *不同* 端点、用 *不同* key，所以这里构建专用 client/func，
而不是依赖全局 OPENAI_* 环境变量。所有值都走 env + 默认值，无硬编码。
"""

import os

from hyperrag.llm import openai_complete_if_cache, openai_embedding

# ---- LLM (Qwen3.6-35B-A3B-FP8, self-hosted vLLM) ---------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", "1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:32768/v1")

# ---- Embedding (bge-large-en-v1.5) -----------------------------------------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9990/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-large-en-v1.5")
# EMBEDDING_MODEL_NAME = "text-embedding-3-small"
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-large-en-v1.5 output dimension
# EMBEDDING_DIM = 1536
# 模型硬上限（BGE = 512）；EMBEDDING_MAX_SEQ_LEN 优先，回退 EMBED_MAX_TOKENS（与 hipporag2 一致）
EMBED_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_SEQ_LEN", os.getenv("EMBED_MAX_TOKENS", "512")))
# 实际截断目标：留 2 token 给 [CLS]/[SEP]。等价旧硬编码 510。
EMBEDDING_MAX_TOKEN_SIZE = max(1, EMBED_MAX_TOKENS - 2)

# ---- 并发控制（可从 env 调整；默认与内核一致）-------------------------------
# 由 Step_1/Step_3 传入 HyperRAG(embedding_func_max_async=..., llm_model_max_async=...)
EMBEDDING_FUNC_MAX_ASYNC = int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "16"))
LLM_FUNC_MAX_ASYNC = int(os.getenv("LLM_FUNC_MAX_ASYNC", os.getenv("MAX_ASYNC", "16")))

# ---- 建图分批大小（Step_1 断点粒度）-----------------------------------------
# Step_1 按此值分批 insert：每批结束时图与向量库落盘，形成断点，中断最多丢一批。
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "500"))

# ---- 把值传播给底层库（OpenAI SDK / 运行时才读 env 的路径）-----------------
os.environ.setdefault("OPENAI_API_KEY", LLM_API_KEY)
os.environ.setdefault("EMBEDDING_API_KEY", EMBEDDING_API_KEY)
os.environ.setdefault("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
    """OpenAI-compatible LLM call shared by index and QA paths.

    按 system_prompt 是否传入区分两条路径（与 hypergraphrag/lightrag 一致）：
      - system_prompt is None  -> 建图 / 关键词抽取（生成性采样）
      - system_prompt is not None -> QA 答案生成（确定性）
    思考链默认关闭，LLM_ENABLE_THINKING=1 可重新开启。
    """
    if history_messages is None:
        history_messages = []
    if os.getenv("LLM_ENABLE_THINKING", "0") != "1":
        _extra = kwargs.setdefault("extra_body", {})
        _extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        _extra["enable_thinking"] = False

    # 温度分路径：QA 确定性 0.0 / 建图·关键词抽取生成性 0.7（对齐 settings.py 约定）
    if system_prompt is not None:
        kwargs.setdefault("temperature", 0.0)
    else:
        kwargs.setdefault("temperature", 0.7)

    # 中性采样参数（对齐 settings.py 默认值；top_k/min_p/repetition_penalty 保持
    # 关闭值，按 settings.py _build_extra_body 语义不下发）
    kwargs.setdefault("max_tokens", 30000)
    kwargs.setdefault("top_p", 1.0)
    kwargs.setdefault("frequency_penalty", 0.0)
    kwargs.setdefault("presence_penalty", 0.0)

    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]):
    return await openai_embedding(
        texts,
        model=EMBEDDING_MODEL_NAME,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )
