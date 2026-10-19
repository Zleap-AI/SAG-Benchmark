"""LightRAG endpoint configuration — env-driven, mirrors hypergraphrag_config.py.

Reads LLM/Embedding endpoints from the unified sag-benchmark/.env.
All projects share the same LLM + Embedding; no per-project endpoint distinction.

重要：env 读取必须发生在 `from lightrag import ...` 之前。lightrag 内核在
import 时会执行 load_dotenv(override=True)（utils.py / operate.py / lightrag.py），
会覆盖 `uv run --env-file` 命令行注入的变量。
"""

import logging
import os

import aiohttp
import numpy as np

# ---- env 读取（先于 lightrag import，见模块 docstring）-----------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", "1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:32768/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")

# ---- Embedding (bge-large-en-v1.5, dim=1024) --------------------------------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9990/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
# 模型硬上限（BGE = 512）；EMBEDDING_MAX_SEQ_LEN 优先，回退 EMBED_MAX_TOKENS（与 hipporag2 一致）
EMBED_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_SEQ_LEN", os.getenv("EMBED_MAX_TOKENS", "512")))
# 实际截断目标：留 2 token 给 [CLS]/[SEP]。等价旧硬编码 510。
EMBEDDING_MAX_TOKEN_SIZE = max(1, EMBED_MAX_TOKENS - 2)

# ---- 建图分批大小（Step_1 断点粒度）-----------------------------------------
# Step_1 按此值分批 insert：每批结束时图与向量库落盘，形成断点，中断最多丢一批。
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "500"))

# ---- 把值传播给底层库（OpenAI SDK / 运行时才读 env 的路径）-----------------
os.environ.setdefault("OPENAI_API_KEY", LLM_API_KEY)
os.environ.setdefault("EMBEDDING_API_KEY", EMBEDDING_API_KEY)
os.environ.setdefault("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)

log = logging.getLogger(__name__)

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """OpenAI-compatible LLM call with thinking disabled by default.

    按 system_prompt 是否传入区分两条路径（与 hypergraphrag/hyperrag 一致）：
      - system_prompt is None  -> 建图 / 关键词抽取（生成性采样）
      - system_prompt is not None -> QA 答案生成（确定性）
    """
    if history_messages is None:
        history_messages = []
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("hashing_kv", None)
    extra = kwargs.setdefault("extra_body", {})
    if os.getenv("LLM_ENABLE_THINKING", "0") != "1":
        extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        extra["enable_thinking"] = False

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


async def embedding_func(texts: list[str]) -> np.ndarray:
    """OpenAI-compatible embedding call with per-text tokenizer-precise truncation.

    Uses the embedding service's /tokenize and /detokenize endpoints for accurate
    token-count truncation. Falls back to char-based estimate (4 chars/token) if
    the tokenizer service is unavailable.
    """
    truncated = await _truncate_embeddings_batch(texts, EMBEDDING_MAX_TOKEN_SIZE)
    return await openai_embed(
        truncated,
        model=EMBEDDING_MODEL_NAME,
        api_key=EMBEDDING_API_KEY,
        base_url=EMBEDDING_BASE_URL,
    )


# ---- Tokenizer-precise truncation helpers ------------------------------------


def _tokenizer_base_url() -> str:
    """Derive the tokenizer service base URL."""
    tok_url = os.getenv("EMBED_TOKENIZER_URL", "").strip()
    if tok_url:
        root = tok_url.rstrip("/")
    else:
        root = EMBEDDING_BASE_URL.rstrip("/")
    # vLLM exposes /tokenize at the server root, not below /v1.
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


async def _post_tokenizer(path: str, payload: dict) -> dict:
    """Post to the tokenizer service with timeout and limited retries."""
    url = f"{_tokenizer_base_url()}{path}"
    headers = {"Authorization": f"Bearer {EMBEDDING_API_KEY}"} if EMBEDDING_API_KEY else {}
    for attempt in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if attempt == 2:
                        raise RuntimeError(f"tokenizer endpoint returned HTTP {resp.status}")
        except Exception:
            if attempt == 2:
                raise
    raise RuntimeError("tokenizer endpoint retries exhausted")


async def _embedding_token_count(text: str) -> int:
    """Return the embedding model's own token count, or raise on bad data."""
    result = await _post_tokenizer(
        "/tokenize",
        {"model": EMBEDDING_MODEL_NAME, "prompt": text},
    )
    if "count" in result:
        return int(result["count"])
    token_ids = result.get("tokens") or result.get("token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)
    raise ValueError("tokenizer response contains neither count nor token ids")


async def _truncate_text_by_tokens(text: str, max_tokens: int) -> str:
    """Return the longest prefix whose embedding-token count fits max_tokens.

    Falls back to a conservative 3 chars/token estimate if the service fails.
    """
    if not text or not text.strip():
        return text
    try:
        if await _embedding_token_count(text) <= max_tokens:
            return text
        lo, hi, best = 0, len(text), ""
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[:mid]
            if await _embedding_token_count(candidate) <= max_tokens:
                best = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        return best
    except Exception:
        log.warning("tokenizer truncation failed, falling back to char estimate")
        return text[: max_tokens * 3]


async def _truncate_embeddings_batch(texts: list[str], max_tokens: int) -> list[str]:
    """Truncate a batch of texts, preserving order."""
    return [await _truncate_text_by_tokens(t, max_tokens) for t in texts]


def build_rag_kwargs(working_dir: str | None = None) -> dict:
    """Keyword args for LightRAG(...).

    Chunk strategy preserved from source project defaults (1200/100).
    Embedding 512-token truncation handled inside embedding_func.
    并发控制：embedding/LLM/并行插入均可从 env 调整（默认与内核一致）。
    """
    kwargs = {
        "llm_model_func": llm_model_func,
        "llm_model_name": LLM_MODEL,
        "chunk_token_size": int(os.getenv("CHUNK_SIZE", "1200")),
        "chunk_overlap_token_size": int(os.getenv("CHUNK_OVERLAP", "100")),
        # 并发：EMBEDDING_FUNC_MAX_ASYNC 控制 embedding 调用并发；
        # LLM_FUNC_MAX_ASYNC 优先，兼容内核既有 MAX_ASYNC；
        # MAX_PARALLEL_INSERT 控制每次批量插入的 doc 数（越小越保守）。
        "embedding_func_max_async": int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "16")),
        "llm_model_max_async": int(os.getenv("LLM_FUNC_MAX_ASYNC", os.getenv("MAX_ASYNC", "16"))),
        "max_parallel_insert": int(os.getenv("MAX_PARALLEL_INSERT", "20")),
        "embedding_func": EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBED_MAX_TOKENS,
            func=embedding_func,
        ),
    }
    if working_dir is not None:
        kwargs["working_dir"] = working_dir
    return kwargs
