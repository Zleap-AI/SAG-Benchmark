"""
Endpoint configuration for HyperGraphRAG —— 全部从仓库根 .env 读取。

LLM 与 embedding 跑在 *不同* 端点、用 *不同* key，所以这里构建专用 client/func，
而不是依赖全局 OPENAI_* 环境变量。所有值都走 env + 默认值，无硬编码。
"""

import os

import httpx
import numpy as np
from hypergraphrag.utils import wrap_embedding_func_with_attrs
from openai import AsyncOpenAI

# ---- LLM (Qwen3.6-35B-A3B-FP8, self-hosted vLLM) ---------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "1")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:32768/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3.6-35B-A3B-FP8")
# ---- 两条路径分开的 LLM 超时 -------------------------------------------------
# 建图（实体抽取 + gleaning）单次调用可能远超 180s；QA（keyword 抽取 + 答案生成）
# 则应快速失败。默认走 QA 超时，Step_1 入口调用 set_index_mode() 切到建图超时。
# connect 单独保持 5s，端点不通时快速失败，不随 read 超时一起放大。
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))  # QA 路径（默认）
LLM_INDEX_TIMEOUT = int(os.getenv("LLM_INDEX_TIMEOUT", "600"))  # 建图路径
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "5"))

# 当前进程处于建图模式还是 QA 模式；由 Step_1 入口显式切换，避免用 prompt 形状猜测路径。
_index_mode = False


def set_index_mode() -> None:
    """Step_1 入口调用：把本进程的 LLM 超时切到建图（更长）档。"""
    global _index_mode
    _index_mode = True


# ---- Embedding (bge-large-en-v1.5) -----------------------------------------
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:9990/v1")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # bge-large-en-v1.5 output dimension
# 模型硬上限（BGE = 512）；EMBEDDING_MAX_SEQ_LEN 优先，回退 EMBED_MAX_TOKENS（与 hipporag2 一致）
EMBED_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_SEQ_LEN", os.getenv("EMBED_MAX_TOKENS", "512")))
# 实际截断目标：留 2 token 给 [CLS]/[SEP]。等价旧硬编码 510。
EMBEDDING_MAX_TOKEN_SIZE = max(1, EMBED_MAX_TOKENS - 2)

# ---- 并发控制（可从 env 调整；默认与内核一致）-------------------------------
# EMBEDDING_FUNC_MAX_ASYNC 控制 embedding 调用并发；LLM_FUNC_MAX_ASYNC 控制 LLM 并发。
EMBEDDING_FUNC_MAX_ASYNC = int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "16"))
LLM_FUNC_MAX_ASYNC = int(os.getenv("LLM_FUNC_MAX_ASYNC", os.getenv("MAX_ASYNC", "32")))

# ---- 建图分批大小（Step_1 断点粒度）-----------------------------------------
# Step_1 按此值分批 insert：每批结束时图与向量库落盘，形成断点，中断最多丢一批。
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "500"))

# ---- 把值传播给底层库（OpenAI SDK / 运行时才读 env 的路径）-----------------
os.environ.setdefault("OPENAI_API_KEY", LLM_API_KEY)
os.environ.setdefault("EMBEDDING_API_KEY", EMBEDDING_API_KEY)
os.environ.setdefault("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL)


# ---- 模块级 client 单例 ----------------------------------------------------
# llm_model_func 每次调用 new 一个 AsyncOpenAI 且从不 close，会在长时间建图时
# 累积大量 TCP 连接（已观察到 CLOSE-WAIT 泄漏）。改为复用单例，让 SDK 的
# 连接池正常工作。client 级不设 timeout，超时按调用路径在下面 per-call 指定。
#
# 依赖单线程 event loop：check-then-assign 之间无 await，__init__ 纯同步，故原子。
# 约束：不要在本进程内改用 asyncio.run()（会关 loop 导致单例 client 失效）。
_llm_client: AsyncOpenAI | None = None
_embedding_client: AsyncOpenAI | None = None


def _get_llm_client() -> AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    return _llm_client


def _get_embedding_client() -> AsyncOpenAI:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)
    return _embedding_client


# tokenize/detokenize 用的 httpx client 同样复用，避免每个 embedding 批次新建连接。
_tokenizer_client: httpx.AsyncClient | None = None


def _get_tokenizer_client() -> httpx.AsyncClient:
    global _tokenizer_client
    if _tokenizer_client is None:
        _tokenizer_client = httpx.AsyncClient(timeout=30)
    return _tokenizer_client


async def llm_model_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs,
) -> str:
    """OpenAI-compatible chat completion against the deployed LLM endpoint.

    Two calling paths share this one function (HyperGraphRAG has a single
    llm_model_func for both index and query). We distinguish them by whether
    the caller passes a system_prompt:
      - system_prompt is None  -> index / keyword-extraction path
        (extract_entities hint_prompt, gleaning, summary, kg_query keyword call).
        Use richer sampling so the model stays generative while building the graph.
      - system_prompt is not None -> answer-generation path (kg_query final call,
        and Step_3 path2). Use temperature=0 for deterministic answers.

    Chain-of-thought ("thinking") is disabled by default too. Set env
    LLM_ENABLE_THINKING=1 to re-enable.

    注意：函数里有两套"路径"概念，别混为一谈 —— 采样参数（temperature）按
    prompt 形状（system_prompt 有无）分叉；超时则按进程模式由 set_index_mode()
    决定，不再从 system_prompt 推断。
    """
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    if history_messages is None:
        history_messages = []

    # ---- sampling params by path ----
    if system_prompt is not None:
        # answer generation: deterministic
        kwargs.setdefault("temperature", 0.0)
    else:
        # index / keyword extraction: generative sampling.
        kwargs.setdefault("temperature", 0.7)

    # 中性采样参数（对齐 settings.py 默认值；top_k/min_p/repetition_penalty 保持
    # 关闭值，按 settings.py _build_extra_body 语义不下发）
    kwargs.setdefault("max_tokens", 30000)
    kwargs.setdefault("top_p", 1.0)
    kwargs.setdefault("frequency_penalty", 0.0)
    kwargs.setdefault("presence_penalty", 0.0)

    if os.getenv("LLM_ENABLE_THINKING", "0") != "1":
        # vLLM reads chat_template_kwargs.enable_thinking; some gateways read a
        # top-level enable_thinking. Set both to be safe.
        _extra = kwargs.setdefault("extra_body", {})
        _extra.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        _extra["enable_thinking"] = False

    client = _get_llm_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # 超时按当前进程模式选档（Step_1 已 set_index_mode 切到建图档；QA 用默认档）。
    # 尊重调用方显式传入的 timeout；否则用当前档。connect 单独保持 5s，端点不通时快速失败。
    _timeout = kwargs.pop("timeout", None)
    if _timeout is None:
        _timeout = LLM_INDEX_TIMEOUT if _index_mode else LLM_TIMEOUT
    if not isinstance(_timeout, httpx.Timeout):
        _timeout = httpx.Timeout(_timeout, connect=LLM_CONNECT_TIMEOUT)
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        timeout=_timeout,
        **kwargs,
    )
    content = response.choices[0].message.content
    if r"\u" in content:
        from hypergraphrag.utils import safe_unicode_decode

        content = safe_unicode_decode(content.encode("utf-8"))
    return content


@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=EMBED_MAX_TOKENS)
async def embedding_func(
    texts: list[str],
) -> np.ndarray:
    """OpenAI-compatible embeddings against the deployed BGE endpoint.

    BGE-large-en-v1.5 hard-caps input at 512 tokens. Truncation uses the BGE
    endpoint's own /tokenize and /detokenize APIs so the token count is exact
    (not tiktoken approximation).
    """
    # /tokenize and /detokenize live at the root (not under /v1).
    _base = EMBEDDING_BASE_URL.rstrip("/")
    if _base.endswith("/v1"):
        _base = _base[:-3]
    TOKENIZE_URL = _base + "/tokenize"
    DETOKENIZE_URL = _base + "/detokenize"
    MAX_TOKENS = EMBEDDING_MAX_TOKEN_SIZE  # leave 2-token headroom for [CLS]/[SEP]

    http_client = _get_tokenizer_client()
    safe_texts = []
    for t in texts:
        # Quick length guard: if text is short enough it can't exceed 512
        # even with the most aggressive tokenizer (~1 char/token worst case).
        if len(t) <= MAX_TOKENS:
            safe_texts.append(t)
            continue

        # Tokenize with the real BGE tokenizer to get exact count.
        try:
            r = await http_client.post(
                TOKENIZE_URL,
                json={"model": EMBEDDING_MODEL_NAME, "prompt": t},
            )
            r.raise_for_status()
            data = r.json()
            token_count = data.get("count", 0)
        except Exception:
            # Fallback: character-level truncation (generous margin).
            safe_texts.append(t[: MAX_TOKENS * 3])
            continue

        if token_count <= MAX_TOKENS:
            safe_texts.append(t)
            continue

        # Truncate via detokenize with the first MAX_TOKENS token IDs.
        try:
            token_ids = data.get("tokens", [])[:MAX_TOKENS]
            if not token_ids:
                safe_texts.append(t[: MAX_TOKENS * 3])
                continue
            rr = await http_client.post(
                DETOKENIZE_URL,
                json={"model": EMBEDDING_MODEL_NAME, "tokens": token_ids},
            )
            rr.raise_for_status()
            truncated = rr.json().get("prompt", t[: MAX_TOKENS * 3])
            safe_texts.append(truncated)
        except Exception:
            safe_texts.append(t[: MAX_TOKENS * 3])

    client = _get_embedding_client()
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=safe_texts,
        encoding_format="float",
    )
    return np.array([dp.embedding for dp in response.data])


def build_rag_kwargs() -> dict:
    """Keyword args to pass into HyperGraphRAG(...) to wire the endpoints.

    chunk_token_size / chunk_overlap_token_size match hyperrag-benchmark (1200/100);
    the embedding_func truncates each text to <= EMBEDDING_MAX_TOKEN_SIZE (510) tokens
    before vectorization, so bge-large's 512-token cap is honored regardless of chunk size.
    """
    return {
        "llm_model_func": llm_model_func,
        "llm_model_name": LLM_MODEL,
        "embedding_func": embedding_func,
        "chunk_token_size": 1200,
        "chunk_overlap_token_size": 100,
        # 并发：EMBEDDING_FUNC_MAX_ASYNC / LLM_FUNC_MAX_ASYNC（env 可调）
        "embedding_func_max_async": EMBEDDING_FUNC_MAX_ASYNC,
        "llm_model_max_async": LLM_FUNC_MAX_ASYNC,
        # node2vec must match the embedding dimension used for vector storage
        "node2vec_params": {
            "dimensions": EMBEDDING_DIM,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        },
    }
