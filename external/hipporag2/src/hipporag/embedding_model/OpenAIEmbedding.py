from copy import deepcopy
from typing import List, Optional
import os

import numpy as np
from openai import OpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI API compatible embedding model.
    Supports both OpenAI official API and custom endpoints (like vLLM, text-embedding-inference).
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        # Priority: parameter > global_config > environment variable
        self.api_key = api_key or (self.global_config.embedding_base_url and os.getenv('EMBEDDING_API_KEY')) or os.getenv('OPENAI_API_KEY', 'dummy-key')
        self.base_url = base_url or self.global_config.embedding_base_url or os.getenv('EMBEDDING_BASE_URL')

        self._init_embedding_config()

        # Load a tokenizer for exact token-based truncation (see _init_tokenizer).
        self._init_tokenizer()

        # Initialize OpenAI client
        logger.info(f"Initializing OpenAI embedding client with base_url: {self.base_url}, model: {self.embedding_model_name}")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Get embedding dimension by encoding a test string
        test_embedding = self._encode_single("test")
        self.embedding_dim = len(test_embedding)
        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "encode_params": {
                "batch_size": self.global_config.embedding_batch_size,
                "max_length": self.global_config.embedding_max_seq_len,
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        response = self.client.embeddings.create(
            model=self.embedding_model_name,
            input=text
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding

    def _init_tokenizer(self) -> None:
        """Load the model's own HF tokenizer for exact token-based truncation.

        Resolution order for the tokenizer source:
          1. env EMBEDDING_TOKENIZER_PATH  (set per-model in run scripts)
          2. embedding_model_name          (works if it is a local path or HF hub id)

        If loading fails, self.tokenizer stays None and truncation falls back to a
        character-based estimate (see _truncate_to_max_tokens).
        """
        self.tokenizer = None
        self._tok_reserve = 0  # number of special tokens the tokenizer/server will auto-add
        source = os.getenv("EMBEDDING_TOKENIZER_PATH") or self.embedding_model_name
        if not source:
            logger.warning("No tokenizer source available; using character-estimate truncation.")
            return
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            try:
                self._tok_reserve = int(self.tokenizer.num_special_tokens_to_add(pair=False))
            except Exception:
                self._tok_reserve = 2  # conservative default (e.g. CLS + SEP)
            logger.info(f"Loaded truncation tokenizer from '{source}' "
                        f"(reserving {self._tok_reserve} special tokens)")
        except Exception as e:
            logger.warning(f"Could not load tokenizer from '{source}' ({e}); "
                           f"falling back to character-estimate truncation.")
            self.tokenizer = None

    def _truncate_to_max_tokens(self, texts: List[str], max_length: int) -> List[str]:
        """Truncate each text so that content + special tokens <= max_length.

        Preferred path: exact token-level truncation via the model's own tokenizer.
        Texts whose UTF-8 byte length already fits the budget are skipped (token count
        <= byte count), which avoids tokenizing the many short passages.

        Fallback path (no tokenizer available, e.g. an offline model like bge whose
        tokenizer files are not on disk): estimate by characters (~4 chars/token for
        English, with a 0.8 safety factor). This keeps good capacity utilisation
        instead of a byte cutoff (which would over-truncate English to ~1/4 of the
        limit). It is only an estimate; provide the model's tokenizer for exact limits.
        """
        if self.tokenizer is not None:
            budget = max(1, max_length - self._tok_reserve)
            out = []
            for text in texts:
                if len(text.encode("utf-8")) <= budget:  # tokens <= bytes -> already safe
                    out.append(text)
                    continue
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if len(ids) > budget:
                    text = self.tokenizer.decode(ids[:budget], skip_special_tokens=True)
                out.append(text)
            return out
        # Fallback: character estimate (no tokenizer). ~4 chars/token * 0.8 safety.
        max_chars = int(max_length * 4 * 0.8)
        return [t[:max_chars] if len(t) > max_chars else t for t in texts]

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encode a batch of texts using OpenAI API.

        Args:
            texts: List of text strings to encode
            **kwargs: Additional parameters (batch_size, instruction, etc.)

        Returns:
            np.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)

        batch_size = params.pop("batch_size", 16)

        # 非对称检索是必选项，不是可选优化：NV-Embed-v2 等模型训练时约定
        # query 侧带任务 instruction、passage/corpus 侧不带，两侧不对称才能让
        # query 向量和 passage 向量落在同一个"检索对齐"的空间里；两侧都加/都不
        # 加，检索效果都会明显下降。HippoRAG.py 里 query 编码时通过
        # batch_encode(..., instruction=get_query_instruction(...)) 传了这个
        # 参数，本方法之前只把它塞进 params 却从没读出来发给远程服务，导致这个
        # 参数在网络请求这一层被悄悄丢弃。这里改成通过 extra_body 转发给
        # /v1/embeddings 接口——serve_nvembed.py 等自建服务原生支持这个扩展
        # 字段（OpenAI 官方客户端不会发送这个字段，转发是安全的，官方 API 只
        # 会忽略未知字段）。
        instruction = params.pop("instruction", "") or ""

        # 指令/模板的拼接统一由服务端负责（服务端设计，路线B）：
        #   - NV-Embed：serve_nvembed.py 原生读 instruction 字段；
        #   - GritLM / GTE-Qwen2：serve_embedding_proxy.py 站在 vLLM 前面，读
        #     instruction 字段后按各自模板（GritLM 的 <|user|>..<|embed|>、
        #     GTE 的 "Instruct:..\nQuery:.."）拼好再转发给 vLLM。
        # 因此客户端保持"傻"的、对所有 API 模型行为一致：只把 instruction 通过
        # extra_body 透传即可，不在这里拼模板（否则会和服务端重复拼接）。
        # ⚠ 前提：GritLM/GTE 必须走带代理的 serve_*.sh；若直连 vanilla vLLM，
        #   instruction 会被丢弃、退化成无指令状态（详见 serve_embedding_proxy.py）。
        logger.debug(f"Encoding {len(texts)} texts with batch_size={batch_size}, instruction={instruction!r}")

        # 预分配输出数组、逐行写入：避免"先攒 12 万个小数组的 list，再 np.array 复制
        # 一份"的双份内存峰值（facts 十几万条时峰值 ~4GB，在无 swap 的共享机上会 OOM）。
        # embedding 维度在 __init__ 已探测好（self.embedding_dim）。
        out = np.empty((len(texts), self.embedding_dim), dtype=np.float32)

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Truncate to max_length TOKENS (exact, via the model's own tokenizer)
            # to avoid API errors on long documents. See _truncate_to_max_tokens.
            max_length = params.get("max_length", self.global_config.embedding_max_seq_len)
            batch_texts = self._truncate_to_max_tokens(batch_texts, max_length)

            extra_body = {"instruction": instruction} if instruction else None

            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch_texts,
                    extra_body=extra_body,
                )

                # 直接写进预分配数组对应行（用 item.index 定位，稳妥不依赖返回顺序）
                for item in response.data:
                    out[i + item.index] = item.embedding

            except Exception as e:
                logger.error(f"Error encoding batch {i//batch_size}: {e}")
                raise

        # Normalize if required —— 原地归一化，避免 (out.T / norm).T 再整份复制一次
        if self.embedding_config.norm:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # 防零向量除零
            out /= norms

        return out
