"""
rerank模型排序服务

提供统一的事项/段落排序能力，支持 302.ai 等兼容的 Rerank API
"""

from typing import List, Optional, Dict, Any

from pipeline.core.config import get_settings
from pipeline.exceptions import AIError
from pipeline.utils import get_logger
import aiohttp

logger = get_logger("ai.rerank")


def _get_rerank_template_name(server_type: str, model: str):
    """根据 server_type 和模型名称判断是否需要加载 rerank 模板，返回模板名；无需模板则返回 None"""
    if server_type != "LOCAL":
        return None
    model_lower = model.lower()
    if "qwen3" in model_lower and "rerank" in model_lower:
        return "qwen3_rerank"
    return None


def _load_rerank_template(template_name: str):
    """从 PromptManager 加载指定 rerank 模板配置，返回模板字段字典；加载失败返回 None"""
    try:
        from pipeline.core.prompt.manager import get_prompt_manager
        pm = get_prompt_manager()
        config = pm.get_template_config(template_name)
        return {
            "prefix": config.get("prefix", ""),
            "suffix": config.get("suffix", ""),
            "query_template": config.get("query_template", ""),
            "doc_template": config.get("doc_template", ""),
            "default_instruction": config.get("default_instruction", ""),
        }
    except Exception as e:
        logger.warning(f"加载 Rerank 模板 '{template_name}' 失败，跳过 prompt 组装: {e}")
        return None


def _format_with_template(query, documents, instruction, tpl):
    """使用 Qwen3-Rerank 模板组装 query 和 documents，返回 (formatted_query, formatted_docs)"""
    formatted_query = tpl["query_template"].format(
        prefix=tpl["prefix"], instruction=instruction, query=query
    )
    formatted_docs = [
        tpl["doc_template"].format(doc=doc, suffix=tpl["suffix"]) for doc in documents
    ]
    return formatted_query, formatted_docs


class RerankClient:
    """
    Rerank客户端

    支持 302.ai Rerank API
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        初始化 Rerank 客户端

        Args:
            model: 模型名称（默认从配置读取）
            base_url: API地址（默认从配置读取）
            api_key: API密钥（默认从配置读取）
        """
        settings = get_settings()

        # Rerank 专用配置（优先），回退到 embedding 配置
        self.model = model or getattr(settings, 'rerank_model_name', None) or "Qwen/Qwen3-Reranker-8B"
        # 基础 URL，拼接 /rerank 端点
        base = (base_url or
                getattr(settings, 'rerank_base_url', None) or
                settings.embedding_base_url or
                "https://api.302.ai/v1")
        self.base_url = base.rstrip('/')
        self.api_key = (api_key or
                       getattr(settings, 'rerank_api_key', None) or
                       settings.embedding_api_key or
                       settings.llm_api_key)

        logger.info(
            f"Rerank客户端初始化完成",
            extra={
                "model": self.model,
                "base_url": self.base_url,
            },
        )

    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_n: Optional[int] = None,
        return_documents: Optional[bool] = False,
        use_prompt_template: bool = False,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        使用 Rerank 模型对文档重新排序

        Args:
            query: 查询文本
            documents: 文档列表，格式 [{"id": "...", "text": "..."}, ...]
            top_n: 返回前 N 个结果
            return_documents: 是否在响应中返回文档（传递给API，不影响客户端返回结构）
            use_prompt_template: 是否注入 Qwen3-Rerank 提示词模板（LOCAL 模式本地部署时需要）
            instruction: 重排指令，仅在 use_prompt_template=True 时生效，默认使用模板中的 default_instruction

        Returns:
            排序后的结果列表，格式：
            [
                {
                    "index": 0,           # 原文档索引
                    "id": "doc_id",       # 文档 ID
                    "score": 0.95         # 相关性分数
                },
                ...
            ]
            注意：不返回 text，调用方通过 id 从原始 documents 获取
        """
        if not documents:
            logger.warning("Rerank 调用失败：文档列表为空")
            return []

        top_n = top_n or len(documents)
        logger.warning(f"Rerank排序 返回 top-{top_n} 个事项/段落的排序结果")
        # 提取文档文本
        texts = [doc.get("text", "") for doc in documents]

        # 如果需要，使用 Rerank 模型专用模板组装 query 和 documents
        if use_prompt_template:
            settings = get_settings()
            tpl_name = _get_rerank_template_name(settings.server_type, self.model)
            if tpl_name:
                tpl = _load_rerank_template(tpl_name)
                if tpl:
                    inst = instruction or tpl["default_instruction"]
                    query, texts = _format_with_template(query, texts, inst, tpl)
                    logger.info(f"已注入 Rerank 提示词模板: {tpl_name}")
                else:
                    logger.warning(f"use_prompt_template=True 但模板 '{tpl_name}' 加载失败，使用原始文本")
            else:
                logger.info(f"模型 '{self.model}' 无专用模板 (server_type={settings.server_type})，使用原始文本")

        # 如果 base_url 已包含 /rerank，直接使用；否则拼接
        url = self.base_url if self.base_url.endswith("/rerank") else f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "top_n": top_n,
            "return_documents": return_documents,  # 不返回文档内容，节省带宽
            "documents": texts
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise AIError(f"Rerank API 返回错误 {resp.status}: {error_text}")

                    resp_json = await resp.json()

            # 解析响应（不返回 text，调用方通过 id 从原始 docs 获取）
            results = []
            for item in resp_json.get("results", []):
                idx = item.get("index")
                results.append({
                    "index": idx,
                    "id": documents[idx].get("id", str(idx)),
                    "score": item.get("relevance_score", item.get("score", 0.0))
                })

            logger.info(f"Rerank 完成: 查询='{query[:30]}...', 文档数={len(documents)}, 返回={len(results)}")
            return results

        except Exception as e:
            logger.error(f"Rerank API 调用失败: {e}")
            raise


# ==================== 全局单例 ====================

# 全局单例
_rerank_client: Optional[RerankClient] = None


def get_rerank_client() -> RerankClient:
    """
    获取全局 Rerank 客户端（单例）

    Returns:
        RerankClient 实例
    """
    global _rerank_client
    if _rerank_client is None:
        _rerank_client = RerankClient()
    return _rerank_client


def reset_rerank_client() -> None:
    """重置全局 Rerank 客户端"""
    global _rerank_client
    _rerank_client = None


async def rerank_documents(
    query: str,
    documents: List[Dict[str, str]],
    top_n: Optional[int] = None,
    use_prompt_template: bool = False,
    instruction: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank 文档的便捷函数

    Args:
        query: 查询文本
        documents: 文档列表
        top_n: 返回前 N 个结果
        use_prompt_template: 是否注入 Qwen3-Rerank 提示词模板
        instruction: 重排指令

    Returns:
        排序后的结果列表
    """
    client = get_rerank_client()
    return await client.rerank(query, documents, top_n,
                               use_prompt_template=use_prompt_template,
                               instruction=instruction)
