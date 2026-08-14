"""
多元事项检索提示词加载层（MLflow Prompt Registry 接入）

设计目标（见 plan）：
- 启动时**一次性**从 MLflow Prompt Registry 加载全部提示词到内存缓存，
  之后运行全程使用缓存副本，不再连 MLflow；
- 只在那一次加载时打印每条 prompt 的来源（mlflow / const / const-fallback）与版本；
- 回退语义：**仅“未注册（RESOURCE_DOES_NOT_EXIST）”才回退到代码常量**；
  MLflow 服务不可达等基础设施故障 → 直接 raise，暴露问题；
- 是否启用由 SAGConfig.use_mlflow_prompts 控制（默认 False=全用代码常量，
  与改动前逐字节一致）。

代码常量（fallback 源）来自 SAG2 prompt 的 _XXX 常量，是唯一数据源；
本模块把它们转成「MLflow 双花括号 {{ var }} 模板」形式统一处理，
使 use_mlflow_prompts 开/关两条路径共用同一套 .format() 渲染逻辑。
"""

from __future__ import annotations

from pipeline.core.ai.models import LLMMessage, LLMRole

from pipeline.utils import get_logger

logger = get_logger("search.prompts")

# MLflow prompt 命名前缀（Registry 内各分条均以此为前缀）
# 注：MLflow prompt 名只允许字母数字/-/_/.，不接受 '/'，故用点号分隔。
_NAME_PREFIX = "multi_search"
_NAME_SEP = "."


def full_prompt_name(short: str) -> str:
    """短名 -> Registry 全名，如 ner_system -> multi_search.ner_system"""
    return f"{_NAME_PREFIX}{_NAME_SEP}{short}"


# 进程级「全文已打印」标志：benchmark 的 engine 池会建多个 SAG2Runtime 实例，
# 各自持有独立 provider；用模块级标志保证「当前 LLM 使用的提示词全文」在
# 整个进程内只打印一遍（后续实例仍会 load 到内存，只是不再重复刷屏）。
_SUMMARY_PRINTED = False


def _to_double_brace(text: str, *fields: str) -> str:
    """把代码常量里的单花括号占位符转成 MLflow 双花括号模板。

    SAG2 prompt 常量用两种占位：位置 `{}`（仅 _NER_TEMPLATE）与具名 `{top_k}` 等。
    这里统一转成 `{{ field }}`，供 mlflow prompt.format(**kwargs) 与本地回退共用。

    - _NER_TEMPLATE 的位置 `{}` 先转成 `{query}` 再统一处理；
    - 其余具名占位 `{f}` → `{{ f }}`。
    """
    if "{}" in text:
        # 仅 _NER_TEMPLATE = "Question: {}"，唯一位置参数，映射到具名 query
        text = text.replace("{}", "{query}")
    for f in fields:
        text = text.replace("{" + f + "}", "{{ " + f + " }}")
    return text


# ---------------------------------------------------------------------------
# 15 条 prompt 的本地回退模板（双花括号形式）
#   键 = 短名（不含前缀）；值 = 模板串
#   无变量的条目原样存放；含变量的按 _to_double_brace 转换
# ---------------------------------------------------------------------------
_NER_SYSTEM_PROMPT = "You're a very effective entity extraction system."
_NER_ONE_SHOT_INPUT = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?
"""
_NER_ONE_SHOT_OUTPUT = """{"named_entities": ["First for Women", "Arthur's Magazine"]}"""
_NER_TEMPLATE = "Question: {}"

_RERANK_SYSTEM_PROMPT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly {top_k} relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of {top_k} relation lines, most useful first)."""
_RERANK_EXAMPLE_1_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
When did Lothair Ii's mother die?

Relationship descriptions:
[53] bertha married to theobald of arles
[54] bertha married to adalbert ii of tuscany
[42] lothair ii son of ermengarde of tours
[43] lothair ii married to teutberga
[41] lothair ii son of emperor lothair i
[60] lothair ii husband of waldrada
[67] waldrada was mistress of lothair ii
"""
_RERANK_EXAMPLE_1_OUTPUT = """{"thought_process": "2-hop question: First find Lothair II's mother (relation [42]: Ermengarde of Tours), then find death date. [41] gives father for family context.", "useful_relations": ["[42]", "[41]", "[43]", "[60]", "[67]"]}"""
_RERANK_EXAMPLE_2_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
What country is the composer of "Erta Eterna" from?

Relationship descriptions:
[12] terra eterna composed by paulo flores
[15] paulo flores born in angola
[18] paulo flores genre is semba
[22] angola located in africa
[25] semba originated in angola
[30] paulo flores nationality angolan
"""
_RERANK_EXAMPLE_2_OUTPUT = """{"thought_process": "2-hop question: First find composer of Terra Eterna ([12]: Paulo Flores), then find his country ([15] born in Angola or [30] nationality Angolan).", "useful_relations": ["[12]", "[15]", "[30]", "[22]", "[25]"]}"""
_RERANK_EXAMPLE_3_INPUT = """I will provide you with a set of relationship descriptions from a knowledge graph. \
Select exactly 5 relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of 5 relation lines, most useful first).

Question:
Who is the director of the film that won the award also won by "The Hurt Locker"?

Relationship descriptions:
[5] the hurt locker won academy award best picture
[8] the hurt locker directed by kathryn bigelow
[12] moonlight won academy award best picture
[15] moonlight directed by barry jenkins
[20] la la land won golden globe best musical
[25] barry jenkins born in miami
"""
_RERANK_EXAMPLE_3_OUTPUT = """{"thought_process": "3-hop question: (1) Find award won by The Hurt Locker ([5]: Academy Award Best Picture), (2) Find another film with same award ([12]: Moonlight), (3) Find director ([15]: Barry Jenkins).", "useful_relations": ["[5]", "[12]", "[15]", "[8]", "[25]"]}"""
_RERANK_TEMPLATE = """Question:
{question}

Relationship descriptions:
{relations}
"""

_FALLBACKS: dict[str, str] = {
    "ner_system": _NER_SYSTEM_PROMPT,
    "ner_oneshot_input": _NER_ONE_SHOT_INPUT,
    "ner_oneshot_output": _NER_ONE_SHOT_OUTPUT,
    "ner_template": _to_double_brace(_NER_TEMPLATE, "query"),
    "rerank_system": _to_double_brace(_RERANK_SYSTEM_PROMPT, "top_k"),
    "rerank_example_1_input": _RERANK_EXAMPLE_1_INPUT,
    "rerank_example_1_output": _RERANK_EXAMPLE_1_OUTPUT,
    "rerank_example_2_input": _RERANK_EXAMPLE_2_INPUT,
    "rerank_example_2_output": _RERANK_EXAMPLE_2_OUTPUT,
    "rerank_example_3_input": _RERANK_EXAMPLE_3_INPUT,
    "rerank_example_3_output": _RERANK_EXAMPLE_3_OUTPUT,
    "rerank_template": _to_double_brace(_RERANK_TEMPLATE, "question", "relations"),
}

PROMPT_NAMES: list[str] = list(_FALLBACKS.keys())


# ---------------------------------------------------------------------------
# chat 组定义（合并需求）：把同属一组的 system + few-shot + template 合并到
# 同一个 MLflow chat prompt（name=multi_search.<group>）。
#
# GROUP_SPECS[group] = 有序的 (role, 源短名) 序列，既是：
#   - registry 拼装 chat prompt 的顺序来源；
#   - 本地回退 _FALLBACKS_CHAT 的拼装来源；
#   - 验证「合并前后消息序列逐条一致」的对比基准。
# 组名即 MLflow 里的短名：multi_search.ner / .rerank。
# 变量集合（仅文档/参考）：ner={query}, rerank={top_k,question,relations}。
# ---------------------------------------------------------------------------
GROUP_SPECS: dict[str, list[tuple[str, str]]] = {
    "ner": [
        ("system", "ner_system"),
        ("user", "ner_oneshot_input"),
        ("assistant", "ner_oneshot_output"),
        ("user", "ner_template"),
    ],
    "rerank": [
        ("system", "rerank_system"),
        ("user", "rerank_example_1_input"),
        ("assistant", "rerank_example_1_output"),
        ("user", "rerank_example_2_input"),
        ("assistant", "rerank_example_2_output"),
        ("user", "rerank_example_3_input"),
        ("assistant", "rerank_example_3_output"),
        ("user", "rerank_template"),
    ],
}

# 组加载/打印顺序
GROUP_NAMES: list[str] = list(GROUP_SPECS.keys())


def build_chat_from_parts(spec: list[tuple[str, str]], parts: dict[str, str]) -> list[dict]:
    """按组序列把「各短名的文本」拼成 chat 消息列表 [{"role","content"}, ...]。

    parts: {短名: content 文本}。用于 registry 从 MLflow 现有分条 prompt 拼、
    以及本地回退从 _FALLBACKS 拼——两处共用同一顺序，保证一致。
    """
    return [{"role": role, "content": parts[short]} for role, short in spec]


# 本地回退（chat 形式）：把 _FALLBACKS（各分条 str）按组拼成 2 个 list[dict]。
# 仅在「MLflow 组未注册」或 use_mlflow=False 时使用。
_FALLBACKS_CHAT: dict[str, list[dict]] = {
    group: build_chat_from_parts(spec, _FALLBACKS) for group, spec in GROUP_SPECS.items()
}


# SAG2-specific prompts remain code-backed and separate from the legacy
# ``ner``/``rerank`` MLflow groups. Moving the literals here keeps sag2.py
# focused on algorithm orchestration without changing the existing registry
# topology or prompt source semantics.
_SAG2_REWRITE_SYSTEM = (
    "You are a helpful assistant that rewrites search queries and extracts entities."
)
_SAG2_REWRITE_TEMPLATE = """Please rewrite the following query to make it more suitable for search, and extract key entities.

Original query: {query}

Current time: {current_timestamp}

Return JSON with "rewritten_query" and "entities" (list of {{"name": "...", "weight": 1.0}}).
Max entities: {max_entities}"""
_SAG2_RERANK_SYSTEM = """I will provide you with a set of relationship descriptions from a knowledge graph. Select exactly {top_k} relationships most useful for answering this multi-hop question.

Return JSON with "thought_process" and "useful_relations" (list of {top_k} relation indices like "[id]", most useful first)."""
_SAG2_RERANK_EXAMPLE_INPUT = """Question:
When did Lothair II's mother die?

Relationship descriptions:
[53] bertha married to theobald of arles
[54] bertha married to adalbert ii of tuscany
[42] lothair ii son of ermengarde of tours
[43] lothair ii married to teutberga
[41] lothair ii son of emperor lothair i
[60] lothair ii husband of waldrada
[67] waldrada was mistress of lothair ii"""
_SAG2_RERANK_EXAMPLE_OUTPUT = """{"thought_process": "Find Lothair II's mother from relation [42].", "useful_relations": ["[42]", "[41]", "[43]", "[60]", "[67]"]}"""
_SAG2_RERANK_TEMPLATE = """Question:
{question}

Relationship descriptions:
{relations}"""



def _render(template: str, fmt: dict) -> str:
    """按双花括号语义渲染模板。

    MLflow 模板用 `{{ var }}`；本地回退也统一为双花括号，
    因此用与 mlflow.PromptVersion.format 等价的替换：`{{ var }}` -> 值。
    无对应 fmt 的占位保持原样（与「无变量原样返回」一致）。
    """
    out = template
    for k, v in fmt.items():
        out = out.replace("{{ " + k + " }}", str(v))
        out = out.replace("{{" + k + "}}", str(v))
    return out


def _normalize_chat_template(template) -> list[dict]:
    """把 MLflow chat prompt 的 .template 规整为纯 [{"role","content"}, ...]。

    实测 .template 返回 list[dict]；但为稳健，兼容元素是对象（带 .role/.content）
    或 dict 两种形态。
    """
    out: list[dict] = []
    for m in template:
        if isinstance(m, dict):
            out.append({"role": m["role"], "content": m["content"]})
        else:
            out.append({"role": getattr(m, "role"), "content": getattr(m, "content")})
    return out


def _is_not_found(exc: Exception) -> bool:
    """判定异常是否为「prompt 未注册」（RESOURCE_DOES_NOT_EXIST）。

    优先用 MLflow 结构化 error_code；兜底按消息文本匹配。
    仅此类异常触发回退；其它（连接失败等）由调用方 raise。
    """
    try:
        from mlflow.exceptions import MlflowException
        from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode

        if isinstance(exc, MlflowException):
            code = getattr(exc, "error_code", None)
            # error_code 在 MlflowException 上是字符串名（如 "RESOURCE_DOES_NOT_EXIST"）
            if code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                return True
    except Exception:
        pass
    msg = str(exc).lower()
    return "does not exist" in msg or "not found" in msg


class PromptProvider:
    """提示词提供者：一次性加载 + 缓存 + 回退 + 打印。

    生命周期 = 一个 SAG2Runtime 实例。load_all() 只真正执行一次，
    之后 get() 全部命中内存缓存，不再触碰 MLflow。
    """

    def __init__(
        self,
        use_mlflow: bool,
        *,
        alias: str = "latest",
        tracking_uri: str | None = None,
        request_timeout: int = 5,
        max_retries: int = 1,
    ):
        self.use_mlflow = use_mlflow
        self.alias = alias  # 加载时使用 prompts:/name@{alias}（默认 latest）
        # 显式传入的 tracking URI（当全局 _tracking_uri 未正确设置时使用）
        self.tracking_uri = tracking_uri
        # 加载期临时收紧 MLflow HTTP 超时/重试，避免服务不可达时卡几分钟才报错。
        # 仅在 load_all 的 MLflow 分支内生效，加载结束即恢复原环境变量。
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self._loaded = False
        # 组名 -> chat 消息列表 [{"role","content"}, ...]（含未渲染的 {{ var }}）
        self._groups: dict[str, list[dict]] = {}
        # 每组来源标签，用于打印汇总："const" / "mlflow vN" / "const-fallback"
        self._sources: dict[str, str] = {}

    def load_all(self) -> None:
        """加载全部 2 组 chat prompt（幂等，只在首次真正执行）。"""
        if self._loaded:
            return

        if not self.use_mlflow:
            # 全用代码常量（chat 形式）：与改动前逐字节一致的路径
            for group in GROUP_NAMES:
                self._groups[group] = _FALLBACKS_CHAT[group]
                self._sources[group] = "const"
            self._loaded = True
            self._log_summary(source_mode="const")
            return

        import os

        import mlflow

        # 防御：确保 tracking URI 为 HTTP URI。
        # mlflow.genai.load_prompt 内部会创建 MlflowClient() 并初始化 tracking store，
        # 若全局 _tracking_uri 未正确设置为 HTTP URI，会回退到本地 SQLite 存储导致崩溃。
        # 用户显式指定了 --use-mlflow-prompts，必须报错，不允许优雅回退到常量。
        _current_uri = mlflow.get_tracking_uri() or ""
        if not _current_uri.startswith(("http://", "https://")):
            _env_uri = os.environ.get("MLFLOW_TRACKING_URI") or ""
            if _env_uri.startswith(("http://", "https://")):
                mlflow.set_tracking_uri(_env_uri)
            elif self.tracking_uri and self.tracking_uri.startswith(("http://", "https://")):
                mlflow.set_tracking_uri(self.tracking_uri)
            else:
                raise RuntimeError(
                    "已启用 MLflow Prompt Registry (use_mlflow_prompts=True)，"
                    "但无法获取有效的 HTTP tracking URI。\n"
                    f"  - mlflow.get_tracking_uri() = {_current_uri!r}\n"
                    f"  - MLFLOW_TRACKING_URI 环境变量 = {os.environ.get('MLFLOW_TRACKING_URI', '未设置')!r}\n"
                    f"  - self.tracking_uri = {self.tracking_uri!r}\n"
                    "请确保先调用 mlflow.set_tracking_uri('http://...') 或设置 MLFLOW_TRACKING_URI 环境变量，"
                    "然后重试。"
                )

        # 临时收紧 HTTP 超时/重试（仅本次加载），结束后恢复，避免污染进程其它 MLflow 调用。
        _env_keys = ("MLFLOW_HTTP_REQUEST_TIMEOUT", "MLFLOW_HTTP_REQUEST_MAX_RETRIES")
        _saved = {k: os.environ.get(k) for k in _env_keys}
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(self.request_timeout)
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(self.max_retries)
        try:
            for group in GROUP_NAMES:
                name = full_prompt_name(group)
                uri = f"prompts:/{name}@{self.alias}"
                try:
                    prompt = mlflow.genai.load_prompt(uri)
                except Exception as exc:  # noqa: BLE001 — 需按类型分流
                    if _is_not_found(exc):
                        self._groups[group] = _FALLBACKS_CHAT[group]
                        self._sources[group] = "const-fallback"
                        continue
                    raise RuntimeError(
                        f"加载 MLflow prompt 失败（非未注册错误，疑似服务不可达/配置问题）: "
                        f"{uri} -> {exc}"
                    ) from exc

                self._groups[group] = _normalize_chat_template(prompt.template)
                self._sources[group] = f"mlflow v{getattr(prompt, 'version', '?')}"
        finally:
            for k, v in _saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self._loaded = True
        self._log_summary(source_mode="mlflow")

    def _log_summary(self, source_mode: str) -> None:
        """一次性打印加载汇总 + 每组 chat prompt 的完整消息序列（存入内存那一刻）。

        目的：让日志里能直接看到「当前 LLM 实际使用的每一段提示词原文」
        （渲染前，含 {{ var }}），便于确认 MLflow 版本内容是否符合预期。

        进程内只打印一遍：由模块级 _SUMMARY_PRINTED 去重，
        避免 benchmark 多实例（engine 池）重复刷屏。
        """
        global _SUMMARY_PRINTED
        if _SUMMARY_PRINTED:
            return
        _SUMMARY_PRINTED = True

        # 顶部：来源/版本一览（快速扫）
        head = [f"  {group:<14} | source={self._sources[group]}" for group in GROUP_NAMES]
        # 逐组全文（每条 message 打 role + content 全文）
        blocks = []
        for group in GROUP_NAMES:
            msgs = self._groups[group]
            body = "\n".join(
                f"  [{i}] role={m['role']}\n{m['content']}" for i, m in enumerate(msgs)
            )
            blocks.append(
                f"{'─' * 70}\n"
                f"● {full_prompt_name(group)}  [source={self._sources[group]}] "
                f"（{len(msgs)} 条消息）\n"
                f"{'─' * 70}\n"
                f"{body}"
            )
        logger.info(
            f"[Prompt 加载汇总] use_mlflow={self.use_mlflow}, "
            f"共 {len(GROUP_NAMES)} 组:\n"
            + "\n".join(head)
            + "\n\n[Prompt 全文] 当前 LLM 实际使用的提示词原文如下（渲染前）：\n"
            + "\n".join(blocks)
        )

    def get_messages(self, group: str, **fmt) -> list[LLMMessage]:
        """取一组 chat prompt，渲染变量后返回 list[LLMMessage]。

        - 对每条 message 的 content 做双花括号渲染（无关变量原样保留）；
        - role 字符串直接映射到 LLMRole（"system"/"user"/"assistant"）。
        缓存缺失属编程错误（组名写错/未 load_all）→ raise。
        """
        if not self._loaded:
            # 防御：正常应由 SAG2Runtime 在 search() 入口先 load_all
            self.load_all()
        if group not in self._groups:
            raise KeyError(f"未知 prompt 组: {group}（可用: {GROUP_NAMES}）")
        messages: list[LLMMessage] = []
        for m in self._groups[group]:
            content = _render(m["content"], fmt) if fmt else m["content"]
            messages.append(LLMMessage(role=LLMRole(m["role"]), content=content))
        return messages

    def get_sag2_rewrite_messages(
        self,
        *,
        query: str,
        current_timestamp: str,
        max_entities: int,
    ) -> list[LLMMessage]:
        """Build the stable SAG2 query-rewrite message sequence."""
        return [
            LLMMessage(role=LLMRole.SYSTEM, content=_SAG2_REWRITE_SYSTEM),
            LLMMessage(
                role=LLMRole.USER,
                content=_SAG2_REWRITE_TEMPLATE.format(
                    query=query,
                    current_timestamp=current_timestamp,
                    max_entities=max_entities,
                ),
            ),
        ]

    def get_sag2_rerank_messages(
        self,
        *,
        question: str,
        relations: str,
        top_k: int,
    ) -> list[LLMMessage]:
        """Build the stable SAG2 LLM-rerank message sequence."""
        return [
            LLMMessage(
                role=LLMRole.SYSTEM,
                content=_SAG2_RERANK_SYSTEM.format(top_k=top_k),
            ),
            LLMMessage(role=LLMRole.USER, content=_SAG2_RERANK_EXAMPLE_INPUT),
            LLMMessage(role=LLMRole.ASSISTANT, content=_SAG2_RERANK_EXAMPLE_OUTPUT),
            LLMMessage(
                role=LLMRole.USER,
                content=_SAG2_RERANK_TEMPLATE.format(
                    question=question,
                    relations=relations,
                ),
            ),
        ]


__all__ = [
    "PromptProvider",
    "PROMPT_NAMES",
    "GROUP_NAMES",
    "GROUP_SPECS",
    "build_chat_from_parts",
    "_NAME_PREFIX",
    "full_prompt_name",
    "_FALLBACKS",
    "_FALLBACKS_CHAT",
]
