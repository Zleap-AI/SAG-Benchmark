"""
Extract 提示词路由层

职责：
- 把 ExtractPromptStrategy 映射为基础模板名
- 处理原策略的 test_mode
- 返回不可变的模板选择结果
- 不读取 YAML，不构建 prompt，不包含业务字段
"""

from dataclasses import dataclass

from .config import ExtractPromptStrategy


@dataclass(frozen=True)
class ExtractPromptRoute:
    template_name: str
    test_mode: bool = False


_ROUTES: dict[ExtractPromptStrategy, ExtractPromptRoute] = {
    ExtractPromptStrategy.ORIGINAL: ExtractPromptRoute(
        template_name="extract",
        test_mode=False,
    ),
    ExtractPromptStrategy.COMPACT: ExtractPromptRoute(
        template_name="extract_compact",
        test_mode=False,
    ),
}


def resolve_extract_prompt_route(
    strategy: ExtractPromptStrategy | str,
    *,
    test_mode: bool,
) -> ExtractPromptRoute:
    """将策略 + test_mode 解析为模板路由。

    original + test_mode=False → template_name="extract"
    original + test_mode=True  → template_name="test_extract"
    compact  + test_mode=False → template_name="extract_compact"
    compact  + test_mode=True  → 应在 config validator 层被拒绝；此处防御性检查
    """
    # pipelineBaseModel 配置会将 Enum 序列化为字符串；路由入口统一恢复枚举，
    # 禁止使用对象 identity 判断。
    strategy = ExtractPromptStrategy(strategy)

    if strategy == ExtractPromptStrategy.ORIGINAL:
        if test_mode:
            # test_extract 是独立 atomic 模板。直接返回有效模板名，避免后续
            # format fallback 或日志路径误回到普通 extract。
            return ExtractPromptRoute(template_name="test_extract", test_mode=False)
        return _ROUTES[ExtractPromptStrategy.ORIGINAL]

    # compact 策略：防御性检查
    if test_mode:
        raise ValueError(
            "compact extract strategy cannot be combined with test_mode. "
            "This should have been caught by config validation."
        )

    return _ROUTES[strategy]
