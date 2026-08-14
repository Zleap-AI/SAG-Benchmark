"""路由测试：resolve_extract_prompt_route 所有有效/无效组合"""

import pytest

from pipeline.modules.extract.config import ExtractPromptStrategy
from pipeline.modules.extract.prompt_router import (
    ExtractPromptRoute,
    resolve_extract_prompt_route,
)


class TestResolveExtractPromptRoute:
    def test_original_no_test_mode(self):
        route = resolve_extract_prompt_route(
            ExtractPromptStrategy.ORIGINAL, test_mode=False
        )
        assert route == ExtractPromptRoute(template_name="extract", test_mode=False)

    def test_original_with_test_mode(self):
        route = resolve_extract_prompt_route(
            ExtractPromptStrategy.ORIGINAL, test_mode=True
        )
        assert route == ExtractPromptRoute(template_name="test_extract", test_mode=False)

    def test_compact_no_test_mode(self):
        route = resolve_extract_prompt_route(
            ExtractPromptStrategy.COMPACT, test_mode=False
        )
        assert route == ExtractPromptRoute(template_name="extract_compact", test_mode=False)

    def test_compact_with_test_mode_raises(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_extract_prompt_route(
                ExtractPromptStrategy.COMPACT, test_mode=True
            )
        assert "compact" in str(exc_info.value).lower()

    def test_route_is_frozen(self):
        route = resolve_extract_prompt_route(
            ExtractPromptStrategy.COMPACT, test_mode=False
        )
        with pytest.raises(Exception):
            route.template_name = "other"  # type: ignore[misc]
