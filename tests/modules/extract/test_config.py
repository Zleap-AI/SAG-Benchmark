"""配置测试：ExtractPromptStrategy 枚举、默认值、组合校验"""

import pytest
from pydantic import ValidationError

from pipeline.modules.extract.config import ExtractBaseConfig, ExtractPromptStrategy


class TestExtractPromptStrategy:
    def test_default_is_compact(self):
        config = ExtractBaseConfig()
        assert config.extract_prompt_strategy == ExtractPromptStrategy.COMPACT

    def test_string_compact_coerced_to_enum(self):
        config = ExtractBaseConfig(extract_prompt_strategy="compact")
        assert config.extract_prompt_strategy == ExtractPromptStrategy.COMPACT

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValidationError):
            ExtractBaseConfig(extract_prompt_strategy="compact_v2")

    def test_invalid_strategy_simple_raises(self):
        with pytest.raises(ValidationError):
            ExtractBaseConfig(extract_prompt_strategy="simple")

    def test_original_without_test_mode_valid(self):
        config = ExtractBaseConfig(
            extract_prompt_strategy="original", test_mode=False
        )
        assert config.extract_prompt_strategy == ExtractPromptStrategy.ORIGINAL

    def test_original_with_test_mode_valid(self):
        config = ExtractBaseConfig(
            extract_prompt_strategy="original", test_mode=True
        )
        assert config.extract_prompt_strategy == ExtractPromptStrategy.ORIGINAL
        assert config.test_mode is True

    def test_compact_without_test_mode_valid(self):
        config = ExtractBaseConfig(
            extract_prompt_strategy="compact", test_mode=False
        )
        assert config.extract_prompt_strategy == ExtractPromptStrategy.COMPACT

    def test_compact_with_test_mode_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            ExtractBaseConfig(extract_prompt_strategy="compact", test_mode=True)
        assert "compact" in str(exc_info.value).lower()
        assert "test_mode" in str(exc_info.value).lower()
