"""Tests for Judge LLM scenario configuration isolation."""

import os
from unittest.mock import patch

import pytest

from pipeline.core.config.settings import Settings


def _clean_settings(**overrides):
    """Create Settings that does NOT read the real .env file."""
    return Settings(_env_file=None, **overrides)


class TestJudgeSettingsDefaults:
    """Verify judge_llm_* fields exist with correct defaults."""

    def test_judge_fields_have_defaults(self):
        s = _clean_settings()
        assert s.judge_llm_api_key == ""
        assert s.judge_llm_model == ""
        assert s.judge_llm_base_url is None
        assert s.judge_llm_max_async == 3
        assert s.judge_llm_timeout == 180
        assert s.judge_llm_max_retries == 3
        assert s.judge_llm_enable_thinking is False
        assert s.judge_llm_max_tokens is None
        assert s.judge_allow_fallback is False

    def test_judge_fields_from_env(self):
        with patch.dict(
            os.environ,
            {
                "JUDGE_LLM_API_KEY": "sk-test-judge",
                "JUDGE_LLM_MODEL": "judge-model",
                "JUDGE_LLM_BASE_URL": "https://judge.api/v1",
                "JUDGE_LLM_MAX_ASYNC": "2",
                "JUDGE_LLM_TIMEOUT": "120",
                "JUDGE_LLM_MAX_RETRIES": "3",
                "JUDGE_LLM_ENABLE_THINKING": "true",
                "JUDGE_LLM_MAX_TOKENS": "4096",
                "JUDGE_ALLOW_FALLBACK": "true",
            },
            clear=True,
        ):
            s = _clean_settings()
            assert s.judge_llm_api_key == "sk-test-judge"
            assert s.judge_llm_model == "judge-model"
            assert s.judge_llm_base_url == "https://judge.api/v1"
            assert s.judge_llm_max_async == 2
            assert s.judge_llm_timeout == 120
            assert s.judge_llm_max_retries == 3
            assert s.judge_llm_enable_thinking is True
            assert s.judge_llm_max_tokens == 4096
            assert s.judge_allow_fallback is True


class TestJudgeScenarioConfig:
    """Verify _build_judge_config produces correct config dict."""

    def test_build_judge_config_with_dedicated_settings(self):
        from pipeline.core.ai.factory import _build_judge_config

        with patch.dict(
            os.environ,
            {
                "JUDGE_LLM_API_KEY": "sk-judge",
                "JUDGE_LLM_MODEL": "qwen-judge",
                "JUDGE_LLM_BASE_URL": "https://judge.api/v1",
                "JUDGE_LLM_TIMEOUT": "180",
                "JUDGE_LLM_MAX_TOKENS": "4096",
                "JUDGE_LLM_MAX_RETRIES": "3",
            },
            clear=True,
        ):
            s = _clean_settings()
            cfg = _build_judge_config(s)
            assert cfg["api_key"] == "sk-judge"
            assert cfg["model"] == "qwen-judge"
            assert cfg["base_url"] == "https://judge.api/v1"
            assert cfg["temperature"] == 0.0
            assert cfg["max_tokens"] == 4096
            assert cfg["timeout"] == 180
            assert cfg["max_retries"] == 3
            assert cfg["enable_thinking"] is False
            assert cfg["top_p"] == 1.0

    def test_build_judge_config_raises_if_missing_key(self):
        from pipeline.core.ai.factory import _build_judge_config
        from pipeline.exceptions import ConfigError

        with patch.dict(os.environ, {}, clear=True):
            s = _clean_settings()
            with pytest.raises(ConfigError, match="JUDGE_LLM_API_KEY"):
                _build_judge_config(s)

    def test_build_judge_config_fallback_enabled(self):
        from pipeline.core.ai.factory import _build_judge_config

        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "sk-main",
                "LLM_MODEL": "main-model",
                "JUDGE_ALLOW_FALLBACK": "true",
            },
            clear=True,
        ):
            s = _clean_settings()
            cfg = _build_judge_config(s)
            assert cfg["api_key"] == "sk-main"
            assert cfg["model"] == "main-model"
            assert cfg["temperature"] == 0.0
            assert cfg["enable_thinking"] is False


class TestThinkingIsolation:
    """Verify enable_thinking is per-client, not global."""

    def test_model_config_has_enable_thinking_field(self):
        from pipeline.core.ai.models import LLMProvider, ModelConfig

        cfg = ModelConfig(
            provider=LLMProvider.OPENAI,
            model="test",
            api_key="sk-test",
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            enable_thinking=True,
        )
        assert cfg.enable_thinking is True

        cfg2 = ModelConfig(
            provider=LLMProvider.OPENAI,
            model="test2",
            api_key="sk-test2",
            temperature=0.7,
            max_tokens=200,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            enable_thinking=False,
        )
        assert cfg2.enable_thinking is False

    def test_llm_config_thinking_not_pollute(self):
        """Judge and general configs have independent enable_thinking."""
        from pipeline.core.ai.models import LLMProvider, ModelConfig

        judge_cfg = ModelConfig(
            provider=LLMProvider.OPENAI,
            model="judge-model",
            api_key="sk-judge",
            temperature=0.0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            enable_thinking=False,
        )

        general_cfg = ModelConfig(
            provider=LLMProvider.OPENAI,
            model="main-model",
            api_key="sk-main",
            temperature=0.7,
            max_tokens=200,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            enable_thinking=True,
        )

        assert judge_cfg.enable_thinking is False
        assert general_cfg.enable_thinking is True
        assert judge_cfg.model != general_cfg.model
        assert judge_cfg.api_key != general_cfg.api_key


class TestJudgeLogSafety:
    """Verify error messages do not leak keys."""

    def test_config_error_no_key_in_message(self):
        from pipeline.core.ai.factory import _build_judge_config
        from pipeline.exceptions import ConfigError

        with patch.dict(
            os.environ,
            {
                "JUDGE_LLM_MODEL": "some-model",
            },
            clear=True,
        ):
            s = _clean_settings()
            with pytest.raises(ConfigError) as exc_info:
                _build_judge_config(s)
            msg = str(exc_info.value)
            assert "JUDGE_LLM_API_KEY" in msg
            # Real key must not appear even if it was in the .env (which we excluded)
            assert "sk-" not in msg.lower() or "sk-" not in msg


class TestRetryClientLifecycle:
    @pytest.mark.asyncio
    async def test_close_delegates_to_wrapped_client(self):
        from pipeline.core.ai.base import LLMRetryClient

        class DummyConfig:
            max_retries = 1

        class DummyClient:
            config = DummyConfig()
            closed = False

            async def close(self):
                self.closed = True

        wrapped = DummyClient()
        client = LLMRetryClient(wrapped)

        assert client.config is wrapped.config
        await client.close()

        assert wrapped.closed is True
