"""Processor 一致性测试：验证 prompt/example/schema 同路由"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.core.ai.models import LLMMessage, LLMRole
from pipeline.modules.extract.config import ExtractBaseConfig, ExtractPromptStrategy
from pipeline.modules.extract.processor import EventProcessor
from pipeline.modules.extract.prompt_router import ExtractPromptRoute


# ---------------------------------------------------------------------------
# Fake PromptManager: 为不同模板返回带唯一标记的配置
# ---------------------------------------------------------------------------
_FAKE_TEMPLATES = {
    "extract": {
        "template": "ORIGINAL_TEMPLATE",
        "strict_requirements": "ORIGINAL_STRICT",
        "examples": {
            "input": '{"marker": "ORIGINAL_EXAMPLE_INPUT"}',
            "output": '{"marker": "ORIGINAL_EXAMPLE_OUTPUT"}',
        },
        "output_schema": {"marker": "ORIGINAL_SCHEMA"},
        "definitions": {
            "entity": {"properties": {"type": {"enum": []}}},
            "event": {"properties": {}},
        },
    },
    "test_extract": {
        "template": "ATOMIC_TEMPLATE",
        "strict_requirements": "ATOMIC_STRICT",
        "examples": {
            "input": '{"marker": "ATOMIC_EXAMPLE_INPUT"}',
            "output": '{"marker": "ATOMIC_EXAMPLE_OUTPUT"}',
        },
        "output_schema": {"marker": "ATOMIC_SCHEMA"},
        "definitions": {
            "entity": {"properties": {"type": {"enum": []}}},
            "event": {"properties": {}},
        },
    },
    "extract_compact": {
        "template": "COMPACT_TEMPLATE",
        "output_schema": {"marker": "COMPACT_SCHEMA"},
        "definitions": {
            "entity": {"properties": {"type": {"enum": []}}},
            "event": {"properties": {}},
        },
    },
}


def _make_fake_prompt_manager():
    pm = MagicMock()
    pm.get_template_config = MagicMock(
        side_effect=lambda name, test_mode=False: _FAKE_TEMPLATES[name]
    )
    pm.render = MagicMock(return_value="RENDERED_TEMPLATE")
    return pm


def _make_fake_llm_client():
    client = MagicMock()
    client.chat_with_schema = AsyncMock(return_value={
        "type": "response",
        "data": {"items": [], "meta": {}},
    })
    return client


def _make_processor(strategy: ExtractPromptStrategy, test_mode: bool = False):
    config = ExtractBaseConfig(
        extract_prompt_strategy=strategy,
        test_mode=test_mode,
    )
    pm = _make_fake_prompt_manager()
    llm = _make_fake_llm_client()
    return EventProcessor(llm_client=llm, prompt_manager=pm, config=config)


class TestProcessorRouteConsistency:
    """验证 _build_system_prompt / _build_messages / _build_schema 同路由"""

    def test_original_build_system_prompt_uses_extract(self):
        processor = _make_processor(ExtractPromptStrategy.ORIGINAL)
        prompt = processor._build_system_prompt()
        # fake template content should come through
        assert "ORIGINAL" in prompt

    def test_compact_build_system_prompt_uses_extract_compact(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        prompt = processor._build_system_prompt()
        assert "COMPACT" in prompt

    def test_original_build_messages_uses_extract_examples(self):
        processor = _make_processor(ExtractPromptStrategy.ORIGINAL)
        messages = processor._build_messages("sys", {"test": True})
        # find the few-shot example assistant message
        assistant_msgs = [m for m in messages if m.role == LLMRole.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert "ORIGINAL_EXAMPLE_OUTPUT" in assistant_msgs[0].content

    def test_compact_build_messages_has_no_few_shot(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        messages = processor._build_messages("sys", {"test": True})
        assistant_msgs = [m for m in messages if m.role == LLMRole.ASSISTANT]
        assert assistant_msgs == []
        assert len(messages) == 2

    def test_original_build_schema_uses_extract_schema(self):
        processor = _make_processor(ExtractPromptStrategy.ORIGINAL)
        schema = processor._build_schema()
        assert schema.get("marker") == "ORIGINAL_SCHEMA"

    def test_compact_build_schema_uses_compact_schema(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        schema = processor._build_schema()
        assert schema.get("marker") == "COMPACT_SCHEMA"

    def test_atomic_builds_only_from_test_extract(self):
        """Atomic 是独立 test_extract 路由，不得回到 extract 或 compact。"""
        processor = _make_processor(ExtractPromptStrategy.ORIGINAL, test_mode=True)

        assert "ATOMIC" in processor._build_system_prompt()
        messages = processor._build_messages("sys", {"test": True})
        assistant_msgs = [m for m in messages if m.role == LLMRole.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert "ATOMIC_EXAMPLE_OUTPUT" in assistant_msgs[0].content
        assert processor._build_schema().get("marker") == "ATOMIC_SCHEMA"

    def test_compact_entity_type_enum_injected(self):
        """验证 compact schema 的 entity type enum 被动态注入"""
        import asyncio

        from pipeline.db.models import EntityType as DBEntityType

        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        fake_types = [
            MagicMock(spec=DBEntityType, type="organization"),
            MagicMock(spec=DBEntityType, type="person"),
        ]

        async def _init_and_check():
            await processor.initialize(fake_types)
            schema = processor._build_schema()
            entity_def = schema["definitions"]["entity"]
            assert entity_def["properties"]["type"]["enum"] == ["organization", "person"]

        asyncio.run(_init_and_check())

    def test_compact_template_is_static_and_does_not_render(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        processor.prompt_manager.get_template_config = MagicMock(return_value={
            "template": "STATIC_COMPACT",
        })
        assert processor._build_system_prompt() == "STATIC_COMPACT"
        processor.prompt_manager.render.assert_not_called()

    def test_atomic_format_fallback_stays_on_test_extract(self):
        processor = _make_processor(ExtractPromptStrategy.ORIGINAL, test_mode=True)
        processor.prompt_manager.get_template_config = MagicMock(
            return_value={
                "template": "{time} {nonexistent_key}",
                "strict_requirements": "",
            }
        )
        processor.prompt_manager.render = MagicMock(return_value="ATOMIC_FALLBACK")

        assert processor._build_system_prompt() == "ATOMIC_FALLBACK"
        processor.prompt_manager.render.assert_called_once()
        assert processor.prompt_manager.render.call_args[0][0] == "test_extract"


class TestCompactOutputContract:
    @staticmethod
    def _valid_result():
        return {
            "type": "response",
            "data": {
                "items": [
                    {
                        "title": "Event",
                        "content": "Event content",
                        "entities": [
                            {
                                "type": "organization",
                                "name": "Organization",
                                "description": "Subject of the event",
                            }
                        ],
                        "is_valid": True,
                    }
                ],
            },
        }

    def test_valid_compact_output_passes(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        processor._validate_output(self._valid_result())

    @pytest.mark.parametrize(
        "field,value",
        [
            ("summary", "摘要"),
            ("category", "分类"),
            ("keywords", ["关键词"]),
            ("priority", "HIGH"),
            ("status", "COMPLETED"),
            ("references", [1]),
        ],
    )
    def test_deleted_event_field_is_rejected(self, field, value):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["items"][0][field] = value

        with pytest.raises(ValueError, match="compact 输出契约校验失败"):
            processor._validate_output(result)

    def test_meta_is_rejected(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["meta"] = {"reason": "不应出现"}

        with pytest.raises(ValueError, match="data 禁止字段 'meta'"):
            processor._validate_output(result)

    def test_missing_required_field_is_rejected(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        del result["data"]["items"][0]["content"]

        with pytest.raises(ValueError, match="缺少必填字段 'content'"):
            processor._validate_output(result)

    def test_empty_items_is_rejected(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["items"] = []

        with pytest.raises(ValueError, match="必须恰好包含 1 个事项"):
            processor._validate_output(result)

    def test_multiple_items_are_rejected(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["items"].append(result["data"]["items"][0].copy())

        with pytest.raises(ValueError, match="必须恰好包含 1 个事项"):
            processor._validate_output(result)

    def test_empty_title_is_rejected(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["items"][0]["title"] = " "

        with pytest.raises(ValueError, match="字段 'title' 必须是非空字符串"):
            processor._validate_output(result)

    @pytest.mark.parametrize("field", ["confidence", "value_type", "value", "unit"])
    def test_unexpected_entity_field_is_rejected(self, field):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["items"][0]["entities"][0][field] = "forbidden"

        with pytest.raises(ValueError, match=f"实体禁止字段 '{field}'"):
            processor._validate_output(result)

    def test_valid_event_requires_entities(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        result["data"]["items"][0]["entities"] = []

        with pytest.raises(ValueError, match="有效事项的 entities 不得为空"):
            processor._validate_output(result)

    def test_children_is_rejected(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        event = result["data"]["items"][0]
        event["children"] = []

        with pytest.raises(ValueError, match="禁止字段 'children'"):
            processor._validate_output(result)

    def test_invalid_event_may_have_empty_entities(self):
        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = self._valid_result()
        event = result["data"]["items"][0]
        event["is_valid"] = False
        event["entities"] = []

        processor._validate_output(result)


class TestProcessorLogging:
    """验证策略感知日志行为"""

    def test_compact_log_does_not_read_reason_confidence(self, caplog):
        import logging

        processor = _make_processor(ExtractPromptStrategy.COMPACT)
        result = {
            "type": "response",
            "data": {
                "items": [{"title": "t", "content": "c", "entities": []}],
            },
        }
        with caplog.at_level(logging.INFO, logger="pipeline.extract.processor"):
            processor._log_extract_result(result)
        log_text = "\n".join(caplog.messages)
        assert "strategy=compact" in log_text
        assert "reason=" not in log_text

    def test_original_log_still_reads_reason_confidence(self, caplog):
        import logging

        processor = _make_processor(ExtractPromptStrategy.ORIGINAL)
        result = {
            "type": "response",
            "data": {
                "items": [],
                "meta": {"reason": "test reason", "confidence": 0.95},
            },
        }
        with caplog.at_level(logging.INFO, logger="pipeline.extract.processor"):
            processor._log_extract_result(result)
        log_text = "\n".join(caplog.messages)
        assert "reason=test reason" in log_text
        assert "confidence=0.95" in log_text
