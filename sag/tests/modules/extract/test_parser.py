"""Parser 兼容性测试：compact event 默认值与 original event 完整字段"""

import uuid
from unittest.mock import MagicMock, patch

from pipeline.modules.extract.config import ExtractConfig, ExtractPromptStrategy
from pipeline.modules.extract.parser import ParseContext, ResultParser


def _make_config(strategy=ExtractPromptStrategy.COMPACT):
    config = MagicMock(spec=ExtractConfig, source_config_id="test-sc")
    config.extract_prompt_strategy = strategy
    return config


def _make_context():
    return ParseContext(
        source_config_id="test-sc",
        source_type="ARTICLE",
        source_id="article-1",
        chunk_id="chunk-1",
    )


def _make_mock_items(count=3):
    items = []
    for i in range(count):
        item = MagicMock()
        item.id = str(uuid.uuid4())
        item.content = f"content-{i + 1}"
        items.append(item)
    return items


class TestCompactEventParsing:
    """构造不含删除字段的 compact event，验证 parser 默认值"""

    def test_compact_event_defaults(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(5)
        compact_event = {
            "title": "测试标题",
            "content": "测试内容",
            "entities": [
                {"type": "person", "name": "张三", "description": "项目负责人"},
            ],
            "is_valid": True,
        }

        events = parser.parse_events(
            raw_items=[compact_event],
            items=items,
            context=_make_context(),
        )

        assert len(events) == 1
        event = events[0]
        assert event.title == "测试标题"
        assert event.content == "测试内容"
        assert event.summary == ""
        assert event.category is None
        assert event.keywords is None
        assert event.priority is None
        assert event.status is None

    def test_compact_references_route_to_all_input_ids(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(3)
        all_ids = [item.id for item in items]

        compact_event = {
            "title": "无引用事件",
            "content": "内容",
            "entities": [],
            "is_valid": True,
            # 无 references 字段
        }

        with patch.object(
            parser, "_parse_references", wraps=parser._parse_references
        ) as original_reference_parser:
            events = parser.parse_events(
                raw_items=[compact_event],
                items=items,
                context=_make_context(),
            )

        assert len(events) == 1
        assert set(events[0].references) == set(all_ids)
        original_reference_parser.assert_not_called()

    def test_compact_entities_parsed(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(3)

        compact_event = {
            "title": "实体事件",
            "content": "内容",
            "entities": [
                {"type": "org", "name": "ACME", "description": "主要供应商"},
                {"type": "person", "name": "李四", "description": "项目经理"},
            ],
            "is_valid": True,
        }

        events = parser.parse_events(
            raw_items=[compact_event],
            items=items,
            context=_make_context(),
        )

        raw_entities = events[0].extra_data["raw_entities"]["entities"]
        assert len(raw_entities) == 2
        assert raw_entities[0]["name"] == "ACME"
        assert raw_entities[1]["name"] == "李四"

    def test_compact_storage_projection_strips_children(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(3)

        compact_event = {
            "title": "父事件",
            "content": "父内容",
            "entities": [],
            "is_valid": True,
            "children": [
                {
                    "title": "子事件1",
                    "content": "子内容1",
                    "entities": [],
                    "is_valid": True,
                },
                {
                    "title": "子事件2",
                    "content": "子内容2",
                    "entities": [],
                    "is_valid": True,
                },
            ],
        }

        events = parser.parse_events(
            raw_items=[compact_event],
            items=items,
            context=_make_context(),
        )

        assert len(events) == 1
        assert events[0].level == 0
        assert events[0].parent_id is None
        assert "children" not in events[0].extra_data["raw_data"]

    def test_compact_storage_projection_strips_typed_entity_values(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(3)

        compact_event = {
            "title": "数值事件",
            "content": "营收100亿",
            "entities": [
                {
                    "type": "metric",
                    "name": "营收",
                    "description": "公司年度营收",
                    "value_type": "float",
                    "value": "100",
                    "unit": "亿",
                },
            ],
            "is_valid": True,
        }

        events = parser.parse_events(
            raw_items=[compact_event],
            items=items,
            context=_make_context(),
        )

        raw_data = events[0].extra_data["raw_data"]
        entity = raw_data["entities"][0]
        assert entity == {
            "type": "metric",
            "name": "营收",
            "description": "公司年度营收",
        }

    def test_compact_raw_data_excludes_deleted_fields(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(3)

        compact_event = {
            "title": "干净事件",
            "content": "内容",
            "entities": [],
            "is_valid": True,
            "children": [{"title": "不应保存"}],
            "summary": "不应保存",
            "references": [1],
        }

        events = parser.parse_events(
            raw_items=[compact_event],
            items=items,
            context=_make_context(),
        )

        raw_data = events[0].extra_data["raw_data"]
        assert set(raw_data) == {"title", "content", "entities", "is_valid"}
        deleted_fields = {
            "keywords",
            "priority",
            "status",
            "category",
            "summary",
            "references",
            "children",
            "value_type",
            "value",
            "unit",
        }
        for field in deleted_fields:
            assert field not in raw_data, f"raw_data 不应包含字段: {field}"

    def test_compact_is_valid_false_filtered(self):
        parser = ResultParser(config=_make_config())
        items = _make_mock_items(3)

        compact_event = {
            "title": "无效事件",
            "content": "无效内容",
            "entities": [],
            "is_valid": False,
        }

        events = parser.parse_events(
            raw_items=[compact_event],
            items=items,
            context=_make_context(),
        )

        assert len(events) == 0


class TestOriginalEventRegression:
    """构造完整 original event，验证原字段仍正常入库"""

    def test_original_event_all_fields(self):
        parser = ResultParser(config=_make_config(ExtractPromptStrategy.ORIGINAL))
        items = _make_mock_items(5)

        original_event = {
            "title": "完整事件",
            "summary": "这是一段摘要",
            "content": "完整内容文本",
            "category": "科技",
            "keywords": ["AI", "GPT-5"],
            "priority": "HIGH",
            "status": "COMPLETED",
            "references": [1, 2, 3],
            "entities": [
                {"type": "org", "name": "OpenAI", "description": "AI公司"},
            ],
            "is_valid": True,
            "children": [],
        }

        events = parser.parse_events(
            raw_items=[original_event],
            items=items,
            context=_make_context(),
        )

        assert len(events) == 1
        event = events[0]
        assert event.title == "完整事件"
        assert event.summary == "这是一段摘要"
        assert event.content == "完整内容文本"
        assert event.category == "科技"
        assert event.keywords == ["AI", "GPT-5"]
        assert event.priority == "HIGH"
        assert event.status == "COMPLETED"
        # references 应该被解析为 UUID 列表
        assert len(event.references) == 3
