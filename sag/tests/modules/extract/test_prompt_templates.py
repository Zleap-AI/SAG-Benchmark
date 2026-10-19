"""模板加载与 schema 契约测试"""

from pathlib import Path

import pytest
import yaml

from pipeline.core.prompt.manager import PromptManager

# ---------------------------------------------------------------------------
# 被删除字段列表（prompt、example、schema 中均不得出现）
# ---------------------------------------------------------------------------
_DELETED_KEYS = {
    "keywords",
    "priority",
    "status",
    "category",
    "summary",
    "references",
    "reason",
    "confidence",
    "meta",
    "children",
    "value_type",
    "value",
    "unit",
}

# compact event 应精确包含的属性
_COMPACT_EVENT_PROPS = {"title", "content", "entities", "is_valid"}

# compact entity 应精确包含的属性
_COMPACT_ENTITY_PROPS = {"type", "name", "description"}

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_EN_EXTRACT = _PROJECT_ROOT / "prompts" / "en" / "extract.yaml"
_EN_COMPACT = _PROJECT_ROOT / "prompts" / "en" / "extract_compact.yaml"


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _recursive_find_keys(data, target_keys, found=None):
    """递归搜索 data 中是否存在 target_keys 中的任何 key"""
    if found is None:
        found = set()
    if isinstance(data, dict):
        for k, v in data.items():
            if k in target_keys:
                found.add(k)
            _recursive_find_keys(v, target_keys, found)
    elif isinstance(data, list):
        for item in data:
            _recursive_find_keys(item, target_keys, found)
    return found


class TestPromptManagerLoadsCompact:
    """验证 PromptManager 可以加载 extract 和 extract_compact"""

    @pytest.fixture(autouse=True)
    def manager(self):
        return PromptManager()

    def test_extract_loads(self, manager):
        config = manager.get_template_config("extract")
        assert "template" in config
        assert "output_schema" in config
        assert "definitions" in config

    def test_extract_compact_loads(self, manager):
        config = manager.get_template_config("extract_compact")
        assert "template" in config
        assert "output_schema" in config
        assert "definitions" in config
        assert "examples" not in config

    def test_extract_compact_top_level_key(self, manager):
        """compact 模板顶层 key 必须是 extract_compact"""
        assert "extract_compact" in manager.template_data

    def test_extract_compact_has_no_inherited_dynamic_sections(self, manager):
        """compact 是静态独立模板，不继承原模板变量、动态规则或 few-shot。"""
        compact = manager.get_template_config("extract_compact")
        assert "variables" not in compact
        assert "strict_requirements" not in compact
        assert "examples" not in compact

    def test_original_not_modified_by_loading(self, manager):
        """验证原 extract 模板加载后配置未被修改"""
        config = manager.get_template_config("extract")
        schema = config.get("output_schema", {})
        meta = schema.get("properties", {}).get("data", {}).get("properties", {}).get("meta", {})
        # 原 schema 的 meta 应包含 reason
        assert "reason" in meta.get("properties", {})


class TestCompactSchemaContract:
    """显式验证英文 compact schema，不依赖运行环境的语言设置。"""

    @pytest.fixture(autouse=True)
    def compact_config(self):
        return _load_yaml(_EN_COMPACT)["extract_compact"]

    def test_compact_event_properties_exact(self, compact_config):
        event_def = compact_config["definitions"]["event"]
        actual = set(event_def["properties"].keys())
        assert actual == _COMPACT_EVENT_PROPS, f"event properties: {actual}"

    def test_compact_entity_properties_exact(self, compact_config):
        entity_def = compact_config["definitions"]["entity"]
        actual = set(entity_def["properties"].keys())
        assert actual == _COMPACT_ENTITY_PROPS, f"entity properties: {actual}"

    def test_compact_output_has_no_meta(self, compact_config):
        schema = compact_config["output_schema"]
        data_schema = schema["properties"]["data"]
        assert set(data_schema["properties"]) == {"items"}
        assert set(data_schema["required"]) == {"items"}

    def test_compact_event_additional_properties_false(self, compact_config):
        assert compact_config["definitions"]["event"].get("additionalProperties") is False

    def test_compact_entity_additional_properties_false(self, compact_config):
        assert compact_config["definitions"]["entity"].get("additionalProperties") is False

    def test_compact_event_required_fields(self, compact_config):
        required = set(compact_config["definitions"]["event"]["required"])
        assert required == {"title", "content", "entities", "is_valid"}

    def test_compact_entity_required_fields(self, compact_config):
        required = set(compact_config["definitions"]["entity"]["required"])
        assert required == {"type", "name", "description"}

    def test_compact_schema_no_deleted_keys(self, compact_config):
        """递归搜索 schema + definitions，确认无删除字段"""
        schema_and_defs = {
            "output_schema": compact_config["output_schema"],
            "definitions": compact_config["definitions"],
        }
        found = _recursive_find_keys(schema_and_defs, _DELETED_KEYS)
        assert not found, f"schema/definitions 中发现禁止字段: {found}"

    def test_compact_has_no_few_shot_examples(self, compact_config):
        assert "examples" not in compact_config

    def test_compact_prompt_text_no_deleted_field_descriptions(self, compact_config):
        """模板正文不包含删除字段的生成指令"""
        template = compact_config.get("template", "")
        forbidden_phrases = [
            "生成摘要",
            "生成分类",
            "生成关键词",
            "判断优先级",
            "判断状态",
            "列出 references",
            "解释 reason",
            "给出 confidence",
        ]
        for phrase in forbidden_phrases:
            assert phrase not in template, f"模板中发现禁止短语: {phrase}"

    def test_compact_schema_has_entity_type_path(self, compact_config):
        """compact entity schema 存在 type enum 路径（用于动态注入）"""
        entity_def = compact_config["definitions"]["entity"]
        assert "type" in entity_def["properties"]

    def test_compact_has_no_nested_or_typed_fields(self, compact_config):
        schema_and_defs = {
            "output_schema": compact_config["output_schema"],
            "definitions": compact_config["definitions"],
        }
        assert not _recursive_find_keys(
            schema_and_defs,
            {"meta", "children", "value_type", "value", "unit"},
        )

    def test_compact_items_min_max(self, compact_config):
        """items 必须恰好包含一个事项。"""
        items = compact_config["output_schema"]["properties"]["data"]["properties"]["items"]
        assert items.get("minItems") == 1
        assert items.get("maxItems") == 1

    def test_invalid_event_can_have_empty_entities(self, compact_config):
        entities = compact_config["definitions"]["event"]["properties"]["entities"]
        assert "minItems" not in entities

    def test_english_template_is_static_and_independent(self, compact_config):
        rendered = compact_config["template"]
        assert (
            "Generate the title, content, entity names, and entity descriptions in English."
            in rendered
        )
        assert "Put the single event object inside `data.items`" in rendered
        assert "data.meta" not in rendered
        assert "simplified Chinese" not in rendered
        for placeholder in (
            "{time}",
            "{timezone}",
            "{custom_background}",
            "{custom_requirements}",
        ):
            assert placeholder not in rendered

    def test_english_prompt_does_not_request_removed_outputs(self, compact_config):
        template = compact_config["template"]
        forbidden_instructions = [
            "reflected in the summary",
            "Generate a single event summary",
            "Output the final summary",
            "corresponding citations",
            "resolve references",
            "children",
            "value_type",
            "`value`",
            "`unit`",
            "data.meta",
            "simplified Chinese",
        ]
        for phrase in forbidden_instructions:
            assert phrase not in template

    def test_compact_preserves_original_role_wording(self, compact_config):
        original = _load_yaml(_EN_EXTRACT)["extract"]["template"]
        role_sentence = (
            "You are a professional content extractor whose core task is to extract "
            "two types of structured information from raw documents: **events** and "
            "**entities**."
        )
        assert role_sentence in original
        assert role_sentence in compact_config["template"]

    def test_english_original_and_atomic_templates_are_separate_files(self):
        """只确认文件边界；本测试不修改 extract/test_extract。"""
        original = _load_yaml(_EN_EXTRACT)
        atomic = _load_yaml(_PROJECT_ROOT / "prompts" / "en" / "test_extract.yaml")
        compact = _load_yaml(_EN_COMPACT)
        assert "extract" in original
        assert "test_extract" in atomic
        assert "extract_compact" in compact
