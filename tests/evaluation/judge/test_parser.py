"""Tests for JSON parser fallback chain."""

import pytest

from pipeline.evaluation.judge.parser import (
    extract_array_fallback,
    extract_json_block,
    safe_json_parse,
    validate_list,
)


class TestSafeJsonParse:
    def test_valid_json_object(self):
        result = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_json_array(self):
        result = safe_json_parse('["a", "b"]')
        assert result == ["a", "b"]

    def test_json5_trailing_comma(self):
        """json5 allows trailing commas that JSON does not."""
        result = safe_json_parse('{"key": "value",}')
        assert result == {"key": "value"}

    def test_json5_single_quotes(self):
        result = safe_json_parse("{'key': 'value'}")
        assert result == {"key": "value"}

    def test_json_repair_broken(self):
        result = safe_json_parse('{"key": "value"')
        assert result == {"key": "value"}

    def test_completely_invalid(self):
        result = safe_json_parse("not json at all!!!")
        assert result == {}


class TestExtractJsonBlock:
    def test_plain_json(self):
        result = extract_json_block('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_in_code_fence(self):
        text = 'Here is some text\n```json\n{"key": "value"}\n```\nMore text'
        # extract_json_block finds the first { ... } pair
        result = extract_json_block(text)
        assert "key" in result
        assert "value" in result

    def test_no_json(self):
        result = extract_json_block("just plain text")
        assert result == "just plain text"

    def test_nested_braces(self):
        result = extract_json_block('{"outer": {"inner": "value"}}')
        assert "outer" in result
        assert "inner" in result


class TestExtractArrayFallback:
    def test_simple_array(self):
        result = extract_array_fallback('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_no_array(self):
        result = extract_array_fallback("no array here")
        assert result == []

    def test_empty_array(self):
        result = extract_array_fallback("[]")
        assert result == []

    def test_single_element(self):
        result = extract_array_fallback('["only"]')
        assert result == ["only"]


class TestValidateList:
    def test_string_list(self):
        result = validate_list(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_dict_list(self):
        result = validate_list([{"key": "val"}, {"key2": "val2"}])
        assert result == [{"key": "val"}, {"key2": "val2"}]

    def test_mixed_types(self):
        result = validate_list(["a", {"key": "val"}, "", "  "])
        assert result == ["a", {"key": "val"}]

    def test_non_list(self):
        assert validate_list("not a list") == []
        assert validate_list(42) == []
        assert validate_list(None) == []

    def test_all_empty_strings(self):
        result = validate_list(["", "  ", "\t"])
        assert result == []


class TestParseWithFallbacks:
    @pytest.mark.asyncio
    async def test_direct_json(self):
        from pipeline.evaluation.judge.parser import parse_with_fallbacks

        result = await parse_with_fallbacks('{"key": ["a", "b"]}', key="key")
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_code_fence_extraction(self):
        from pipeline.evaluation.judge.parser import parse_with_fallbacks

        text = '```json\n{"items": ["x", "y"]}\n```'
        result = await parse_with_fallbacks(text, key="items")
        assert result == ["x", "y"]

    @pytest.mark.asyncio
    async def test_array_fallback(self):
        from pipeline.evaluation.judge.parser import parse_with_fallbacks

        result = await parse_with_fallbacks('some text ["a", "b"] more', key="missing_key")
        assert result == ["a", "b"]

    @pytest.mark.asyncio
    async def test_no_key_returns_data(self):
        from pipeline.evaluation.judge.parser import parse_with_fallbacks

        result = await parse_with_fallbacks('{"x": 1}')
        assert result == {"x": 1}

    @pytest.mark.asyncio
    async def test_missing_key_no_array(self):
        from pipeline.evaluation.judge.parser import parse_with_fallbacks

        result = await parse_with_fallbacks('{"other": 1}', key="missing")
        assert result == []
