"""Tests for locate_json_string_body_from_string two-pass JSON parsing.

Usage:  python external/hypergraphrag/tests/test_utils.py
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hypergraphrag.utils import locate_json_string_body_from_string


class LocateJSONTests(unittest.TestCase):
    def test_valid_json_with_apostrophe(self):
        result = locate_json_string_body_from_string(
            'The answer is {"key": "don\'t worry"} end.'
        )
        self.assertIsNotNone(result)
        self.assertIn("don't", result)
        self.assertIn('"key": "don\'t worry"', result)

    def test_single_quote_fallback(self):
        result = locate_json_string_body_from_string(
            "{'name': 'hello', 'value': 42}"
        )
        self.assertIsNotNone(result)
        self.assertIn('"name": "hello"', result)

    def test_json_with_surrounding_text(self):
        result = locate_json_string_body_from_string(
            'Here is the extraction result: {"entities": ["Alice", "Bob"]} Thank you.'
        )
        self.assertIsNotNone(result)
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)

    def test_no_json_returns_none(self):
        result = locate_json_string_body_from_string("This is just plain text.")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        self.assertIsNone(locate_json_string_body_from_string(""))

    def test_valid_double_quoted_json(self):
        result = locate_json_string_body_from_string(
            '{"name": "test", "count": 5}'
        )
        self.assertIsNotNone(result)
        self.assertIn('"name": "test"', result)

    def test_preserves_apostrophe_in_value(self):
        result = locate_json_string_body_from_string(
            '{"text": "God\'s Gift to Women"}'
        )
        self.assertIsNotNone(result)
        self.assertTrue(
            "God's Gift" in result or "God\\'s Gift" in result
        )


if __name__ == "__main__":
    unittest.main()
