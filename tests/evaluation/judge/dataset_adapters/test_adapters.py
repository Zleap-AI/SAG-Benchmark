"""Schema and conversion contracts for built-in dataset adapters."""

import pytest

from pipeline.evaluation.judge.dataset_adapters.errors import DatasetSchemaError
from pipeline.evaluation.judge.dataset_adapters.hotpotqa import HotpotQAAdapter
from pipeline.evaluation.judge.dataset_adapters.musique import MusiqueAdapter
from pipeline.evaluation.judge.dataset_adapters.sample import SampleAdapter
from pipeline.evaluation.judge.dataset_adapters.two_wiki import TwoWikiAdapter


def _hotpot_row():
    return {
        "question": "Who?",
        "answer": "A",
        "supporting_facts": [["Title", 99]],
        "context": [["Title", ["Gold sentence one. ", "Gold sentence two."]]],
    }


def test_hotpotqa_uses_complete_title_document_and_ignores_sentence_index():
    sample = HotpotQAAdapter().parse_sample(_hotpot_row(), 0)
    assert sample.id == 0
    assert sample.evidences == ("Title\nGold sentence one. Gold sentence two.",)


def test_test_hotpotqa_is_an_explicit_hotpot_alias():
    assert "test_hotpotqa" in HotpotQAAdapter().descriptor.aliases


def test_two_wiki_ignores_relation_triples():
    row = _hotpot_row()
    row["evidences"] = [["Title", "relation", "Value"]]
    sample = TwoWikiAdapter().parse_sample(row, 0)
    assert sample.evidences == ("Title\nGold sentence one.  Gold sentence two.",)


def test_musique_uses_bare_supporting_paragraph_text():
    row = {
        "id": "musique-sample-1",
        "question": "Who?",
        "answer": "A",
        "paragraphs": [
            {"title": "Gold title", "paragraph_text": "gold", "is_supporting": True},
            {"title": "Noise title", "paragraph_text": "noise", "is_supporting": False},
        ],
    }
    sample = MusiqueAdapter().parse_sample(row, 0)
    assert sample.dataset_sample_id == "musique-sample-1"
    assert sample.evidences == ("gold",)


def test_sample_uses_title_when_present_and_keeps_legacy_bare_text_when_absent():
    titled = {
        "question": "Who?",
        "answer": "A",
        "paragraphs": [{"title": "Gold title", "text": "gold", "is_supporting": True}],
    }
    legacy = {
        "question": "Who?",
        "answer": "A",
        "paragraphs": [{"text": "legacy gold", "is_supporting": True}],
    }
    adapter = SampleAdapter()

    assert adapter.parse_sample(titled, 0).evidences == ("Gold title\ngold",)
    assert adapter.parse_sample(legacy, 0).evidences == ("legacy gold",)


def test_sample_uses_native_id_as_dataset_sample_id():
    row = {
        "id": "sample/question_1.json",
        "question": "Who?",
        "answer": "A",
        "paragraphs": [{"text": "gold", "is_supporting": True}],
    }

    assert SampleAdapter().parse_sample(row, 0).dataset_sample_id == "sample/question_1.json"


@pytest.mark.parametrize("bad_id", [123, "   "])
def test_sample_rejects_invalid_native_id(bad_id):
    row = {
        "id": bad_id,
        "question": "Who?",
        "answer": "A",
        "paragraphs": [{"text": "gold", "is_supporting": True}],
    }

    with pytest.raises(DatasetSchemaError, match="non-empty"):
        SampleAdapter().parse_sample(row, 0)


@pytest.mark.parametrize("bad_id", [123, "   "])
def test_musique_rejects_invalid_native_id(bad_id):
    row = {
        "id": bad_id,
        "question": "Who?",
        "answer": "A",
        "paragraphs": [{"paragraph_text": "gold", "is_supporting": True}],
    }

    with pytest.raises(DatasetSchemaError, match="non-empty"):
        MusiqueAdapter().parse_sample(row, 0)


def test_musique_rejects_supporting_empty_text():
    row = {
        "question": "Who?",
        "answer": "A",
        "paragraphs": [{"title": "Gold title", "paragraph_text": " ", "is_supporting": True}],
    }
    with pytest.raises(DatasetSchemaError, match="must not be empty"):
        MusiqueAdapter().parse_sample(row, 0)


def test_hotpotqa_ignores_out_of_range_fact_index():
    row = _hotpot_row()
    row["supporting_facts"] = [["Title", -2]]
    assert HotpotQAAdapter().parse_sample(row, 0).evidences == (
        "Title\nGold sentence one. Gold sentence two.",
    )


def test_two_wiki_out_of_range_index_still_selects_complete_title_document():
    row = {
        "question": "Who?",
        "answer": "A",
        "supporting_facts": [["Broken", 9], ["Valid", 0]],
        "context": [["Broken", ["one sentence"]], ["Valid", ["gold"]]],
    }
    sample = TwoWikiAdapter().parse_sample(row, 0)
    assert sample.evidences == ("Broken\none sentence", "Valid\ngold")
    assert sample.metadata == {}


def test_two_wiki_missing_title_produces_no_document_but_keeps_valid_titles():
    row = {
        "question": "Who?",
        "answer": "A",
        "supporting_facts": [["Missing", 0], ["Valid", 7]],
        "context": [["Valid", ["gold", "context"]]],
    }
    assert TwoWikiAdapter().parse_sample(row, 0).evidences == ("Valid\ngold context",)


@pytest.mark.parametrize("answer", ["", "  ", [], [""], ["valid", " "]])
def test_dataset_adapter_rejects_empty_answers(answer):
    row = _hotpot_row()
    row["answer"] = answer
    with pytest.raises(DatasetSchemaError, match="non-empty"):
        HotpotQAAdapter().parse_sample(row, 0)
