"""DatasetLoader / load_utils 的单元测试。

使用 tmp_path 合成的小数据集，不依赖真实的 94MB narrativeqa 文件。
"""

import json

import pytest

from pipeline.evaluation.utils.load_utils import DatasetLoader


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_narrativeqa_fixtures(tmp_path):
    """写一份小的 NarrativeQA 数据集，不把 document 字段当作 gold_docs。"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    corpus = [
        {"idx": "docA_0", "title": "Title A", "text": "chunk0"},
        {"idx": "docA_1", "title": "Title A", "text": "chunk1"},
        {"idx": "docA_2", "title": "Title A", "text": "chunk2"},
        {"idx": "docB_0", "title": "Title B", "text": "chunk0"},
        {"idx": "docB_1", "title": "Title B", "text": "chunk1"},
    ]
    _write_json(dataset_dir / "narrativeqa_corpus.json", corpus)

    samples = [
        {
            "document": {"id": "docA"},
            "question": "Question A?",
            "answer": ["ans1", "ans2"],
        },
        {
            "document": {"id": "docB"},
            "question": "Question B?",
            "answer": ["ans3", "ans4"],
        },
    ]
    _write_json(dataset_dir / "narrativeqa.json", samples)

    return dataset_dir


def test_supported_datasets_registration():
    assert "narrativeqa" in DatasetLoader.SUPPORTED_DATASETS


def test_narrativeqa_without_gold_docs_keeps_empty_recall_labels(tmp_path):
    dataset_dir = _make_narrativeqa_fixtures(tmp_path)
    loader = DatasetLoader("narrativeqa", dataset_dir=str(dataset_dir))

    gold_docs = loader.get_gold_docs_for_recall()

    assert gold_docs == [[], []]


def test_narrativeqa_questions_and_answers(tmp_path):
    dataset_dir = _make_narrativeqa_fixtures(tmp_path)
    loader = DatasetLoader("narrativeqa", dataset_dir=str(dataset_dir))

    assert loader.get_questions() == ["Question A?", "Question B?"]
    assert loader.get_gold_answers() == [{"ans1", "ans2"}, {"ans3", "ans4"}]


def test_narrativeqa_corpus_can_be_saved_as_markdown(tmp_path):
    dataset_dir = _make_narrativeqa_fixtures(tmp_path)
    output_dir = tmp_path / "markdown"
    loader = DatasetLoader("narrativeqa", dataset_dir=str(dataset_dir))

    result = loader.save_as_markdown(
        output_dir=str(output_dir), chunks_per_file=3, force_regenerate=True
    )

    assert result["stats"]["total_chunks"] == 5
    assert result["stats"]["num_files"] == 2
    assert result["stats"]["last_file_chunks"] == 2
    first_part = (output_dir / "narrativeqa_part1.md").read_text(encoding="utf-8")
    assert "# Title A" in first_part
    assert "chunk0" in first_part


def test_hotpotqa_branch_regression(tmp_path):
    """narrativeqa 分支不应影响 hotpotqa 风格数据的既有行为。"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # hotpotqa 需要 corpus 文件存在（get_docs/load_all 会读取），这里给空的即可
    _write_json(dataset_dir / "hotpotqa_corpus.json", [])
    samples = [
        {
            "question": "Q?",
            "answer": ["A"],
            "supporting_facts": [["wiki1", 0]],
            "context": [
                ["wiki1", ["sentence1", "sentence2"]],
                ["wiki2", ["unrelated"]],
            ],
        }
    ]
    _write_json(dataset_dir / "hotpotqa.json", samples)

    loader = DatasetLoader("hotpotqa", dataset_dir=str(dataset_dir))
    gold_docs = loader.get_gold_docs_for_recall()

    assert gold_docs is not None
    assert gold_docs[0] == ["wiki1\nsentence1sentence2"]


def test_discover_requires_complete_source_pair(tmp_path):
    _write_json(tmp_path / "complete.json", [])
    _write_json(tmp_path / "complete_corpus.json", [])
    _write_json(tmp_path / "orphan_corpus.json", [])

    assert DatasetLoader.discover_datasets(tmp_path) == ["complete"]


def test_docs_support_paragraph_text_and_stable_deduplication(tmp_path):
    _write_json(tmp_path / "demo.json", [])
    _write_json(
        tmp_path / "demo_corpus.json",
        [
            {"title": "T", "paragraph_text": "body"},
            {"title": "T", "text": "body"},
            {"title": "", "text": ""},
        ],
    )

    assert DatasetLoader("demo", tmp_path).get_docs() == ["T\nbody"]


def test_question_records_keep_source_alignment(tmp_path):
    _write_json(tmp_path / "hotpotqa_corpus.json", [])
    _write_json(
        tmp_path / "hotpotqa.json",
        [
            {
                "_id": "q-1",
                "question": "Q?",
                "answer": "A",
                "answer_aliases": ["alias"],
                "supporting_facts": [["wiki1", 0]],
                "context": [["wiki1", ["sentence"]]],
            }
        ],
    )

    assert DatasetLoader("hotpotqa", tmp_path).get_question_records() == [
        {
            "id": "q-1",
            "question": "Q?",
            "gold_answers": ["A", "alias"],
            "gold_docs": ["wiki1\nsentence"],
            "gold_ref": "wiki1\nsentence",
        }
    ]


def test_narrativeqa_question_records_use_row_index_identity(tmp_path):
    """narrativeqa 无原生 id：身份用行号，且不得取 document.id（不唯一）。"""
    dataset_dir = _make_narrativeqa_fixtures(tmp_path)

    records = DatasetLoader("narrativeqa", dataset_dir=str(dataset_dir)).get_question_records()

    assert [record["id"] for record in records] == ["0", "1"]
    assert [record["question"] for record in records] == ["Question A?", "Question B?"]
    assert records[0]["gold_answers"] == ["ans1", "ans2"]
    assert records[0]["gold_docs"] == []


def test_narrativeqa_duplicate_document_id_does_not_collide(tmp_path):
    """同一 document.id 的多行必须拿到不同身份，不能触发 duplicate sample ID。"""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_json(dataset_dir / "narrativeqa_corpus.json", [])
    _write_json(
        dataset_dir / "narrativeqa.json",
        [
            {"document": {"id": "docA"}, "question": "Q1?", "answer": ["a1"]},
            {"document": {"id": "docA"}, "question": "Q2?", "answer": ["a2"]},
            {"document": {"id": "docA"}, "question": "Q3?", "answer": ["a3"]},
        ],
    )

    records = DatasetLoader("narrativeqa", dataset_dir=str(dataset_dir)).get_question_records()

    assert [record["id"] for record in records] == ["0", "1", "2"]


def test_question_records_still_reject_missing_native_id(tmp_path):
    """带原生 id 的数据集不受回退影响：缺 id 仍必须显式报错。"""
    _write_json(tmp_path / "hotpotqa_corpus.json", [])
    _write_json(
        tmp_path / "hotpotqa.json",
        [
            {
                "question": "Q?",
                "answer": "A",
                "supporting_facts": [["wiki1", 0]],
                "context": [["wiki1", ["sentence"]]],
            }
        ],
    )

    with pytest.raises(ValueError, match="non-empty string sample ID"):
        DatasetLoader("hotpotqa", tmp_path).get_question_records()
