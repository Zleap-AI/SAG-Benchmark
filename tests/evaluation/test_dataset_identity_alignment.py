"""守护测试：loader 的行身份必须与 judge 的 dataset_sample_id 逐个数据集一致。

身份口径分别写在 DatasetLoader 和各 dataset adapter 里，两侧一旦漂移，
judge 的 match_dataset_sample_id 会匹配失败。这里用合成夹具交叉校验。
"""

import json
from pathlib import Path

import pytest

from pipeline.evaluation.judge.dataset_io import GroundTruthRepository
from pipeline.evaluation.utils.load_utils import DatasetLoader

# 每个数据集一份最小合成行，形状取自真实数据集（真实 narrativeqa 过大，不跑真实文件）。
FIXTURES: dict[str, tuple[list[dict], list[str | None]]] = {
    "sample": (
        [
            {
                "id": "sample/question_1.json",
                "question": "Q1?",
                "answer": ["A1"],
                "paragraphs": [{"title": "T", "text": "gold", "is_supporting": True}],
            }
        ],
        ["sample/question_1.json"],
    ),
    "musique": (
        [
            {
                "id": "musique-1",
                "question": "Q1?",
                "answer": "A1",
                "paragraphs": [{"paragraph_text": "gold", "is_supporting": True}],
            }
        ],
        ["musique-1"],
    ),
    "hotpotqa": (
        [
            {
                "_id": "hp-1",
                "question": "Q1?",
                "answer": "A1",
                "supporting_facts": [["W", 0]],
                "context": [["W", ["s"]]],
            }
        ],
        ["hp-1"],
    ),
    "2wikimultihopqa": (
        [
            {
                "_id": "2w-1",
                "question": "Q1?",
                "answer": "A1",
                "supporting_facts": [["W", 0]],
                "context": [["W", ["s"]]],
            }
        ],
        ["2w-1"],
    ),
    # HotpotQAAdapter 的注册别名，覆盖 registry 的别名解析分支。
    "test_hotpotqa": (
        [
            {
                "_id": "thp-1",
                "question": "Q1?",
                "answer": "A1",
                "supporting_facts": [["W", 0]],
                "context": [["W", ["s"]]],
            }
        ],
        ["thp-1"],
    ),
    "narrativeqa": (
        [
            {"document": {"id": "docA"}, "question": "Q1?", "answer": ["a1"]},
            {"document": {"id": "docA"}, "question": "Q2?", "answer": ["a2"]},
        ],
        ["0", "1"],
    ),
}


@pytest.mark.parametrize("dataset_name", sorted(FIXTURES))
def test_loader_identity_matches_judge_dataset_sample_id(dataset_name, tmp_path):
    rows, expected = FIXTURES[dataset_name]
    (tmp_path / f"{dataset_name}.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / f"{dataset_name}_corpus.json").write_text("[]", encoding="utf-8")

    loader_ids = [
        record["id"]
        for record in DatasetLoader(dataset_name, dataset_dir=str(tmp_path)).get_question_records()
    ]
    judge_ids = [
        sample.dataset_sample_id
        for sample in GroundTruthRepository(Path(tmp_path)).load(dataset_name)
    ]

    # 断言字面量：防止两侧被同一个改动一起改坏却仍然「一致」。
    assert loader_ids == expected
    assert loader_ids == judge_ids


def test_timestamped_copy_keeps_base_dataset_identity(tmp_path):
    """时间戳副本必须与原数据集同口径：judge 侧 resolver 会剥后缀，loader 也必须剥。"""
    rows, expected = FIXTURES["narrativeqa"]
    dataset_name = "narrativeqa_20260101_120000"
    (tmp_path / f"{dataset_name}.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / f"{dataset_name}_corpus.json").write_text("[]", encoding="utf-8")

    loader_ids = [
        record["id"]
        for record in DatasetLoader(dataset_name, dataset_dir=str(tmp_path)).get_question_records()
    ]
    judge_ids = [
        sample.dataset_sample_id
        for sample in GroundTruthRepository(Path(tmp_path)).load(dataset_name)
    ]

    assert loader_ids == expected
    assert loader_ids == judge_ids
