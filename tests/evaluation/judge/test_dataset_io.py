import json

from pipeline.evaluation.judge.dataset_io import load_evidence_map


def test_load_evidence_map_resolves_hotpot_supporting_facts(tmp_path):
    dataset = [
        {
            "question": "Q",
            "supporting_facts": [["Alpha", 1], ["Beta", 0]],
            "context": [
                ["Alpha", ["not evidence", "alpha evidence"]],
                ["Beta", ["beta evidence"]],
            ],
        }
    ]
    (tmp_path / "test_hotpotqa.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )

    evidence = load_evidence_map("test_hotpotqa", str(tmp_path))

    assert evidence == {0: ["alpha evidence", "beta evidence"]}


def test_load_evidence_map_prefers_explicit_evidence(tmp_path):
    dataset = [
        {
            "evidence": ["explicit"],
            "supporting_facts": [["Alpha", 0]],
            "context": [["Alpha", ["hotpot evidence"]]],
        }
    ]
    (tmp_path / "sample.json").write_text(
        json.dumps(dataset), encoding="utf-8"
    )

    assert load_evidence_map("sample", str(tmp_path)) == {0: ["explicit"]}
