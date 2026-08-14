import json

from pipeline.evaluation.judge.adapters.graphrag import graphrag_adapter


def test_graphrag_adapter_joins_one_based_eval_to_zero_based_retrieval(tmp_path):
    run = tmp_path
    (run / "evaluation").mkdir()
    (run / "response").mkdir()
    (run / "evaluation" / "emf1_graphrag_test.json").write_text(
        json.dumps(
            {
                "per_example": [
                    {
                        "question_index": 1,
                        "question": "question A",
                        "predicted_answer": "answer A",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (run / "response" / "graphrag_test_result.json").write_text(
        json.dumps(
            [
                {
                    "question_index": 0,
                    "question": "question A",
                    "retrieved_docs": ["correct context"],
                }
            ]
        ),
        encoding="utf-8",
    )

    rows = graphrag_adapter(str(run), "test")

    assert rows[0]["context"] == "correct context"
