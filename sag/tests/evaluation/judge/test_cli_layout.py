"""Tests for canonical Judge CLI layout and overwrite controls."""

import os
import time
from argparse import Namespace

import pytest

from pipeline.evaluation.judge.artifacts import ArtifactLayoutResolver, JudgeArtifactStore
from pipeline.evaluation.judge.errors import JudgeConfigurationError
from scripts.run_llm_judge import (
    _default_external_output_root,
    _new_run_id,
    _normalize_auto_args,
    _print_metric_scores,
    _resolve_artifact_root,
    _resolve_convert_artifact_root,
    _resolve_evaluation_data_file,
    _resolve_resume_judge_model,
    _resolve_source_run_id,
    _route_metrics,
    _validate_run_options,
    build_parser,
    cmd_auto,
)


def test_print_metric_scores_reads_requested_metrics_from_summary(tmp_path, capsys):
    layout = ArtifactLayoutResolver.judge_run(tmp_path, "judge-model", "run1")
    layout.judge_run_dir.mkdir(parents=True)
    layout.summary_file.write_text(
        '{"average_scores":{"qa_em":0.5,"qa_f1":0.75},'
        '"indexing_metrics":{"node_count":12},'
        '"metric_valid_counts":{"qa_em":4,"qa_f1":4}}',
        encoding="utf-8",
    )
    JudgeArtifactStore().write_run_manifest(
        layout, {"judge_run_id": "run1", "judge_model": "judge-model"}
    )

    _print_metric_scores(
        artifact_root=tmp_path,
        judge_run_id="run1",
        metrics=("qa_f1", "qa_em", "node_count"),
        project=None,
        dataset=None,
        source_run_id=None,
    )

    output = capsys.readouterr().out
    assert "qa_f1: 0.7500 (valid samples: 4)" in output
    assert "qa_em: 0.5000 (valid samples: 4)" in output
    assert "node_count: 12.0000" in output


def test_resume_model_is_resolved_from_existing_run(tmp_path):
    root = tmp_path / "run"
    layout = ArtifactLayoutResolver.judge_run(root, "qwen-model", "run1")
    JudgeArtifactStore().write_run_manifest(
        layout,
        {"judge_run_id": "run1", "judge_model": "qwen-model"},
    )
    assert _resolve_resume_judge_model(root, "run1") == "qwen-model"


def test_evaluate_defaults_are_safe_and_reproducible():
    args = build_parser().parse_args(["evaluate", "--data-file", "/tmp/p.json"])
    assert args.max_concurrent == 3
    assert args.context_top_k == 5
    assert args.force is False
    assert args.force_metrics is False


def test_unified_metric_option_is_selectable_with_auto_routing():
    evaluate = build_parser().parse_args(
        [
            "evaluate",
            "--data-file",
            "/tmp/p.json",
            "--metrics",
            "qa_em,qa_f1,rouge_score,context_relevancy",
        ]
    )
    assert evaluate.metrics == "qa_em,qa_f1,rouge_score,context_relevancy"
    assert evaluate.only_metrics is None

    # The unified list is routed to the correct phase by metric name.
    gen, ret, run_indexing = _route_metrics(("qa_em", "qa_f1", "rouge_score", "context_relevancy"))
    assert gen == ("qa_em", "qa_f1", "rouge_score")
    assert ret == ("context_relevancy",)
    assert run_indexing is False

    # The ``indexing`` sentinel toggles the indexing phase.
    gen, ret, run_indexing = _route_metrics(("qa_em", "indexing"))
    assert gen == ("qa_em",)
    assert ret == ()
    assert run_indexing is True


def test_legacy_only_metrics_remains_available_for_evaluate_command():
    args = build_parser().parse_args(
        ["evaluate", "--data-file", "/tmp/p.json", "--only-metrics", "qa_em"]
    )
    assert args.only_metrics == "qa_em"
    assert args.metrics is None


def test_route_metrics_splits_mixed_selection_preserving_order():
    gen, ret, run_indexing = _route_metrics(
        ("evidence_recall", "qa_em", "context_relevancy", "rouge_score")
    )
    assert gen == ("qa_em", "rouge_score")
    assert ret == ("evidence_recall", "context_relevancy")
    assert run_indexing is False


def test_route_metrics_rejects_unknown_metric():
    from pipeline.evaluation.judge.errors import JudgeConfigurationError

    with pytest.raises(JudgeConfigurationError, match="em"):
        _route_metrics(("qa_em", "em"))
    with pytest.raises(JudgeConfigurationError, match="qa_em"):
        _route_metrics(("em",))


def test_route_metrics_splits_correctly():
    """Unified --metrics routes by metric name into (gen, ret, run_indexing)."""
    assert _route_metrics(("qa_em", "context_relevancy")) == (
        ("qa_em",),
        ("context_relevancy",),
        False,
    )
    assert _route_metrics(("indexing",)) == ((), (), True)
    with pytest.raises(JudgeConfigurationError):
        _route_metrics(("invalid_metric",))


def test_convert_uses_project_output_root_by_default():
    args = build_parser().parse_args(["convert", "--project", "hyperrag"])
    assert args.input_root is None
    assert _default_external_output_root("hyperrag").name == "outputs"
    assert _default_external_output_root("hyperrag").parent.name == "hyperrag"


def test_convert_accepts_custom_input_root():
    args = build_parser().parse_args(
        ["convert", "--project", "lightrag", "--input-root", "/custom/results"]
    )
    assert args.input_root == "/custom/results"


def test_evaluation_can_resolve_predictions_from_artifact_root(tmp_path):
    root = tmp_path / "run"
    predictions = root / "evaluation" / "predictions" / "predictions_narrativeqa.json"
    predictions.parent.mkdir(parents=True)
    predictions.write_text("{}")
    args = build_parser().parse_args(
        [
            "evaluate",
            "--artifact-run-root",
            str(root),
            "--dataset",
            "narrativeqa",
        ]
    )
    assert _resolve_evaluation_data_file(args) == predictions.resolve()


def test_evaluation_data_file_remains_an_explicit_override():
    args = build_parser().parse_args(
        [
            "evaluate",
            "--project",
            "hyperrag",
            "--dataset",
            "narrativeqa",
        ]
    )
    assert args.data_file is None


def test_evaluation_resolves_three_layer_mirror(tmp_path):
    root = tmp_path / "run"
    predictions = (
        root
        / "evaluation"
        / "hyperrag"
        / "narrativeqa"
        / "run-001"
        / "predictions"
        / "predictions_narrativeqa.json"
    )
    predictions.parent.mkdir(parents=True)
    predictions.write_text("{}")
    args = build_parser().parse_args(
        [
            "evaluate",
            "--artifact-run-root",
            str(root),
            "--project",
            "hyperrag",
            "--dataset",
            "narrativeqa",
            "--source-run-id",
            "run-001",
        ]
    )
    assert _resolve_evaluation_data_file(args) == predictions.resolve()


@pytest.mark.parametrize("command", ["evaluate"])
def test_top_k_alias_selects_the_global_prediction_limit(command):
    args = build_parser().parse_args([command, "--data-file", "/tmp/p.json", "--top-k", "7"])
    assert args.num_samples == 7


@pytest.mark.parametrize(
    "argv",
    [
        ["evaluate", "--data-file", "/tmp/p.json", "--output-file", "x.json"],
        ["convert", "--project", "hipporag2", "--input-root", "/tmp", "--out-root", "/x"],
    ],
)
def test_removed_deprecated_output_options_are_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_resume_and_new_run_id_are_mutually_exclusive():
    args = Namespace(
        resume_run_id="old",
        judge_run_id="new",
        force=False,
        force_metrics=False,
    )
    with pytest.raises(JudgeConfigurationError, match="mutually exclusive"):
        _validate_run_options(args)


def test_force_metrics_requires_resume_and_explicit_metrics():
    with pytest.raises(JudgeConfigurationError, match="requires --resume-run-id"):
        _validate_run_options(
            Namespace(
                resume_run_id=None,
                judge_run_id=None,
                force=False,
                force_metrics=True,
                only_metrics="qa_em",
            )
        )
    with pytest.raises(JudgeConfigurationError, match="requires --only-metrics"):
        _validate_run_options(
            Namespace(
                resume_run_id="old",
                judge_run_id=None,
                force=False,
                force_metrics=True,
                only_metrics=None,
            )
        )


def test_force_metrics_and_force_kind_are_mutually_exclusive():
    with pytest.raises(JudgeConfigurationError, match="mutually exclusive"):
        _validate_run_options(
            Namespace(
                resume_run_id="old",
                judge_run_id=None,
                force=True,
                force_metrics=True,
                only_metrics="qa_em",
            )
        )


def test_force_metrics_rejects_indexing_only_selection():
    with pytest.raises(JudgeConfigurationError, match="at least one real metric"):
        _validate_run_options(
            Namespace(
                resume_run_id="old",
                judge_run_id=None,
                force=False,
                force_metrics=True,
                only_metrics="indexing",
            )
        )


def test_artifact_root_is_inferred_from_canonical_predictions(tmp_path):
    root = tmp_path / "run"
    predictions = root / "evaluation" / "predictions" / "predictions_test.json"
    args = Namespace(artifact_run_root=None, data_file=str(predictions))
    assert _resolve_artifact_root(args) == root.resolve()


def test_noncanonical_predictions_require_explicit_artifact_root(tmp_path):
    args = Namespace(artifact_run_root=None, data_file=str(tmp_path / "predictions.json"))
    with pytest.raises(JudgeConfigurationError, match="Cannot determine"):
        _resolve_artifact_root(args)


def test_convert_predictions_dir_can_infer_artifact_root(tmp_path):
    root = tmp_path / "run"
    predictions_dir = root / "evaluation" / "predictions"
    args = Namespace(artifact_run_root=None, predictions_dir=str(predictions_dir))
    assert _resolve_convert_artifact_root(args, tmp_path) == root.resolve()


def test_convert_predictions_dir_can_infer_three_layer_artifact_root(tmp_path):
    root = tmp_path / "run"
    predictions_dir = root / "evaluation" / "hyperrag" / "narrativeqa" / "run-001" / "predictions"
    args = Namespace(artifact_run_root=None, predictions_dir=str(predictions_dir))
    assert _resolve_convert_artifact_root(args, tmp_path) == root.resolve()


def test_artifact_root_inferred_from_three_layer_predictions(tmp_path):
    root = tmp_path / "run"
    predictions = (
        root
        / "evaluation"
        / "hyperrag"
        / "narrativeqa"
        / "run-001"
        / "predictions"
        / "predictions_narrativeqa.json"
    )
    predictions.parent.mkdir(parents=True)
    predictions.write_text("[]")
    args = Namespace(artifact_run_root=None, data_file=str(predictions))
    assert _resolve_artifact_root(args) == root.resolve()


def test_generated_run_ids_do_not_collide():
    assert _new_run_id() != _new_run_id()


def test_source_run_id_resolves_latest_by_mtime_not_name(tmp_path):
    layer = tmp_path / "evaluation" / "hyperrag" / "test"
    for name in ("zebra", "apple"):
        pred = layer / name / "predictions" / "predictions_test.json"
        pred.parent.mkdir(parents=True)
        pred.write_text("[]")
    now = time.time()
    os.utime(layer / "zebra", (now - 100, now - 100))
    os.utime(layer / "apple", (now, now))

    assert _resolve_source_run_id(tmp_path, "hyperrag", "test", None) == "apple"


# ---------------------------------------------------------------------------
# Auto mode (no subcommand → convert then all)
# ---------------------------------------------------------------------------


def test_auto_mode_parses_without_subcommand():
    args = build_parser().parse_args(
        [
            "--project",
            "lightrag",
            "--dataset",
            "musique",
            "--input-root",
            "/custom/in",
            "--output-root",
            "/custom/out",
        ]
    )
    assert args.command is None
    assert args.project == "lightrag"
    assert args.dataset == "musique"
    assert args.input_root == "/custom/in"
    assert args.output_root == "/custom/out"


def test_auto_mode_respects_subcommand_parsing_still():
    # Existing subcommand-first usage is unaffected by the new top-level options.
    args = build_parser().parse_args(["evaluate", "--data-file", "/tmp/p.json"])
    assert args.command == "evaluate"
    assert args.data_file == "/tmp/p.json"


def test_normalize_output_root_maps_to_artifact_run_root(tmp_path):
    args = build_parser().parse_args(
        ["--project", "lightrag", "--dataset", "musique", "--output-root", str(tmp_path)]
    )
    _normalize_auto_args(args)
    assert args.artifact_run_root == str(tmp_path)
    assert args.datasets == ["musique"]


def test_normalize_folds_single_dataset_and_defaults():
    args = build_parser().parse_args(["--project", "lightrag", "--dataset", "musique"])
    _normalize_auto_args(args)
    assert args.datasets == ["musique"]
    assert args.artifact_run_root is None  # default: repository root
    assert args.retry_failed is False


def test_normalize_missing_project_raises():
    args = build_parser().parse_args(["--dataset", "musique"])
    with pytest.raises(JudgeConfigurationError, match="--project"):
        _normalize_auto_args(args)


def test_normalize_missing_dataset_raises():
    args = build_parser().parse_args(["--project", "lightrag"])
    with pytest.raises(JudgeConfigurationError, match="dataset"):
        _normalize_auto_args(args)


def test_normalize_output_and_artifact_root_mutually_exclusive():
    args = build_parser().parse_args(
        ["--project", "p", "--dataset", "d", "--output-root", "/o", "--artifact-run-root", "/a"]
    )
    with pytest.raises(JudgeConfigurationError, match="mutually exclusive"):
        _normalize_auto_args(args)


def test_normalize_rejects_resume_and_fixed_run_id():
    for extra in (["--resume-run-id", "old"], ["--judge-run-id", "new"]):
        args = build_parser().parse_args(["--project", "p", "--dataset", "d", *extra])
        with pytest.raises(JudgeConfigurationError, match="auto mode"):
            _normalize_auto_args(args)


def test_cmd_auto_runs_convert_then_evaluate(tmp_path, monkeypatch):
    """Auto mode must chain convert → evaluate, one evaluate per dataset, with
    roots normalised so evaluation reads the predictions the convert just wrote."""
    from scripts import run_llm_judge as mod

    calls: list[tuple[str, str | None, str | None]] = []

    def fake_convert(args):
        calls.append(
            ("convert", args.datasets[0] if args.datasets else None, args.artifact_run_root)
        )
        return 0

    def fake_evaluate(args):
        calls.append(("evaluate", args.dataset, args.artifact_run_root))
        return 0

    monkeypatch.setattr(mod, "cmd_convert", fake_convert)
    monkeypatch.setattr(mod, "cmd_evaluate", fake_evaluate)

    out = tmp_path / "out"
    args = build_parser().parse_args(
        ["--project", "lightrag", "--datasets", "ds1", "ds2", "--output-root", str(out)]
    )
    rc = cmd_auto(args)

    assert rc == 0
    assert calls == [
        ("convert", "ds1", str(out)),
        ("evaluate", "ds1", str(out)),
        ("evaluate", "ds2", str(out)),
    ]


def test_cmd_auto_dry_run_skips_evaluate(tmp_path, monkeypatch):
    from scripts import run_llm_judge as mod

    calls: list[str] = []

    def fake_convert(args):
        calls.append("convert")
        return 0

    monkeypatch.setattr(mod, "cmd_convert", fake_convert)
    monkeypatch.setattr(mod, "cmd_evaluate", lambda args: calls.append("evaluate") or 0)

    args = build_parser().parse_args(
        ["--project", "p", "--dataset", "d", "--output-root", str(tmp_path), "--dry-run"]
    )
    rc = cmd_auto(args)

    assert rc == 0
    assert calls == ["convert"]


def test_cmd_auto_propagates_convert_failure(tmp_path, monkeypatch):
    from scripts import run_llm_judge as mod

    monkeypatch.setattr(mod, "cmd_convert", lambda args: 3)
    monkeypatch.setattr(
        mod, "cmd_evaluate", lambda args: (_ for _ in ()).throw(AssertionError("no evaluate"))
    )

    args = build_parser().parse_args(
        ["--project", "p", "--dataset", "d", "--output-root", str(tmp_path)]
    )
    assert cmd_auto(args) == 3


def test_cmd_auto_warns_on_multi_dataset_indexing(tmp_path, monkeypatch, caplog):
    """Auto mode with multiple datasets and an `indexing` token must warn that
    a single --base-path is recomputed once per dataset."""
    from scripts import run_llm_judge as mod

    monkeypatch.setattr(mod, "cmd_convert", lambda args: 0)
    monkeypatch.setattr(mod, "cmd_evaluate", lambda args: 0)

    args = build_parser().parse_args(
        [
            "--project",
            "p",
            "--datasets",
            "ds1",
            "ds2",
            "--metrics",
            "qa_em,indexing",
            "--framework",
            "graphml",
            "--base-path",
            str(tmp_path),
            "--output-root",
            str(tmp_path),
        ]
    )
    rc = cmd_auto(args)
    assert rc == 0
    assert any("runs indexing once per dataset" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "argv",
    [
        # Single dataset: indexing runs once, no warning needed.
        [
            "--project",
            "p",
            "--dataset",
            "d1",
            "--metrics",
            "qa_em,indexing",
            "--framework",
            "graphml",
            "--base-path",
            "/x",
            "--output-root",
        ],
        # Multiple datasets but no indexing token: nothing to warn about.
        [
            "--project",
            "p",
            "--datasets",
            "ds1",
            "ds2",
            "--metrics",
            "qa_em",
            "--output-root",
        ],
    ],
)
def test_cmd_auto_no_warning_when_indexing_unambiguous(argv, tmp_path, monkeypatch, caplog):
    """Negative control for the multi-dataset indexing warning."""
    from scripts import run_llm_judge as mod

    monkeypatch.setattr(mod, "cmd_convert", lambda args: 0)
    monkeypatch.setattr(mod, "cmd_evaluate", lambda args: 0)

    args = build_parser().parse_args([*argv, str(tmp_path)])
    assert cmd_auto(args) == 0
    assert not any("runs indexing once per dataset" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Indexing sentinel integration: --metrics indexing must not crash the
# evaluate core, and missing --framework/--base-path must fail before any LLM
# call is spent.
# ---------------------------------------------------------------------------


def _make_predictions(tmp_path):
    """Create a canonical three-layer predictions file and return its path."""
    pred = (
        tmp_path
        / "evaluation"
        / "sag"
        / "narrativeqa"
        / "run-001"
        / "predictions"
        / "predictions_narrativeqa.json"
    )
    pred.parent.mkdir(parents=True)
    pred.write_text("[]", encoding="utf-8")
    return pred


def test_indexing_metrics_missing_framework_fails_before_llm(tmp_path, monkeypatch):
    """`--metrics indexing` without --framework/--base-path must raise before
    the LLM client is created (regression for the old fail-after-work bug)."""
    from scripts import run_llm_judge as mod

    pred = _make_predictions(tmp_path)
    llm_created = []

    async def fake_create_llm_client(scenario):
        llm_created.append(scenario)
        raise AssertionError("LLM client must not be created for a config error")

    monkeypatch.setattr(mod, "create_llm_client", fake_create_llm_client)

    args = build_parser().parse_args(
        [
            "evaluate",
            "--data-file",
            str(pred),
            "--metrics",
            "indexing",
            "--framework",
            "graphml",  # framework present, base-path absent
        ]
    )
    with pytest.raises(JudgeConfigurationError, match="framework and --base-path"):
        import asyncio

        asyncio.run(mod._run_evaluate_core(args, requested_metrics=("indexing",)))
    assert llm_created == []


def test_indexing_only_routes_to_no_gen_no_ret(tmp_path, monkeypatch):
    """`--metrics indexing` alone yields empty gen/ret and run_indexing=True."""
    from scripts import run_llm_judge as mod

    pred = _make_predictions(tmp_path)

    async def fake_create_llm_client(scenario):
        raise AssertionError("no LLM needed for indexing-only run")

    monkeypatch.setattr(mod, "create_llm_client", fake_create_llm_client)

    class _FakeManifest:
        judge_run_id = "run-001"
        judge_model = "indexing"

    class _FakeService:
        def __init__(self, *a, **k):
            self.indexing_calls = []

        def run_indexing(self, **kwargs):
            self.indexing_calls.append(kwargs)
            return {"metrics": {"num_nodes": 12.0}}

    calls = {}

    def fake_service_cls(**kwargs):
        calls["service"] = _FakeService()
        return calls["service"]

    monkeypatch.setattr(mod, "JudgeEvaluationService", fake_service_cls)
    # route: indexing-only → run_indexing True, gen/ret empty; needs_llm False.
    gen, ret, run_indexing = mod._route_metrics(("indexing",))
    assert gen == () and ret == () and run_indexing is True

    # Drive the core with framework/base-path present and assert run_indexing
    # dispatches to the service without touching run_generation/run_retrieval.
    args = build_parser().parse_args(
        [
            "evaluate",
            "--data-file",
            str(pred),
            "--metrics",
            "indexing",
            "--framework",
            "graphml",
            "--base-path",
            str(tmp_path),
        ]
    )
    import asyncio

    rc = asyncio.run(mod._run_evaluate_core(args, requested_metrics=("indexing",)))
    assert rc == 0
    svc = calls["service"]
    assert len(svc.indexing_calls) == 1
    assert svc.indexing_calls[0]["framework"] == "graphml"
    # Standalone indexing mirrors the `indexing` subcommand's model dir and
    # starts a fresh run (no resume).
    assert svc.indexing_calls[0]["judge_model"] == "indexing"
    assert svc.indexing_calls[0]["resume_run_id"] is None
    # No gen/ret results may be written for an indexing-only selection.
    gen_written = list(
        (tmp_path / "evaluation" / "sag" / "narrativeqa" / "run-001").rglob("*generation*")
    )
    ret_written = list(
        (tmp_path / "evaluation" / "sag" / "narrativeqa" / "run-001").rglob("*retrieval*")
    )
    assert gen_written == []
    assert ret_written == []


def test_indexing_only_resume_appends_to_existing_run(tmp_path, monkeypatch):
    """`evaluate --metrics indexing --resume-run-id X` must append to run X in
    its original model dir, not silently start a fresh standalone run."""
    from pipeline.evaluation.judge.artifacts import ArtifactLayoutResolver, JudgeArtifactStore
    from scripts import run_llm_judge as mod

    pred = _make_predictions(tmp_path)
    # Persist an existing run under a non-default model dir to resume into,
    # in the same three-layer lineage the predictions file implies.
    layout = ArtifactLayoutResolver.judge_run(
        tmp_path,
        "qwen-model",
        "run-001",
        project="sag",
        dataset="narrativeqa",
        source_run_id="run-001",
    )
    JudgeArtifactStore().write_run_manifest(
        layout, {"judge_run_id": "run-001", "judge_model": "qwen-model"}
    )

    async def fake_create_llm_client(scenario):
        raise AssertionError("no LLM needed for indexing-only resume")

    monkeypatch.setattr(mod, "create_llm_client", fake_create_llm_client)

    class _FakeService:
        def __init__(self, *a, **k):
            self.indexing_calls = []

        def run_indexing(self, **kwargs):
            self.indexing_calls.append(kwargs)
            return {"metrics": {"num_edges": 34.0}}

    calls = {}

    def fake_service_cls(**kwargs):
        calls["service"] = _FakeService()
        return calls["service"]

    monkeypatch.setattr(mod, "JudgeEvaluationService", fake_service_cls)

    args = build_parser().parse_args(
        [
            "evaluate",
            "--data-file",
            str(pred),
            "--metrics",
            "indexing",
            "--framework",
            "graphml",
            "--base-path",
            str(tmp_path),
            "--resume-run-id",
            "run-001",
        ]
    )
    import asyncio

    rc = asyncio.run(mod._run_evaluate_core(args, requested_metrics=("indexing",)))
    assert rc == 0
    svc = calls["service"]
    assert len(svc.indexing_calls) == 1
    # Resume intent is preserved: same run id, original model dir, resume set.
    assert svc.indexing_calls[0]["run_id"] == "run-001"
    assert svc.indexing_calls[0]["judge_model"] == "qwen-model"
    assert svc.indexing_calls[0]["resume_run_id"] == "run-001"
