#!/usr/bin/env python3
"""
Step 0: 数据集准备 —— 语料 → CSV + QA → questions.jsonl。

通过 subprocess 复用 graphrag_benchmark/dataset.py（不改动原脚本的职责边界）。
完成后将 contexts/questions 写入 caches/{ds}/ 产物汇聚层（hyperrag 风格）。

用法:
  uv run python reproduce/Step_0_prepare_dataset.py --dataset musique
  uv run python reproduce/Step_0_prepare_dataset.py --dataset musique --smoke
  uv run python reproduce/Step_0_prepare_dataset.py --dataset 2wikimultihopqa --limit-docs 100 --limit-q 50
"""

import argparse
import json
import sys

from _common import (
    CACHE_ROOT,
    DATASETS,
    PROJECT_ROOT,
    REPOSITORY_ROOT,
    load_env,
    log,
    pin_venv,
    run_streamed,
    workspace_dir,
)

EVALUATION_UTILS = REPOSITORY_ROOT / "pipeline" / "evaluation" / "utils"
if str(EVALUATION_UTILS) not in sys.path:
    sys.path.insert(0, str(EVALUATION_UTILS))

from load_utils import DatasetLoader
from reproduce_dataset import ReproduceDatasetExporter


def main():
    load_env()
    pin_venv()

    ap = argparse.ArgumentParser(description="Step 0: 数据集准备")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument(
        "--dataset-root",
        default=str(REPOSITORY_ROOT / "dataset"),
        help="原始 QA/corpus 目录；默认仓库根 dataset/",
    )
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：50 文档 + 20 题")
    ap.add_argument("--limit-docs", type=int, default=0, help="文档数上限（0=全部）")
    ap.add_argument("--limit-q", type=int, default=0, help="问题数上限（0=全部）")
    ap.add_argument("--force", action="store_true", help="统一接口，prepare 阶段每次都会重建 input")
    args = ap.parse_args()

    ds = args.dataset
    root = workspace_dir(ds)

    limit_docs = args.limit_docs
    limit_q = args.limit_q
    if args.smoke:
        if not args.limit_docs:
            limit_docs = 50
        if not args.limit_q:
            limit_q = 20
        log(f"smoke 模式: limit-docs={limit_docs}, limit-q={limit_q}")

    cmd = [sys.executable, "-m", "graphrag_benchmark.dataset", "--dataset", ds]
    if args.dataset_root:
        cmd.extend(["--dataset-root", args.dataset_root])
    if limit_docs:
        cmd.extend(["--limit-docs", str(limit_docs)])
    if limit_q:
        cmd.extend(["--limit-q", str(limit_q)])

    log(f"Step_0: {ds} 数据准备")
    rc = run_streamed(cmd, cwd=str(PROJECT_ROOT))
    if rc != 0:
        log(f"prepare 失败，rc={rc}")
        sys.exit(1)

    # 验证 graphrag 引擎工作区产物
    input_csv = root / "input" / f"{ds}.csv"
    questions_jsonl = root / "questions.jsonl"
    settings_yaml = root / "settings.yaml"
    missing = [p.name for p in (input_csv, questions_jsonl, settings_yaml) if not p.exists()]
    if missing:
        log(f"产物缺失: {missing}")
        sys.exit(1)
    log(
        f"graphrag 产物: {input_csv.name} ({input_csv.stat().st_size} bytes) / "
        f"{questions_jsonl.name} ({questions_jsonl.stat().st_size} bytes) / {settings_yaml.name}"
    )

    # ── 写入 caches/<ds>/ 产物汇聚层（contexts/ + questions/）────────
    loader = DatasetLoader(ds, args.dataset_root)
    manifest = ReproduceDatasetExporter(loader, CACHE_ROOT, subdir=ds).export(
        limit_documents=limit_docs,
        limit_questions=limit_q,
    )
    manifest_path = CACHE_ROOT / ds / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log(
        f"caches 产物: documents={manifest['counts']['documents']} / "
        f"questions={manifest['counts']['questions']} / manifest={manifest_path}"
    )

    log("Step_0 完成")


if __name__ == "__main__":
    main()
