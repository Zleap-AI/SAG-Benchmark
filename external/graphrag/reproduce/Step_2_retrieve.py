#!/usr/bin/env python3
"""Step 2 — GraphRAG 检索（local search，完整图上下文）。

调用源项目检索器 graphrag_benchmark/retrieval.py，产物写入 outputs/<ds>/response/：
  graphrag_<ds>_result.json        （完整图上下文版，SAG 格式）
成本：GRAPHRAG_COST_PHASE=query → caches/<ds>/cost.json。
"""

import argparse
import os
import sys
import time

from _common import (
    OUTPUT_ROOT,
    PROJECT_ROOT,
    ensure_overrides,
    load_env,
    log,
    pin_venv,
    run_streamed,
    workspace_dir,
)


def main():
    load_env()
    pin_venv()
    ensure_overrides()

    ap = argparse.ArgumentParser(description="Step 2: GraphRAG 检索（local search）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    args = ap.parse_args()

    ds = args.dataset
    root = workspace_dir(ds)

    questions = root / "questions.jsonl"
    if not questions.exists():
        log(f"缺少 {questions}，请先运行 Step_0_prepare_dataset.py")
        sys.exit(1)
    if not (root / "output" / "entities.parquet").exists():
        log("缺少 index 产物，请先运行 Step_1_build_index.py")
        sys.exit(1)

    resp_dir = OUTPUT_ROOT / ds / "response"
    resp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GRAPHRAG_COST_FILE"] = str(root / "cost.json")
    os.environ["GRAPHRAG_COST_PHASE"] = "query"

    cmd = [
        sys.executable,
        "-m",
        "graphrag_benchmark.retrieval",
        "--root",
        str(root),
        "--questions",
        str(questions),
        "--out",
        str(resp_dir / f"graphrag_{ds}_result.json"),
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    t0 = time.perf_counter()
    log(f"[retrieve] {ds}: 开始检索")
    rc = run_streamed(cmd, cwd=str(PROJECT_ROOT))
    if rc != 0:
        log(f"检索失败 rc={rc}")
        sys.exit(rc)

    log(f"[retrieve] {ds}: 完成，用时 {time.perf_counter() - t0:.0f}s")
    log(f"  result: {resp_dir / f'graphrag_{ds}_result.json'}")


if __name__ == "__main__":
    main()
