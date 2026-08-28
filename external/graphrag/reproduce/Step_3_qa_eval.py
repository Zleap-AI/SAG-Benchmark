#!/usr/bin/env python3
"""Step 3 — QA 评估（完整图上下文版）。

复用仓库根 scripts/run_qa_benchmark.py（SAG 同款 prompt / 答案抽取 / EM/F1，口径零漂移）。
输入：Step_2 产物 graphrag_<ds>_result.json；输出：outputs/<ds>/qa/qa_results.json。
"""

import argparse
import json
import sys
import time
from pathlib import Path

from _common import (
    OUTPUT_ROOT,
    REPOSITORY_ROOT,
    load_env,
    log,
    pin_venv,
    run_streamed,
)


def run_qa(ds, input_file, out_dir, qa_top_k, max_concurrency, limit, bench_size):
    """调仓库根 scripts/run_qa_benchmark.py，返回 returncode。

    该脚本每 bench_size 题打印一次 `QA 进度: i/total`，run_streamed 逐行 tee 到 stdout。
    """
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--env-file",
        ".env",
        "python",
        "scripts/run_qa_benchmark.py",
        "--dataset-name",
        ds,
        "--input",
        str(input_file),
        "--qa-top-k",
        str(qa_top_k),
        "--max-concurrency",
        str(max_concurrency),
        "--bench-size",
        str(bench_size),
        "--output-dir",
        str(out_dir),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    return run_streamed(cmd, cwd=str(REPOSITORY_ROOT))


def print_summary(out_dir: Path) -> None:
    """打印 EM/F1 摘要（与统一规格 E 一致）。"""
    print("==> [摘要] 指标")
    p = out_dir / "qa" / "qa_results.json"
    if not p.exists():
        return
    try:
        m = json.loads(p.read_text(encoding="utf-8")).get("metrics", {})
        em = m.get("exact_match", float("nan"))
        f1 = m.get("f1", float("nan"))
        print(f"  [QA] ExactMatch={em:.4f}  F1={f1:.4f}")
    except (OSError, ValueError) as e:
        print(f"  [QA] 读取失败: {e!r}")


def main():
    load_env()
    pin_venv()

    ap = argparse.ArgumentParser(description="Step 3: QA 评估（EM/F1）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--qa-top-k", type=int, default=5)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0=全部")
    ap.add_argument("--bench-size", type=int, default=20, help="每 N 题打印一次 QA 进度")
    args = ap.parse_args()

    ds = args.dataset
    out_dir = OUTPUT_ROOT / ds

    resp = out_dir / "response"
    input_file = resp / f"graphrag_{ds}_result.json"
    if not input_file.exists():
        log(f"缺少 {input_file}，请先运行 Step_2_retrieve.py")
        sys.exit(1)

    t0 = time.perf_counter()
    log(f"[qa] {ds} 开始")
    rc = run_qa(
        ds,
        input_file,
        out_dir / "qa",
        args.qa_top_k,
        args.max_concurrency,
        args.limit,
        args.bench_size,
    )
    if rc != 0:
        log(f"QA 失败 rc={rc}")
        sys.exit(rc)
    log(f"[qa] {ds} 完成，用时 {time.perf_counter() - t0:.0f}s")

    print_summary(out_dir)
    log(f"[qa_eval] {ds}: 完成 → {out_dir}")


if __name__ == "__main__":
    main()
