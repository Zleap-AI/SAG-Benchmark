#!/usr/bin/env python
"""
把 HippoRAG2 的数据集语料 + QA 转成 GraphRAG 需要的 csv + questions.jsonl。

支持数据集: musique / 2wikimultihopqa / hotpotqa / test_hotpotqa
产物写入 caches/<ds>/（graphrag 引擎工作区）：settings.yaml + input/<ds>.csv + questions.jsonl

用法:
  # 冒烟（50篇+20题）
  python -m graphrag_benchmark.dataset --dataset musique --limit-docs 50 --limit-q 20
  # 全量
  python -m graphrag_benchmark.dataset --dataset 2wikimultihopqa
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../external/graphrag
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]  # 仓库根（sag-benchmark-external）
SETTINGS_TEMPLATE = PROJECT_ROOT / "reproduce" / "settings.yaml"

# 跨仓库依赖：load_utils 位于仓库根 pipeline/evaluation/utils，不在本项目 uv 环境里。
# 此 path hack 依赖仓库根结构，待仓库级统一配置后清理（与 reproduce/Step_0 同款）。
EVALUATION_UTILS = REPOSITORY_ROOT / "pipeline" / "evaluation" / "utils"
if str(EVALUATION_UTILS) not in sys.path:
    sys.path.insert(0, str(EVALUATION_UTILS))

from load_utils import DatasetLoader
from sample_identity import index_unique_records


def _fmt_dur(seconds: float) -> str:
    """把秒格式化成 08m12s / 1h23m45s（与 reproduce/progress.py 口径一致）。"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def setup_root(root: Path):
    """确保 graphrag 引擎工作区有 input/ 与 settings.yaml。

    settings.yaml 从 reproduce/settings.yaml 模板复制（每次覆盖，模板是唯一真相源）。
    模板里的 ${VAR} 占位符由 graphrag_benchmark.config.apply_env() 回填 env 后，
    内核 load_config() 展开。prompts/ 目录不需要：settings.yaml 未设 prompt 字段时
    内核回退到内置提示词。
    """
    (root / "input").mkdir(parents=True, exist_ok=True)
    if not SETTINGS_TEMPLATE.exists():
        raise FileNotFoundError(f"缺少 settings 模板: {SETTINGS_TEMPLATE}")
    shutil.copy2(SETTINGS_TEMPLATE, root / "settings.yaml")
    print(f"[setup] settings.yaml → {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--dataset-root",
        default=str(REPOSITORY_ROOT / "dataset"),
        help="原始 QA/corpus 目录；默认仓库根 dataset/",
    )
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--limit-q", type=int, default=0)
    args = ap.parse_args()

    ds = args.dataset
    loader = DatasetLoader(ds, args.dataset_root)
    loader.validate_source_pair()
    root = PROJECT_ROOT / "caches" / ds

    t_start = time.perf_counter()

    # 确保 root 有完整配置
    setup_root(root)

    # clean old input
    input_dir = root / "input"
    for old in list(input_dir.glob("*.txt")) + list(input_dir.glob("*.csv")):
        old.unlink()

    # ---- 语料 → csv ----
    corpus = loader.load_corpus()
    if args.limit_docs:
        corpus = corpus[: args.limit_docs]
    total_docs = len(corpus)
    csv_path = input_dir / f"{ds}.csv"
    _last_log = 0.0  # 固定节流：每 2 秒最多一行，避免万级文档刷屏
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "title", "text"])
        for i, doc in enumerate(corpus):
            w.writerow(
                [
                    i,
                    doc.get("title", ""),
                    doc.get("text", doc.get("paragraph_text", "")),
                ]
            )
            now = time.perf_counter()
            done = i + 1
            if now - _last_log >= 2.0 or done == total_docs:
                _last_log = now
                elapsed = now - t_start
                eta = elapsed * (total_docs - done) / done if done else None
                eta_s = f"  预计剩余 {_fmt_dur(eta)}" if eta is not None else ""
                print(
                    f"[corpus] 转换 {done}/{total_docs}  已用 {_fmt_dur(elapsed)}{eta_s}",
                    flush=True,
                )
    print(f"[corpus] {total_docs} 行 → {csv_path}")

    # ---- QA → questions.jsonl ----
    qa = loader.get_question_records()
    if args.limit_q:
        qa = qa[: args.limit_q]
    index_unique_records(qa, "id", "GraphRAG questions")

    out_q = root / "questions.jsonl"
    with open(out_q, "w", encoding="utf-8") as f:
        for s in qa:
            answers = s["gold_answers"]
            rec = {
                "id": s["id"],
                "question": s["question"],
                "answer": answers[0] if answers else "",
                "gold_aliases": answers,
                "gold_docs": s["gold_docs"],
                "gold_ref": s["gold_ref"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[QA] {len(qa)} 题 → {out_q}")

    t_elapsed = time.perf_counter() - t_start
    print(f"[prepare] {ds} done ({t_elapsed:.1f}s)")


if __name__ == "__main__":
    main()
