#!/usr/bin/env python3
"""Step 1 — GraphRAG 索引构建（进程内调用 graphrag.api.build_index）。

前置：Step_0_prepare_dataset.py（生成 caches/<ds>/input/<ds>.csv + settings.yaml）。
产物：caches/<ds>/output/（entities/relationships/communities/text_units parquet + lancedb）。
成本：GRAPHRAG_COST_PHASE=index → caches/<ds>/cost.json（含 index 耗时）。

不走 `graphrag index` 子进程：CLI 的 --logger 只接受 rich/print/none 枚举，无法传入
自定义 ProgressLogger；进程内调用才能打印 已完成/总数 + ETA。
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 项目根加入 sys.path，供 `import graphrag_benchmark`（与 hyperrag Step_1 同款）。
# uv 不支持 [tool.uv] pythonpath，此为标准做法。
sys.path.append(str(Path(__file__).resolve().parent.parent))

from _common import (
    ensure_overrides,
    file_lock,
    load_env,
    log,
    pin_venv,
    workspace_dir,
)
from graphrag_benchmark.config import apply_env
from progress import PipelineOutlineCallbacks, ProgressState, StepProgressLogger


def main():
    load_env()
    pin_venv()
    ensure_overrides()

    ap = argparse.ArgumentParser(description="Step 1: GraphRAG 索引构建")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--force", action="store_true", help="强制重建索引（默认跳过已有产物）")
    ap.add_argument("--skip-validation", action="store_true", help="跳过建索引前的端点连通性自检")
    ap.add_argument("--progress-interval", type=float, default=20.0, help="进度打印最小间隔（秒）")
    args = ap.parse_args()

    ds = args.dataset
    root = workspace_dir(ds)

    if not (root / "settings.yaml").exists():
        log(
            f"[index] 缺少 {root / 'settings.yaml'}，请先运行 Step_0_prepare_dataset.py --dataset {ds}"
        )
        sys.exit(1)
    if not (root / "input").exists() or not list((root / "input").glob("*.csv")):
        log(f"[index] 缺少 input/*.csv，请先运行 Step_0_prepare_dataset.py --dataset {ds}")
        sys.exit(1)

    with file_lock(root / "run.lock", "index"):
        if not args.force and (root / "output" / "entities.parquet").exists():
            log(f"[index] {ds}: index 产物已存在，跳过（用 --force 强制重建）")
            return

        cost_file = root / "cost.json"
        cost_file.unlink(missing_ok=True)
        # 断点续跑：存在 extract_graph 增量缓存时保留 cache/output，
        # graphrag 会按缓存跳过已完成的 chunk（幂等续跑，不重复调用 LLM）。
        has_incremental = (root / "cache" / "extract_graph").is_dir() and any(
            (root / "cache" / "extract_graph").iterdir()
        )
        if args.force or not has_incremental:
            shutil.rmtree(root / "cache", ignore_errors=True)
            shutil.rmtree(root / "output", ignore_errors=True)
            log(f"[index] {ds}: 清理旧缓存后全新构建")
        else:
            n_cached = len(list((root / "cache" / "extract_graph").iterdir()))
            log(
                f"[index] {ds}: 检测到 extract_graph 增量缓存（{n_cached} 个 chunk），保留并断点续跑"
            )

        os.environ["GRAPHRAG_COST_FILE"] = str(cost_file)
        os.environ["GRAPHRAG_COST_PHASE"] = "index"

        # 回填 settings.yaml 的 ${VAR} 占位符，必须在 load_config 之前
        apply_env()
        from graphrag.api import build_index
        from graphrag.config.enums import IndexingMethod
        from graphrag.config.load_config import load_config
        from graphrag.config.logging import enable_logging_with_config
        from graphrag.index.validate_config import validate_config_names

        config = load_config(root.resolve(), None, {})
        # graphrag 内核自带的 logs/indexing-engine.log（详细 DEBUG 落盘，非本项目添加）
        enable_logging_with_config(config, verbose=False)

        state = ProgressState()
        progress = StepProgressLogger(state, min_interval_s=args.progress_interval)
        if not args.skip_validation:
            # 上游自带的 LLM/embedding 连通性自检：端点配错时立刻 exit(1)，不静默重试
            validate_config_names(progress, config)

        t0 = time.time()
        log(f"[index] {ds} 开始建索引（root={root}）")
        outputs = asyncio.run(
            build_index(
                config=config,
                method=IndexingMethod.Standard,
                callbacks=[PipelineOutlineCallbacks(state)],
                progress_logger=progress,
            )
        )
        elapsed = time.time() - t0

        errors = [e for o in outputs if o.errors for e in o.errors]
        if errors:
            log(
                f"[index] {ds} 失败：{len(errors)} 个错误，详见 {root / 'logs' / 'indexing-engine.log'}"
            )
            for e in errors[:3]:
                log(f"  {type(e).__name__}: {e}")
            sys.exit(1)

        # 写 index 耗时进 cost.json
        try:
            m = {}
            if cost_file.exists() and cost_file.stat().st_size > 0:
                m = json.loads(cost_file.read_text(encoding="utf-8"))
            m.setdefault("index", {})["elapsed_s"] = round(elapsed, 3)
            cost_file.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError) as e:
            log(f"[index] ⚠ cost.json 写入失败: {e!r}")

        log(f"[index] {ds}: 完成，耗时 {elapsed:.0f}s → {root / 'output'}")


if __name__ == "__main__":
    main()
