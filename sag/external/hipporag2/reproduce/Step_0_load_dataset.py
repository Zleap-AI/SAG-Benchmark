#!/usr/bin/env python3
"""Step 0: normalize the shared benchmark dataset for HippoRAG2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_UTILS = REPOSITORY_ROOT / "pipeline" / "evaluation" / "utils"
if str(EVALUATION_UTILS) not in sys.path:
    sys.path.insert(0, str(EVALUATION_UTILS))
from load_utils import DatasetLoader
from reproduce_dataset import export_reproduce_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="加载共享多跳 QA 数据集（HippoRAG2）")
    parser.add_argument("--data_name")
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=REPOSITORY_ROOT / "dataset",
        help="共享数据集目录；默认仓库根 dataset/",
    )
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "caches")
    parser.add_argument(
        "--source-id",
        default=None,
        help="绑定 run_upload.py 的 source_config_id；默认自动发现最新上传批次",
    )
    parser.add_argument("--list", action="store_true", help="列出共享目录中的有效数据集")
    return parser


def resolve_source_id(
    data_name: str, explicit: str | None, repo_root: Path, cache_root: Path | None = None
) -> str:
    """解析批次目录（两级："数据集/批次"）。
    优先级：显式 --source-id > 最新 upload 批次 > 复用已有缓存 > 数据集名。"""
    if explicit:
        prefix = f"{data_name}-"
        batch = explicit[len(prefix) :] if explicit.startswith(prefix) else explicit
        return f"{data_name}/{batch}"
    model = (os.getenv("LLM_MODEL", "") or "").split("/")[-1]
    upload_root = repo_root / "pipeline" / "evaluation" / "source" / "SAG" / model / data_name
    if upload_root.is_dir():
        timestamps = sorted(d for d in upload_root.iterdir() if d.is_dir())
        if timestamps:
            return f"{data_name}/{timestamps[-1].name}"
    # 无 upload 记录：复用已有缓存（caches/<data_name>/ 下最新批次子目录；或旧版扁平 caches/<data_name>）
    cache_root = cache_root or Path("caches")
    ds_dir = cache_root / data_name
    if ds_dir.is_dir():
        # 只认 upload 时间戳格式的批次子目录（YYYYMMDD_HHMMSS）；contexts/questions 等固定目录不算批次
        subs = sorted(
            d
            for d in ds_dir.iterdir()
            if d.is_dir()
            and len(d.name) == 15
            and d.name[8] == "_"
            and d.name[:8].isdigit()
            and d.name[9:].isdigit()
        )
        if subs:
            latest = max(subs, key=lambda d: d.stat().st_mtime)
            print(
                f"[source-id] 无 upload 记录，复用已有缓存批次: {data_name}/{latest.name}",
                file=sys.stderr,
            )
            return f"{data_name}/{latest.name}"
        print(f"[source-id] 无 upload 记录，复用已有扁平缓存: {data_name}", file=sys.stderr)
        return data_name
    print(
        f"[source-id] 未找到 upload 记录或已有缓存，按数据集名新建：caches/{data_name}/。",
        file=sys.stderr,
    )
    return data_name


def main() -> None:
    args = build_parser().parse_args()
    available = DatasetLoader.discover_datasets(args.dataset_root)
    if args.list:
        print("\n".join(available))
        return
    if not args.data_name:
        raise SystemExit("--data_name 必填；可先用 --list 查看有效数据集")
    if args.data_name not in available:
        raise SystemExit(f"数据集 {args.data_name!r} 不完整或不存在；可选: {', '.join(available)}")

    source_id = resolve_source_id(args.data_name, args.source_id, REPOSITORY_ROOT, args.cache_root)
    manifest = export_reproduce_dataset(
        args.data_name,
        args.cache_root,
        dataset_root=args.dataset_root,
        subdir=source_id,
    )
    manifest["source_config_id"] = source_id
    manifest_path = Path(args.cache_root) / source_id / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(
        f"完成 {args.data_name}: documents={manifest['counts']['documents']}, "
        f"questions={manifest['counts']['questions']}"
    )
    print(f"manifest: {manifest_path} (source_config_id={source_id})")


if __name__ == "__main__":
    main()
