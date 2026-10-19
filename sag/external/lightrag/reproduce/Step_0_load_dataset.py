#!/usr/bin/env python3
"""Step 0: normalize the shared benchmark dataset for LightRAG."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="加载共享多跳 QA 数据集（LightRAG）")
    parser.add_argument("--data-name", "--data_name", dest="data_name")
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=REPOSITORY_ROOT / "dataset",
        help="共享数据集目录；默认仓库根 dataset/",
    )
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "caches")
    parser.add_argument("--list", action="store_true", help="列出共享目录中的有效数据集")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    available = DatasetLoader.discover_datasets(args.dataset_root)
    if args.list:
        print("\n".join(available))
        return
    if not args.data_name:
        raise SystemExit("--data-name 必填；可先用 --list 查看有效数据集")
    if args.data_name not in available:
        raise SystemExit(f"数据集 {args.data_name!r} 不完整或不存在；可选: {', '.join(available)}")

    manifest = export_reproduce_dataset(
        args.data_name,
        args.cache_root,
        dataset_root=args.dataset_root,
        legacy_sidecars=True,
    )
    print(
        f"完成 {args.data_name}: documents={manifest['counts']['documents']}, "
        f"questions={manifest['counts']['questions']}"
    )
    print(f"manifest: {Path(args.cache_root) / args.data_name / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
