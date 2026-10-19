"""Build and verify the full local EnterpriseRAG-Bench index without model calls."""

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sag2 import CHUNK_POLICY, DEFAULT_CHUNK_CHARS, CorpusIndex, compact_json, corpus_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT.parent.parent / "datasets/EnterpriseRAG-Bench"
    )
    parser.add_argument("--index-workers", type=int, default=8)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/enterprise/index")
    parser.add_argument("--index-path", type=Path, default=ROOT / ".cache/sag2/enterprise.sqlite")
    args = parser.parse_args()
    if args.index_workers < 1:
        parser.error("--index-workers must be positive")
    if args.chunk_chars < 2:
        parser.error("--chunk-chars must be at least 2")
    started = time.perf_counter()
    source = args.dataset_dir / "documents"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.rglob("*.txt"))
    downloaded = json.loads((args.dataset_dir / "download_manifest.json").read_text())
    expected_count = downloaded["extraction"]["documents"]
    if len(paths) != expected_count:
        raise ValueError(f"Expected {expected_count} documents, found {len(paths)}")
    print(f"Inventory: {len(paths):,} source documents", flush=True)
    inventory = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        inventory.update(
            compact_json([str(path.relative_to(source)), stat.st_size, stat.st_mtime_ns]).encode()
        )
    fingerprint = inventory.hexdigest()
    index_path = args.index_path
    config = {
        "dataset": "EnterpriseRAG-Bench",
        "release": downloaded["release"],
        "source_documents": len(paths),
        "source_inventory_sha256": fingerprint,
        "algorithm_sha256": corpus_fingerprint(ROOT / "scripts/sag2.py"),
        "chunking_sha256": corpus_fingerprint(ROOT / "scripts/chunking.py"),
        "index_workers": args.index_workers,
        "chunk_chars": args.chunk_chars,
        "chunk_policy": CHUNK_POLICY,
        "chunk_length_limit": "soft; complete sentences may exceed the target",
        "full_documents_stored": False,
        "chunk_text_stored": False,
        "source_directory": str(source.resolve()),
        "title_source": "Filename slug; complete document body preserved",
        "index_path": str(index_path),
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Building full index with {args.index_workers} processes: {index_path}", flush=True)
    indexed_source_ids = set()
    content_digest = hashlib.sha256()

    def documents():
        last = time.perf_counter()
        with (output / "documents.jsonl").open("w") as mapping:
            for position, path in enumerate(paths):
                raw = path.read_bytes()
                text = raw.decode("utf-8")
                source_id, slug = path.stem.split("__", 1)
                digest = hashlib.sha256(raw).hexdigest()
                relative = str(path.relative_to(source))
                record = {
                    "document_id": position + 1,
                    "source_id": source_id,
                    "path": relative,
                    "sha256": digest,
                }
                mapping.write(compact_json(record) + "\n")
                content_digest.update(compact_json([relative, digest]).encode())
                indexed_source_ids.add(source_id)
                # CorpusIndex removes the final chunk suffix when storing source_id.
                yield {
                    "idx": source_id + "_0",
                    "title": slug.replace("-", " "),
                    "text": text,
                    "path": relative,
                    "sha256": digest,
                }
                if time.perf_counter() - last >= 15:
                    elapsed = time.perf_counter() - started
                    progress = {
                        "phase": "indexing",
                        "documents": position + 1,
                        "total_documents": len(paths),
                        "elapsed_seconds": round(elapsed, 1),
                    }
                    (output / "progress.json").write_text(json.dumps(progress) + "\n")
                    print(
                        f"Indexed {position + 1:,}/{len(paths):,} ({(position + 1) / len(paths):.1%}), {elapsed:.1f}s",
                        flush=True,
                    )
                    last = time.perf_counter()
        print(
            "All source documents processed; merging index and loading term statistics...",
            flush=True,
        )

    build_started = time.perf_counter()
    index = CorpusIndex(
        index_path,
        documents(),
        fingerprint,
        args.chunk_chars,
        index_workers=args.index_workers,
        source_dir=source,
    )
    build_seconds = time.perf_counter() - build_started
    try:
        if not indexed_source_ids:
            indexed_source_ids = {
                row[0] for row in index.db.execute("SELECT DISTINCT source_id FROM chunks")
            }
        print("Verifying SQLite and full-text index integrity...", flush=True)
        integrity = index.db.execute("PRAGMA integrity_check").fetchall()
        if [row[0] for row in integrity] != ["ok"]:
            raise ValueError([tuple(row) for row in integrity])
        with index.db:
            index.db.execute("INSERT INTO chunk_fts(chunk_fts) VALUES('integrity-check')")
        questions = [
            json.loads(line)
            for line in (args.dataset_dir / "questions.jsonl").read_text().splitlines()
        ]
        gold_ids = {doc_id for q in questions for doc_id in q["expected_doc_ids"]}
        missing = sorted(gold_ids - indexed_source_ids)
        max_chunk_chars = index.db.execute(
            "SELECT MAX(characters) FROM chunks"
        ).fetchone()[0]
        if missing:
            raise ValueError(f"Missing sources: {missing}")
        result = {
            **config,
            "status": "complete",
            "indexed_documents": index.document_count,
            "unique_source_ids": len(indexed_source_ids),
            "index_chunks": index.total,
            "index_signature": index.db.execute("SELECT signature FROM metadata").fetchone()[0],
            "observed_max_chunk_chars": max_chunk_chars,
            "index_bytes": index_path.stat().st_size,
            "build_seconds": round(build_seconds, 2),
            "total_seconds": round(time.perf_counter() - started, 2),
            "sqlite_integrity": "ok",
            "fts_integrity": "ok",
            "evaluation_questions": len(questions),
            "gold_source_ids": len(gold_ids),
            "missing_gold_source_ids": missing,
        }
        if index.document_count != len(paths):
            raise ValueError(
                f"Unexpected document deduplication: {index.document_count}/{len(paths)}"
            )
        if content_digest.hexdigest() != hashlib.sha256().hexdigest():
            result["source_content_sha256"] = content_digest.hexdigest()
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "progress.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        index.close()


if __name__ == "__main__":
    main()
