import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chunking import chunk_document
from sag2 import CorpusIndex, JevClient, rank_dpp
from enterprise_index import ReadOnlyIndex
from scripts import build_enterprise_index

DESKTOP = json.loads(Path(__file__).with_name("desktop_chunks.json").read_text())


@pytest.mark.parametrize("case", DESKTOP["cases"], ids=lambda case: case["name"])
def test_chunks_match_desktop(case):
    chunks = chunk_document(case["source"], case["target"], case["path"])
    for chunk in chunks:
        chunk["spans"] = [list(span) for span in chunk["spans"]]
    assert chunks == case["expected"]


def test_long_sentence_and_complete_overlap_keep_heading():
    first = "完整条件" * 100 + "。"
    long = "超长句子" * 150 + "。"
    last = "只有满足上述条件才能发布。"
    source = f"# 发布指南\n## 生产环境\n{first}{long}{last}"
    chunks = chunk_document(source, 512)
    assert [chunk["content"] for chunk in chunks] == [first, first + long, long + last]
    assert all(chunk["heading"] == "发布指南 / 生产环境" for chunk in chunks)


def test_plain_text_uses_sentences_and_keeps_source_offsets():
    source = "完整句子。" * 200
    chunks = chunk_document(source, 512, path="document.txt")
    assert len(chunks) > 2
    for chunk in chunks:
        assert len(chunk["content"]) <= 512
        assert chunk["content"].endswith("。")
        assert chunk["content"] == source[chunk["spans"][0][0] : chunk["spans"][0][1]]


@pytest.mark.parametrize("legacy_cache", [False, True])
def test_json_corpus_preserves_hash_prefixed_plain_text(tmp_path, legacy_cache):
    source = "# Director Photo Service Term President. Ralph Canine served under Truman."
    if legacy_cache:
        previous = CorpusIndex(
            tmp_path / "index.sqlite",
            [{"title": "Directors", "text": source, "path": "document.md"}],
            "test",
        )
        assert previous.document_count == 0
        signature = json.loads(previous.db.execute("SELECT signature FROM metadata").fetchone()[0])
        legacy_signature = json.dumps(signature[:-1], separators=(",", ":"))
        previous.db.execute("UPDATE metadata SET signature=?", (legacy_signature,))
        previous.db.execute("UPDATE index_statistics SET signature=?", (legacy_signature,))
        previous.db.commit()
        previous.close()
    index = CorpusIndex(tmp_path / "index.sqlite", [{"title": "Directors", "text": source}], "test")
    try:
        assert index.document_count == 1
        hits = index.retrieve('"truman"', set(), 10)
        assert len(hits) == 1
        assert hits[0]["content"] == source
        assert hits[0]["heading"] == ""
    finally:
        index.close()


def test_index_and_jev_keep_section_context(tmp_path):
    source = "# 发布指南\n## 生产环境\n" + "发布前必须核对授权。" * 100
    index = CorpusIndex(tmp_path / "index.sqlite", [{"title": "指南", "text": source, "path": "guide.md"}], "test", chunk_chars=512)
    try:
        hits = index.retrieve('"授权"', set(), 10)
        assert len(hits) > 1
        assert all(hit["heading"] == "发布指南 / 生产环境" for hit in hits)
        for hit in hits:
            hit["jev_score"] = 0.8
        assert len(rank_dpp(hits, index.diversity_idf)) == len(hits)
        client = JevClient(None, "http://offline/decisions", "test", "unused")
        client.request = AsyncMock(return_value={
            "model": "test", "answers": {f"candidate_{hits[0]['id']}": {"type": "noul", "noul": 0.8}}
        })
        asyncio.run(client.score("如何发布？", hits[:1], [hits[-1]]))
        payload = client.request.call_args.args[0]
        assert payload["questions"][f"candidate_{hits[0]['id']}"]["instructions"]["candidate_heading"] == "发布指南 / 生产环境"
        assert json.loads(payload["state"])["previously_retrieved_evidence"][0]["heading"] == "发布指南 / 生产环境"
    finally:
        index.close()


def test_file_backed_index_keeps_exact_ranges_and_headings(tmp_path):
    source = "# 发布指南\r\n## 生产环境\r\n" + ("授权规则必须完整保留。\r\n\r\n" * 60)
    path = tmp_path / "dsid_1__Guide.md"
    path.write_bytes(source.encode())
    corpus = [{"title": "Guide", "text": source, "path": path.name,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    writable = CorpusIndex(tmp_path / "index.sqlite", corpus, "test", chunk_chars=512, source_dir=tmp_path)
    signature = writable.db.execute("SELECT signature FROM metadata").fetchone()[0]
    readonly = ReadOnlyIndex(tmp_path / "index.sqlite", signature, tmp_path)
    try:
        expected = chunk_document(source, 512, path=path)
        hits = readonly.retrieve('"授权"', set(), 100)
        assert len(hits) == len(expected)
        assert {hit["content"] for hit in hits} == {chunk["content"] for chunk in expected}
        assert all(hit["heading"] == "发布指南 / 生产环境" for hit in hits)
        assert any(len(json.loads(hit["spans"])) > 1 for hit in hits)
        assert readonly.diversity_idf == writable.diversity_idf
        assert path.read_bytes() == source.encode()
    finally:
        readonly.close()
        writable.close()


def test_readonly_index_requires_rebuild_for_previous_chunk_policy(tmp_path):
    with pytest.raises(ValueError, match="rebuild"):
        ReadOnlyIndex(tmp_path / "index.sqlite", '["fixture", 1000, 4, null]', tmp_path)


def test_chunk_length_counts_returned_text_not_gaps(tmp_path):
    source = "# A\nfirst.\n" + "\n" * 10000 + "second."
    index = CorpusIndex(tmp_path / "index.sqlite", [{"title": "Guide", "text": source, "path": "guide.md"}], "test")
    try:
        hit = index.retrieve('"first"', set(), 10)[0]
        assert hit["content"] == "first.\n\nsecond."
        assert hit["characters"] == 15
    finally:
        index.close()


def test_enterprise_builder_accepts_complete_sentence_above_target(tmp_path, monkeypatch):
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "dsid_1__Guide.txt").write_text("😀" * 1500 + "。")
    (tmp_path / "download_manifest.json").write_text(json.dumps({
        "extraction": {"documents": 1}, "release": "fixture"
    }))
    (tmp_path / "questions.jsonl").write_text(json.dumps({"expected_doc_ids": ["dsid_1"]}))
    monkeypatch.setattr("sys.argv", ["build_enterprise_index", "--dataset-dir", str(tmp_path),
        "--index-workers", "1", "--output-dir", str(tmp_path / "output"),
        "--index-path", str(tmp_path / "index.sqlite")])
    build_enterprise_index.main()
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["chunk_chars"] == 2200
    assert summary["observed_max_chunk_chars"] == 3001
    assert summary["status"] == "complete"


def test_streamed_file_index_persists_complete_statistics(tmp_path):
    source = "Evidence about Project Aurora. Additional retrieval evidence."
    path = tmp_path / "dsid_1__Project-Aurora.txt"
    path.write_bytes(source.encode())
    corpus = [{"title": "Project Aurora", "text": source, "path": path.name,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
    index_path = tmp_path / "index.sqlite"
    streamed = CorpusIndex(index_path, iter(corpus), "fixture", source_dir=tmp_path)
    signature = streamed.db.execute("SELECT signature FROM metadata").fetchone()[0]
    try:
        assert ("project", "aurora") in streamed.title_phrases
        assert "evidence" in streamed.diversity_idf
        expected = streamed.diversity_idf.copy()
    finally:
        streamed.close()
    # Repair metadata left by the previous exhausted-generator path, without reindexing.
    with sqlite3.connect(index_path) as db:
        data = json.loads(db.execute("SELECT data FROM index_statistics").fetchone()[0])
        data.pop("source_statistics_complete")
        data.update(title_phrases=[], aliases={}, diversity_idf={})
        db.execute("UPDATE index_statistics SET data=?", (json.dumps(data),))
    reopened = CorpusIndex(index_path, iter(corpus), "fixture", source_dir=tmp_path)
    reader = ReadOnlyIndex(index_path, signature, tmp_path)
    try:
        assert reopened.diversity_idf == reader.diversity_idf == expected
        assert reopened.title_phrases == reader.title_phrases
    finally:
        reader.close()
        reopened.close()
