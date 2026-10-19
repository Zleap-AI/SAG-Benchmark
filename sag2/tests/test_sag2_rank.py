import sqlite3

import pytest

from scripts import run_sag2_benchmark  # noqa: F401 - exposes the standalone scripts

sag2 = __import__("sag2")


@pytest.mark.parametrize("rare_documents", [80, 140])
def test_pruned_and_full_search_match_sqlite_scores(tmp_path, rare_documents):
    rare = [f"rare{i}" for i in range(5)]
    common = [f"common{i}" for i in range(4)]
    corpus = [
        {
            "title": f"Document {i}",
            "text": " ".join(common + rare * 8 if i < rare_documents else common),
        }
        for i in range(300)
    ]
    index = sag2.CorpusIndex(tmp_path / "prune.sqlite", corpus, "prune", chunk_chars=10000)
    try:
        expression = " OR ".join(map(sag2.quote_term, rare + common))
        excluded = {1, 3}
        expected = [
            tuple(row)
            for row in index.db.execute(
                "SELECT rowid,bm25(chunk_fts,2,2,1) AS score FROM chunk_fts "
                "WHERE chunk_fts MATCH ? AND rowid NOT IN (1,3) ORDER BY score,rowid LIMIT 100",
                (expression,),
            )
        ]
        actual = index.retrieve(expression, excluded, 100)
        assert [(row["id"], row["score"]) for row in actual] == expected
    finally:
        index.close()


def test_native_statistics_preserve_duplicate_phrases_and_column_filters():
    from sag2_rank import load_rank_extension

    with sqlite3.connect(":memory:") as db:
        load_rank_extension(db)
        db.execute("CREATE VIRTUAL TABLE terms USING fts5(title,heading,body)")
        db.executemany(
            "INSERT INTO terms VALUES(?,?,?)",
            [
                ("alpha", "beta", "alpha beta alpha"),
                ("beta", "alpha", "beta gamma"),
                ("alpha beta", "gamma", "gamma alpha"),
            ],
        )
        for expression in (
            '"alpha" OR "alpha" OR "beta"',
            'title : "alpha" OR body : "beta"',
            '"alpha beta" OR "gamma"',
        ):
            stats = db.execute(
                "SELECT sag_stats(terms) FROM terms WHERE terms MATCH ? LIMIT 1", (expression,)
            ).fetchone()[0]
            expected = db.execute(
                "SELECT rowid,bm25(terms,2,2,1) FROM terms WHERE terms MATCH ? ORDER BY rowid",
                (expression,),
            ).fetchall()
            actual = db.execute(
                "SELECT rowid,sag_bm25(terms,?,2,2,1) FROM terms WHERE terms MATCH ? ORDER BY rowid",
                (stats, expression),
            ).fetchall()
            assert actual == expected
