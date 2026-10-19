"""Read a full EnterpriseRAG index without modifying its database."""

import hashlib
import json
import math
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict

import regex
from sag2 import INDEX_SCHEMA, CorpusIndex, diversity_tokens, fold_latin_accents, tokenize
from chunking import source_content


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class ReadOnlyIndex(CorpusIndex):
    def __init__(self, path, signature, source_dir, statistics=None, *, lexical_statistics=None):
        if json.loads(signature)[2] != INDEX_SCHEMA:
            raise ValueError("Chunk policy changed; rebuild the full index")
        self.source_dir = source_dir
        self.db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        if self.db.execute("SELECT signature FROM metadata").fetchone()[0] != signature:
            raise ValueError("Full index signature does not match the completed build")
        self._initialize_queries()
        self._load_lexical_statistics(path, signature, lexical_statistics)
        if statistics is not None:
            self.frequencies, self.total, self.document_count = statistics
            return
        self.db.execute("CREATE VIRTUAL TABLE temp.vocab USING fts5vocab(main,chunk_fts,'col')")
        self.frequencies = {
            row["term"]: row["doc"]
            for row in self.db.execute("SELECT term,doc FROM temp.vocab WHERE col='body'")
        }
        self.total = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        self.document_count = self.db.execute(
            "SELECT COUNT(DISTINCT document_id) FROM chunks"
        ).fetchone()[0]

    def _load_lexical_statistics(self, path, signature, shared):
        if shared is None:
            cache = path.with_suffix(".lexical-statistics.json")
            if cache.exists():
                data = json.loads(cache.read_text())
                if data["index_signature"] != signature:
                    raise ValueError("Lexical statistics do not match the current index")
            else:
                titles, names, frequencies = set(), defaultdict(set), Counter()
                count = 0
                started = last = time.perf_counter()
                passages = iter(self.db.execute(
                    "SELECT document_id,spans FROM chunks ORDER BY document_id,id"
                ))
                passage = next(passages, None)
                for source in self.db.execute(
                    "SELECT document_id,path,sha256 FROM source_files ORDER BY document_id"
                ):
                    file = self.source_dir / source["path"]
                    raw = file.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
                        raise ValueError(
                            f"Source changed since indexing; rebuild the index: {file}"
                        )
                    title = file.stem.split("__", 1)[1].replace("-", " ").strip()
                    words = tuple(tokenize(title))
                    if len(words) >= 2:
                        titles.add(words)
                    name = unicodedata.normalize("NFKC", title).casefold()
                    names[fold_latin_accents(name)].add(name)
                    text = raw.decode("utf-8")
                    terms = set(diversity_tokens(text))
                    while passage is not None and passage["document_id"] == source["document_id"]:
                        terms.update(diversity_tokens(source_content(text, json.loads(passage["spans"]))))
                        passage = next(passages, None)
                    frequencies.update(terms)
                    count += 1
                    if time.perf_counter() - last >= 15:
                        print(
                            f"Lexical statistics: {count:,} documents, {time.perf_counter() - started:.1f}s",
                            flush=True,
                        )
                        last = time.perf_counter()
                data = {
                    "index_signature": signature,
                    "document_count": count,
                    "title_phrases": sorted(titles),
                    "aliases": {
                        key: next(iter(values))
                        for key, values in names.items()
                        if len(values) == 1 and key not in values
                    },
                    "diversity_idf": {
                        term: math.log((1 + count) / (1 + frequency)) + 1
                        for term, frequency in frequencies.items()
                    },
                }
                save(cache, data)
            aliases = data["aliases"]
            shared = {
                "title_phrases": {tuple(words) for words in data["title_phrases"]},
                "aliases": aliases,
                "diversity_idf": data["diversity_idf"],
                "name_pattern": regex.compile(
                    r"(?<![\p{L}\p{N}])(?:"
                    + "|".join(
                        regex.escape(key)
                        for key in sorted(aliases, key=lambda value: (-len(value), value))
                    )
                    + r")(?![\p{L}\p{N}])",
                    regex.IGNORECASE,
                )
                if aliases
                else None,
            }
        self.lexical_statistics = shared
        self.__dict__.update(shared)

    def retrieve_expansion(self, expression, terms, excluded, limit):
        print(f"Retrieving expansion: {len(terms)} terms across the full index", flush=True)
        return super().retrieve_expansion(expression, terms, excluded, limit)
