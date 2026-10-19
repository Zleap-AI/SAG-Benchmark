"""Load the three paper datasets without the former server pipeline."""

import json
from pathlib import Path

from sample_identity import index_unique_records


class DatasetLoader:
    def __init__(self, name, directory):
        self.name = name
        self.samples_path = Path(directory) / f"{name}.json"
        self.corpus_path = Path(directory) / f"{name}_corpus.json"

    def validate_source_pair(self):
        for path in (self.samples_path, self.corpus_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing dataset source: {path}")

    def load_corpus(self):
        return json.loads(self.corpus_path.read_text())

    def load_samples(self):
        return json.loads(self.samples_path.read_text())

    def get_docs(self):
        return list(
            dict.fromkeys(
                f"{row['title'].strip()}\n{row['text']}" for row in self.load_corpus()
            )
        )

    def get_question_records(self):
        records = []
        for sample in self.load_samples():
            if self.name == "musique":
                gold = [
                    f"{p['title']}\n{p.get('paragraph_text') or p.get('text') or ''}"
                    for p in sample["paragraphs"]
                    if p["is_supporting"]
                ]
            else:
                titles = {fact[0] for fact in sample["supporting_facts"]}
                separator = "" if self.name == "hotpotqa" else " "
                gold = [
                    f"{title}\n{separator.join(sentences)}"
                    for title, sentences in sample["context"]
                    if title in titles
                ]
            records.append(
                {
                    "id": sample.get("_id", sample.get("id")),
                    "question": sample["question"].strip(),
                    "gold_docs": list(dict.fromkeys(gold)),
                }
            )
        index_unique_records(records, "id", f"{self.name} questions")
        return records
