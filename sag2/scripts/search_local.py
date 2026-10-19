"""Search a local JSON corpus with SAG2 and local model weights; no network calls."""

import argparse
import asyncio
import json
from pathlib import Path

from local_reranker import LocalReranker, LocalScorer
from run_sag2_benchmark import unique_documents
from sag2 import DEFAULT_CHUNK_CHARS, CorpusIndex, corpus_fingerprint, search_sag2


async def run(args):
    engine = LocalReranker(
        args.model_path, args.reranker, args.backend, args.device, args.batch_size
    )
    with args.corpus.open() as stream:
        corpus = json.load(stream)
    if (
        not isinstance(corpus, list)
        or not corpus
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("title"), str)
            or not isinstance(row.get("text"), str)
            for row in corpus
        )
    ):
        raise ValueError('Corpus must be a nonempty JSON list of {"title": "...", "text": "..."}')
    fingerprint = corpus_fingerprint(args.corpus)
    index = CorpusIndex(args.cache_dir / f"{fingerprint}.sqlite", corpus, fingerprint, args.chunk_chars)
    try:
        scorer = LocalScorer(engine, args.cache_dir / "responses")
        result = await search_sag2(
            index, scorer, args.query, candidate_k=args.candidate_k, max_hops=args.expansions
        )
        passages = unique_documents(result["candidates"], args.top_k)
        print(
            json.dumps(
                {
                    "query": args.query,
                    "model_api_requests": 0,
                    "passages": [
                        {
                            "rank": i,
                            "title": p["title"],
                            "heading": p["heading"],
                            "text": p["full_doc"],
                            "score": p["jev_score"],
                        }
                        for i, p in enumerate(passages, 1)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        index.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reranker", choices=["qwen", "jina"], default="qwen")
    parser.add_argument("--backend", choices=["transformers", "mlx"], default="transformers")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--expansions", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/local-search"))
    args = parser.parse_args()
    if min(args.batch_size, args.candidate_k, args.top_k) <= 0 or args.top_k > args.candidate_k:
        parser.error(
            "Batch size and candidate counts must be positive; top-k cannot exceed candidate-k"
        )
    if args.chunk_chars < 2:
        parser.error("--chunk-chars must be at least 2")
    if not args.query.strip():
        parser.error("Query must not be empty")
    if args.backend == "mlx" and args.reranker != "jina":
        parser.error("MLX requires --reranker jina")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
