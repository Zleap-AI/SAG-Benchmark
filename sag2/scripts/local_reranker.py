"""Offline rerank models and the scoring adapter used by Local SAG2."""

import asyncio
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

from model_api import write_json
from sag2 import CRITERION, EVIDENCE_SCORE_NOTE, compact_json

STANDARD_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query."
QWEN_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LocalReranker:
    def __init__(self, path: Path, model: str, backend: str, device: str, batch_size: int):
        self.path = path.resolve(strict=True)
        self.model, self.backend, self.batch_size = model, backend, batch_size
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        hashes = {}
        for file in sorted(self.path.rglob("*")):
            if (
                file.is_file()
                and file.suffix in (".safetensors", ".json", ".py", ".txt", ".jinja")
                and "__pycache__" not in file.parts
            ):
                with file.open("rb") as stream:
                    hashes[str(file.relative_to(self.path))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
        if not any(name.endswith(".safetensors") for name in hashes):
            raise ValueError("Local model directory contains no safetensors weights")
        self.lock = asyncio.Lock()
        if backend == "mlx":
            if model != "jina":
                raise ValueError("The MLX backend supports the official Jina model only")
            import mlx.core as mx

            sys.path.insert(0, str(self.path))
            self.module = importlib.import_module("rerank")
            if Path(self.module.__file__).resolve() != self.path / "rerank.py":
                raise ValueError("Another rerank module is already loaded; use a fresh process")
            mx.set_default_device(mx.gpu)
            self.engine = self.module.MLXReranker(model_path=str(self.path))
            # Keep the full SAG2 evidence context; the official default truncates it.
            self.engine.max_query_length = self.engine.max_length
            self.device = str(mx.default_device())
        else:
            import torch
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

            self.torch = torch
            self.device = (
                device
                if device != "auto"
                else (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.path), local_files_only=True, padding_side="left"
            )
            loader = AutoModel if model == "jina" else AutoModelForCausalLM
            self.engine = (
                loader.from_pretrained(
                    str(self.path),
                    local_files_only=True,
                    trust_remote_code=model == "jina",
                    dtype=torch.float32 if self.device == "cpu" else torch.bfloat16,
                    attn_implementation="sdpa",
                )
                .to(self.device)
                .eval()
            )
            if model == "jina":
                self.engine._tokenizer = self.tokenizer
                self.module = importlib.import_module(type(self.engine).__module__)
            else:
                self.prefix = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
                self.suffix = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
                self.no_id = self.tokenizer.convert_tokens_to_ids("no")
                self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.config = {
            "model": model,
            "backend": backend,
            "device": self.device,
            "batch_size": batch_size,
            "model_files_sha256": hashes,
            "source_truncation": False,
        }

    def rank(self, query: str, documents: list[str], instruction: str) -> dict:
        if self.model == "jina":
            full_query = instruction + "\n\n" + query
            if self.backend == "mlx":
                processed_query, processed_docs, blocks = self.engine._make_blocks(
                    full_query, documents
                )
                if processed_query != full_query or processed_docs != documents:
                    raise ValueError(
                        "Jina input would be truncated; reduce candidate count or context"
                    )
                sizes = [
                    len(self.engine._tokenize_prompt(self.module._format_prompt(full_query, block)))
                    for block in blocks
                ]
                if max(sizes) > self.engine.max_length:
                    raise ValueError(
                        "Jina prompt exceeds the model context; source was not truncated"
                    )
                rows = self.engine.rerank(full_query, documents)
            else:
                prompt = self.module.format_docs_prompts_func(
                    full_query,
                    documents,
                    special_tokens=self.engine.special_tokens,
                    no_thinking=True,
                )
                encoded = self.tokenizer([prompt], return_tensors="pt", truncation=False).to(
                    self.device
                )
                sizes = [encoded["input_ids"].shape[1]]
                if sizes[0] > self.engine.config.max_position_embeddings:
                    raise ValueError(
                        "Jina prompt exceeds the model context; source was not truncated"
                    )
                with self.torch.inference_mode():
                    scores = self.engine(**encoded).scores.reshape(-1).float().cpu().tolist()
                rows = [{"index": i, "relevance_score": score} for i, score in enumerate(scores)]
            tokens = sum(sizes)
        else:
            texts = [
                f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
                for doc in documents
            ]
            ids = [
                self.prefix + self.tokenizer.encode(text, add_special_tokens=False) + self.suffix
                for text in texts
            ]
            if any(len(row) > self.engine.config.max_position_embeddings for row in ids):
                raise ValueError("Qwen prompt exceeds the model context; source was not truncated")
            tokens, rows = sum(map(len, ids)), []
            for start in range(0, len(ids), self.batch_size):
                inputs = self.tokenizer.pad(
                    {"input_ids": ids[start : start + self.batch_size]},
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    logits = self.engine(**inputs, use_cache=False, logits_to_keep=1).logits[
                        :, -1, :
                    ]
                    scores = (
                        self.torch.softmax(logits[:, [self.no_id, self.yes_id]].float(), dim=-1)[
                            :, 1
                        ]
                        .cpu()
                        .tolist()
                    )
                rows.extend(
                    {"index": start + i, "relevance_score": score} for i, score in enumerate(scores)
                )
        return {
            "model": f"local/{self.model}/{self.backend}",
            "results": rows,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


class LocalScorer:
    def __init__(self, engine: LocalReranker, directory: Path, protocol: str = "sag2"):
        self.engine, self.directory, self.protocol = engine, directory, protocol

    async def score(self, query: str, candidates: list[dict], evidence: list[dict]):
        instruction = CRITERION if self.protocol == "sag2" else STANDARD_INSTRUCTION
        if self.protocol == "sag2":
            state = {
                "original_question": query,
                "previously_retrieved_evidence": [
                    {
                        "title": c["title"],
                        "passage": c["content"],
                        **(
                            {"previous_jev_score": c["previous_jev_score"]}
                            if "previous_jev_score" in c
                            else {}
                        ),
                    }
                    for c in evidence
                ],
            }
            if any("previous_jev_score" in c for c in evidence):
                state["evidence_score_note"] = EVIDENCE_SCORE_NOTE
            query = compact_json(state)
        documents = [c["title"] + "\n" + c["content"] for c in candidates]
        request = {
            "query": query,
            "instruction": instruction,
            "documents": documents,
            "model": self.engine.config,
        }
        digest = hashlib.sha256(compact_json(request).encode()).hexdigest()
        path = self.directory / f"{digest}.json"
        async with self.engine.lock:
            reused = path.exists()
            if reused:
                response = json.loads(path.read_text())["response"]
            else:
                write_json(path.with_suffix(".request.json"), request)
                start = time.perf_counter()
                response = await asyncio.to_thread(self.engine.rank, query, documents, instruction)
                write_json(path, {"response": response, "seconds": time.perf_counter() - start})
        rows = response["results"]
        if len(rows) != len(candidates) or {r["index"] for r in rows} != set(
            range(len(candidates))
        ):
            raise ValueError("Local reranker result indices do not match the candidate list")
        scores = {}
        for row in rows:
            score = row["relevance_score"]
            minimum = -1 if self.engine.model == "jina" else 0
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not minimum <= score <= 1
            ):
                raise ValueError("Invalid local reranker score")
            scores[row["index"]] = (score + 1) / 2 if self.engine.model == "jina" else score
        detail = {
            "model": response["model"],
            "candidate_count": len(candidates),
            "reused": reused,
            **response["usage"],
        }
        return [
            dict(c, jev_score=scores[i], jev_model=response["model"])
            for i, c in enumerate(candidates)
        ], {
            "requests": 1,
            "judgments": len(candidates),
            "batches": [detail],
            **response["usage"],
        }
