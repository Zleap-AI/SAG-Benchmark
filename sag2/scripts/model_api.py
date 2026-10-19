"""Provider-neutral HTTP requests with local replay and usage accounting."""

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import aiohttp

from sag2 import JevClient, compact_json


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def token_usage(response: dict) -> dict:
    usage = response.get("usage") or {}
    inputs = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    outputs = usage.get("output_tokens", usage.get("completion_tokens", 0))
    output_details = (
        usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    )
    return {
        "input_tokens": inputs,
        "cached_input_tokens": details.get("cached_tokens", 0),
        "cache_write_input_tokens": details.get("cache_write_tokens", 0),
        "output_tokens": outputs,
        "reasoning_output_tokens": output_details.get("reasoning_tokens", 0),
    }


class RequestJournal:
    """Replay successful identical payloads; never send credentials to the journal."""

    def __init__(self, session: aiohttp.ClientSession, directory: Path):
        self.session = session
        self.directory = directory
        self.records = []

    async def post(self, service: str, endpoint: str, api_key: str, payload: dict) -> dict:
        digest = hashlib.sha256(compact_json([endpoint, payload]).encode()).hexdigest()
        path = self.directory / service / f"{digest}.json"
        reused = path.exists()
        if reused:
            record = json.loads(path.read_text())
            if "response" not in record:
                raise ValueError(
                    f"Inspect the saved successful HTTP response before retrying: {path}"
                )
            data = record["response"]
        else:
            attempt = len(list(path.parent.glob(f"{digest}.attempt-*.json")))
            limited_retries = 0
            while True:
                attempt += 1
                status, raw, error = None, None, None
                started = time.monotonic()
                try:
                    async with self.session.post(
                        endpoint, json=payload, headers={"Authorization": f"Bearer {api_key}"}
                    ) as response:
                        status = response.status
                        raw = (await response.text()).replace(api_key, "<REDACTED>")
                except (TimeoutError, aiohttp.ClientConnectionError) as exc:
                    error = type(exc).__name__
                write_json(
                    path.with_name(f"{digest}.attempt-{attempt:04d}.json"),
                    {
                        "status": status,
                        "error": error,
                        "seconds": time.monotonic() - started,
                        "body": raw,
                    },
                )
                if status == 200:
                    record = {"endpoint": endpoint, "request": payload, "raw_response": raw}
                    write_json(path, record)
                    data = json.loads(raw)
                    if not isinstance(data, dict) or data.get("error"):
                        raise ValueError(
                            f"Provider returned an invalid successful response: {path}"
                        )
                    record["response"] = data
                    write_json(path, record)
                    break
                retryable = error is not None or status in (408, 429, 499, 500, 502, 503, 504)
                if not retryable or (status != 499 and limited_retries >= 3):
                    raise RuntimeError(
                        f"{service} request failed: HTTP {status}, {error}; see {path.parent}"
                    )
                if status != 499:
                    limited_retries += 1
                await asyncio.sleep(min(30, 2 ** min(attempt - 1, 5)))
        self.records.append(
            {
                "service": service,
                "reused": reused,
                "payload_sha256": digest,
                "usage": token_usage(data),
                "provider_cost_usd": (data.get("usage") or {}).get("cost"),
                "actual_model": data.get("model"),
            }
        )
        return data

    def summary(self, service: str) -> dict:
        rows = [r for r in self.records if r["service"] == service]
        tokens = Counter({key: 0 for key in token_usage({})})
        for row in rows:
            tokens.update(row["usage"])
        costs = [r["provider_cost_usd"] for r in rows]
        return {
            **tokens,
            "requests": len(rows),
            "reused_requests": sum(r["reused"] for r in rows),
            "provider_cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            "actual_models": sorted({r["actual_model"] for r in rows if r["actual_model"]}),
        }


class RecordedJev(JevClient):
    def __init__(self, journal: RequestJournal, endpoint: str, model: str, api_key: str):
        super().__init__(journal.session, endpoint, model, api_key)
        self.journal = journal

    async def request(self, payload: dict) -> dict:
        return await self.journal.post("jev", self.endpoint, self.api_key, payload)
