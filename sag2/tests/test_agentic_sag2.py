import asyncio
from collections import Counter
import json

import aiohttp
from aiohttp import web
import pytest

from agentic_sag2 import SAG2Tool, agent_prompt
from model_api import RequestJournal
from scripts import run_sag2_benchmark as benchmark


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_native_tools_keep_question_history_reasoning_and_resume(tmp_path, monkeypatch, api):
    """Real HTTP and real SAG2, with provider responses fixed at the API boundary."""

    async def scenario():
        calls = Counter()
        payloads, jev_states = [], []
        question = "Anchor?"
        corpus = [
            {"title": f"Passage {i}", "text": f"Anchor bridge evidence {i}."} for i in range(24)
        ]
        (tmp_path / "hotpotqa_corpus.json").write_text(json.dumps(corpus))
        sample = {
            "_id": "question-1",
            "question": question,
            "supporting_facts": [["Passage 0", 0]],
            "context": [["Passage 0", ["Anchor bridge evidence 0."]]],
        }
        (tmp_path / "hotpotqa.json").write_text(json.dumps([sample]))

        async def jev(request):
            calls["jev"] += 1
            payload = await request.json()
            assert request.headers["Authorization"] == "Bearer secret-jev"
            jev_states.append(json.loads(payload["state"]))
            return web.json_response(
                {
                    "model": "test-jev",
                    "answers": {key: {"type": "noul", "noul": 0.7} for key in payload["questions"]},
                    "usage": {"input_tokens": 100, "output_tokens": 20, "cost": 0.001},
                }
            )

        async def llm(request):
            calls["llm"] += 1
            p = await request.json()
            payloads.append(p)
            assert request.headers["Authorization"] == "Bearer secret-llm"
            assert p["model"] == "configured-model"
            history = p["input"] if api == "responses" else p["messages"]
            assert history[1] == {"role": "user", "content": question}
            outputs = [
                json.loads(m["output"] if api == "responses" else m["content"])
                for m in history
                if m.get("type") == "function_call_output" or m.get("role") == "tool"
            ]
            if len(outputs) < 2:
                args = {
                    "query": question if not outputs else "Bridge",
                    "confirmed_relations": [],
                    "unresolved_relations": ["Find the bridge evidence"],
                }
                if outputs:
                    assert len(outputs[0]["passages"]) == 20
                    assert "score" not in outputs[0]["passages"][0]
                    args["confirmed_relations"] = [
                        {
                            "subject": "Anchor",
                            "relation": "has evidence",
                            "object": "bridge",
                            "evidence_ids": [outputs[0]["passages"][0]["id"]],
                        }
                    ]
                call = {
                    "type": "function_call",
                    "name": "sag2_search",
                    "arguments": json.dumps(args),
                    "call_id": f"call-{len(outputs)}",
                }
                if api == "responses":
                    output = [
                        {
                            "type": "reasoning",
                            "id": f"reason-{len(outputs)}",
                            "summary": [],
                            "encrypted_content": "retained",
                        },
                        call,
                    ]
                else:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "retained",
                        "tool_calls": [
                            {
                                "id": call["call_id"],
                                "type": "function",
                                "function": {"name": call["name"], "arguments": call["arguments"]},
                            }
                        ],
                    }
            else:
                assert outputs[1]["confirmed_relations"]
                ids = list(dict.fromkeys(p["id"] for out in outputs for p in out["passages"]))[:10]
                final = json.dumps(
                    {"ranked_ids": ids, "confirmed_relations": [], "unresolved_relations": []}
                )
                if api == "responses":
                    assert any(item.get("encrypted_content") == "retained" for item in history)
                    output = [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": final}],
                        }
                    ]
                else:
                    assert any(item.get("reasoning_content") == "retained" for item in history)
                    message = {"role": "assistant", "content": final}
            if api == "responses":
                return web.json_response(
                    {
                        "id": "response",
                        "status": "completed",
                        "model": "configured-model",
                        "output": output,
                        "usage": {
                            "input_tokens": 50,
                            "output_tokens": 10,
                            "output_tokens_details": {"reasoning_tokens": 3},
                        },
                    }
                )
            return web.json_response(
                {
                    "model": "configured-model",
                    "choices": [
                        {
                            "message": message,
                            "finish_reason": "tool_calls" if len(outputs) < 2 else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 10,
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                }
            )

        app = web.Application()
        app.router.add_post("/decisions", jev)
        app.router.add_post("/v1/responses", llm)
        app.router.add_post("/v1/chat/completions", llm)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        monkeypatch.setenv("LLM_API_KEY", "secret-llm")
        monkeypatch.setenv("OPENROUTER_API_KEY", "secret-jev")
        args = benchmark.build_parser().parse_args(
            [
                "--mode",
                "agentic",
                "--dataset",
                "hotpotqa",
                "--dataset-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--cache-dir",
                str(tmp_path / "cache"),
                "--index-workers",
                "1",
                "--concurrency",
                "1",
                "--llm-api",
                api,
                "--llm-model",
                "configured-model",
                "--llm-base-url",
                f"http://127.0.0.1:{port}/v1",
                "--jev-base-url",
                f"http://127.0.0.1:{port}/decisions",
                "--max-hops",
                "0",
            ]
        )
        try:
            await benchmark.run(args)
            assert calls == {"jev": 2, "llm": 3}
            assert all(state["original_question"] == question for state in jev_states)
            assert jev_states[1]["current_search_query"] == "Bridge"
            assert jev_states[1]["retrieval_history"][0]["query"] == question
            summary = json.loads((tmp_path / "output/hotpotqa/metrics.json").read_text())
            assert summary["search_calls"] == 2
            assert summary["llm_usage"]["input_tokens"] == 150
            assert summary["llm_usage"]["reasoning_output_tokens"] == 9
            assert summary["llm_provider_cost_usd"] is None
            config = (tmp_path / "output/hotpotqa/config.json").read_text()
            assert "secret-" not in config
            args.resume, args.output_dir = args.output_dir, None
            await benchmark.run(args)
            assert calls == {"jev": 2, "llm": 3}
            # Replay at the request boundary also avoids a new billable call.
            async with aiohttp.ClientSession() as session:
                journal = RequestJournal(session, tmp_path / "output/hotpotqa/requests/0")
                await journal.post(
                    "llm",
                    f"http://127.0.0.1:{port}/v1/"
                    + ("responses" if api == "responses" else "chat/completions"),
                    "secret-llm",
                    payloads[0],
                )
                assert journal.summary("llm")["reused_requests"] == 1
            assert calls["llm"] == 3
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_tool_budget_and_first_query_rejections_do_not_search():
    tool = SAG2Tool(None, None, "Original?", {"max_tool_calls": 1})
    args = {"query": "Rewritten?", "confirmed_relations": [], "unresolved_relations": []}
    assert not asyncio.run(tool.call(json.dumps(args)))["search_executed"]
    tool.calls.append({})
    args["query"] = "Original?"
    assert "budget" in asyncio.run(tool.call(json.dumps(args)))["error"]


def test_best_prompt_unchanged_and_other_budgets_configurable():
    import hashlib

    text = agent_prompt({"max_tool_calls": 3, "return_k": 20, "top_k": 10})
    # Source file includes one terminal newline.
    assert (
        hashlib.sha256((text + "\n").encode()).hexdigest()
        == "2a1c1e857f053f6eb854061b77df8b814df107944900805341b4e4ee9dcaf84b"
    )
    text = agent_prompt({"max_tool_calls": 2, "return_k": 15, "top_k": 5})
    assert "at most 2 times" in text and "15 passages" in text and "exactly 5 distinct IDs" in text


@pytest.mark.parametrize("status,body", [(403, "provider denied secret-value"), (200, "not JSON")])
def test_provider_denial_and_invalid_success_never_retry(tmp_path, status, body):
    async def scenario():
        calls = []

        async def handler(request):
            calls.append(await request.json())
            return web.Response(status=status, text=body)

        app = web.Application()
        app.router.add_post("/responses", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                journal = RequestJournal(session, tmp_path)
                endpoint = f"http://127.0.0.1:{port}/responses"
                with pytest.raises((RuntimeError, ValueError)):
                    await journal.post("llm", endpoint, "secret-value", {"model": "test"})
                assert len(calls) == 1
                assert all("secret-value" not in p.read_text() for p in tmp_path.rglob("*.json"))
                if status == 200:
                    with pytest.raises(ValueError, match="Inspect the saved"):
                        await journal.post("llm", endpoint, "secret-value", {"model": "test"})
                    assert len(calls) == 1
        finally:
            await runner.cleanup()

    asyncio.run(scenario())
