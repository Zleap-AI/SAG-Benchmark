"""Native API tool calling around SAG2; the model chooses searches and final ranking."""

from copy import deepcopy
import json
from pathlib import Path
import time

from model_api import RequestJournal, write_json
from sag2 import compact_json, search_sag2

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def relation_state(value: dict, available: set[int]) -> dict:
    confirmed, unresolved = value["confirmed_relations"], value["unresolved_relations"]
    if not isinstance(confirmed, list) or not isinstance(unresolved, list):
        raise ValueError("Both relation fields must be lists")
    for link in confirmed:
        if not isinstance(link, dict) or set(link) != {
            "subject",
            "relation",
            "object",
            "evidence_ids",
        }:
            raise ValueError("Confirmed relations need subject, relation, object and evidence_ids")
        if any(
            not isinstance(link[k], str) or not link[k].strip()
            for k in ("subject", "relation", "object")
        ):
            raise ValueError(
                "Confirmed relations require nonempty subjects, relationships and objects"
            )
        ids = link["evidence_ids"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or i not in available for i in ids)
        ):
            raise ValueError("Confirmed relations must cite previously returned passage IDs")
    if any(not isinstance(v, str) or not v.strip() for v in unresolved):
        raise ValueError("Unresolved relations must be explicit nonempty questions")
    return {"confirmed_relations": confirmed, "unresolved_relations": unresolved}


class SAG2Tool:
    def __init__(self, index, jev, question: str, options: dict):
        self.index, self.jev, self.question, self.options = index, jev, question, options
        self.calls, self.decisions, self.evidence = [], [], {}

    async def call(self, arguments: str) -> dict:
        try:
            args = json.loads(arguments)
            if not isinstance(args, dict) or set(args) != {
                "query",
                "confirmed_relations",
                "unresolved_relations",
            }:
                raise ValueError("Expected query, confirmed_relations and unresolved_relations")
            if len(self.calls) >= self.options["max_tool_calls"]:
                raise ValueError("Search budget exhausted; rank the already retrieved passages")
            if not isinstance(args["query"], str) or not args["query"].strip():
                raise ValueError("query must be a nonempty string")
            if not self.calls and args["query"] != self.question:
                raise ValueError("The first query must match the original question exactly")
            state = relation_state(args, set(self.evidence))
        except (ValueError, TypeError, KeyError) as error:
            return {
                "error": str(error),
                "search_executed": False,
                "calls_remaining": self.options["max_tool_calls"] - len(self.calls),
            }
        self.decisions.append({"action": "search", "query": args["query"], **state})
        context = {
            "original_question": self.question,
            "search_history": [
                {"call": n, "query": c["query"], "top20_ids": [p["id"] for p in c["returned"]]}
                for n, c in enumerate(self.calls, 1)
            ],
            "previous_decisions": deepcopy(self.decisions),
            "evidence": list(self.evidence.values()),
            **state,
        }
        found = await search_sag2(
            self.index,
            self.jev,
            args["query"],
            candidate_k=self.options["candidate_k"],
            seed_k=self.options["seed_k"],
            max_hops=self.options["max_hops"],
            expansion_term_limit=self.options["expansion_term_limit"],
            context=context,
        )
        returned = found["candidates"][: self.options["return_k"]]
        self.calls.append(
            {"query": args["query"], "context": context, "returned": returned, **found}
        )
        for passage in returned:
            self.evidence.setdefault(passage["id"], passage)
        return {
            "query": args["query"],
            "calls_used": len(self.calls),
            "calls_remaining": self.options["max_tool_calls"] - len(self.calls),
            **state,
            "relation_state_note": "Model-maintained source claims, not independently verified facts. Recheck them against returned text and update after reading these passages.",
            "passages": [
                {"id": p["id"], "title": p["title"], "text": p["content"], "rank": i,
                 **({"heading": p["heading"]} if p.get("heading") else {})}
                for i, p in enumerate(returned, 1)
            ],
        }


def agent_prompt(options: dict) -> str:
    prompt = (PROMPTS / "agentic_sag2.txt").read_text().strip()
    if options["max_tool_calls"] != 3:
        n = options["max_tool_calls"]
        prompt = prompt.replace("at most three times", f"at most {n} times").replace(
            "used three calls", f"used {n} calls"
        )
    if options["return_k"] != 20:
        prompt = prompt.replace("twenty passages", f"{options['return_k']} passages")
    if options["top_k"] != 10:
        prompt = prompt.replace(
            "exactly ten distinct IDs", f"exactly {options['top_k']} distinct IDs"
        )
    return prompt


async def run_agentic(
    index, jev, question: str, journal: RequestJournal, llm: dict, api_key: str, options: dict
) -> dict:
    started = time.perf_counter()
    tool = SAG2Tool(index, jev, question, options)
    schema = json.loads((PROMPTS / "sag2_tool.json").read_text())
    final_schema = json.loads((PROMPTS / "agentic_result.json").read_text())
    final_schema["properties"]["ranked_ids"].update(
        minItems=options["top_k"], maxItems=options["top_k"]
    )
    description = (
        f"Search the unmodified corpus with SAG2 ({options['max_hops']} expansion rounds), returning up to {options['return_k']} raw passages. "
        "The first query must be the exact original question. After the first search, supply the complete updated relation state and target one specific unresolved relationship using grounded entity names. "
        f"Cite only previously returned passage IDs. At most {options['max_tool_calls']} searches may execute."
    )
    history = [
        {"role": "system", "content": agent_prompt(options)},
        {"role": "user", "content": question},
    ]
    trace = []
    endpoint = llm["base_url"].rstrip("/") + (
        "/responses" if llm["api"] == "responses" else "/chat/completions"
    )
    for _ in range(options["max_tool_calls"] + 4):
        payload = {"model": llm["model"], "parallel_tool_calls": False}
        definition = {
            "name": "sag2_search",
            "description": description,
            "parameters": schema,
            "strict": True,
        }
        if llm["api"] == "responses":
            payload.update(
                input=history,
                store=False,
                include=["reasoning.encrypted_content"],
                tools=[{"type": "function", **definition}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "retrieval_result",
                        "schema": final_schema,
                        "strict": True,
                    }
                },
            )
            if llm["reasoning_effort"] != "default":
                payload["reasoning"] = {"effort": llm["reasoning_effort"]}
        else:
            payload.update(
                messages=history,
                tools=[{"type": "function", "function": definition}],
                response_format={"type": "json_object"},
            )
            if llm["reasoning_effort"] != "default":
                payload["reasoning_effort"] = llm["reasoning_effort"]
        response = await journal.post("llm", endpoint, api_key, payload)
        if llm["api"] == "responses":
            if response.get("status") != "completed":
                raise ValueError(
                    f"LLM response incomplete: {response.get('status')}; inspect request journal"
                )
            output = response["output"]
            history.extend(output)
            calls = [item for item in output if item["type"] == "function_call"]
            final_text = "".join(
                part["text"]
                for item in output
                if item["type"] == "message"
                for part in item["content"]
                if part["type"] == "output_text"
            )
        else:
            choice = response["choices"][0]
            if choice["finish_reason"] not in ("stop", "tool_calls"):
                raise ValueError(
                    f"LLM response incomplete: {choice['finish_reason']}; inspect request journal"
                )
            message = choice["message"]
            history.append(
                {
                    k: v
                    for k, v in message.items()
                    if k
                    in (
                        "role",
                        "content",
                        "tool_calls",
                        "reasoning_content",
                        "reasoning_details",
                        "refusal",
                    )
                }
            )
            calls = [
                {
                    "name": c["function"]["name"],
                    "arguments": c["function"]["arguments"],
                    "call_id": c["id"],
                }
                for c in message.get("tool_calls", [])
            ]
            final_text = message.get("content") or ""
        if not calls:
            final = json.loads(final_text)
            ids = final["ranked_ids"]
            if not tool.calls or not isinstance(ids, list) or len(ids) != options["top_k"]:
                raise ValueError("Final output must rank the required number of retrieved passages")
            if any(type(i) is not int or i not in tool.evidence for i in ids) or len(
                set(ids)
            ) != len(ids):
                raise ValueError("Final ranking contains duplicate or unretrieved passage IDs")
            relation_state(final, set(tool.evidence))
            write_json(
                journal.directory.parent / "conversation.json",
                {"messages": history, "tool_trace": trace, "final": final},
            )
            return {
                "algorithm": "Agentic SAG2",
                "candidates": [tool.evidence[i] for i in ids],
                "baseline": tool.calls[0]["baseline"],
                "ranked_chunk_ids": ids,
                "final": final,
                "search_calls": len(tool.calls),
                "trace": tool.calls,
                "jev_usage": journal.summary("jev"),
                "llm_usage": journal.summary("llm"),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        for call in calls:
            if call["name"] != "sag2_search":
                raise ValueError(f"Unexpected tool: {call['name']}")
            result = await tool.call(call["arguments"])
            trace.append({"call": call, "result": result})
            if llm["api"] == "responses":
                history.append(
                    {
                        "type": "function_call_output",
                        "call_id": call["call_id"],
                        "output": compact_json(result),
                    }
                )
            else:
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["call_id"],
                        "content": compact_json(result),
                    }
                )
        write_json(
            journal.directory.parent / "conversation.json",
            {"messages": history, "tool_trace": trace},
        )
    raise ValueError("LLM did not finish within the tool-call and format-correction budget")
