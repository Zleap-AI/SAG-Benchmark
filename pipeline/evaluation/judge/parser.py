"""JSON parser for Judge LLM responses — no LangChain dependency.

Replicates the fallback chain from external/judge/Evaluation/metrics/utils.py
but replaces `llm.ainvoke(prompt, config={"callbacks": callbacks})` with
pipeline-native `await llm.chat(...)`.
"""

import json
import re
from typing import Any

import json_repair

# json5 is kept for compatibility — the original Judge uses json5.loads as a
# fallback and removing it would change golden fixture parse behavior.
try:
    import json5
except ImportError:
    json5 = None


def safe_json_parse(text: str) -> dict | list:
    """Try JSON -> json5 -> json_repair parsing."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if json5 is not None:
        try:
            return json5.loads(text)
        except Exception:
            pass
    try:
        repaired = json_repair.repair_json(text)
        return json.loads(repaired)
    except Exception:
        return {}


def extract_json_block(text: str) -> str:
    """Extract first JSON object block from text."""
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else text


def extract_array_fallback(text: str) -> list[str]:
    """Fallback: extract array from text."""
    match = re.search(r"\[([\s\S]*?)\]", text)
    if not match:
        return []
    items = re.split(r",\s*", match.group(1))
    return [i.strip(" \"'") for i in items if i.strip()]


def validate_list(items: Any) -> list[Any]:
    """Ensure clean list of strings or dicts."""
    if not isinstance(items, list):
        return []
    cleaned = []
    for i in items:
        if isinstance(i, str) and i.strip():
            cleaned.append(i.strip())
        elif isinstance(i, dict):
            cleaned.append(i)
    return cleaned


async def parse_with_fallbacks(
    raw_text: str,
    key: str | None = None,
    llm: Any = None,
) -> list | dict:
    """Multi-tier JSON parsing with optional LLM self-heal.

    Replicates the original JSONHandler.parse_with_fallbacks() behavior:
    1. Direct JSON parse
    2. Extract JSON block from code fences
    3. Extract array fallback
    4. Optional LLM self-heal
    """
    content = re.sub(r"```(?:json)?|```", "", raw_text).strip()

    # 1. Direct parse
    data = safe_json_parse(content)
    if key and isinstance(data, dict) and key in data:
        return validate_list(data[key])
    elif not key and data:
        return data

    # 2. Extract block
    json_block = extract_json_block(content)
    data = safe_json_parse(json_block)
    if key and isinstance(data, dict) and key in data:
        return validate_list(data[key])
    elif not key and data:
        return data

    # 3. Fallback array
    if key:
        fallback_array = extract_array_fallback(content)
        if fallback_array:
            return validate_list(fallback_array)

    # 4. Self-heal via LLM
    if llm is not None:
        healed = await _heal_with_llm(raw_text, key, llm)
        if healed:
            return healed

    return [] if key else {}


async def _heal_with_llm(
    invalid_text: str,
    key: str | None,
    llm: Any,
) -> list | dict:
    """Ask LLM to return valid JSON."""
    from pipeline.core.ai.models import LLMMessage, LLMRole

    key_instruction = f' with a key "{key}"' if key else ""
    repair_prompt = (
        f"Return ONLY valid JSON{key_instruction}.\n"
        f"Invalid output was:\n{invalid_text}\n"
    )
    try:
        response = await llm.chat(
            [LLMMessage(role=LLMRole.USER, content=repair_prompt)],
            temperature=0.0,
        )
        repaired_text = re.sub(r"```(?:json)?|```", "", response.content).strip()
        data = safe_json_parse(repaired_text)
        if key and isinstance(data, dict) and key in data:
            return validate_list(data[key])
        elif not key and data:
            return data
    except Exception:
        pass
    return [] if key else {}
