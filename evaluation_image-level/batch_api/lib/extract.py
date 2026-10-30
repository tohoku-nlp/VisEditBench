from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCORE_KEYS = ("instruction_adherence", "diagram_readability", "content_preservation")


def _get_custom_id(record: dict[str, Any]) -> str | None:
    """Get custom_id / key in a provider-agnostic way."""
    return record.get("custom_id") or record.get("key")


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (for Qwen thinking models)."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_json_scores(text: str) -> dict[str, int] | None:
    """Parse score JSON; retry after removing thinking tags if needed."""
    for t in (text.strip(), _strip_thinking(text)):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and all(type(obj.get(k)) is int and 1 <= obj[k] <= 5 for k in SCORE_KEYS):
                return {k: obj[k] for k in SCORE_KEYS}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


def extract_scores_openai(record: dict[str, Any]) -> dict[str, int] | None:
    """Extract scores from an OpenAI chat/completions batch response."""
    try:
        content = (
            record["response"]["body"]["choices"][0]["message"]["content"]
        )
        return _parse_json_scores(content)
    except (KeyError, IndexError, TypeError):
        return None


def extract_scores_gemini(record: dict[str, Any]) -> dict[str, int] | None:
    """Extract scores from a Gemini batch response."""
    try:
        parts = record["response"]["candidates"][0]["content"]["parts"]
        for part in reversed(parts):
            if "text" in part:
                scores = _parse_json_scores(part["text"])
                if scores:
                    return scores
    except (KeyError, IndexError, TypeError):
        pass
    return None


def extract_scores_claude(record: dict[str, Any]) -> dict[str, int] | None:
    """Extract scores from a Claude batch response."""
    try:
        content = record["result"]["message"]["content"]
        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "tool_use":
                input_obj = item.get("input")
                if isinstance(input_obj, dict) and all(type(input_obj.get(k)) is int and 1 <= input_obj[k] <= 5 for k in SCORE_KEYS):
                    try:
                        return {k: int(input_obj[k]) for k in SCORE_KEYS}
                    except (ValueError, TypeError):
                        pass

            if item.get("type") == "text":
                scores = _parse_json_scores(item.get("text", ""))
                if scores:
                    return scores
    except (KeyError, IndexError, TypeError):
        pass
    return None


def extract_scores_from_record(
    record: dict[str, Any], provider: str
) -> tuple[str | None, dict[str, int] | None]:
    """Extract (custom_id, scores) from one batch response record."""
    custom_id = _get_custom_id(record)
    if provider == "openai":
        scores = extract_scores_openai(record)
    elif provider == "google":
        scores = extract_scores_gemini(record)
    elif provider == "anthropic":
        scores = extract_scores_claude(record)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return custom_id, scores


def extract_scores_from_jsonl(
    output_jsonl: Path, provider: str
) -> list[dict[str, Any]]:
    """Read batch output JSONL and return a list of score records.

    Returns:
        [{"custom_id": ..., "instruction_adherence": ..., ...}, ...]
    """
    results: list[dict[str, Any]] = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id, scores = extract_scores_from_record(record, provider)
            if custom_id is None or scores is None:
                continue
            results.append({"custom_id": custom_id, **scores})
    return results
