from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```latex ... ``` or ``` ... ```)."""
    text = text.strip()
    # Remove leading ```latex or ``` with optional language specifier
    text = re.sub(r"^```(?:latex|tex)?\s*\n?", "", text, flags=re.IGNORECASE)
    # Remove trailing ```
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def extract_text_claude(record: dict[str, Any]) -> str | None:
    """Extract text from a Claude batch API response record."""
    result = record.get("result", {})
    if not result:
        return None
    message = result.get("message", {})
    if not message:
        return None
    content = message.get("content", [])
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            return item.get("text", "")
    return None


def extract_text_gemini(record: dict[str, Any]) -> str | None:
    """Extract text from a Gemini batch API response record."""
    response = record.get("response", {})
    if not response:
        return None
    candidates = response.get("candidates", [])
    if not candidates:
        return None
    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    # Return the last text part (skip thinking signature parts)
    for part in reversed(parts):
        if isinstance(part, dict) and "text" in part:
            return part["text"]
    return None


def extract_text_openai(record: dict[str, Any]) -> str | None:
    """Extract text from an OpenAI batch API response record."""
    response = record.get("response", {})
    if not response:
        return None
    body = response.get("body", {})
    if not body:
        return None
    output = body.get("output", [])
    for item in output:
        if isinstance(item, dict) and item.get("type") == "message":
            content = item.get("content", [])
            for c in content:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    return c.get("text", "")
    return None


def _strip_thinking_tags(text: str) -> str:
    """<think>...</think> ブロックを除去して実際の出力だけを返す。"""
    marker = "</think>"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx + len(marker):]
    return text.strip()


def extract_text_openrouter(record: dict[str, Any]) -> str | None:
    """OpenRouter (chat completions形式) のレスポンスレコードからテキストを抽出する。"""
    response = record.get("response", {})
    if not response:
        return None
    body = response.get("body", {})
    if not body:
        return None
    choices = body.get("choices", [])
    if not choices:
        return None
    content = choices[0].get("message", {}).get("content", "") or ""
    if not content:
        return None
    return _strip_thinking_tags(content)


def get_custom_id(record: dict[str, Any]) -> str | None:
    """Get the custom_id from a batch response record (works for all providers)."""
    # Claude and OpenAI use "custom_id"
    if "custom_id" in record:
        return record["custom_id"]
    # Gemini uses "key"
    if "key" in record:
        return record["key"]
    return None


def extract_tikz_from_response(
    record: dict[str, Any],
    provider: str,
) -> tuple[str | None, str | None]:
    """
    Extract TikZ code and custom_id from a batch API response record.

    Returns:
        (custom_id, tikz_text) — either may be None on failure.
    """
    custom_id = get_custom_id(record)

    if provider == "anthropic":
        text = extract_text_claude(record)
    elif provider == "google":
        text = extract_text_gemini(record)
    elif provider == "openai":
        text = extract_text_openai(record)
    elif provider == "openrouter":
        text = extract_text_openrouter(record)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if text is None:
        return custom_id, None

    text = strip_code_fences(text)
    return custom_id, text


def custom_id_to_key(custom_id: str, system_instruction_type: str) -> str:
    """Claude用に短縮されたIDからも、先頭の完全なサンプルキーを復元する。"""
    match = re.match(r"^(\d{8}_(?:ADD|REMOVE|UPDATE|LAYOUT|STRUCTURE)_[123]_edit_\d+)(?:_|$)", custom_id)
    if match:
        return match[1]
    suffix = f"_{system_instruction_type}"
    if custom_id.endswith(suffix):
        return custom_id[: -len(suffix)]
    return custom_id


def extract_and_save(
    output_jsonl: Path,
    save_dir: Path,
    provider: str,
    system_instruction_type: str,
) -> tuple[int, int]:
    """
    Read a batch API output JSONL and save each response as a .tex file.

    Returns:
        (success_count, error_count)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    errors = 0

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id, tikz_text = extract_tikz_from_response(record, provider)

            if custom_id is None:
                errors += 1
                continue

            key = custom_id_to_key(custom_id, system_instruction_type)

            if tikz_text is None:
                errors += 1
                continue

            tex_path = save_dir / f"{key}.tex"
            tex_path.write_text(tikz_text, encoding="utf-8")
            success += 1

    return success, errors
