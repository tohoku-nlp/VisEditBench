import base64
import json
from pathlib import Path


def encode_image(image_path: str) -> str:
    """Encode an image file to base64 and return the result."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def read_jsonl(path: Path):
    """Read a JSONL file line by line and return as a generator."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL: {path}:{line_no}") from e


def strip_thinking(content: str | None) -> str:
    """Remove <think>...</think> blocks and return only the text that follows.

    Returns empty string if content is None (case where reasoning is stored
    in a separate field for thinking models).
    """
    if content is None:
        return ""
    marker = "</think>"
    idx = content.find(marker)
    if idx != -1:
        content = content[idx + len(marker):]
    return content.strip()


def extract_content(choice) -> str | None:
    """Extract the body text from an OpenAI response choice.

    For thinking models, message.content may be None and the thinking content
    may be stored in message.reasoning or message.reasoning_content.
    In that case, return the text including the thinking block, and let the
    caller use strip_thinking to extract the final answer.
    """
    content = choice.message.content
    if content is not None:
        return content
    # Fallback 1: reasoning field (used in qwen3.5-397b etc.)
    reasoning = getattr(choice.message, "reasoning", None)
    if reasoning:
        return reasoning
    # Fallback 2: reasoning_content field
    reasoning_content = getattr(choice.message, "reasoning_content", None)
    if reasoning_content:
        return reasoning_content
    return None
