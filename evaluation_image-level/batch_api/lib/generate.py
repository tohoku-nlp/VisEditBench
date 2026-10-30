from __future__ import annotations

import base64
import hashlib
import json
import io
import re
from pathlib import Path
from typing import Any

from .config import EvalConfig

CLAUDE_MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# JSON schema for structured output (all providers use 1-5 scale)
# ---------------------------------------------------------------------------

EVAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction_adherence": {"type": "integer", "minimum": 1, "maximum": 5},
        "diagram_readability": {"type": "integer", "minimum": 1, "maximum": 5},
        "content_preservation": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": [
        "instruction_adherence",
        "diagram_readability",
        "content_preservation",
    ],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Common criterion descriptions (used in all 3 system prompts)
# ---------------------------------------------------------------------------

_CRITERIA_BODY = r"""
Evaluate based on the three criteria below and return ONLY a JSON object with integer scores.

1. instruction_adherence (1-5)
   How well does the edited diagram satisfy the requested changes in the instructions?
   Focus on whether each requested change is correctly applied. Do not penalize for unintended changes in this criterion.
   Note: In this criterion, evaluate only whether the requested changes are correctly applied. Do not consider any changes that were not specified in the instructions; those should be evaluated under 3. content_preservation.
   5 (Perfect): All requested changes are correctly applied according to the instructions.
   4 (Good): Most requested changes are correctly applied, with only minor omissions or inaccuracies.
   3 (Fair): Some requested changes are correctly applied, but several important ones are missing or inaccurate.
   2 (Poor): Only a few requested changes are correctly applied, and many are missing or inaccurate.
   1 (Very Poor): Few or none of the requested changes are correctly applied.

2. diagram_readability (1-5)
   How easy is it to clearly see and understand the elements in the edited diagram?
   Focus only on visibility and legibility. Ignore whether the edits follow the instructions.
   Note: In this criterion, carefully check readability details as well, including any small cut-offs, overlaps, occlusions, or other visibility issues.
   5 (Perfect): All elements are clearly visible and easy to read, with no overlaps, occlusions, elements cut off, or size issues.
   4 (Good): Most elements are clearly visible, with only minor issues such as slight overlap, small elements, or elements partially cut off that do not affect overall readability.
   3 (Fair): Some elements are difficult to see, and this noticeably affects readability, but the diagram is still understandable overall.
   2 (Poor): Many elements are hard to see due to overlap, occlusion, small size, or being cut off, making the diagram difficult to interpret.
   1 (Very Poor): The diagram is largely unreadable, with severe visibility issues such as extensive overlap, occlusion, or elements being cut off.

3. content_preservation (1-5)
   To what extent does the edited diagram avoid unintended changes?
   Focus only on changes that were not requested. Do not consider whether the requested changes are correct.
   Evaluate any changes not specified in the instructions, including major layout modifications. Check carefully for subtle or minor unintended changes.
   5 (Perfect): No unintended changes are present; all non-requested parts of the diagram remain unchanged.
   4 (Good): Very few unintended changes are present, and they have minimal impact on the overall diagram.
   3 (Fair): Some unintended changes are present and somewhat affect the diagram.
   2 (Poor): Many unintended changes are introduced, affecting large parts of the diagram.
   1 (Very Poor): Extensive unintended changes are present, and much of the original content is not preserved.
"""

_RETURN_JSON = r"""
Return ONLY valid JSON (no markdown, no explanation):
{"instruction_adherence": <1-5>, "diagram_readability": <1-5>, "content_preservation": <1-5>}
"""

_IMPORTANT_NOTES = r"""
Important Notes for Evaluation:
- Invalid Edited Output: If the edited output is not a diagram, please rate all three criteria as 1 (Very Poor). Examples include a completely blank (white) image and a text-only image.
- Whitespace/Margins: If the edited diagram is present but surrounded by whitespace or margins, ignore the whitespace and evaluate only the diagram content itself.
- Cropped or Cut-Off Diagrams: If the diagram appears cropped or cut off (missing elements at edges), evaluate this under the Diagram Readability criterion, as it directly affects visibility.
"""

_CRITERIA_TEXT = _CRITERIA_BODY + _IMPORTANT_NOTES + _RETURN_JSON
_CRITERIA_TEXT_NO_NOTES = _CRITERIA_BODY + _RETURN_JSON

# ---------------------------------------------------------------------------
# System prompts per judge_input_type
# ---------------------------------------------------------------------------


_PREAMBLE_W_ORG_IMAGE = r"""You are an expert judge evaluating TikZ diagram editing quality.

You are given:
1. The ORIGINAL diagram image (before any edits)
2. An annotated version of the original diagram image containing visual edit instructions
3. The EDITED diagram image (after applying the visual edit instructions)

Use the annotated image to understand what changes were requested.
Compare the ORIGINAL and EDITED diagrams to evaluate quality.
"""

_PREAMBLE_W_ORG_TIKZ = r"""You are an expert judge evaluating TikZ diagram editing quality.

You are given:
1. The ORIGINAL TikZ code of the diagram
2. An annotated version of the original diagram image containing visual edit instructions
3. The EDITED TikZ code (after applying the visual edit instructions)

Use the annotated image to understand what changes were requested.
Compare the ORIGINAL and EDITED TikZ code to evaluate quality.
"""

_PREAMBLE_W_ORG_IMAGE_W_ORG_TIKZ = r"""You are an expert judge evaluating TikZ diagram editing quality.

You are given:
1. The ORIGINAL diagram image (before any edits)
2. The ORIGINAL TikZ code of the diagram
3. An annotated version of the original diagram image containing visual edit instructions
4. The EDITED diagram image (after applying the visual edit instructions)
5. The EDITED TikZ code (after applying the visual edit instructions)

Use the annotated image to understand what changes were requested.
Compare the ORIGINAL and EDITED diagrams and TikZ code to evaluate quality.
"""

JUDGE_PROMPTS: dict[str, str] = {
    "w_org_image": _PREAMBLE_W_ORG_IMAGE + _CRITERIA_TEXT,
    "w_org_tikz": _PREAMBLE_W_ORG_TIKZ + _CRITERIA_TEXT,
    "w_org_image_w_org_tikz": _PREAMBLE_W_ORG_IMAGE_W_ORG_TIKZ + _CRITERIA_TEXT,
}

JUDGE_PROMPTS_NO_NOTES: dict[str, str] = {
    "w_org_image": _PREAMBLE_W_ORG_IMAGE + _CRITERIA_TEXT_NO_NOTES,
    "w_org_tikz": _PREAMBLE_W_ORG_TIKZ + _CRITERIA_TEXT_NO_NOTES,
    "w_org_image_w_org_tikz": _PREAMBLE_W_ORG_IMAGE_W_ORG_TIKZ
    + _CRITERIA_TEXT_NO_NOTES,
}


# notes_variant is deprecated; use include_important_notes.

# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------


def _encode_b64(image_path: Path) -> str:
    """Encode an image file as Base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _encode_data_url(image_path: Path) -> str:
    """Encode an image file as a data URL."""
    return f"data:image/png;base64,{_encode_b64(image_path)}"


def _as_local_image_path(src: str | Path) -> Path:
    """Resolve input into a local filesystem image path.

    URL-based image input is intentionally unsupported.
    """
    if isinstance(src, Path):
        return src
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", src):
        raise ValueError(f"URL-based image input is not supported: {src}")
    return Path(src)


# ---------------------------------------------------------------------------
# OpenAI chat/completions request builders
# ---------------------------------------------------------------------------

_OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evaluation_scores",
        "strict": True,
        "schema": EVAL_JSON_SCHEMA,
    },
}



def _openai_img(url: str, detail: str = "auto") -> dict:
    block: dict = {"type": "image_url", "image_url": {"url": url}}
    if detail != "auto":
        block["image_url"]["detail"] = detail
    return block


def _openai_img_or_url(src: str | Path, detail: str = "auto") -> dict:
    """Always send local images as data URLs (no remote URL pass-through)."""
    return _openai_img(_encode_data_url(_as_local_image_path(src)), detail)


def _openai_text(text: str) -> dict:
    return {"type": "text", "text": text}


def build_request_openai(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_content: list[dict],
    temperature: float = 0,
) -> dict:
    """Build an OpenAI chat/completions batch request."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": _OPENAI_RESPONSE_FORMAT,
            "temperature": temperature,
        },
    }


def build_request_openai_w_org_image(
    custom_id: str,
    model: str,
    org_image_path: str | Path,
    vi_image_path: str | Path,
    edited_image_path: str | Path,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build an OpenAI request for the w_org_image pattern."""
    user_content = [
        _openai_img_or_url(org_image_path),
        _openai_img_or_url(vi_image_path),
        _openai_img_or_url(edited_image_path),
    ]
    return build_request_openai(
        custom_id, model, system_prompt, user_content, temperature
    )


def build_request_openai_w_org_tikz(
    custom_id: str,
    model: str,
    org_tikz_code: str,
    vi_image_path: str | Path,
    edited_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build an OpenAI request for the w_org_tikz pattern."""
    user_content = [
        _openai_text(f"Original TikZ code:\n{org_tikz_code}"),
        _openai_img_or_url(vi_image_path),
        _openai_text(f"Edited TikZ code:\n{edited_tikz_code}"),
    ]
    return build_request_openai(
        custom_id, model, system_prompt, user_content, temperature
    )


def build_request_openai_w_org_image_w_org_tikz(
    custom_id: str,
    model: str,
    org_image_path: str | Path,
    org_tikz_code: str,
    vi_image_path: str | Path,
    edited_image_path: str | Path,
    edited_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build an OpenAI request for the w_org_image_w_org_tikz pattern."""
    user_content = [
        _openai_img_or_url(org_image_path),
        _openai_text(f"Original TikZ code:\n{org_tikz_code}"),
        _openai_img_or_url(vi_image_path),
        _openai_img_or_url(edited_image_path),
        _openai_text(f"Edited TikZ code:\n{edited_tikz_code}"),
    ]
    return build_request_openai(
        custom_id, model, system_prompt, user_content, temperature
    )




# ---------------------------------------------------------------------------
# Gemini request builders
# ---------------------------------------------------------------------------

_GEMINI_GENERATION_CONFIG = {
    "responseMimeType": "application/json",
    "responseJsonSchema": EVAL_JSON_SCHEMA,
}



def _gemini_img(image_path: Path) -> dict:
    return {"inlineData": {"mimeType": "image/png", "data": _encode_b64(image_path)}}


def _gemini_img_url(url: str) -> dict:
    """Pass a public GCS URL as fileData (using raw https:// URL)."""
    return {"fileData": {"mimeType": "image/png", "fileUri": url}}


def _gemini_text(text: str) -> dict:
    return {"text": text}


def _build_gemini_request(
    custom_id: str,
    system_prompt: str,
    parts: list[dict],
    temperature: float = 0,
) -> dict:
    """Build a Gemini batch-format request."""
    generation_config = {**_GEMINI_GENERATION_CONFIG, "temperature": temperature}
    return {
        "key": custom_id,
        "request": {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        },
    }


def _gemini_img_or_url(src: str | Path) -> dict:
    """Always send local images as inlineData (no remote URL pass-through)."""
    return _gemini_img(_as_local_image_path(src))


def build_request_gemini_w_org_image(
    custom_id: str,
    org_image_path: str | Path,
    vi_image_path: str | Path,
    edited_image_path: str | Path,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Gemini request for the w_org_image pattern."""
    parts = [
        _gemini_img_or_url(org_image_path),
        _gemini_img_or_url(vi_image_path),
        _gemini_img_or_url(edited_image_path),
    ]
    return _build_gemini_request(custom_id, system_prompt, parts, temperature)


def build_request_gemini_w_org_tikz(
    custom_id: str,
    org_tikz_code: str,
    vi_image_path: str | Path,
    edited_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Gemini request for the w_org_tikz pattern."""
    parts = [
        _gemini_text(f"Original TikZ code:\n{org_tikz_code}"),
        _gemini_img_or_url(vi_image_path),
        _gemini_text(f"Edited TikZ code:\n{edited_tikz_code}"),
    ]
    return _build_gemini_request(custom_id, system_prompt, parts, temperature)


def build_request_gemini_w_org_image_w_org_tikz(
    custom_id: str,
    org_image_path: str | Path,
    org_tikz_code: str,
    vi_image_path: str | Path,
    edited_image_path: str | Path,
    edited_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Gemini request for the w_org_image_w_org_tikz pattern."""
    parts = [
        _gemini_img_or_url(org_image_path),
        _gemini_text(f"Original TikZ code:\n{org_tikz_code}"),
        _gemini_img_or_url(vi_image_path),
        _gemini_img_or_url(edited_image_path),
        _gemini_text(f"Edited TikZ code:\n{edited_tikz_code}"),
    ]
    return _build_gemini_request(custom_id, system_prompt, parts, temperature)




# ---------------------------------------------------------------------------
# Claude request builders
# ---------------------------------------------------------------------------


def _claude_img(image_path: Path) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _encode_b64(image_path),
        },
    }


def _claude_img_url(url: str) -> dict:
    """Pass a public GCS URL directly (no Base64 encoding)."""
    return {
        "type": "image",
        "source": {
            "type": "url",
            "url": url,
        },
    }


def _claude_img_or_url(src: str | Path) -> dict:
    """Always send local images as base64 (no remote URL pass-through)."""
    return _claude_img(_as_local_image_path(src))


def _claude_text(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_claude_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    content: list[dict],
    temperature: float = 0,
) -> dict:
    """Build a Claude batch-format request."""
    tool = {
        "name": "evaluation_scores",
        "description": "Return evaluation scores as structured JSON.",
        "input_schema": EVAL_JSON_SCHEMA,
    }
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_prompt,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "temperature": temperature,
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "evaluation_scores"},
            "messages": [{"role": "user", "content": content}],
        },
    }


def build_request_claude_w_org_image(
    custom_id: str,
    model: str,
    org_image_path: str | Path,
    vi_image_path: str | Path,
    edited_image_path: str | Path,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Claude request for the w_org_image pattern."""
    content = [
        _claude_img_or_url(org_image_path),
        _claude_img_or_url(vi_image_path),
        _claude_img_or_url(edited_image_path),
    ]
    return _build_claude_request(custom_id, model, system_prompt, content, temperature)


def build_request_claude_w_org_tikz(
    custom_id: str,
    model: str,
    org_tikz_code: str,
    vi_image_path: str | Path,
    edited_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Claude request for the w_org_tikz pattern."""
    content = [
        _claude_text(f"Original TikZ code:\n{org_tikz_code}"),
        _claude_img_or_url(vi_image_path),
        _claude_text(f"Edited TikZ code:\n{edited_tikz_code}"),
    ]
    return _build_claude_request(custom_id, model, system_prompt, content, temperature)


def build_request_claude_w_org_image_w_org_tikz(
    custom_id: str,
    model: str,
    org_image_path: str | Path,
    org_tikz_code: str,
    vi_image_path: str | Path,
    edited_image_path: str | Path,
    edited_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Claude request for the w_org_image_w_org_tikz pattern."""
    content = [
        _claude_img_or_url(org_image_path),
        _claude_text(f"Original TikZ code:\n{org_tikz_code}"),
        _claude_img_or_url(vi_image_path),
        _claude_img_or_url(edited_image_path),
        _claude_text(f"Edited TikZ code:\n{edited_tikz_code}"),
    ]
    return _build_claude_request(custom_id, model, system_prompt, content, temperature)


# ---------------------------------------------------------------------------
# JSONL generation
# ---------------------------------------------------------------------------


def _make_custom_id(seed: str, exp: EvalConfig) -> str:
    """Generate custom_id for a request (shorten for Claude 64-char limit)."""
    raw = f"{seed}__{exp.judge_input_type}__{exp.edited_input_type}"
    if exp.judge_provider == "anthropic" and len(raw) > 64:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        prefix_len = 64 - 1 - len(digest)
        raw = f"{raw[:prefix_len]}_{digest}"
    return raw


def _derive_tikz_path(img_path: Path) -> Path:
    """Derive the corresponding TikZ path from an image path."""
    return img_path.parent.parent / "tikz" / f"{img_path.stem}.tex"


def _resolve_edited_paths(
    record: dict[str, Any],
    metadata_fields: dict[str, str],
    edited_input_type: str,
    model_candidates: list[str],
) -> tuple[str | None, Path | None, Path | None]:
    """Resolve model from edited_image_paths and return (model, img_path, tikz_path)."""
    field = metadata_fields.get("edited_image_paths", "edited_image_paths")
    all_paths = record.get(field, {})
    by_input = all_paths.get(edited_input_type, {})

    for candidate in model_candidates:
        img_path_str = by_input.get(candidate)
        if img_path_str is not None:
            img_path = Path(img_path_str)
            tikz_path = _derive_tikz_path(img_path)
            return candidate, img_path, tikz_path

    return None, None, None


def _build_json_line(
    exp: EvalConfig,
    custom_id: str,
    org_image_path: Path,
    vi_image_path: Path,
    org_tikz_code: str,
    edited_image_path: Path | None,
    edited_tikz_code: str,
) -> dict[str, Any]:
    """Build request object from judge_input_type and judge_provider."""
    jit = exp.judge_input_type
    temperature = getattr(exp, "temperature", 0.0)

    # `notes_variant` is deprecated.
    # Use `include_important_notes` to control important-note text.
    if exp.include_important_notes:
        prompts = JUDGE_PROMPTS
    else:
        prompts = JUDGE_PROMPTS_NO_NOTES
    system_prompt = prompts[jit]
    if exp.judge_provider == "openai":
        if jit == "w_org_image":
            return build_request_openai_w_org_image(
                custom_id,
                exp.judge_model,
                org_image_path,
                vi_image_path,
                edited_image_path,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        elif jit == "w_org_tikz":
            return build_request_openai_w_org_tikz(
                custom_id,
                exp.judge_model,
                org_tikz_code,
                vi_image_path,
                edited_tikz_code,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        else:
            return build_request_openai_w_org_image_w_org_tikz(
                custom_id,
                exp.judge_model,
                org_image_path,
                org_tikz_code,
                vi_image_path,
                edited_image_path,
                edited_tikz_code,
                system_prompt=system_prompt,
                temperature=temperature,
            )
    elif exp.judge_provider == "google":
        if jit == "w_org_image":
            return build_request_gemini_w_org_image(
                custom_id,
                org_image_path,
                vi_image_path,
                edited_image_path,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        elif jit == "w_org_tikz":
            return build_request_gemini_w_org_tikz(
                custom_id,
                org_tikz_code,
                vi_image_path,
                edited_tikz_code,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        else:
            return build_request_gemini_w_org_image_w_org_tikz(
                custom_id,
                org_image_path,
                org_tikz_code,
                vi_image_path,
                edited_image_path,
                edited_tikz_code,
                system_prompt=system_prompt,
                temperature=temperature,
            )
    elif exp.judge_provider == "anthropic":
        if jit == "w_org_image":
            return build_request_claude_w_org_image(
                custom_id,
                exp.judge_model,
                org_image_path,
                vi_image_path,
                edited_image_path,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        elif jit == "w_org_tikz":
            return build_request_claude_w_org_tikz(
                custom_id,
                exp.judge_model,
                org_tikz_code,
                vi_image_path,
                edited_tikz_code,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        else:
            return build_request_claude_w_org_image_w_org_tikz(
                custom_id,
                exp.judge_model,
                org_image_path,
                org_tikz_code,
                vi_image_path,
                edited_image_path,
                edited_tikz_code,
                system_prompt=system_prompt,
                temperature=temperature,
            )
    else:
        raise ValueError(f"Unsupported judge_provider: {exp.judge_provider}")


def _load_url_map(json_path: Path | None) -> dict[str, str]:
    """Load filename->URL mapping; return empty dict when path is None/missing."""
    if json_path is None or not json_path.exists():
        return {}
    return json.loads(json_path.read_text(encoding="utf-8"))


def _load_edited_url_map(json_path: Path | None) -> dict[str, str]:
    """Load edited-image GCS URL mapping.
    Supports both list format ([{local_path, gcs_url}]) and dict format ({filename: url}).
    """
    if json_path is None or not json_path.exists():
        return {}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {item["local_path"]: item["gcs_url"] for item in data}
    return data  # Keep {filename: url} format as-is.


def _resolve_img(local_path: Path, url_map: dict[str, str]) -> str | Path:
    """Return URL when mapping exists; otherwise return local path."""
    url = url_map.get(Path(local_path).name) or url_map.get(str(local_path))
    return url if url else local_path


def generate_jsonl_for_experiment(
    exp: EvalConfig,
    metadata_fields: dict[str, str],
    limit: int | None,
    preview_first: bool,
    allowed_keys: set[str] | None = None,
    row_selectors: list[dict[str, str]] | None = None,
    strict_row_match: bool = False,
    gemini_use_local: bool = False,
) -> tuple[int, dict[str, Any] | None]:
    """Generate evaluation input JSONL from experiment settings.

    Returns:
        (generated_count, first_request_preview)
    """
    org_url_map = _load_url_map(exp.org_url_json)
    vi_url_map = _load_url_map(exp.vi_url_json)
    edited_url_map = _load_edited_url_map(exp.edited_url_json)
    if exp.judge_input_type not in JUDGE_PROMPTS:
        raise ValueError(f"Unsupported judge_input_type: {exp.judge_input_type}")

    needs_edited_image = exp.judge_input_type in (
        "w_org_image",
        "w_org_image_w_org_tikz",
    )
    needs_tikz = exp.judge_input_type in ("w_org_tikz", "w_org_image_w_org_tikz")

    mode = "a" if exp.append else "w"
    save_path = Path(exp.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    preview_obj: dict[str, Any] | None = None
    local_image_mode = True

    with (io.StringIO() if preview_first else open(save_path, mode, encoding="utf-8")) as f_out:
        if row_selectors is not None:
            # Row-level metadata selection mode using AMT CSV
            metadata_by_key: dict[str, dict[str, Any]] = {}
            with open(exp.metadata_path, "r", encoding="utf-8") as f_meta:
                for line in f_meta:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    metadata_by_key[record[metadata_fields["key"]]] = record

            missing_rows: list[str] = []
            rows = row_selectors if limit is None else row_selectors[:limit]

            for row in rows:
                key = row["key"]
                record = metadata_by_key.get(key)
                if record is None:
                    missing_rows.append(
                        f"{row.get('question_id', 'unknown')}: key not found -> {key}"
                    )
                    continue

                selector_model = row.get("edited_model", "")
                seen: set[str] = set()
                model_candidates: list[str] = []
                for m in [selector_model, exp.edited_model_type]:
                    if m and m not in seen:
                        seen.add(m)
                        model_candidates.append(m)

                org_image_path = Path(record[metadata_fields["org_image_path"]])
                vi_image_path = Path(
                    record[metadata_fields["visual_instruction_image_path"]]
                )
                org_tikz_code_path = Path(record[metadata_fields["org_tikz_code_path"]])

                resolved_model, edited_image_path, edited_tikz_path = (
                    _resolve_edited_paths(
                        record, metadata_fields, exp.edited_input_type, model_candidates
                    )
                )

                if resolved_model is None:
                    missing_rows.append(
                        f"{row.get('question_id', 'unknown')}: edited image not found "
                        f"key={key}, tried={model_candidates}, input_type={exp.edited_input_type}"
                    )
                    continue

                if (
                    strict_row_match
                    and selector_model
                    and resolved_model != selector_model
                ):
                    missing_rows.append(
                        f"{row.get('question_id', 'unknown')}: model mismatch "
                        f"key={key}, requested={selector_model}, resolved={resolved_model}"
                    )
                    continue

                if needs_edited_image and not edited_image_path.exists():
                    missing_rows.append(
                        f"{row.get('question_id', 'unknown')}: edited image not found -> {edited_image_path}"
                    )
                    continue

                if needs_tikz and not edited_tikz_path.exists():
                    missing_rows.append(
                        f"{row.get('question_id', 'unknown')}: edited tikz not found -> {edited_tikz_path}"
                    )
                    continue

                org_tikz_code = ""
                edited_tikz_code = ""
                if needs_tikz:
                    if org_tikz_code_path.exists():
                        org_tikz_code = org_tikz_code_path.read_text(encoding="utf-8")
                    edited_tikz_code = edited_tikz_path.read_text(encoding="utf-8")

                seed = row.get("question_id") or row.get("row_id") or key
                custom_id = _make_custom_id(seed, exp)
                _org = org_image_path
                _vi = vi_image_path
                _edited = edited_image_path
                json_line = _build_json_line(
                    exp,
                    custom_id,
                    _org,
                    _vi,
                    org_tikz_code,
                    _edited,
                    edited_tikz_code,
                )

                if preview_first:
                    return 0, json_line

                f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                count += 1

            if strict_row_match and missing_rows:
                details = "\n".join(missing_rows[:20])
                raise ValueError(
                    f"strict_row_match failed: generated={count}, expected={len(rows)}\n{details}"
                )

            return count, preview_obj

        # allowed_keys / limit mode (without AMT CSV)
        with open(exp.metadata_path, "r", encoding="utf-8") as f_meta:
            for line in f_meta:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record[metadata_fields["key"]]
                if allowed_keys is not None and key not in allowed_keys:
                    continue

                org_image_path = Path(record[metadata_fields["org_image_path"]])
                vi_image_path = Path(
                    record[metadata_fields["visual_instruction_image_path"]]
                )
                org_tikz_code_path = Path(record[metadata_fields["org_tikz_code_path"]])

                resolved_model, edited_image_path, edited_tikz_path = (
                    _resolve_edited_paths(
                        record,
                        metadata_fields,
                        exp.edited_input_type,
                        [exp.edited_model_type],
                    )
                )
                if resolved_model is None:
                    continue

                if needs_edited_image and not edited_image_path.exists():
                    continue
                if needs_tikz and not edited_tikz_path.exists():
                    continue

                org_tikz_code = ""
                edited_tikz_code = ""
                if needs_tikz:
                    if org_tikz_code_path.exists():
                        org_tikz_code = org_tikz_code_path.read_text(encoding="utf-8")
                    edited_tikz_code = edited_tikz_path.read_text(encoding="utf-8")

                custom_id = _make_custom_id(key, exp)
                _org = org_image_path
                _vi = vi_image_path
                _edited = edited_image_path
                json_line = _build_json_line(
                    exp,
                    custom_id,
                    _org,
                    _vi,
                    org_tikz_code,
                    _edited,
                    edited_tikz_code,
                )

                if preview_first:
                    return 0, json_line

                f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break

    return count, preview_obj
