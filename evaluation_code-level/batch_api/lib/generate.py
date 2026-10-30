from __future__ import annotations

import hashlib
import json
import io
from pathlib import Path
from typing import Any

from .config import EvalConfig

CLAUDE_MAX_TOKENS = 8192

# ---------------------------------------------------------------------------
# JSON schema for structured output (all providers use 1-5 scale)
# ---------------------------------------------------------------------------

# Schema for TikZ code evaluation
EVAL_JSON_SCHEMA_CODE = {
    "type": "object",
    "properties": {
        "analysis": {
            "type": "object",
            "properties": {
                "reference_intent": {"type": "string"},
                "generated_intent": {"type": "string"},
                "comparison_critique": {"type": "string"},
            },
            "required": ["reference_intent", "generated_intent", "comparison_critique"],
            "additionalProperties": False,
        },
        "metrics": {
            "type": "object",
            "properties": {
                "Instruction Adherence": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                    "additionalProperties": False,
                },
                "Content Preservation": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {"type": "string"},
                    },
                    "required": ["score", "reason"],
                    "additionalProperties": False,
                },
            },
            "required": ["Instruction Adherence", "Content Preservation"],
            "additionalProperties": False,
        },
        "overall_alignment_score": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["analysis", "metrics", "overall_alignment_score"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Common criterion descriptions (used in all 3 system prompts)
# ---------------------------------------------------------------------------

EVALUATE_CODE_SYSTEM_PROMPT = r"""You are an expert LaTeX developer and geometry reasoning specialist.
Your task is to evaluate the quality of a generated TikZ code edit (Gen) by comparing it with a reference edit (Ref) based on the original code (Org).

You are given:
1. Original TikZ (Org)
2. Reference TikZ (Ref)
3. Generated TikZ (Gen)

Evaluation Steps
   1.  Analysis of Reference Intent (Org vs Ref): Identify all intended visual modifications (geometric, stylistic, and structural) required to reach the ground truth. This serves as the "Requirement List."
   2.  Analysis of Generated Intent (Org vs Gen): Identify all visual modifications actually performed by the model.
   3.  Comparative Mapping: Cross-reference the findings. Determine which requirements were met, which were missed, and if any unintended visual changes occurred.
   4.  Scoring: Assign integer scores (1-5) based on the criteria below, focusing on visual and semantic impact.

Metric A: Instruction Adherence (1-5)
   This metric measures how much of the "intended visual edit" from Step 1 was successfully captured in Gen.
   5 (Perfect): All requested visual changes are correctly applied according to the instructions.
   4 (Good): Most requested visual changes are correctly applied, with only minor omissions or inaccuracies.
   3 (Fair): Some requested visual changes are correctly applied, but several important ones are missing or inaccurate.
   2 (Poor): Only a few requested visual changes are correctly applied, and many are missing or inaccurate.
   1 (Very Poor): Few or none of the requested visual changes are correctly applied.

Metric B: Content Preservation (1-5)
   This metric measures if Gen preserves the visual integrity of the original parts that were NOT supposed to change.
   5 (Perfect): No unintended visual changes are present; all non-requested graphical elements remain identical to Org.
   4 (Good): Very few unintended visual changes are present, and they have minimal impact on the overall appearance.
   3 (Fair): Some unintended visual changes are present (e.g., a node moved slightly or a color changed) and somewhat affect the diagram.
   2 (Poor): Many unintended visual changes are introduced, significantly altering parts of the diagram that should have been preserved.
   1 (Very Poor): Extensive unintended visual changes are present, and the original visual content is largely lost or unrecognizable.

Important Notes for Evaluation:
- Visual Focus: Ignore trivial code-level differences (e.g., white spaces, variable names, or different syntax for the same visual result) unless they break the compilation.
- Output Format: Output must be a single, valid JSON object. Do not include any conversational text.
- Score Integrity: Scores must be integers (1, 2, 3, 4, or 5).

Return ONLY valid JSON (no markdown, no explanation):
{"analysis":{"reference_intent":"Summary of intentional visual changes from Org to Ref","generated_intent":"Summary of visual changes actually performed from Org to Gen","comparison_critique":"Focus on visual alignment: What was missed or visually corrupted in Gen?"},"metrics":{"Instruction Adherence":{"score":<1-5>,"reason":"Justification based on visual completeness criteria"},"Content Preservation":{"score":<1-5>,"reason":"Justification based on visual preservation of original parts"}},"overall_alignment_score":<1-5>}
"""

CONTENT_PROMPT = r"""
Original TikZ (Org):
{original_tikz_code}

Reference TikZ (Ref):
{reference_tikz_code}

Generated TikZ (Gen):
{generated_tikz_code}
"""


# ---------------------------------------------------------------------------
# OpenAI chat/completions request builders
# ---------------------------------------------------------------------------

_OPENAI_RESPONSE_FORMAT_CODE = {
    "type": "json_schema",
    "json_schema": {
        "name": "code_evaluation",
        "strict": True,
        "schema": EVAL_JSON_SCHEMA_CODE,
    },
}


def _openai_text(text: str) -> dict:
    return {"type": "text", "text": text}


def build_request_openai(
    custom_id: str,
    model: str,
    system_prompt: str,
    user_content: list[dict],
    temperature: float = 0,
) -> dict:
    """Build a batch request in OpenAI chat completions format."""
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
            # response_format may be overridden by the caller
            "response_format": _OPENAI_RESPONSE_FORMAT_CODE,
            "temperature": temperature,
        },
    }


# ---------------------------------------------------------------------------
# Gemini request builders
# ---------------------------------------------------------------------------

_GEMINI_GENERATION_CONFIG = {
    "responseMimeType": "application/json",
    "responseJsonSchema": EVAL_JSON_SCHEMA_CODE,
}


def _gemini_text(text: str) -> dict:
    return {"text": text}


def _build_gemini_request(
    custom_id: str,
    system_prompt: str,
    parts: list[dict],
    temperature: float = 0,
) -> dict:
    """Build a request in Gemini batch format."""
    generation_config = {**_GEMINI_GENERATION_CONFIG, "temperature": temperature}
    return {
        "key": custom_id,
        "request": {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": parts}],
            "generationConfig": generation_config,
        },
    }


# ---------------------------------------------------------------------------
# Claude request builders
# ---------------------------------------------------------------------------


def _claude_text(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_claude_request(
    custom_id: str,
    model: str,
    system_prompt: str,
    content: list[dict],
    temperature: float = 0,
) -> dict:
    """Build a request in Claude batch format."""
    tool = {
        "name": "evaluation_scores",
        "description": "Return evaluation scores as structured JSON.",
        "input_schema": EVAL_JSON_SCHEMA_CODE,
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


# ---------------------------------------------------------------------------
# TikZ Code Evaluation request builders
# ---------------------------------------------------------------------------


def build_request_openai_code_evaluation(
    custom_id: str,
    model: str,
    org_tikz_code: str,
    reference_tikz_code: str,
    generated_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build an OpenAI TikZ code evaluation request."""
    user_content_text = CONTENT_PROMPT.format(
        original_tikz_code=org_tikz_code,
        reference_tikz_code=reference_tikz_code,
        generated_tikz_code=generated_tikz_code,
    )
    user_content = [_openai_text(user_content_text)]
    return build_request_openai(
        custom_id, model, system_prompt, user_content, temperature
    )


def build_request_gemini_code_evaluation(
    custom_id: str,
    org_tikz_code: str,
    reference_tikz_code: str,
    generated_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Gemini TikZ code evaluation request."""
    user_content_text = CONTENT_PROMPT.format(
        original_tikz_code=org_tikz_code,
        reference_tikz_code=reference_tikz_code,
        generated_tikz_code=generated_tikz_code,
    )
    parts = [_gemini_text(user_content_text)]
    return _build_gemini_request(custom_id, system_prompt, parts, temperature)


def build_request_claude_code_evaluation(
    custom_id: str,
    model: str,
    org_tikz_code: str,
    reference_tikz_code: str,
    generated_tikz_code: str,
    system_prompt: str = "",
    temperature: float = 0,
) -> dict:
    """Build a Claude TikZ code evaluation request."""
    user_content_text = CONTENT_PROMPT.format(
        original_tikz_code=org_tikz_code,
        reference_tikz_code=reference_tikz_code,
        generated_tikz_code=generated_tikz_code,
    )
    content = [_claude_text(user_content_text)]
    return _build_claude_request(custom_id, model, system_prompt, content, temperature)


# ---------------------------------------------------------------------------
# JSONL generation
# ---------------------------------------------------------------------------


def _make_custom_id(seed: str, exp: EvalConfig) -> str:
    """Generate a custom ID for request identification. Shortened with hash due to Claude's 64-character limit."""
    raw = f"{seed}__code_evaluation"
    if exp.judge_provider == "anthropic" and len(raw) > 64:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        prefix_len = 64 - 1 - len(digest)
        raw = f"{raw[:prefix_len]}_{digest}"
    return raw


def _build_json_line(
    exp: EvalConfig,
    custom_id: str,
    org_image_path: Path | None,
    vi_image_path: Path | None,
    org_tikz_code: str,
    edited_image_path: Path | None,
    edited_tikz_code: str,
    reference_tikz_code: str = "",
) -> dict[str, Any]:
    """Build a request object based on judge_provider."""
    temperature = getattr(exp, "temperature", 0.0)
    system_prompt = EVALUATE_CODE_SYSTEM_PROMPT
    if exp.judge_provider == "openai":
        return build_request_openai_code_evaluation(
            custom_id,
            exp.judge_model,
            org_tikz_code,
            reference_tikz_code,
            edited_tikz_code,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    elif exp.judge_provider == "google":
        return build_request_gemini_code_evaluation(
            custom_id,
            org_tikz_code,
            reference_tikz_code,
            edited_tikz_code,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    elif exp.judge_provider == "anthropic":
        return build_request_claude_code_evaluation(
            custom_id,
            exp.judge_model,
            org_tikz_code,
            reference_tikz_code,
            edited_tikz_code,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported judge_provider: {exp.judge_provider}")


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
    """Generate evaluation input JSONL based on experiment configuration.

    Returns:
        (number of generated entries, first request object for preview)
    """
    mode = "a" if exp.append else "w"
    save_path = Path(exp.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    preview_obj: dict[str, Any] | None = None

    def _read_code_text(
        record: dict[str, Any],
        field_candidates: list[str],
        row_label: str,
        code_label: str,
    ) -> str:
        """Resolve the TikZ code path from candidate fields and read the content."""
        for field_key in field_candidates:
            actual_key = metadata_fields.get(field_key, field_key)
            path_value = record.get(actual_key)
            if not path_value:
                continue
            code_path = Path(path_value)
            if not code_path.exists():
                raise FileNotFoundError(
                    f"{row_label}: {code_label} not found -> {code_path}"
                )
            if code_path.is_dir():
                raise IsADirectoryError(
                    f"{row_label}: {code_label} is a directory -> {code_path}"
                )
            return code_path.read_text(encoding="utf-8")
        tried = ", ".join(field_candidates)
        raise ValueError(
            f"{row_label}: missing required {code_label} path; tried: {tried}"
        )

    def _build_code_request(record: dict[str, Any], seed: str) -> dict[str, Any]:
        """Build a code evaluation request for a single record."""
        org_tikz_code = _read_code_text(
            record,
            ["org_tikz_code_path"],
            seed,
            "org tikz code",
        )
        reference_tikz_code = _read_code_text(
            record,
            ["reference_tikz_code_path", "reference_tikz_path"],
            seed,
            "reference tikz code",
        )
        generated_tikz_code = ""
        generated_path_resolved = False
        generated_candidates = [
            "edited_tikz_code_path",
            "generated_tikz_code_path",
            "edited_code_path",
            "generated_code_path",
        ]
        try:
            generated_tikz_code = _read_code_text(
                record,
                generated_candidates,
                seed,
                "generated tikz code",
            )
            generated_path_resolved = True
        except (ValueError, FileNotFoundError, IsADirectoryError):
            generated_path_resolved = False

        # Fallback: resolve from edited_tikz_code_paths[input_type][model]
        if not generated_path_resolved:
            nested = record.get("edited_tikz_code_paths", {})
            input_type = getattr(exp, "edited_input_type", "") or ""
            model = getattr(exp, "edited_model_type", "") or ""
            path_str = nested.get(input_type, {}).get(model)
            if not path_str:
                raise ValueError(
                    f"{seed}: generated tikz code not found; "
                    f"tried fields={generated_candidates}, "
                    f"and edited_tikz_code_paths['{input_type}']['{model}']"
                )
            code_path = Path(path_str)
            if not code_path.exists():
                raise FileNotFoundError(
                    f"{seed}: generated tikz code not found -> {code_path}"
                )
            if code_path.is_dir():
                raise IsADirectoryError(
                    f"{seed}: generated tikz code is a directory -> {code_path}"
                )
            generated_tikz_code = code_path.read_text(encoding="utf-8")

        custom_id = _make_custom_id(seed, exp)
        return _build_json_line(
            exp,
            custom_id,
            None,
            None,
            org_tikz_code,
            None,
            generated_tikz_code,
            reference_tikz_code=reference_tikz_code,
        )

    with (io.StringIO() if preview_first else open(save_path, mode, encoding="utf-8")) as f_out:
        if row_selectors is not None:
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

                seed = row.get("question_id") or row.get("row_id") or key
                json_line = _build_code_request(record, seed)

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

        with open(exp.metadata_path, "r", encoding="utf-8") as f_meta:
            for line in f_meta:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = record[metadata_fields["key"]]
                if allowed_keys is not None and key not in allowed_keys:
                    continue

                # attachで確認済みの未生成レコードは、コード評価対象から除外する。
                status = record.get("generation_status", {}).get(exp.edited_input_type, {}).get(exp.edited_model_type)
                if status == "missing_generation":
                    print(f"Skipping missing generated code: {key}")
                    continue

                # If question_id exists, prioritize it in custom_id to ensure 1 request = 1 ID
                # (avoids collisions in AMT-style metadata where the same key may appear in multiple rows)
                seed = record.get("question_id") or key
                json_line = _build_code_request(record, seed)

                if preview_first:
                    return 0, json_line

                f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break

    return count, preview_obj
