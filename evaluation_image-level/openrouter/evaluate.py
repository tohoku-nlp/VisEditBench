"""OpenRouter-based image evaluation runner for the main_v2 dataset.

This script sends three images per sample (original, visual-instruction, edited) to
OpenRouter and writes JSONL outputs with parsed scores.

Environment variable:
    OPENROUTER_API_KEY
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_content, strip_thinking  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
log_file = SCRIPT_DIR / "log" / "evaluate_mainv2.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Model setup
MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_METADATA_FIELD = "edited_image_paths"

# Evaluation prompt (aligned with batch_api settings)
SYSTEM_PROMPT = """\
You are an expert judge evaluating TikZ diagram editing quality.

You are given:
1. The ORIGINAL diagram image (before any edits)
2. An annotated version of the original diagram image containing visual edit instructions
3. The EDITED diagram image (after applying the visual edit instructions)

Use the annotated image to understand what changes were requested.
Compare the ORIGINAL and EDITED diagrams to evaluate quality.

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

Important Notes for Evaluation:
- Invalid Edited Output: If the edited output is not a diagram, please rate all three criteria as 1 (Very Poor). Examples include a completely blank (white) image and a text-only image.
- Whitespace/Margins: If the edited diagram is present but surrounded by whitespace or margins, ignore the whitespace and evaluate only the diagram content itself.
- Cropped or Cut-Off Diagrams: If the diagram appears cropped or cut off (missing elements at edges), evaluate this under the Diagram Readability criterion, as it directly affects visibility.

Return ONLY valid JSON (no markdown, no explanation):
{"instruction_adherence": <1-5>, "diagram_readability": <1-5>, "content_preservation": <1-5>}
"""

SCORE_KEYS = ("instruction_adherence", "diagram_readability", "content_preservation")
ROOT = Path(__file__).resolve().parents[1]

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

OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "evaluation_scores",
        "strict": True,
        "schema": EVAL_JSON_SCHEMA,
    },
}


# ---------------------------------------------------------------------------
# GCS URL -> local path mapping
# ---------------------------------------------------------------------------


def _url_to_local(url: str) -> Path:
    """Convert a public GCS URL to a local project path.

    org_images/{f}                          → data/org_images/{f}
    visual_instruction_images_allred/{f}    → data/visual_instruction_images_allred/{f}
    main_v2/{model}/{f}                     → work/editing/visual/w_org_image_w_org_tikz/{model}/images/{f}
    """
    import re as _re

    url = url.replace("https://storage.googleapis.com/diagram_vis_edit_eval/", "")
    if url.startswith("org_images/"):
        return ROOT / "data" / url
    if url.startswith("visual_instruction_images_allred/"):
        return ROOT / "data" / url
    m = _re.match(r"main_v2/([^/]+)/(.+)", url)
    if m:
        model, fname = m.group(1), m.group(2)
        return (
            ROOT
            / "work/editing/visual/w_org_image_w_org_tikz"
            / model
            / "images"
            / fname
        )
    raise ValueError(f"Unknown GCS URL pattern: {url}")


def _img_block(source: str, use_local: bool) -> dict:
    """Return an OpenAI image block from local image bytes only."""
    if source.startswith(("http://", "https://")):
        local = _url_to_local(source)
    else:
        local = Path(source).expanduser()
    if not local.exists():
        raise FileNotFoundError(f"Local image not found: {local}")
    import base64 as _b64

    data = _b64.b64encode(local.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def _load_rows_from_metadata(
    metadata_path: str,
    limit: int | None,
    edited_model_type: str,
    edited_input_type: str,
    edited_image_paths_field: str = DEFAULT_METADATA_FIELD,
) -> list[dict]:
    """Load evaluation rows from metadata JSONL."""
    rows: list[dict] = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record.get("question_id") or record.get("key") or "").strip()
            if not qid:
                continue

            org = record.get("img_org_url") or record.get("org_image_path")
            vi = record.get("img_visual_url") or record.get("visual_instruction_image_path")
            edited = record.get("img_edited_url")
            if edited is None:
                all_paths = record.get(edited_image_paths_field, {})
                by_input = all_paths.get(edited_input_type, {}) if isinstance(all_paths, dict) else {}
                edited = by_input.get(edited_model_type) if isinstance(by_input, dict) else None

            if not org or not vi or not edited:
                logger.warning(f"[skip] metadata row missing required fields: question_id={qid}")
                continue

            rows.append(
                {
                    "question_id": qid,
                    "img_org_url": str(org).strip(),
                    "img_visual_url": str(vi).strip(),
                    "img_edited_url": str(edited).strip(),
                }
            )
    return rows[:limit] if limit is not None else rows


def _parse_scores(text: str | None) -> dict | None:
    """Extract score JSON from model text output."""
    if not text:
        return None
    cleaned = strip_thinking(text)
    for t in (cleaned, text.strip()):
        # Try after removing markdown code fences
        for candidate in (t, re.sub(r"```json\s*|\s*```", "", t).strip()):
            try:
                obj = json.loads(candidate)
                if all(k in obj for k in SCORE_KEYS):
                    return {k: int(obj[k]) for k in SCORE_KEYS}
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        # Fallback: parse the first object-like block
        m = re.search(r"\{[^{}]+\}", t)
        if m:
            try:
                obj = json.loads(m.group())
                if all(k in obj for k in SCORE_KEYS):
                    return {k: int(obj[k]) for k in SCORE_KEYS}
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
    return None


def _evaluate_one(
    client: OpenAI,
    row: dict,
    model_id: str,
    enable_thinking: bool,
    max_tokens: int,
    use_local: bool = False,
    retry: int = 3,
    wait: float = 5.0,
) -> dict:
    """Evaluate one row and return a result record."""
    qid = row["question_id"]

    kwargs: dict = {
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # "response_format": OPENAI_RESPONSE_FORMAT,
        "response_format": {"type": "json_object"},
    }
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}

    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                    },
                    {
                        "role": "user",
                        "content": [
                            _img_block(row["img_org_url"], use_local),
                            _img_block(row["img_visual_url"], use_local),
                            _img_block(row["img_edited_url"], use_local),
                        ],
                    },
                ],
                **kwargs,
            )
            # Validate response shape
            if not resp or not resp.choices or len(resp.choices) == 0:
                error_info = getattr(resp, "error", None) if resp else None
                raise ValueError(
                    f"Invalid API response: choices is empty or None. "
                    f"Error info: {error_info}"
                )
            choice = resp.choices[0]
            if choice is None:
                error_info = getattr(resp, "error", None) if resp else None
                raise ValueError(
                    f"Invalid API response: choices[0] is None. "
                    f"Error info: {error_info}"
                )
            raw = extract_content(choice)
            scores = _parse_scores(raw)
            if scores:
                logger.info(f"[{qid}] ok: {scores}")
            else:
                logger.warning(f"[{qid}] parse failed: {str(raw)[:80]}")
            return {
                "question_id": qid,
                "model": model_id,
                "enable_thinking": enable_thinking,
                "raw": raw,
                "scores": scores,
                "error": None if scores else "parse_failed",
            }
        except Exception as e:
            logger.exception(
                f"[{qid}] attempt {attempt + 1}/{retry} failed with {type(e).__name__}"
            )
            if attempt < retry - 1:
                time.sleep(wait * (attempt + 1))

    return {
        "question_id": qid,
        "model": model_id,
        "enable_thinking": enable_thinking,
        "raw": None,
        "scores": None,
        "error": f"failed after {retry} attempts",
    }


def _load_failed_qids(jsonl_path: Path) -> set[str]:
    """Return question_ids with missing scores from an output JSONL."""
    failed: set[str] = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not d.get("scores"):
                failed.add(d["question_id"])
    return failed


def _write_score_output(output_path: Path, score_output_path: Path) -> None:
    """Rebuild score-only JSONL by removing raw model text."""
    if not output_path.exists():
        return

    score_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, encoding="utf-8") as src, open(
        score_output_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            record.pop("raw", None)
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"score JSONL written -> {score_output_path}")


def run_evaluation(
    output_path: Path,
    metadata_path: str,
    edited_model_type: str,
    model_id: str = MODEL_ID,
    score_output_path: Path | None = None,
    edited_input_type: str = "w_org_image_w_org_tikz",
    edited_image_paths_field: str = DEFAULT_METADATA_FIELD,
    enable_thinking: bool = False,
    concurrency: int = 8,
    max_tokens: int = 8192,
    limit: int | None = None,
    use_local: bool = False,
    retry_failed_from: Path | None = None,
) -> None:
    """Run parallel evaluation and append results to output_path.

    Args:
        retry_failed_from: If set, evaluate only failed question_ids from that JSONL.
        use_local: If True, load local files and send base64 data URLs.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set")

    if score_output_path is None:
        score_output_path = output_path.with_name("score.jsonl")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    rows = _load_rows_from_metadata(
        metadata_path=metadata_path,
        limit=limit,
        edited_model_type=edited_model_type,
        edited_input_type=edited_input_type,
        edited_image_paths_field=edited_image_paths_field,
    )

    # --retry-failed: keep only failed keys from a previous run
    if retry_failed_from is not None:
        failed_qids = _load_failed_qids(retry_failed_from)
        rows = [r for r in rows if r["question_id"] in failed_qids]
        logger.info(f"retry_failed_from: {len(rows)} rows selected")

    # resume: skip already successful question_ids
    done: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("scores"):
                    done.add(d["question_id"])
        logger.info(f"Skipping {len(done)} already successful rows")

    rows = [r for r in rows if r["question_id"] not in done]
    if not rows:
        logger.info("All rows are already completed")
        _write_score_output(output_path, score_output_path)
        return

    logger.info(
        f"Model={model_id}, thinking={enable_thinking}, use_local={use_local}, "
        f"rows={len(rows)}, concurrency={concurrency}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = [{}] * len(rows)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(
                _evaluate_one, client, row, model_id, enable_thinking, max_tokens, use_local
            ): i
            for i, row in enumerate(rows)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == len(rows):
                logger.info(f"Progress: {completed}/{len(rows)}")

    with open(output_path, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok = [r for r in results if r.get("scores")]
    ng = [r for r in results if not r.get("scores")]
    logger.info(f"Done: success={len(ok)}, failed={len(ng)} -> {output_path}")
    if ng:
        logger.warning(f"Failed question_id list: {[r['question_id'] for r in ng]}")

    _write_score_output(output_path, score_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run image-based evaluation through OpenRouter"
    )
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--score_output",
        default=None,
        help="Optional score-only JSONL path (default: score.jsonl next to output)",
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Metadata JSONL path",
    )
    parser.add_argument(
        "--edited-model-type",
        required=True,
        help="Edited model key under edited_image_paths",
    )
    parser.add_argument(
        "--edited-input-type",
        default="w_org_image_w_org_tikz",
        help="Edited input type key under edited_image_paths",
    )
    parser.add_argument(
        "--edited-image-paths-field",
        default=DEFAULT_METADATA_FIELD,
        help="Metadata field name for edited image mapping",
    )
    parser.add_argument(
        "--model",
        default=MODEL_ID,
        help=f"OpenRouter judge model id (default: {MODEL_ID})",
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", help="Enable thinking mode"
    )
    parser.add_argument(
        "--concurrency", type=int, default=8, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens per response",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only first N rows"
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local image files and send as base64 data URLs",
    )
    parser.add_argument(
        "--retry-failed",
        default=None,
        metavar="JSONL",
        help="Re-evaluate only failed question_ids from a previous output JSONL",
    )
    args = parser.parse_args()

    run_evaluation(
        output_path=Path(args.output),
        score_output_path=Path(args.score_output) if args.score_output else None,
        metadata_path=args.metadata_path,
        edited_model_type=args.edited_model_type,
        model_id=args.model,
        edited_input_type=args.edited_input_type,
        edited_image_paths_field=args.edited_image_paths_field,
        enable_thinking=args.enable_thinking,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        limit=args.limit,
        use_local=args.use_local,
        retry_failed_from=Path(args.retry_failed) if args.retry_failed else None,
    )


if __name__ == "__main__":
    main()
