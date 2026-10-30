"""
Script to generate evaluation Batch API input JSONL directly from AMT input CSV (with GCS URLs).

Target CSV columns:
  question_id, img_org_url, img_visual_url, img_edited_url

No metadata JSONL or URL mapping JSON required.

Usage example (from project root):
  python evaluation_code-level/batch_api/steps/01_generate_jsonl_from_url_csv.py \\
      --config configs/code_eval.toml \\
      --amt-csv amt/amt_input_main_v2_1.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.config import load_config  # noqa: E402
from lib.generate import (  # noqa: E402
    JUDGE_PROMPTS,
    JUDGE_PROMPTS_NO_NOTES,
    _build_claude_request,
    _build_gemini_request,
    _claude_img_url,
    _encode_b64,
    _gemini_img,
    _gemini_img_url,
    _make_custom_id,
    _openai_img,
    build_request_openai,
)

log_file = Path("evaluation_code-level/batch_api/log/evaluation_step_generate_jsonl_url_csv.log")
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


_GCS_PREFIX = "https://storage.googleapis.com/diagram_vis_edit_eval/"
_GCS_ORG_PREFIX = _GCS_PREFIX + "org_images/"
_GCS_VI_PREFIX = _GCS_PREFIX + "visual_instruction_images_allred/"
_GCS_MAIN_V2_PREFIX = _GCS_PREFIX + "main_v2/"


def _gcs_url_to_local(url: str) -> Path:
    """Convert a GCS URL to a local file path.

    Supported patterns:
    - org_images/{fname}           -> data/org_images/{fname}
    - visual_instruction_*/{fname} -> data/visual_instruction_images_allred/{fname}
    - main_v2/{model}/{fname}      -> work/editing/visual/w_org_image_w_org_tikz/{model}/images/{fname}
    """
    if url.startswith(_GCS_ORG_PREFIX):
        fname = url[len(_GCS_ORG_PREFIX):]
        return Path("data/org_images") / fname
    if url.startswith(_GCS_PREFIX + "visual_instruction_"):
        fname = url.split("/")[-1]
        return Path("data/visual_instruction_images_allred") / fname
    if url.startswith(_GCS_MAIN_V2_PREFIX):
        rest = url[len(_GCS_MAIN_V2_PREFIX):]
        model, fname = rest.split("/", 1)
        return Path("work/editing/visual/w_org_image_w_org_tikz") / model / "images" / fname
    raise ValueError(f"Unknown GCS URL pattern: {url}")


def _gemini_img_local(url: str) -> dict:
    """Convert a GCS URL to a local path and return as base64 inlineData."""
    local_path = _gcs_url_to_local(url)
    return _gemini_img(local_path)


def _load_url_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load an AMT URL CSV and return a list of rows."""
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "question_id": (row.get("question_id") or "").strip(),
                    "img_org_url": (row.get("img_org_url") or "").strip(),
                    "img_visual_url": (row.get("img_visual_url") or "").strip(),
                    "img_edited_url": (row.get("img_edited_url") or "").strip(),
                }
            )
    return rows


def generate_jsonl_from_url_csv(
    exp,
    rows: list[dict[str, str]],
    preview_first: bool = False,
    use_local: bool = False,
) -> tuple[int, dict | None]:
    """Generate evaluation JSONL from URL CSV rows.

    Args:
        exp: EvalConfig instance
        rows: List of rows from the URL CSV
        preview_first: If True, return only the first entry (does not save to file)
        use_local: If True, convert GCS URLs to local paths and pass as base64 encoded

    Returns:
        (number of generated entries, request object for preview)
    """
    if exp.judge_input_type not in JUDGE_PROMPTS:
        raise ValueError(f"Unsupported judge_input_type: {exp.judge_input_type}")

    if exp.judge_provider not in ("anthropic", "google", "openai"):
        raise ValueError(f"Unsupported judge_provider: {exp.judge_provider}")

    # notes_variant is deprecated; switch using include_important_notes
    prompts = JUDGE_PROMPTS if exp.include_important_notes else JUDGE_PROMPTS_NO_NOTES
    system_prompt = prompts[exp.judge_input_type]

    save_path = Path(exp.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(save_path, "w", encoding="utf-8") as f_out:
        for row in rows:
            qid = row["question_id"]
            org_url = row["img_org_url"]
            vi_url = row["img_visual_url"]
            edited_url = row["img_edited_url"]

            if not (qid and org_url and vi_url and edited_url):
                logger.warning(f"Skip incomplete row: {row}")
                continue

            custom_id = _make_custom_id(qid, exp)

            if exp.judge_provider == "anthropic":
                content = [
                    _claude_img_url(org_url),
                    _claude_img_url(vi_url),
                    _claude_img_url(edited_url),
                ]
                request = _build_claude_request(
                    custom_id=custom_id,
                    model=exp.judge_model,
                    system_prompt=system_prompt,
                    content=content,
                )
            elif exp.judge_provider == "openai":
                detail = getattr(exp, "image_detail", "auto")
                user_content = [
                    _openai_img(org_url, detail),
                    _openai_img(vi_url, detail),
                    _openai_img(edited_url, detail),
                ]
                request = build_request_openai(
                    custom_id=custom_id,
                    model=exp.judge_model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                )
            else:  # google
                _img = _gemini_img_local if use_local else _gemini_img_url
                parts = [
                    _img(org_url),
                    _img(vi_url),
                    _img(edited_url),
                ]
                request = _build_gemini_request(
                    custom_id=custom_id,
                    system_prompt=system_prompt,
                    parts=parts,
                    temperature=getattr(exp, "temperature", 0.0),
                )

            if preview_first:
                return 0, request

            f_out.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1

    return count, None


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(
        description="Generate evaluation Batch API input JSONL from AMT URL CSV"
    )
    parser.add_argument("--config", required=True, help="Path to experiment configuration (TOML)")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Experiment name to run (all experiments if not specified)",
    )
    parser.add_argument(
        "--amt-csv",
        required=True,
        help="AMT input CSV (question_id / img_org_url / img_visual_url / img_edited_url)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview first entry only (does not save to file)",
    )
    parser.add_argument(
        "--use-local",
        action="store_true",
        help="Convert GCS URLs to local paths and pass images as base64 encoded (for Gemini)",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    _, configs = load_config(config_path)
    csv_path = Path(args.amt_csv).expanduser()
    rows = _load_url_rows(csv_path)
    logger.info(f"Loaded {len(rows)} rows from {csv_path}")

    if args.config_name:
        configs = [c for c in configs if c.name == args.config_name]
        if not configs:
            raise ValueError(f"Config not found: {args.config_name}")

    for exp in configs:
        logger.info(f"Processing experiment: {exp.name}")
        logger.info(f"  include_important_notes: {exp.include_important_notes}")
        logger.info(f"  use_local: {args.use_local}")
        count, preview_obj = generate_jsonl_from_url_csv(
            exp=exp,
            rows=rows,
            preview_first=args.preview,
            use_local=args.use_local,
        )

        if args.preview and preview_obj is not None:
            logger.info("Preview (first request):")
            print(json.dumps(preview_obj, indent=2, ensure_ascii=False))
        else:
            logger.info(f"Generated {count} lines -> {exp.save_path}")


if __name__ == "__main__":
    main()
