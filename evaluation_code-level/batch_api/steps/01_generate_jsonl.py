from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.config import load_config  # noqa: E402
from lib.generate import generate_jsonl_for_experiment  # noqa: E402

log_file = Path("evaluation_code-level/batch_api/log/evaluation_step_generate_jsonl.log")
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

KEY_PATTERN = re.compile(r"(\d{8}_[A-Z]+_\d+_edit_\d+)")


def load_allowed_keys_from_amt_csv(path: Path) -> set[str]:
    keys: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edited_url = (row.get("img_edited_url") or "").strip()
            match = KEY_PATTERN.search(Path(edited_url).name)
            if match:
                keys.add(match.group(1))
    return keys


def load_row_selectors_from_amt_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            edited_url = (row.get("img_edited_url") or "").strip()
            match = KEY_PATTERN.search(Path(edited_url).name)
            if not match:
                continue
            model = ""
            parts = edited_url.split("/")
            for prefix in ("pilot", "main"):
                if prefix in parts:
                    idx = parts.index(prefix)
                    if idx + 1 < len(parts):
                        model = parts[idx + 1]
                    break
            rows.append(
                {
                    "row_id": str(i),
                    "question_id": (row.get("question_id") or "").strip(),
                    "key": match.group(1),
                    "edited_model": model,
                }
            )
    return rows


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(
        description="Generate evaluation Batch API input JSONL"
    )
    parser.add_argument("--config", required=True, help="Path to experiment configuration (TOML)")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Experiment name to run (all experiments if not specified)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of entries to process from the start of metadata (all if not specified)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview first entry only (does not save to file)",
    )
    parser.add_argument(
        "--amt-csv",
        type=str,
        default=None,
        help="AMT input CSV to extract evaluation target keys from (extracts key from img_edited_url)",
    )
    parser.add_argument(
        "--strict-csv-rows",
        action="store_true",
        help="Require strict row-level matching with CSV (error if any rows are missing)",
    )
    parser.add_argument(
        "--gemini-use-local",
        action="store_true",
        help="Pass Gemini images as local inlineData instead of URLs",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    metadata_fields, configs = load_config(config_path)
    allowed_keys: set[str] | None = None
    row_selectors: list[dict[str, str]] | None = None
    if args.amt_csv:
        csv_path = Path(args.amt_csv).expanduser()
        if args.strict_csv_rows:
            row_selectors = load_row_selectors_from_amt_csv(csv_path)
            logger.info(f"Loaded {len(row_selectors)} rows from {args.amt_csv}")
        else:
            allowed_keys = load_allowed_keys_from_amt_csv(csv_path)
            logger.info(f"Loaded {len(allowed_keys)} keys from {args.amt_csv}")

    if args.config_name:
        configs = [c for c in configs if c.name == args.config_name]
        if not configs:
            raise ValueError(f"Config not found: {args.config_name}")

    for exp in configs:
        logger.info(f"Processing experiment: {exp.name}")
        count, preview_obj = generate_jsonl_for_experiment(
            exp=exp,
            metadata_fields=metadata_fields,
            limit=args.limit,
            preview_first=args.preview,
            allowed_keys=allowed_keys,
            row_selectors=row_selectors,
            strict_row_match=args.strict_csv_rows,
            gemini_use_local=args.gemini_use_local,
        )

        if args.preview and preview_obj is not None:
            logger.info("Preview (first request):")
            print(json.dumps(preview_obj, indent=2, ensure_ascii=False))
        else:
            logger.info(f"Generated {count} lines -> {exp.save_path}")


if __name__ == "__main__":
    main()
