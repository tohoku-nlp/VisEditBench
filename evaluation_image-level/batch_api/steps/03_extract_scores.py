"""Extract scores from batch output JSONL and save them to CSV."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.config import load_config  # noqa: E402
from lib.extract import SCORE_KEYS, extract_scores_from_jsonl  # noqa: E402

log_file = Path("evaluation_image-level/batch_api/log/evaluation_step_extract_scores.log")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract scores from batch output and save CSV.")
    parser.add_argument("--config", required=True, help="Path to experiment config (TOML).")
    parser.add_argument(
        "--config-name",
        default=None,
        help="Experiment name (all if omitted).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: runs/<name>/scores_<name>.csv).",
    )
    parser.add_argument(
        "--batch-output-path",
        default=None,
        help="Batch output JSONL path (default: runs/<name>/output_<name>.jsonl).",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    _, configs = load_config(config_path)

    if args.config_name:
        configs = [c for c in configs if c.name == args.config_name]
        if not configs:
            raise ValueError(f"Config not found: {args.config_name}")

    for exp in configs:
        if args.batch_output_path:
            output_jsonl = Path(args.batch_output_path).expanduser()
        else:
            output_jsonl = exp.save_path.parent / f"output_{exp.name}.jsonl"

        if not output_jsonl.exists():
            logger.warning(f"Output JSONL not found, skipping: {output_jsonl}")
            continue

        if args.output_csv:
            csv_path = Path(args.output_csv).expanduser()
        else:
            csv_path = exp.save_path.parent / f"scores_{exp.name}.csv"

        logger.info(f"Extracting scores: {exp.name}")
        results = extract_scores_from_jsonl(output_jsonl, exp.judge_provider)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["custom_id", *SCORE_KEYS],
            )
            writer.writeheader()
            writer.writerows(results)

        logger.info(f"  Saved {len(results)} rows -> {csv_path}")


if __name__ == "__main__":
    main()
