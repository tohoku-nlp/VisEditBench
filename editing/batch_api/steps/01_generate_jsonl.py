from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.config import load_config  # noqa: E402
from lib.generate import generate_jsonl_for_experiment  # noqa: E402

log_file = Path("editing/batch_api/log/step_generate_jsonl.log")
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
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(
        description="Generate input JSONL for the Batch API."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config TOML",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Experiment name to run (all experiments if omitted)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of metadata rows to process (all rows if omitted)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Print preview of the first request without saving to file",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    metadata_fields, configs = load_config(config_path)

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
        )

        if args.preview and preview_obj is not None:
            logger.info("Preview (first request):")
            print(json.dumps(preview_obj, indent=2, ensure_ascii=False))
        else:
            logger.info(f"Generated {count} lines -> {exp.save_path}")


if __name__ == "__main__":
    main()
