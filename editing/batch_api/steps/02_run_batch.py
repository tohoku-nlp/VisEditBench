from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.batch_run import (  # noqa: E402
    run_batch_claude,
    run_batch_gemini,
    run_batch_openai,
)
from lib.config import load_config  # noqa: E402

log_file = Path("editing/batch_api/log/step_run_batch.log")
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
        description="Submit requests to the Batch API and retrieve results."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config TOML",
    )
    parser.add_argument(
        "--config-name",
        required=True,
        help="Experiment name to run",
    )
    parser.add_argument(
        "--batch-output-path",
        default=None,
        help="Output path for batch results (default: runs/<name>/output_*.jsonl)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Batch status polling interval in seconds (default: 120)",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    metadata_fields, configs = load_config(config_path)

    selected = [c for c in configs if c.name == args.config_name]
    if not selected:
        raise ValueError(f"Config not found: {args.config_name}")
    cfg = selected[0]

    if args.batch_output_path:
        output_path = Path(args.batch_output_path).expanduser()
    else:
        detail_prefix = f"{cfg.image_detail}_" if cfg.image_detail != "auto" else ""
        output_path = cfg.save_path.parent / f"output_{detail_prefix}{cfg.name}.jsonl"

    logger.info(f"Running batch for config: {cfg.name}")
    logger.info(f"  Provider: {cfg.provider}")
    logger.info(f"  Model: {cfg.model}")
    logger.info(f"  Input: {cfg.save_path}")
    logger.info(f"  Output: {output_path}")

    if cfg.provider == "openai":
        run_batch_openai(
            cfg.save_path,
            output_path,
            cfg.name,
            args.poll_interval,
        )
    elif cfg.provider == "google":
        run_batch_gemini(
            cfg.save_path,
            output_path,
            cfg.model,
            args.poll_interval,
        )
    elif cfg.provider == "anthropic":
        run_batch_claude(
            cfg.save_path,
            output_path,
            args.poll_interval,
        )
    else:
        raise ValueError(f"Unsupported provider: {cfg.provider}")

    logger.info(f"Batch completed. Output saved to: {output_path}")


if __name__ == "__main__":
    main()
