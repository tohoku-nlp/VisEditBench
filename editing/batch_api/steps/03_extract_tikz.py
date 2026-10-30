from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.config import load_config  # noqa: E402
from lib.extract import extract_and_save  # noqa: E402

log_file = Path("editing/batch_api/log/step_extract_tikz.log")
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
        description="Extract TikZ code from Batch API output JSONL and save as .tex files."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config TOML",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="Experiment name to process (all experiments if omitted)",
    )
    parser.add_argument(
        "--batch-output-path",
        default=None,
        help="Input JSONL path for batch results (default: runs/<name>/output_*.jsonl)",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output directory for TikZ files (default: work/editing/visual/<pattern>/<model>/tikz)",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    _, configs = load_config(config_path)

    if args.config_name:
        configs = [c for c in configs if c.name == args.config_name]
        if not configs:
            raise ValueError(f"Config not found: {args.config_name}")

    for exp in configs:
        logger.info(f"Extracting TikZ for experiment: {exp.name}")

        if args.batch_output_path:
            output_jsonl = Path(args.batch_output_path).expanduser()
        else:
            output_jsonl = exp.save_path.parent / f"output_{exp.name}.jsonl"

        if not output_jsonl.exists():
            logger.warning(f"Output JSONL not found: {output_jsonl} — skipping")
            continue

        if args.save_dir:
            save_dir = Path(args.save_dir).expanduser()
        else:
            model_dir = exp.model.split("/")[-1].replace(":", "_")
            pattern_dir = exp.system_instruction_type.replace("visual_instruction_", "")
            save_dir = (
                Path("work")
                / "editing"
                / "visual"
                / pattern_dir
                / model_dir
                / "tikz"
            )

        success, errors = extract_and_save(
            output_jsonl=output_jsonl,
            save_dir=save_dir,
            provider=exp.provider,
            system_instruction_type=exp.system_instruction_type,
        )
        logger.info(
            f"  Saved {success} TikZ files to {save_dir} "
            f"({errors} errors)"
        )


if __name__ == "__main__":
    main()
