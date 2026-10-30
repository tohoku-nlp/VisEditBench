"""Send requests to the Batch API and retrieve results."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.batch_run import run_batch_claude, run_batch_gemini, run_batch_openai  # noqa: E402
from lib.config import load_config  # noqa: E402

log_file = Path("evaluation_code-level/batch_api/log/evaluation_step_run_batch.log")
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


def _split_jsonl(input_path: Path, num_splits: int, work_dir: Path) -> list[Path]:
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Input JSONL is empty: {input_path}")

    n = min(max(1, num_splits), len(lines))
    chunk_size = (len(lines) + n - 1) // n
    work_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Path] = []
    for idx in range(n):
        start = idx * chunk_size
        end = min((idx + 1) * chunk_size, len(lines))
        if start >= end:
            continue
        chunk_path = work_dir / f"input_part_{idx + 1:02d}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.writelines(lines[start:end])
        chunks.append(chunk_path)
    return chunks


def _merge_jsonl(parts: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as dst:
        for part in parts:
            with open(part, "r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        dst.write(line)


def _run_one_split(exp, input_path: Path, output_path: Path, poll_interval: int) -> None:
    if exp.judge_provider == "openai":
        run_batch_openai(input_path, output_path, exp.name, poll_interval)
    elif exp.judge_provider == "google":
        run_batch_gemini(input_path, output_path, exp.judge_model, poll_interval)
    elif exp.judge_provider == "anthropic":
        run_batch_claude(input_path, output_path, poll_interval)
    else:
        raise ValueError(f"Unsupported judge_provider: {exp.judge_provider}")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(description="Send requests to the evaluation Batch API")
    parser.add_argument("--config", required=True, help="Path to experiment configuration (TOML)")
    parser.add_argument("--config-name", required=True, help="Experiment name to run")
    parser.add_argument(
        "--batch-output-path",
        default=None,
        help="Save path for Batch results (default: runs/<name>/output_<name>.jsonl)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Polling interval (seconds)",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=3,
        help="Number of splits for input JSONL (default: 3)",
    )

    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    _, configs = load_config(config_path)

    selected = [e for e in configs if e.name == args.config_name]
    if not selected:
        raise ValueError(f"Config not found: {args.config_name}")
    exp = selected[0]

    if args.batch_output_path:
        output_path = Path(args.batch_output_path).expanduser()
    else:
        output_path = exp.save_path.parent / f"output_{exp.name}.jsonl"

    logger.info(f"ConfigItem: {exp.name}")
    logger.info(f"  Provider: {exp.judge_provider}, Model: {exp.judge_model}")
    logger.info(f"  Input:  {exp.save_path}")
    logger.info(f"  Output: {output_path}")

    split_dir = output_path.parent / f".split_{exp.name}"
    input_parts = _split_jsonl(exp.save_path, args.num_splits, split_dir)
    output_parts: list[Path] = []

    logger.info(f"Split execution: {len(input_parts)} part(s)")
    for idx, input_part in enumerate(input_parts, 1):
        output_part = split_dir / f"output_part_{idx:02d}.jsonl"
        output_parts.append(output_part)
        logger.info(f"  Running part {idx}/{len(input_parts)}: {input_part}")
        _run_one_split(exp, input_part, output_part, args.poll_interval)

    _merge_jsonl(output_parts, output_path)

    logger.info(f"Batch completed -> {output_path}")


if __name__ == "__main__":
    main()
