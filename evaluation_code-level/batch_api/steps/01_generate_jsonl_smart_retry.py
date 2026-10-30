"""
Script to generate evaluation JSONL for Gemini using smart retry.

- Model without output JSONL -> Generate all entries
- Model with existing output JSONL -> Generate only failed keys (URL errors, etc.)

Usage (from project root):
    python evaluation_code-level/batch_api/steps/01_generate_jsonl_smart_retry.py \
        --config configs/code_eval.toml

    # Run only a specific experiment
    python evaluation_code-level/batch_api/steps/01_generate_jsonl_smart_retry.py \
        --config configs/code_eval.toml \
        --config-name all_claude_haiku_4_5_gemini31pro_judge
"""

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

log_file = Path("evaluation_code-level/batch_api/log/evaluation_step_generate_jsonl_smart_retry.log")
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


def _strip_key(raw: str) -> str:
    """Remove the suffix (__xxx) from custom_id or key and return the metadata key."""
    return raw.split("__")[0]


def get_failed_keys(output_path: Path) -> set[str]:
    """Return keys (with suffix removed) that failed score retrieval from the output JSONL."""
    succeeded: set[str] = set()
    failed: set[str] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            key = _strip_key(d.get("key", ""))
            if d.get("error"):
                failed.add(key)
                continue
            cands = d.get("response", {}).get("candidates", [])
            if not cands:
                failed.add(key)
                continue
            parts = cands[0].get("content", {}).get("parts", [])
            has_score = False
            for p in parts:
                try:
                    obj = json.loads(p.get("text", ""))
                    if "instruction_adherence" in obj:
                        has_score = True
                        break
                except Exception:
                    pass
            (succeeded if has_score else failed).add(key)
    return failed


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(
        description="Generate Gemini evaluation JSONL using smart retry"
    )
    parser.add_argument("--config", required=True, help="Path to experiment configuration (TOML)")
    parser.add_argument(
        "--config-name", default=None, help="Experiment name to run (all experiments if not specified)"
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser()
    metadata_fields, configs = load_config(config_path)

    if args.config_name:
        configs = [c for c in configs if c.name == args.config_name]
        if not configs:
            raise ValueError(f"Config not found: {args.config_name}")

    for exp in configs:
        output_path = Path(exp.save_path).parent / f"output_{exp.name}.jsonl"

        if not output_path.exists():
            # Generate all entries
            logger.info(f"[FULL] {exp.name} — no output, generating all entries")
            allowed_keys = None
        else:
            # Only failed keys
            failed_keys = get_failed_keys(output_path)
            if not failed_keys:
                logger.info(f"[SKIP] {exp.name} — all entries succeeded, skipping")
                continue
            logger.info(
                f"[RETRY] {exp.name} — regenerating {len(failed_keys)} failed entries only"
            )
            allowed_keys = failed_keys

        count, _ = generate_jsonl_for_experiment(
            exp=exp,
            metadata_fields=metadata_fields,
            limit=None,
            preview_first=False,
            allowed_keys=allowed_keys,
        )
        logger.info(f"  -> {count} entries generated: {exp.save_path}")


if __name__ == "__main__":
    main()
