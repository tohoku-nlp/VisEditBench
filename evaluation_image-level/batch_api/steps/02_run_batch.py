"""Submit Batch API requests and retrieve results."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from lib.batch_run import run_batch_claude, run_batch_gemini, run_batch_openai  # noqa: E402
from lib.config import load_config  # noqa: E402
from lib.extract import extract_scores_from_record  # noqa: E402

log_file = Path("evaluation_image-level/batch_api/log/evaluation_step_run_batch.log")
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
CLAUDE_IMAGE_MAX_EDGE = 8000


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


def _collect_failed_custom_ids_claude(output_jsonl: Path) -> set[str]:
    failed: set[str] = set()
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id, scores = extract_scores_from_record(record, "anthropic")
            if custom_id and scores is None:
                failed.add(custom_id)
    return failed


def _resize_base64_png_if_needed(data_b64: str) -> str:
    try:
        from PIL import Image  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Pillow is required for Claude retry resizing. Install with: pip install pillow"
        ) from e

    raw = base64.b64decode(data_b64)
    with Image.open(io.BytesIO(raw)) as img:
        width, height = img.size
        longest = max(width, height)
        if longest <= CLAUDE_IMAGE_MAX_EDGE:
            return data_b64
        scale = CLAUDE_IMAGE_MAX_EDGE / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        out = io.BytesIO()
        resized.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")


def _build_claude_retry_input_with_resize(
    original_input_jsonl: Path, failed_ids: set[str], retry_input_jsonl: Path
) -> int:
    count = 0
    retry_input_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(original_input_jsonl, "r", encoding="utf-8") as src, open(
        retry_input_jsonl, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            req = json.loads(line)
            cid = req.get("custom_id")
            if cid not in failed_ids:
                continue

            messages = req.get("params", {}).get("messages", [])
            for msg in messages:
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "image":
                        continue
                    source = item.get("source", {})
                    if not isinstance(source, dict):
                        continue
                    if source.get("type") != "base64":
                        continue
                    data_b64 = source.get("data")
                    if not isinstance(data_b64, str) or not data_b64:
                        continue
                    source["data"] = _resize_base64_png_if_needed(data_b64)

            dst.write(json.dumps(req, ensure_ascii=False) + "\n")
            count += 1
    return count


def _merge_with_override(base_output: Path, retry_output: Path, final_output: Path) -> None:
    merged: dict[str, dict] = {}
    order: list[str] = []

    def ingest(path: Path, override: bool) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec.get("custom_id")
                if not cid:
                    continue
                if cid not in merged:
                    order.append(cid)
                    merged[cid] = rec
                elif override:
                    merged[cid] = rec

    ingest(base_output, override=False)
    ingest(retry_output, override=True)

    final_output.parent.mkdir(parents=True, exist_ok=True)
    with open(final_output, "w", encoding="utf-8") as f:
        for cid in order:
            f.write(json.dumps(merged[cid], ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    parser = argparse.ArgumentParser(description="Submit evaluation requests to Batch API.")
    parser.add_argument("--config", required=True, help="Path to experiment config (TOML).")
    parser.add_argument("--config-name", required=True, help="Experiment name to run.")
    parser.add_argument(
        "--batch-output-path",
        default=None,
        help="Output path for batch results (default: runs/<name>/output_<name>.jsonl).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=120,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=3,
        help="Number of splits for input JSONL (default: 3).",
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

    if exp.judge_provider == "anthropic":
        failed_ids = _collect_failed_custom_ids_claude(output_path)
        if failed_ids:
            logger.info(
                f"Claude retry: {len(failed_ids)} failed item(s) will be retried with <=8000px resize."
            )
            retry_input = split_dir / "retry_input_resized.jsonl"
            retry_count = _build_claude_retry_input_with_resize(
                exp.save_path, failed_ids, retry_input
            )
            if retry_count > 0:
                retry_parts = _split_jsonl(retry_input, args.num_splits, split_dir / "retry_parts")
                retry_outputs: list[Path] = []
                for idx, retry_part in enumerate(retry_parts, 1):
                    out_part = split_dir / f"retry_output_part_{idx:02d}.jsonl"
                    retry_outputs.append(out_part)
                    logger.info(
                        f"  Retry part {idx}/{len(retry_parts)}: {retry_part}"
                    )
                    run_batch_claude(retry_part, out_part, args.poll_interval)
                retry_merged = split_dir / "retry_output_merged.jsonl"
                _merge_jsonl(retry_outputs, retry_merged)
                _merge_with_override(output_path, retry_merged, output_path)
                logger.info("Claude retry merge completed.")

    logger.info(f"Batch completed -> {output_path}")


if __name__ == "__main__":
    main()
