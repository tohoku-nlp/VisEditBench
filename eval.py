#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_step(script: Path, args: list[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def add_common_batch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to experiment config TOML")
    parser.add_argument("--config-name", default=None, help="Experiment name in TOML")


def handle_code_batch(args: argparse.Namespace) -> None:
    base = REPO_ROOT / "evaluation_code-level" / "batch_api" / "steps"
    if args.mode in ("generate", "pipeline"):
        step_args = ["--config", args.config]
        if args.config_name:
            step_args += ["--config-name", args.config_name]
        if args.limit is not None:
            step_args += ["--limit", str(args.limit)]
        if args.preview:
            step_args += ["--preview"]
        if args.amt_csv:
            step_args += ["--amt-csv", args.amt_csv]
        if args.strict_csv_rows:
            step_args += ["--strict-csv-rows"]
        if args.gemini_use_local:
            step_args += ["--gemini-use-local"]
        run_step(base / "01_generate_jsonl.py", step_args)

    if args.mode in ("run", "pipeline"):
        if not args.config_name:
            raise ValueError("--config-name is required for run/pipeline in code-batch mode")
        step_args = ["--config", args.config, "--config-name", args.config_name]
        if args.batch_output_path:
            step_args += ["--batch-output-path", args.batch_output_path]
        if args.poll_interval is not None:
            step_args += ["--poll-interval", str(args.poll_interval)]
        if args.num_splits is not None:
            step_args += ["--num-splits", str(args.num_splits)]
        run_step(base / "02_run_batch.py", step_args)

    if args.mode in ("extract", "pipeline"):
        step_args = ["--config", args.config]
        if args.config_name:
            step_args += ["--config-name", args.config_name]
        if args.output_csv:
            step_args += ["--output-csv", args.output_csv]
        if args.batch_output_path:
            step_args += ["--batch-output-path", args.batch_output_path]
        run_step(base / "03_extract_scores.py", step_args)


def handle_image_batch(args: argparse.Namespace) -> None:
    base = REPO_ROOT / "evaluation_image-level" / "batch_api" / "steps"
    if args.mode in ("generate", "pipeline"):
        step_args = ["--config", args.config]
        if args.config_name:
            step_args += ["--config-name", args.config_name]
        if args.limit is not None:
            step_args += ["--limit", str(args.limit)]
        if args.preview:
            step_args += ["--preview"]
        if args.amt_csv:
            step_args += ["--amt-csv", args.amt_csv]
        if args.strict_csv_rows:
            step_args += ["--strict-csv-rows"]
        if args.gemini_use_local:
            step_args += ["--gemini-use-local"]
        run_step(base / "01_generate_jsonl.py", step_args)

    if args.mode in ("run", "pipeline"):
        if not args.config_name:
            raise ValueError("--config-name is required for run/pipeline in image-batch mode")
        step_args = ["--config", args.config, "--config-name", args.config_name]
        if args.batch_output_path:
            step_args += ["--batch-output-path", args.batch_output_path]
        if args.poll_interval is not None:
            step_args += ["--poll-interval", str(args.poll_interval)]
        if args.num_splits is not None:
            step_args += ["--num-splits", str(args.num_splits)]
        run_step(base / "02_run_batch.py", step_args)

    if args.mode in ("extract", "pipeline"):
        step_args = ["--config", args.config]
        if args.config_name:
            step_args += ["--config-name", args.config_name]
        if args.output_csv:
            step_args += ["--output-csv", args.output_csv]
        if args.batch_output_path:
            step_args += ["--batch-output-path", args.batch_output_path]
        run_step(base / "03_extract_scores.py", step_args)


def handle_image_vllm(args: argparse.Namespace) -> None:
    script = REPO_ROOT / "evaluation_image-level" / "vllm" / "run_vllm_evaluation.py"
    step_args = [
        "--input_path", args.input_path,
        "--output_path", args.output_path,
        "--hosts", *args.hosts,
        "--ports", *args.ports,
        "--model", args.model,
    ]
    if args.token:
        step_args += ["--token", args.token]
    if args.num_processes_per_server is not None:
        step_args += ["--num_processes_per_server", str(args.num_processes_per_server)]
    if args.seed is not None:
        step_args += ["--seed", str(args.seed)]
    if args.max_tokens is not None:
        step_args += ["--max_tokens", str(args.max_tokens)]
    if args.tex_output_dir:
        step_args += ["--tex_output_dir", args.tex_output_dir]
    if args.enable_thinking:
        step_args += ["--enable_thinking"]
    run_step(script, step_args)


def handle_image_openrouter(args: argparse.Namespace) -> None:
    script = REPO_ROOT / "evaluation_image-level" / "openrouter" / "evaluate.py"
    step_args = [
        "--output", args.output,
        "--metadata-path", args.metadata_path,
        "--edited-model-type", args.edited_model_type,
    ]
    if args.model:
        step_args += ["--model", args.model]
    if args.score_output:
        step_args += ["--score_output", args.score_output]
    if args.edited_input_type:
        step_args += ["--edited-input-type", args.edited_input_type]
    if args.edited_image_paths_field:
        step_args += ["--edited-image-paths-field", args.edited_image_paths_field]
    if args.enable_thinking:
        step_args += ["--enable_thinking"]
    if args.concurrency is not None:
        step_args += ["--concurrency", str(args.concurrency)]
    if args.max_tokens is not None:
        step_args += ["--max_tokens", str(args.max_tokens)]
    if args.limit is not None:
        step_args += ["--limit", str(args.limit)]
    if args.use_local:
        step_args += ["--use-local"]
    if args.retry_failed:
        step_args += ["--retry-failed", args.retry_failed]
    run_step(script, step_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for VisEditBench evaluation experiments."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    code_batch = subparsers.add_parser(
        "code-batch",
        help="Code-based evaluation via Batch API (OpenAI/Gemini/Claude)",
    )
    code_batch.add_argument("mode", choices=["generate", "run", "extract", "pipeline"])
    add_common_batch_args(code_batch)
    code_batch.add_argument("--limit", type=int, default=None)
    code_batch.add_argument("--preview", action="store_true")
    code_batch.add_argument("--amt-csv", default=None)
    code_batch.add_argument("--strict-csv-rows", action="store_true")
    code_batch.add_argument("--gemini-use-local", action="store_true")
    code_batch.add_argument("--batch-output-path", default=None)
    code_batch.add_argument("--output-csv", default=None)
    code_batch.add_argument("--poll-interval", type=int, default=120)
    code_batch.add_argument("--num-splits", type=int, default=3)
    code_batch.set_defaults(handler=handle_code_batch)

    image_batch = subparsers.add_parser(
        "image-batch",
        help="Image-based evaluation via Batch API (OpenAI/Gemini/Claude)",
    )
    image_batch.add_argument("mode", choices=["generate", "run", "extract", "pipeline"])
    add_common_batch_args(image_batch)
    image_batch.add_argument("--limit", type=int, default=None)
    image_batch.add_argument("--preview", action="store_true")
    image_batch.add_argument("--amt-csv", default=None)
    image_batch.add_argument("--strict-csv-rows", action="store_true")
    image_batch.add_argument("--gemini-use-local", action="store_true")
    image_batch.add_argument("--batch-output-path", default=None)
    image_batch.add_argument("--output-csv", default=None)
    image_batch.add_argument("--poll-interval", type=int, default=120)
    image_batch.add_argument("--num-splits", type=int, default=3)
    image_batch.set_defaults(handler=handle_image_batch)

    image_vllm = subparsers.add_parser(
        "image-vllm",
        help="Image-based evaluation via vLLM server",
    )
    image_vllm.add_argument("--input-path", required=True)
    image_vllm.add_argument("--output-path", required=True)
    image_vllm.add_argument("--hosts", nargs="+", required=True)
    image_vllm.add_argument("--ports", nargs="+", required=True)
    image_vllm.add_argument("--model", required=True)
    image_vllm.add_argument("--token", default="")
    image_vllm.add_argument("--num-processes-per-server", type=int, default=1)
    image_vllm.add_argument("--seed", type=int, default=42)
    image_vllm.add_argument("--max-tokens", type=int, default=128000)
    image_vllm.add_argument("--tex-output-dir", default=None)
    image_vllm.add_argument("--enable-thinking", action="store_true")
    image_vllm.set_defaults(handler=handle_image_vllm)

    image_openrouter = subparsers.add_parser(
        "image-openrouter",
        help="Image-based evaluation via OpenRouter",
    )
    image_openrouter.add_argument("--output", required=True)
    image_openrouter.add_argument("--metadata-path", required=True)
    image_openrouter.add_argument("--edited-model-type", required=True)
    image_openrouter.add_argument("--model", default="Qwen/Qwen3.5-397B-A17B")
    image_openrouter.add_argument("--score-output", default=None)
    image_openrouter.add_argument("--edited-input-type", default="w_org_image_w_org_tikz")
    image_openrouter.add_argument("--edited-image-paths-field", default="edited_image_paths")
    image_openrouter.add_argument("--enable-thinking", action="store_true")
    image_openrouter.add_argument("--concurrency", type=int, default=8)
    image_openrouter.add_argument("--max-tokens", type=int, default=8192)
    image_openrouter.add_argument("--limit", type=int, default=None)
    image_openrouter.add_argument("--use-local", action="store_true")
    image_openrouter.add_argument("--retry-failed", default=None)
    image_openrouter.set_defaults(handler=handle_image_openrouter)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "preview", False) and args.mode != "generate":
        parser.error("--preview is only available with generate")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
