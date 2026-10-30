#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_step(script: Path, args: list[str]) -> None:
    """Run a sub-script as a subprocess with the given arguments."""
    cmd = [sys.executable, str(script), *args]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def add_common_batch_args(parser: argparse.ArgumentParser) -> None:
    """Add --config and --config-name arguments shared by all batch sub-commands."""
    parser.add_argument("--config", required=True, help="Path to experiment config TOML")
    parser.add_argument("--config-name", default=None, help="Experiment name in TOML")


def handle_batch(args: argparse.Namespace) -> None:
    """Dispatch batch editing pipeline steps (generate / run / extract / pipeline)."""
    base = REPO_ROOT / "editing" / "batch_api" / "steps"

    if args.mode in ("generate", "pipeline"):
        step_args = ["--config", args.config]
        if args.config_name:
            step_args += ["--config-name", args.config_name]
        if args.limit is not None:
            step_args += ["--limit", str(args.limit)]
        if args.preview:
            step_args += ["--preview"]
        run_step(base / "01_generate_jsonl.py", step_args)

    if args.mode in ("run", "pipeline"):
        if not args.config_name:
            raise ValueError("--config-name is required for run/pipeline mode")
        step_args = ["--config", args.config, "--config-name", args.config_name]
        if args.batch_output_path:
            step_args += ["--batch-output-path", args.batch_output_path]
        if args.poll_interval is not None:
            step_args += ["--poll-interval", str(args.poll_interval)]
        run_step(base / "02_run_batch.py", step_args)

    if args.mode in ("extract", "pipeline"):
        step_args = ["--config", args.config]
        if args.config_name:
            step_args += ["--config-name", args.config_name]
        if args.batch_output_path:
            step_args += ["--batch-output-path", args.batch_output_path]
        if args.save_dir:
            step_args += ["--save-dir", args.save_dir]
        run_step(base / "03_extract_tikz.py", step_args)


def handle_vllm(args: argparse.Namespace) -> None:
    """Dispatch vLLM-based editing inference."""
    script = REPO_ROOT / "editing" / "vllm" / "run_vllm_edit_inference.py"
    step_args = [
        "--mode", args.mode,
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


def handle_openrouter(args: argparse.Namespace) -> None:
    """Dispatch OpenRouter-based editing inference."""
    script = REPO_ROOT / "editing" / "openrouter" / "edit_images.py"
    step_args = [
        "--mode", args.mode,
        "--jsonl_path", args.jsonl_path,
    ]
    if args.model:
        step_args += ["--model", args.model]
    if args.api_key:
        step_args += ["--api_key", args.api_key]
    if args.save_dir:
        step_args += ["--save_dir", args.save_dir]
    if args.start is not None:
        step_args += ["--start", str(args.start)]
    if args.limit is not None:
        step_args += ["--limit", str(args.limit)]
    if args.overwrite:
        step_args += ["--overwrite"]
    if args.only_keys:
        step_args += ["--only_keys", args.only_keys]
    if args.max_tokens is not None:
        step_args += ["--max_tokens", str(args.max_tokens)]
    if args.enable_thinking:
        step_args += ["--enable_thinking"]
    run_step(script, step_args)


def handle_diffusers(args: argparse.Namespace) -> None:
    """Dispatch Diffusers-based editing inference (Qwen-Image-Edit-2511)."""
    script = REPO_ROOT / "editing" / "diffusers" / "edit_images_by_qwen-image-edit-2511.py"
    step_args = [
        "--mode", args.mode,
        "--save_dir", args.save_dir,
    ]
    if args.metadata_path:
        step_args += ["--metadata_path", args.metadata_path]
    if args.model:
        step_args += ["--model", args.model]
    if args.device:
        step_args += ["--device", args.device]
    if args.seed is not None:
        step_args += ["--seed", str(args.seed)]
    if args.skip_existing:
        step_args += ["--skip_existing"]
    if args.limit is not None:
        step_args += ["--limit", str(args.limit)]
    if args.keys:
        step_args += ["--keys", *args.keys]
    run_step(script, step_args)


def handle_verbalize(args: argparse.Namespace) -> None:
    """論文Figure9の4条件を、モデル別設定と固定subsetを使って準備する。"""
    from viseditbench.workflow import prepare_experiment
    import json
    result = prepare_experiment("verbalization", Path(args.output), Path(args.data_dir),
                                models=args.models, conditions=args.conditions, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all sub-commands."""
    parser = argparse.ArgumentParser(
        description="Unified entry point for VisEditBench diagram editing experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verbalize = subparsers.add_parser("verbalize", help="Prepare the paper's four prompting conditions (offline)")
    verbalize.add_argument("--output", required=True)
    verbalize.add_argument("--data-dir", default="data")
    verbalize.add_argument("--models", nargs="+")
    verbalize.add_argument("--conditions", nargs="+", choices=["direct", "verbalize_first", "gold_text", "gold_text_visual"])
    verbalize.add_argument("--limit", type=int)
    verbalize.set_defaults(handler=handle_verbalize)

    # ---- batch ----------------------------------------------------------------
    batch = subparsers.add_parser(
        "batch",
        help="TikZ editing via Batch API (OpenAI / Gemini / Claude)",
    )
    batch.add_argument("mode", choices=["generate", "run", "extract", "pipeline"])
    add_common_batch_args(batch)
    batch.add_argument("--limit", type=int, default=None,
                       help="Process only the first N metadata rows")
    batch.add_argument("--preview", action="store_true",
                       help="Preview the first request without saving")
    batch.add_argument("--batch-output-path", default=None,
                       help="Override output JSONL path for batch results")
    batch.add_argument("--save-dir", default=None,
                       help="Override output directory for extracted TikZ files")
    batch.add_argument("--poll-interval", type=int, default=120,
                       help="Batch status polling interval in seconds (default: 120)")
    batch.set_defaults(handler=handle_batch)

    # ---- vllm -----------------------------------------------------------------
    vllm = subparsers.add_parser(
        "vllm",
        help="TikZ editing via vLLM server",
    )
    vllm.add_argument("--mode", required=True,
                      choices=["visual", "visual_w_org_tikz", "text_w_org_tikz"])
    vllm.add_argument("--input-path", required=True, dest="input_path")
    vllm.add_argument("--output-path", required=True, dest="output_path")
    vllm.add_argument("--hosts", nargs="+", required=True)
    vllm.add_argument("--ports", nargs="+", required=True)
    vllm.add_argument("--model", required=True)
    vllm.add_argument("--token", default="")
    vllm.add_argument("--num-processes-per-server", type=int, default=1,
                      dest="num_processes_per_server")
    vllm.add_argument("--seed", type=int, default=42)
    vllm.add_argument("--max-tokens", type=int, default=128000, dest="max_tokens")
    vllm.add_argument("--tex-output-dir", default=None, dest="tex_output_dir")
    vllm.add_argument("--enable-thinking", action="store_true", dest="enable_thinking")
    vllm.set_defaults(handler=handle_vllm)

    # ---- openrouter -----------------------------------------------------------
    openrouter = subparsers.add_parser(
        "openrouter",
        help="TikZ editing via OpenRouter",
    )
    openrouter.add_argument(
        "--mode", required=True,
        choices=[
            "visual_instruction",
            "visual_instruction_w_org_image",
            "visual_instruction_w_org_tikz",
            "visual_instruction_w_org_image_w_org_tikz",
            "text_instruction_w_org_image_w_org_tikz",
        ],
    )
    openrouter.add_argument("--model", default="qwen/qwen3-vl-235b-a22b-instruct",
                            help="OpenRouter model ID or alias")
    openrouter.add_argument("--api-key", default=None, dest="api_key",
                            help="OpenRouter API key (falls back to OPENROUTER_API_KEY)")
    openrouter.add_argument("--jsonl-path", default="data/metadata/edit_metadata.jsonl",
                            dest="jsonl_path", help="Input JSONL file path")
    openrouter.add_argument("--save-dir", default=None, dest="save_dir",
                            help="Output directory for TikZ files")
    openrouter.add_argument("--start", type=int, default=0,
                            help="Start index (0-based, default: 0)")
    openrouter.add_argument("--limit", type=int, default=None,
                            help="Maximum number of records to process")
    openrouter.add_argument("--overwrite", action="store_true",
                            help="Overwrite existing output files")
    openrouter.add_argument("--only-keys", default=None, dest="only_keys",
                            help="Comma-separated list of target keys")
    openrouter.add_argument("--max-tokens", type=int, default=128000, dest="max_tokens")
    openrouter.add_argument("--enable-thinking", action="store_true", dest="enable_thinking")
    openrouter.set_defaults(handler=handle_openrouter)

    # ---- diffusers ------------------------------------------------------------
    diffusers = subparsers.add_parser(
        "diffusers",
        help="Image editing via Diffusers (Qwen-Image-Edit-2511)",
    )
    diffusers.add_argument("--mode", required=True,
                           choices=["visual", "text_instruction", "visual_with_org_tikz"])
    diffusers.add_argument("--save-dir", required=True, dest="save_dir",
                           help="Output directory for edited images")
    diffusers.add_argument("--metadata-path", default="data/metadata/edit_metadata.jsonl",
                           dest="metadata_path", help="Path to edit_metadata.jsonl")
    diffusers.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511",
                           help="HuggingFace model ID")
    diffusers.add_argument("--device", default="cuda:0",
                           help="Target device (e.g., cuda:0, cuda:1)")
    diffusers.add_argument("--seed", type=int, default=0)
    diffusers.add_argument("--skip-existing", action="store_true", dest="skip_existing",
                           help="Skip files that already exist in the output directory")
    diffusers.add_argument("--limit", type=int, default=None,
                           help="Maximum number of records to process")
    diffusers.add_argument("--keys", nargs="+", default=None,
                           help="Process only the specified keys")
    diffusers.set_defaults(handler=handle_diffusers)

    return parser


def main() -> None:
    """Parse arguments and dispatch to the appropriate handler."""
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "preview", False) and args.mode != "generate":
        parser.error("--preview is only available with batch generate")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
