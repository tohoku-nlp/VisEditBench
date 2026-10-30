"""
Unified script to generate edited TikZ code via OpenRouter.

Modes:
  - visual_instruction
  - visual_instruction_w_org_image
  - visual_instruction_w_org_tikz
  - visual_instruction_w_org_image_w_org_tikz
  - text_instruction_w_org_image_w_org_tikz

Notes:
  - Image inputs are always sent from local files as base64 data URLs.
  - GCS URL workflows are intentionally excluded.
"""

import argparse
import base64
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

MODEL_NAME_MAP = {
    "qwen3-vl-235b-instruct": "qwen/qwen3-vl-235b-a22b-instruct",
    "qwen3-vl-235b-thinking": "qwen/qwen3-vl-235b-a22b-thinking",
    "qwen3-vl-32b-instruct": "qwen/qwen3-vl-32b-instruct",
    "qwen3-vl-30b-instruct": "qwen/qwen3-vl-30b-a3b-instruct",
    "qwen3-vl-30b-thinking": "qwen/qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-8b-instruct": "qwen/qwen3-vl-8b-instruct",
    "qwen3-vl-8b-thinking": "qwen/qwen3-vl-8b-thinking",
    "qwen3-5-397b": "Qwen/Qwen3.5-397B-A17B",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

VISUAL_INSTRUCTION_SYSTEM_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
- an annotated version of the diagram image containing visual edit instructions.

Use input:
- Use the annotated image to understand the structure and content of the diagram, and to identify the requested edits.

Apply the requested edits and produce LaTeX/TikZ code representing the final diagram after the edits have been applied.
Your output must describe only the final edited diagram, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

VISUAL_W_ORG_IMAGE_SYSTEM_PROMPT = r"""You are an expert at converting diagram images into LaTeX/TikZ.

You are given two images of the same diagram:
1. the original diagram image, before any edit instructions were added, and
2. an annotated version of that same diagram image containing written edit instructions.

Use both images together:
- Use the original image to understand the diagram clearly without overlaid markings.
- Use the annotated image to identify the requested edits.
- Use both the original image to capture the underlying content and the annotated image to understand the intended modifications.

Apply the requested edits and produce LaTeX/TikZ code that represents the final diagram after those edits have been applied.
Your output must describe the edited final state of the diagram itself, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

VISUAL_W_ORG_TIKZ_SYSTEM_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original TikZ code of a diagram,
2. an annotated version of that diagram image containing visual edit instructions.

Use all inputs together:
- Use the original TikZ code to understand the structure and content of the diagram.
- Use the annotated image to identify the requested edits.

Apply the requested edits and produce LaTeX/TikZ code representing the final diagram after the edits have been applied.
Your output must describe only the final edited diagram, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

VISUAL_W_ORG_IMAGE_W_ORG_TIKZ_SYSTEM_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original TikZ code of a diagram,
2. the original diagram image, and
3. an annotated version of that diagram image containing visual edit instructions.

Use all inputs together:
- Use the original TikZ code and the original image to understand the structure and content of the diagram.
- Use the annotated image to identify the requested edits.
- Use both the original image to capture the underlying content and the annotated image to understand the intended modifications.

Apply the requested edits and produce LaTeX/TikZ code representing the final diagram after the edits have been applied.
Your output must describe only the final edited diagram, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

TEXT_W_ORG_IMAGE_W_ORG_TIKZ_SYSTEM_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original diagram image,
2. the original TikZ code of a diagram, and
3. a text instruction that describes how to edit the diagram.

Use all inputs together:
- Use the original TikZ code and the original image to understand the structure and content of the diagram.
- Use the text instruction to identify the requested edits.

Apply the requested edits and produce LaTeX/TikZ code representing the final diagram after the edits have been applied.
Your output must describe only the final edited diagram, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

ORG_TIKZ_USER_PROMPT_TEMPLATE = r"""Original TikZ code:
{original_tikz_code}
"""

ORG_TIKZ_TEXT_USER_PROMPT_TEMPLATE = r"""Original TikZ code:
{original_tikz_code}

Text instruction:
{text_instruction}
"""


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def strip_thinking(content: str | None) -> str:
    if not content:
        return ""
    start_tag = "<think>"
    end_tag = "</think>"
    while True:
        s = content.find(start_tag)
        if s == -1:
            break
        e = content.find(end_tag, s + len(start_tag))
        if e == -1:
            content = content[:s]
            break
        content = content[:s] + content[e + len(end_tag) :]
    return content.strip()


def extract_content(choice) -> str | None:
    message = getattr(choice, "message", None)
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif hasattr(item, "type") and getattr(item, "type") == "text":
                parts.append(getattr(item, "text", ""))
        return "\n".join(parts).strip() if parts else None
    return None


def resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    return MODEL_NAME_MAP.get(key, name)


def make_image_content(image_path: str) -> dict:
    b64 = encode_image(image_path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


def require_existing_path(path_str: str, label: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def infer_tikz(
    client: OpenAI,
    model: str,
    mode: str,
    record: dict,
    max_tokens: int | None,
    enable_thinking: bool,
) -> str:
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if enable_thinking:
        kwargs["extra_body"] = {"enable_thinking": True}

    if mode == "visual_instruction":
        vi_path = record.get("visual_instruction_image_path")
        if not vi_path:
            raise ValueError("missing visual_instruction_image_path")
        require_existing_path(vi_path, "visual image")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VISUAL_INSTRUCTION_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [make_image_content(vi_path)],
            },
        ]

    elif mode == "visual_instruction_w_org_image":
        org_path = record.get("org_image_path")
        vi_path = record.get("visual_instruction_image_path")
        if not org_path or not vi_path:
            raise ValueError("missing org_image_path or visual_instruction_image_path")
        require_existing_path(org_path, "org image")
        require_existing_path(vi_path, "visual image")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VISUAL_W_ORG_IMAGE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    make_image_content(org_path),
                    make_image_content(vi_path),
                ],
            },
        ]

    elif mode == "visual_instruction_w_org_tikz":
        vi_path = record.get("visual_instruction_image_path")
        tikz_path_str = record.get("org_tikz_code_path")
        if not vi_path or not tikz_path_str:
            raise ValueError("missing visual_instruction_image_path or org_tikz_code_path")
        require_existing_path(vi_path, "visual image")
        tikz_path = require_existing_path(tikz_path_str, "tikz file")

        user_prompt = ORG_TIKZ_USER_PROMPT_TEMPLATE.format(
            original_tikz_code=tikz_path.read_text(encoding="utf-8")
        )

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VISUAL_W_ORG_TIKZ_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    make_image_content(vi_path),
                ],
            },
        ]

    elif mode == "visual_instruction_w_org_image_w_org_tikz":
        org_path = record.get("org_image_path")
        vi_path = record.get("visual_instruction_image_path")
        tikz_path_str = record.get("org_tikz_code_path")
        if not org_path or not vi_path or not tikz_path_str:
            raise ValueError("missing org_image_path, visual_instruction_image_path or org_tikz_code_path")
        require_existing_path(org_path, "org image")
        require_existing_path(vi_path, "visual image")
        tikz_path = require_existing_path(tikz_path_str, "tikz file")

        user_prompt = ORG_TIKZ_USER_PROMPT_TEMPLATE.format(
            original_tikz_code=tikz_path.read_text(encoding="utf-8")
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": VISUAL_W_ORG_IMAGE_W_ORG_TIKZ_SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    make_image_content(org_path),
                    make_image_content(vi_path),
                ],
            },
        ]

    elif mode == "text_instruction_w_org_image_w_org_tikz":
        org_path = record.get("org_image_path")
        tikz_path_str = record.get("org_tikz_code_path")
        text_instruction = record.get("text_instruction")
        if not org_path or not tikz_path_str:
            raise ValueError("missing org_image_path or org_tikz_code_path")
        if not isinstance(text_instruction, str) or not text_instruction.strip():
            raise ValueError("missing text_instruction")
        require_existing_path(org_path, "org image")
        tikz_path = require_existing_path(tikz_path_str, "tikz file")

        user_prompt = ORG_TIKZ_TEXT_USER_PROMPT_TEMPLATE.format(
            original_tikz_code=tikz_path.read_text(encoding="utf-8"),
            text_instruction=text_instruction,
        )

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": TEXT_W_ORG_IMAGE_W_ORG_TIKZ_SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    make_image_content(org_path),
                ],
            },
        ]

    else:
        raise ValueError(f"unknown mode: {mode}")

    res = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    if not res.choices:
        error = getattr(res, "error", None)
        raise RuntimeError(f"choices=None: {error}")

    return strip_thinking(extract_content(res.choices[0]))


def default_save_dir(mode: str, model_dir_name: str) -> Path:
    if mode == "visual_instruction":
        return Path("work") / "editing" / "visual" / "visual_instruction" / model_dir_name / "tikz"
    if mode == "visual_instruction_w_org_image":
        return Path("work") / "editing" / "visual" / "w_org_image" / model_dir_name / "tikz"
    if mode == "visual_instruction_w_org_tikz":
        return Path("work") / "editing" / "visual" / "w_org_tikz" / model_dir_name / "tikz"
    if mode == "visual_instruction_w_org_image_w_org_tikz":
        return Path("work") / "editing" / "visual" / "w_org_image_w_org_tikz" / model_dir_name / "tikz"
    if mode == "text_instruction_w_org_image_w_org_tikz":
        return Path("work") / "editing" / "text" / "w_org_image_w_org_tikz" / model_dir_name / "tikz"
    raise ValueError(f"unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified script for generating edited TikZ code via OpenRouter."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "visual_instruction",
            "visual_instruction_w_org_image",
            "visual_instruction_w_org_tikz",
            "visual_instruction_w_org_image_w_org_tikz",
            "text_instruction_w_org_image_w_org_tikz",
        ],
        help="Select input mode",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-vl-235b-a22b-instruct",
        help="OpenRouter model ID (or alias).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="data/metadata/edit_metadata.jsonl",
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Output directory. If omitted, mode-specific default is used.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based).")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to process.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--only_keys",
        type=str,
        default=None,
        help="Comma-separated target keys (e.g., key1,key2).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128000,
        help="Maximum generation tokens.",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode (passes enable_thinking=true).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "Specify OpenRouter API key via --api_key or OPENROUTER_API_KEY."
        )

    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {jsonl_path}")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    resolved_model = resolve_model_name(args.model)

    model_dir_name = args.model.split("/")[-1].replace(":", "_")
    if args.enable_thinking and not model_dir_name.endswith("-thinking"):
        model_dir_name = f"{model_dir_name}-thinking"

    save_dir = Path(args.save_dir) if args.save_dir else default_save_dir(args.mode, model_dir_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    only_keys = None
    if args.only_keys:
        only_keys = {k.strip() for k in args.only_keys.split(",") if k.strip()}

    processed = 0
    for idx, record in enumerate(read_jsonl(jsonl_path)):
        if idx < args.start:
            continue
        if args.limit is not None and processed >= args.limit:
            break

        key = record.get("key", "unknown")
        if only_keys and key not in only_keys:
            continue

        save_path = save_dir / f"{key}.tex"
        if save_path.exists() and not args.overwrite:
            print(f"skip (exists): {save_path}")
            processed += 1
            continue

        try:
            tikz_code = infer_tikz(
                client=client,
                model=resolved_model,
                mode=args.mode,
                record=record,
                max_tokens=args.max_tokens,
                enable_thinking=args.enable_thinking,
            )
            if not tikz_code:
                print(f"error: {key} -> empty response")
            else:
                save_path.write_text(tikz_code, encoding="utf-8")
                print(f"saved: {save_path}")
        except Exception as e:
            print(f"error: {key} -> {e}")

        processed += 1


if __name__ == "__main__":
    main()
