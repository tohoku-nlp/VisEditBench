"""
Unified Diffusers script for diagram editing with Qwen-Image-Edit-2511.

Switch among three experiment modes in one file:
  - visual: original image + visual instruction image
  - text_instruction: original image + text instruction
  - visual_with_org_tikz: original image + visual instruction image + original TikZ code

Usage:
    python diffusers/edit_images_by_qwen-image-edit-2511.py \
        --mode visual \
        --metadata_path data/metadata/edit_metadata.jsonl \
        --save_dir work/editing/visual/w_org_image/qwen-image-edit-2511

    python diffusers/edit_images_by_qwen-image-edit-2511.py \
        --mode text_instruction \
        --metadata_path data/metadata/edit_metadata.jsonl \
        --save_dir work/editing/text/w_org_image/qwen-image-edit-2511

    python diffusers/edit_images_by_qwen-image-edit-2511.py \
        --mode visual_with_org_tikz \
        --metadata_path data/metadata/edit_metadata.jsonl \
        --save_dir work/editing/visual/w_org_image_w_org_tikz/qwen-image-edit-2511
"""

import argparse
import json
import traceback
from pathlib import Path

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

VISUAL_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given two images of the same diagram:
1. the original diagram image, before any edit instructions were added, and
2. an annotated version of that same diagram image containing visual edit instructions.

Use both images together:
- Use the original image to understand the diagram clearly without overlaid markings.
- Use the annotated image to identify the requested edits.
- Use both the original image to capture the underlying content and the annotated image to understand the intended modifications.

Edit the first (original) image by following the visual instructions shown in the second (annotated) image.

Preserve the original layout, geometry, proportions, line quality, labels, and sharp edges exactly.
Only apply the modifications indicated by the visual instructions in the annotated image.
Do not introduce any additional changes or reinterpretations.
Ensure that all elements not explicitly targeted by the visual instructions remain unchanged.
"""

SYSTEM_PROMPT_W_ORG_IMAGE_W_TEXT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original diagram image, and
2. a text instruction that describes how to edit the diagram.

Use all inputs together:
- Use the original image to understand the current diagram structure and content.
- Use the text instruction to identify the requested edits.

Edit the original image by following the text instruction exactly.

Preserve the original layout, geometry, proportions, line quality, labels, and sharp edges exactly.
Only apply the modifications requested by the text instruction.
Do not introduce any additional changes or reinterpretations.
Ensure that all elements not explicitly targeted by the instruction remain unchanged.

Return only the edited diagram image.
"""

USER_PROMPT_W_TEXT = r"""Text instruction:
{text_instruction}
"""

VISUAL_WITH_TIKZ_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original TikZ code of a diagram,
2. the original diagram image, and
3. an annotated version of that diagram image containing visual edit instructions.

Use all inputs together:
- Use the original TikZ code and the original image to understand the structure and content of the diagram precisely.
- Use the annotated image to identify the requested edits.
- Use both the original image to capture the underlying content and the annotated image to understand the intended modifications.

Edit the first (original) image by following the visual instructions shown in the second (annotated) image.
Use the TikZ code as additional reference to understand the diagram structure.

Preserve the original layout, geometry, proportions, line quality, labels, and sharp edges exactly.
Only apply the modifications indicated by the visual instructions in the annotated image.
Do not introduce any additional changes or reinterpretations.
Ensure that all elements not explicitly targeted by the visual instructions remain unchanged.

Original TikZ code:
{original_tikz_code}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified editing script for Qwen-Image-Edit-2511 (Diffusers)"
    )
    parser.add_argument(
        "--mode",
        choices=["visual", "text_instruction", "visual_with_org_tikz"],
        required=True,
        help="Select editing mode",
    )
    parser.add_argument(
        "--metadata_path",
        default="data/metadata/edit_metadata.jsonl",
        help="Path to edit_metadata.jsonl",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="Output directory for edited images",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit-2511",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Target device (e.g., cuda:0, cuda:1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files that already exist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process (for testing)",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=None,
        help="Filter by specific keys (e.g., --keys 00017027_UPDATE_3_edit_0)",
    )
    return parser.parse_args()


def run_edit(
    pipeline: QwenImageEditPlusPipeline,
    mode: str,
    record: dict,
    generator: torch.Generator,
):
    org_path = Path(record["org_image_path"])

    if not org_path.exists():
        raise FileNotFoundError(f"org not found: {org_path}")

    org_image = Image.open(org_path).convert("RGB")

    if mode == "visual":
        vis_path = Path(record["visual_instruction_image_path"])
        if not vis_path.exists():
            raise FileNotFoundError(f"visual not found: {vis_path}")
        vis_image = Image.open(vis_path).convert("RGB")

        return pipeline(
            image=[org_image, vis_image],
            prompt=VISUAL_PROMPT,
            generator=generator,
        )

    if mode == "text_instruction":
        text_instruction = record.get("text_instruction")
        if not text_instruction:
            raise ValueError("text_instruction not found")

        prompt = (
            SYSTEM_PROMPT_W_ORG_IMAGE_W_TEXT
            + "\n\n"
            + USER_PROMPT_W_TEXT.format(text_instruction=text_instruction)
        )

        return pipeline(
            image=org_image,
            prompt=prompt,
            generator=generator,
        )

    if mode == "visual_with_org_tikz":
        vis_path = Path(record["visual_instruction_image_path"])
        tikz_path = Path(record["org_tikz_code_path"])

        if not vis_path.exists():
            raise FileNotFoundError(f"visual not found: {vis_path}")
        if not tikz_path.exists():
            raise FileNotFoundError(f"tikz not found: {tikz_path}")

        vis_image = Image.open(vis_path).convert("RGB")
        original_tikz_code = tikz_path.read_text(encoding="utf-8")
        prompt = VISUAL_WITH_TIKZ_PROMPT.format(original_tikz_code=original_tikz_code)

        return pipeline(
            image=[org_image, vis_image],
            prompt=prompt,
            generator=generator,
        )

    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    args = parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=True)
    print(f"Model loaded (mode={args.mode})")

    success = skip = fail = 0

    with open(args.metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            key = record["key"]

            if args.keys is not None and key not in args.keys:
                continue

            if args.limit is not None and success + fail >= args.limit:
                break

            save_path = save_dir / f"{key}.png"
            if args.skip_existing and save_path.exists():
                skip += 1
                continue

            try:
                with torch.inference_mode():
                    output = run_edit(
                        pipeline=pipeline,
                        mode=args.mode,
                        record=record,
                        generator=torch.manual_seed(args.seed),
                    )
                output.images[0].save(save_path)
                print(f"saved: {save_path}")
                success += 1
            except Exception as e:
                print(f"error ({key}): {e}")
                traceback.print_exc()
                fail += 1

    print(f"\nCompleted: {success} success / {skip} skipped / {fail} failed")


if __name__ == "__main__":
    main()
