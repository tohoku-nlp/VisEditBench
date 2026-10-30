"""
Compile all .tex files found under *tikz* subdirectories of a base directory
and save the resulting PNG images to sibling *images* subdirectories.

Directory mapping example:
    .../qwen3-vl-4b-instruct/tikz/foo.tex
        -> .../qwen3-vl-4b-instruct/images/foo.png

Usage:
    # Local execution (no container)
    python editing/compile_tikz_batch.py \
        --base_dir work/editing/visual

    # With Apptainer / Singularity
    python editing/compile_tikz_batch.py \
        --base_dir work/editing/visual \
        --apptainer path/to/texlive.sif \
        --bind /path/to/work \
        --env apptainer

    # Limit to a specific model directory
    python editing/compile_tikz_batch.py \
        --base_dir work/editing/visual/qwen3-vl-4b-instruct
"""

import argparse
import re
import subprocess
from pathlib import Path

from tqdm import tqdm


def extract_latex_content(text: str) -> str:
    """Extract a complete LaTeX document from text, stripping any surrounding content."""
    match = re.search(r"\\documentclass.*?\\end\{document\}", text, re.DOTALL)
    if match:
        return match.group()
    return "No LaTeX content found."


def build_cmd_prefix(env: str, apptainer: str | None, bind: str | None) -> list[str]:
    """Return the container runtime command prefix, or an empty list for local execution."""
    if not apptainer:
        return []
    runner = "apptainer" if env == "apptainer" else "singularity"
    cmd = [runner, "exec"]
    if bind:
        cmd.extend(["--bind", f"{bind}:{bind}"])
    cmd.append(apptainer)
    return cmd


def compile_tex_to_png(
    tex_file: Path,
    png_dir: Path,
    args: argparse.Namespace,
) -> bool:
    """古い成果物を成功扱いせず、単一のLaTeX文書をPNGへ変換する。"""
    stem = tex_file.stem
    tmp_dir = png_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    prefix = build_cmd_prefix(args.env, args.apptainer, args.bind)
    pdf_file = tmp_dir / f"{stem}.pdf"
    cropped_pdf = tmp_dir / f"{stem}_crop.pdf"
    png_out = png_dir / f"{stem}.png"

    if png_out.exists() and not args.overwrite:
        print(f"  skip (exists): {png_out.name}")
        return True

    # 上書き時に失敗した場合、以前の画像を評価へ渡さない。
    png_out.unlink(missing_ok=True)
    (png_dir / f"{stem}-1.png").unlink(missing_ok=True)

    try:
        # Step 0: extract LaTeX document, stripping code fences or surrounding text
        raw_text = tex_file.read_text(encoding="utf-8")
        extracted = extract_latex_content(raw_text)
        if extracted == "No LaTeX content found.":
            print(f"  error (no LaTeX content): {tex_file.name}")
            return False
        tmp_tex = tmp_dir / f"{stem}.tex"
        tmp_tex.write_text(extracted, encoding="utf-8")
        # 再実行時の古いPDFをコンパイル成功と誤認しない。
        pdf_file.unlink(missing_ok=True)
        cropped_pdf.unlink(missing_ok=True)

        # Step 1: tex -> pdf
        pdflatex_cmd = prefix + [
            "pdflatex",
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-output-directory",
            str(tmp_dir),
            str(tmp_tex),
        ]
        if not getattr(args, "historical_success_policy", False):
            pdflatex_cmd.insert(len(prefix) + 1, "-halt-on-error")
        result = subprocess.run(
            pdflatex_cmd, timeout=30, capture_output=True, text=True
        )
        if not pdf_file.exists() or (result.returncode != 0 and not getattr(args, "historical_success_policy", False)):
            print(f"  error (pdflatex failed): {tex_file.name}")
            return False

        # Step 2: pdf -> cropped pdf
        pdfcrop_cmd = prefix + [
            "pdfcrop",
            "--margins",
            "10 10 10 10",
            str(pdf_file),
            str(cropped_pdf),
        ]
        subprocess.run(pdfcrop_cmd, timeout=10, capture_output=True)

        target_pdf = cropped_pdf if cropped_pdf.exists() else pdf_file

        # Step 3: pdf -> png (pdftoppm outputs <stem>-1.png, renamed to <stem>.png)
        png_base = str(png_dir / stem)
        pdftoppm_cmd = prefix + [
            "pdftoppm",
            "-png",
            "-r",
            str(args.dpi),
            "-f",
            "1",
            "-l",
            "1",
            str(target_pdf),
            png_base,
        ]
        subprocess.run(pdftoppm_cmd, timeout=30, capture_output=True, check=True)

        tmp_png = png_dir / f"{stem}-1.png"
        if tmp_png.exists():
            tmp_png.rename(png_out)

        if png_out.exists():
            print(f"  saved: {png_out.name}")
            return True
        else:
            print(f"  error (png not created): {tex_file.name}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  timeout: {tex_file.name}")
        return False
    except Exception as e:
        print(f"  error: {tex_file.name} -> {e}")
        return False


def main():
    """Entry point: discover tex directories under base_dir and compile each .tex to PNG."""
    parser = argparse.ArgumentParser(
        description="Compile .tex files in tikz subdirectories to PNG images."
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Base directory to search for tex subdirectories recursively.",
    )
    parser.add_argument(
        "--apptainer",
        default=None,
        help="Path to Apptainer/Singularity image. Omit to use the local pdflatex.",
    )
    parser.add_argument(
        "--bind",
        default=None,
        help="Directory to bind-mount into the container (passed to --bind).",
    )
    parser.add_argument(
        "--env",
        default="apptainer",
        choices=["apptainer", "singularity"],
        help="Container runtime: 'apptainer' or 'singularity' (default: apptainer).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for PNG conversion (default: 300).",
    )
    parser.add_argument(
        "--tex_dir",
        default="tikz",
        help="Name of the subdirectory containing .tex files (default: tikz).",
    )
    parser.add_argument(
        "--png_dir",
        default="images",
        help="Name of the output subdirectory for PNG files (default: images).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNG files.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    tex_dirs = sorted(base_dir.rglob(args.tex_dir))
    tex_dirs = [d for d in tex_dirs if d.is_dir()]

    if not tex_dirs:
        print(f"No '{args.tex_dir}' directories found under {base_dir}")
        return

    print(f"Found {len(tex_dirs)} tex directories.")

    total_ok = total_ng = 0

    for tex_dir in tex_dirs:
        png_dir = tex_dir.parent / args.png_dir
        tex_files = sorted(tex_dir.glob("*.tex"))
        print(
            f"\n[{tex_dir.relative_to(base_dir)}]  {len(tex_files)} files -> {png_dir.name}/"
        )

        for tex_file in tqdm(tex_files, leave=False):
            ok = compile_tex_to_png(tex_file, png_dir, args)
            if ok:
                total_ok += 1
            else:
                total_ng += 1

    print(f"\nDone: {total_ok} succeeded / {total_ng} failed")


if __name__ == "__main__":
    main()
