"""既存HFスナップショットを使い、一時コピー内でCLI全体をオフライン検証する。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viseditbench.data import prepare, read_jsonl, verify, write_jsonl


def run(root: Path, *arguments: str) -> None:
    """APIを呼ばないCLIだけを実行し、失敗時に出力を表示する。"""
    import os
    environment = dict(os.environ, MPLCONFIGDIR=str(root / ".cache/mpl"), XDG_CACHE_HOME=str(root / ".cache/xdg"))
    result = subprocess.run([sys.executable, *arguments], cwd=root, capture_output=True, text=True, env=environment)
    if result.returncode:
        raise RuntimeError(f"Command failed: {arguments}\n{result.stdout}\n{result.stderr}")


def main() -> None:
    """全件変換と1件の模擬応答ワークフローを検証し、集計だけを表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--supplement", type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    with tempfile.TemporaryDirectory(prefix="viseditbench-validation-") as temporary:
        sandbox = Path(temporary)
        full = sandbox / "full_data"
        manifest = prepare(snapshot, full)
        verified = verify(full / "metadata/edit_metadata.jsonl")
        full_rows = read_jsonl(full / "metadata/edit_metadata.jsonl")
        root = sandbox / "repository"
        shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".env", "data", "work", "runs", "log", "logs", "*.egg-info"))
        # 子プロセスも外部接続を禁止する。API用のキーは読み込まない。
        (root / "sitecustomize.py").write_text(
            'import socket\n'
            'def block(*args, **kwargs):\n'
            '    """オフライン検証時の外部通信を拒否する。"""\n'
            '    raise RuntimeError("Network forbidden during local validation")\n'
            'socket.socket.connect = block\n', encoding="utf-8")
        # sitecustomizeを確実に読み込ませ、模擬応答による確認に限定する。
        import os
        previous_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(root)
        try:
            for script in ["dataset.py", "edit.py", "eval.py", "results.py", "paper.py"]:
                run(root, script, "--help")
            run(root, "dataset.py", "prepare", "--snapshot", str(snapshot), "--limit", "1")
            run(root, "dataset.py", "verify")
            for name in ["gemini31pro_visual", "gpt54_visual", "claude_sonnet45_visual"]:
                run(root, "edit.py", "batch", "generate", "--config", "configs/editing.toml", "--config-name", name)
                assert len(read_jsonl(root / "runs" / name / "input.jsonl")) == 1
            row = read_jsonl(root / "data/metadata/edit_metadata.jsonl")[0]
            request = read_jsonl(root / "runs/gemini31pro_visual/input.jsonl")[0]
            write_jsonl(root / "runs/gemini31pro_visual/output_gemini31pro_visual.jsonl", [
                {"key": request["key"], "response": {"candidates": [{"content": {"parts": [{"text": row["target_tikz"]}]}}]}}
            ])
            run(root, "edit.py", "batch", "extract", "--config", "configs/editing.toml", "--config-name", "gemini31pro_visual", "--save-dir", "work/gemini31pro_visual/tikz")
            images = root / "work/gemini31pro_visual/images"
            images.mkdir(parents=True)
            shutil.copy2(root / row["target_image_path"], images / (row["key"] + ".png"))
            run(root, "dataset.py", "attach", "--model", "gemini-3.1-pro-preview", "--tex-dir", "work/gemini31pro_visual/tikz", "--image-dir", "work/gemini31pro_visual/images", "--output", "runs/gemini31pro_visual/metadata.jsonl")
            for kind in ["image", "code"]:
                arguments = ["eval.py", kind + "-batch", "generate", "--config", f"configs/{kind}_eval.toml", "--config-name", f"gemini31pro_visual_{kind}_eval"]
                if kind == "image":
                    arguments.append("--gemini-use-local")
                run(root, *arguments)
            judge = read_jsonl(root / "runs/gemini31pro_visual_image_eval/input.jsonl")[0]
            scores = dict(instruction_adherence=5, content_preservation=5, diagram_readability=5)
            write_jsonl(root / "runs/gemini31pro_visual_image_eval/output_gemini31pro_visual_image_eval.jsonl", [
                {"key": judge["key"], "response": {"candidates": [{"content": {"parts": [{"text": json.dumps(scores)}]}}]}}
            ])
            run(root, "eval.py", "image-batch", "extract", "--config", "configs/image_eval.toml", "--config-name", "gemini31pro_visual_image_eval", "--output-csv", "runs/scores.csv")
            run(root, "results.py", "--metadata", "runs/gemini31pro_visual/metadata.jsonl", "--scores", "runs/scores.csv", "--model", "gemini-3.1-pro-preview", "--output", "runs/summary.json")
            summary = json.loads((root / "runs/summary.json").read_text())
            assert summary["complete"] and summary["overall"]["n"] == 1 and summary["overall"]["FSR_count"] == 1
            if args.supplement:
                run(root, "dataset.py", "prepare", "--snapshot", str(snapshot), "--force")
                run(root, "dataset.py", "supplement", "--source", str(args.supplement.resolve()))
                run(root, "edit.py", "verbalize", "--models", "gemini-3.1-pro-preview", "--limit", "1", "--output", "runs/four_conditions")
                for filename in json.loads((root / "runs/four_conditions/plan.json").read_text())["jobs"]:
                    job_path = Path(filename)
                    job = json.loads(job_path.read_text())
                    row = read_jsonl(job_path.parent / "metadata.jsonl")[0]
                    key = row["key"]
                    code = row["target_tikz"]
                    if job["condition"] == "verbalize_first":
                        code = "<instruction>Mock instruction for offline validation.</instruction>\n" + code
                    responses = job_path.parent / "mock_generation.jsonl"
                    write_jsonl(responses, [{"key": key, "response": {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": code}]}}]}}])
                    run(root, "paper.py", "import-responses", "--job", str(job_path), "--responses", str(responses))
                    images = job_path.parent / "images"
                    images.mkdir()
                    shutil.copy2(root / row["target_image_path"], images / (key + ".png"))
                    # このCLI試験の画像は参照PNGによる模擬出力。実TikZ描画はunittest側で確認する。
                    run(root, "paper.py", "render", "--job", str(job_path))
                    run(root, "paper.py", "judge-prepare", "--job", str(job_path))
                    judge = job_path.parent / "judges/gemini-3.6-flash/image/repeat_00/job.json"
                    responses = judge.parent / "mock_judgment.jsonl"
                    write_jsonl(responses, [{"key": key, "response": {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(scores)}]}}]}}])
                    run(root, "paper.py", "import-responses", "--job", str(judge), "--responses", str(responses))
                    run(root, "paper.py", "report", "--job", str(judge))
                    assert json.loads((judge.parent / "summary.json").read_text())["overall"]["FSR_count"] == 1
                run(root, "paper.py", "reproduce", "--output", "runs/paper_reproduction")
                report = json.loads((root / "runs/paper_reproduction/reproduction_report.json").read_text())
                assert report["main_models"] == 32 and report["figure9"]["verbalization_better_than_visual"] == 15
        finally:
            if previous_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous_pythonpath
        print(json.dumps({"full_snapshot_samples": manifest["samples"], "verified": verified,
            "edit_type": dict(Counter(row["edit_type"] for row in full_rows)),
            "edit_operation": dict(Counter(str(row["edit_operation"]) for row in full_rows)),
            "base_snapshot_gold_text_samples": manifest["text_instructions_present"],
            "prepared_supplement_gold_text_samples": json.loads((root / "data/dataset_manifest.json").read_text())["text_instructions_present"] if args.supplement else 0,
            "cli_smoke_test": "passed (1 sample, mocked generation and judge, network blocked)",
            "four_condition_and_archived_analysis": "passed" if args.supplement else "not requested"}, indent=2))


if __name__ == "__main__":
    main()
