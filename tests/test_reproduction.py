"""実データや外部APIを使わず、データ準備から評価集計まで検証する。"""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import shutil
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from viseditbench.data import attach, digest, download, prepare, read_jsonl, verify, write_jsonl
from viseditbench.results import FIELDS, score_rows, summarize

ROOT = Path(__file__).resolve().parents[1]
MODE = "w_org_image_w_org_tikz"
PROMPT = "visual_instruction_" + MODE
MODEL = "gemini-3.1-pro-preview"
DOCUMENT = r"""\documentclass{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}\draw (0,0) -- (1,0);\end{tikzpicture}
\end{document}"""


def load_backend(name: str, directory: Path) -> types.SimpleNamespace:
    """同名libの衝突を避け、各バックエンドを別パッケージとして読み込む。"""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / "__init__.py",
                                                     submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return types.SimpleNamespace(**{part: importlib.import_module(name + "." + part)
                                   for part in ["config", "generate", "extract"]})


def load_file(name: str, path: Path) -> types.ModuleType:
    """副作用のない単独モジュールをテスト用の名前で読み込む。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproductionTests(unittest.TestCase):
    """合成した3件で公開ワークフローの境界と失敗時の集計を確認する。"""

    def setUp(self) -> None:
        """一時データセットを作り、テスト中の外部接続を禁止する。"""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        network_guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        network_guard.start()
        self.addCleanup(network_guard.stop)
        self.snapshot = self.base / "snapshot"
        self.metadata = self.snapshot / "dataset/metadata.jsonl"
        self.rows = []
        for number, edit in enumerate(["ADD", "STRUCTURE", "LAYOUT"], 1):
            key = f"{number:08d}_{edit}_{number}_edit_0"
            row = dict(id=key, edit_type=edit, edit_operation=number,
                       source_tikz=DOCUMENT, target_tikz=DOCUMENT, uri="https://example.org/diagram")
            for field, folder in [("source_image", "source_images"), ("visual_instruction", "visual_instructions"),
                                  ("target_image", "target_images")]:
                row[field] = f"{folder}/{key}.png"
                image = self.metadata.parent / row[field]
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 16), "white").save(image)
            self.rows.append(row)
        write_jsonl(self.metadata, self.rows)
        self.data = self.base / "data"
        self.edit = load_backend("test_edit_backend", ROOT / "editing/batch_api/lib")
        self.image_eval = load_backend("test_image_backend", ROOT / "evaluation_image-level/batch_api/lib")
        self.code_eval = load_backend("test_code_backend", ROOT / "evaluation_code-level/batch_api/lib")

    def prepare_data(self, **kwargs) -> list[dict]:
        """共通のメタデータを生成し、そのレコードを返す。"""
        prepare(self.snapshot, self.data, **kwargs)
        return read_jsonl(self.data / "metadata/edit_metadata.jsonl")

    def test_prepare_preserves_images_code_order_and_manifest(self) -> None:
        """画像・コード・ID・順序が変換で変わらないことを検証する。"""
        rows = self.prepare_data(revision="test-commit")
        self.assertEqual([r["key"] for r in rows], [r["id"] for r in self.rows])
        self.assertEqual(verify(self.data / "metadata/edit_metadata.jsonl")["samples"], 3)
        manifest = json.loads((self.data / "dataset_manifest.json").read_text())
        self.assertEqual(manifest["revision"], "test-commit")
        self.assertEqual(manifest["text_instructions_present"], 0)
        for row, source in zip(rows, self.rows):
            self.assertNotIn("text_instruction", row)
            self.assertEqual(Path(row["org_tikz_code_path"]).read_text(), DOCUMENT)
            self.assertEqual(digest(Path(row["org_image_path"])), digest(self.metadata.parent / source["source_image"]))
        for path, expected in manifest["files_sha256"].items():
            self.assertEqual(digest(Path(path)), expected)
        with self.assertRaises(FileExistsError):
            prepare(self.snapshot, self.data)

    def test_imagefolder_aliases_root_metadata_and_limit(self) -> None:
        """HFのfile_name形式とメタデータ直下形式を受け入れる。"""
        for row in self.rows:
            for field in ["source_image", "visual_instruction", "target_image"]:
                row[field + "_file_name"] = "dataset/" + row.pop(field)
        self.metadata.unlink()
        write_jsonl(self.snapshot / "metadata_hf.jsonl", self.rows)
        rows = self.prepare_data(limit=1)
        self.assertEqual(len(rows), 1)
        manifest = json.loads((self.data / "dataset_manifest.json").read_text())
        self.assertEqual(manifest["full_dataset_samples"], 3)

    def test_invalid_metadata_is_rejected_before_writes(self) -> None:
        """重複ID、種別不一致、パストラバーサル、欠損画像を拒否する。"""
        variants = []
        duplicate = copy.deepcopy(self.rows)
        duplicate.append(duplicate[0])
        variants.append(duplicate)
        for field, value in [("edit_type", "REMOVE"), ("source_image", "../outside.png"),
                             ("source_image", "source_images/missing.png"), ("source_tikz", "")]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows[0]["source_image"]):
                write_jsonl(self.metadata, rows)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    prepare(self.snapshot, self.data)
                self.assertFalse(self.data.exists())

    def test_supplemental_text_is_explicitly_joined(self) -> None:
        """外部のgold指示だけをIDで結合し、指示を自動生成しない。"""
        supplement = self.base / "text.jsonl"
        write_jsonl(supplement, [{"id": self.rows[0]["id"], "text_instruction": "Move the node."}])
        rows = self.prepare_data(text_instructions=supplement)
        self.assertEqual(rows[0]["text_instruction"], "Move the node.")
        self.assertNotIn("text_instruction", rows[1])
        write_jsonl(supplement, [{"id": "unknown", "text_instruction": "Wrong dataset."}])
        with self.assertRaises(ValueError):
            prepare(self.snapshot, self.data, force=True, text_instructions=supplement)

    def test_download_pins_resolved_dataset_commit(self) -> None:
        """ネットワークを使わずHF呼び出しの種類・コミット固定を検証する。"""
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as fetch:
            api.return_value.repo_info.return_value.sha = "resolved-sha"
            fetch.return_value = str(self.snapshot)
            snapshot, sha = download("tohoku-nlp/VisEditBench", "requested-tag", self.data)
            api.return_value.repo_info.assert_called_once_with("tohoku-nlp/VisEditBench", repo_type="dataset", revision="requested-tag")
            self.assertEqual(sha, "resolved-sha")
            self.assertEqual(snapshot, self.snapshot)
            self.assertEqual(fetch.call_args.kwargs["revision"], "resolved-sha")
            self.assertEqual(fetch.call_args.kwargs["repo_type"], "dataset")
            self.assertIn("dataset/visual_instructions/*.png", fetch.call_args.kwargs["allow_patterns"])

    def test_three_providers_generate_and_extract_full_sample_keys(self) -> None:
        """3社の要求生成と模擬応答抽出を通し、Claudeの短縮IDも復元する。"""
        rows = self.prepare_data()
        fields, configs = self.edit.config.load_config(ROOT / "configs/editing.toml")
        for config in configs:
            with self.subTest(provider=config.provider):
                config.metadata_path = self.data / "metadata/edit_metadata.jsonl"
                config.save_path = self.base / config.name / "input.jsonl"
                count, _ = self.edit.generate.generate_jsonl_for_experiment(config, fields, None, False)
                self.assertEqual(count, 3)
                requests = read_jsonl(config.save_path)
                responses = []
                for request in requests:
                    key = request.get("key") or request["custom_id"]
                    if config.provider == "google":
                        response = {"key": key, "response": {"candidates": [{"content": {"parts": [{"text": DOCUMENT}]}}]}}
                    elif config.provider == "openai":
                        response = {"custom_id": key, "response": {"body": {"output": [{"type": "message", "content": [{"type": "output_text", "text": DOCUMENT}]}]}}}
                    else:
                        self.assertLessEqual(len(key), 64)
                        response = {"custom_id": key, "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": DOCUMENT}]}}}
                    responses.append(response)
                output = config.save_path.with_name("output.jsonl")
                write_jsonl(output, responses)
                tex = config.save_path.parent / "tikz"
                self.assertEqual(self.edit.extract.extract_and_save(output, tex, config.provider, PROMPT), (3, 0))
                self.assertEqual(sorted(p.stem for p in tex.glob("*.tex")), sorted(r["key"] for r in rows))

    def create_attached_outputs(self) -> Path:
        """画像あり・コンパイル失敗・未生成の3状態を含むメタデータを作る。"""
        rows = self.prepare_data()
        tex, images = self.base / "work/tikz", self.base / "work/images"
        tex.mkdir(parents=True)
        images.mkdir()
        for row in rows[:2]:
            (tex / (row["key"] + ".tex")).write_text(DOCUMENT)
        shutil.copy2(rows[0]["org_image_path"], images / (rows[0]["key"] + ".png"))
        output = self.base / "attached.jsonl"
        result = attach(self.data / "metadata/edit_metadata.jsonl", output, MODEL, MODE, tex, images)
        self.assertEqual(result["statuses"], {"ready": 1, "compile_failed": 1, "missing_generation": 1})
        return output

    def test_evaluation_requests_skip_missing_outputs(self) -> None:
        """画像評価は描画成功1件、コード評価は生成コード2件だけを送る。"""
        attached = self.create_attached_outputs()
        for backend, filename, expected in [(self.image_eval, "image_eval.toml", 1), (self.code_eval, "code_eval.toml", 2)]:
            fields, configs = backend.config.load_config(ROOT / "configs" / filename)
            config = configs[0]
            config.metadata_path, config.save_path = attached, self.base / filename / "input.jsonl"
            count, _ = backend.generate.generate_jsonl_for_experiment(config, fields, None, False)
            self.assertEqual(count, expected)
            self.assertEqual(len(read_jsonl(config.save_path)), expected)

    def test_full_denominator_and_pending_judge_distinction(self) -> None:
        """失敗を1点として全件分母に含め、採点未完了時には最終FSRを出さない。"""
        attached = self.create_attached_outputs()
        scores = self.base / "scores.jsonl"
        write_jsonl(scores, [{"key": self.rows[0]["id"], **dict.fromkeys(FIELDS, 5)}])
        summary, cases = summarize(attached, scores, MODEL, MODE)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["overall"]["n"], 3)
        self.assertAlmostEqual(summary["overall"]["FSR_percent"], 100 / 3)
        self.assertEqual([r["FSR"] for r in cases], [1, 0, 0])
        write_jsonl(scores, [])
        with self.assertRaisesRegex(ValueError, "unscored"):
            summarize(attached, scores, MODEL, MODE)
        partial, _ = summarize(attached, scores, MODEL, MODE, allow_partial=True)
        self.assertIsNone(partial["overall"])
        self.assertFalse(partial["complete"])
        self.assertEqual(partial["unscored"], 1)

    def test_unknown_duplicate_and_invalid_score_rows_rejected(self) -> None:
        """誤ったIDの混入、二重採点、非整数値をFSRへ混ぜない。"""
        attached = self.create_attached_outputs()
        scores = self.base / "scores.jsonl"
        good = {"key": self.rows[0]["id"], **dict.fromkeys(FIELDS, 5)}
        for rows in [[good, good], [dict(good, key="99999999_ADD_1_edit_0")]]:
            write_jsonl(scores, rows)
            with self.assertRaises(ValueError):
                summarize(attached, scores, MODEL, MODE)
        for value in [True, 5.0, 0, 6, "5"]:
            write_jsonl(scores, [dict(good, instruction_adherence=value)])
            with self.assertRaises(ValueError):
                score_rows(scores)

    def test_image_and_code_parsers_reject_fractional_and_boolean_scores(self) -> None:
        """プロバイダ応答の不正評点を整数へ丸めず拒否する。"""
        image = dict.fromkeys(FIELDS, 5)
        code = {"metrics": {"Instruction Adherence": {"score": 5}, "Content Preservation": {"score": 4}}, "overall_alignment_score": 4}
        self.assertEqual(self.image_eval.extract._parse_json_scores(json.dumps(image)), image)
        self.assertEqual(self.code_eval.extract._parse_json_scores(json.dumps(code))["content_preservation"], 4)
        for value in [True, 4.5, 5.0, 0, 6, "5"]:
            invalid_image = dict(image, instruction_adherence=value)
            self.assertIsNone(self.image_eval.extract._parse_json_scores(json.dumps(invalid_image)))
            invalid_code = copy.deepcopy(code)
            invalid_code["metrics"]["Instruction Adherence"]["score"] = value
            self.assertIsNone(self.code_eval.extract._parse_json_scores(json.dumps(invalid_code)))

    def test_preview_preserves_existing_requests(self) -> None:
        """3種類の要求プレビューで既存の要求ファイルが空にならないことを確認する。"""
        attached = self.create_attached_outputs()
        for backend, name in [(self.edit, "editing.toml"), (self.image_eval, "image_eval.toml"), (self.code_eval, "code_eval.toml")]:
            fields, configs = backend.config.load_config(ROOT / "configs" / name)
            config = configs[0]
            config.metadata_path, config.save_path = attached, self.base / name / "input.jsonl"
            config.save_path.parent.mkdir()
            config.save_path.write_text("keep existing requests\n")
            count, preview = backend.generate.generate_jsonl_for_experiment(config, fields, 1, True)
            self.assertEqual(count, 0)
            self.assertIsInstance(preview, dict)
            self.assertEqual(config.save_path.read_text(), "keep existing requests\n")

    def test_vllm_healthcheck_is_mocked(self) -> None:
        """実サーバーへ接続せず標準healthエンドポイントへの要求を確認する。"""
        client = load_file("test_http_client", ROOT / "editing/vllm/src/http_client.py")
        with patch.object(client.requests, "get") as get:
            self.assertTrue(client.wait_for_server_ready("http://127.0.0.1:8000/", {}, 1, 0, 1, Mock()))
            self.assertEqual(get.call_args.kwargs["url"], "http://127.0.0.1:8000/health")

    @unittest.skipUnless(all(shutil.which(tool) for tool in ["pdflatex", "pdfcrop", "pdftoppm"]), "TeX/Poppler not installed")
    def test_real_tikz_render_and_failed_overwrite_removes_stale_image(self) -> None:
        """実際のTikZを描画し、再コンパイル失敗後に古い成功画像が残らないことを確認する。"""
        compiler = load_file("test_compiler", ROOT / "editing/compile_tikz_batch.py")
        tex = self.base / "render/tikz/test.tex"
        tex.parent.mkdir(parents=True)
        tex.write_text(DOCUMENT)
        images = self.base / "render/images"
        args = argparse.Namespace(env="apptainer", apptainer=None, bind=None, overwrite=True, dpi=72)
        self.assertTrue(compiler.compile_tex_to_png(tex, images, args))
        with Image.open(images / "test.png") as image:
            self.assertGreater(image.width, 10)
        tex.write_text(DOCUMENT.replace(r"\draw", r"\undefinedcommand"))
        self.assertFalse(compiler.compile_tex_to_png(tex, images, args))
        self.assertFalse((images / "test.png").exists())


if __name__ == "__main__":
    unittest.main()
