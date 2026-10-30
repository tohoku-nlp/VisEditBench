"""論文の条件・対象ID・再送・評点・コンテナ定義をネットワークなしで検証する。"""

from __future__ import annotations

import base64
import fcntl
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import test_reproduction as fixtures
from viseditbench.analysis import aggregate, human_statistics, ordinal_alpha, stability_statistics
from viseditbench.data import digest, install_supplement, read_jsonl, write_jsonl
from viseditbench.image_edit import run_images
from viseditbench.workflow import (CONDITIONS, IMAGE_FIELDS, build_request, collect_job, experiment_groups,
    prepare_experiment, prepare_judge, read_json, registry, render_job, report_job, retry_job,
    save_json, server_command, store_generation, submit_job, prepare_stability, report_stability)

DOCUMENT, ROOT = fixtures.DOCUMENT, fixtures.ROOT


class PaperWorkflowTests(unittest.TestCase):
    """3件の合成データに全モデル・条件・API境界を通して確認する。"""

    setUp = fixtures.ReproductionTests.setUp
    prepare_data = fixtures.ReproductionTests.prepare_data

    def fixture(self) -> list[dict]:
        """goldと異なる2つのsubsetを持つ合成データを作る。"""
        rows = self.prepare_data()
        for row in rows:
            row["text_instruction"] = "Move the left node."
        write_jsonl(self.data / "metadata/edit_metadata.jsonl", rows)
        save_json(self.data / "reproduction/cohorts.json", {"figure6": [r["key"] for r in rows],
            "figure7": [r["key"] for r in rows[:2]], "legacy_verbalization": [r["key"] for r in rows[1:]]})
        return rows

    def one_job(self, model: str = "gemini-3.1-pro-preview", condition: str = "verbalize_first", limit: int = 1) -> Path:
        """指定条件の小さな要求runを作り、そのjobファイルを返す。"""
        self.fixture()
        plan = prepare_experiment("verbalization", self.base / "run", self.data,
                                  models=[model], conditions=[condition], limit=limit)
        return Path(plan["jobs"][0])

    def google_response(self, key: str, text: str) -> dict:
        """正常なGoogle Batch応答を模擬する。"""
        return {"key": key, "response": {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}}

    def test_all_models_and_four_conditions_are_materialized(self) -> None:
        """全29コードモデルの4条件で実データ用プレースホルダーを埋められる。"""
        row = self.fixture()[0]
        count = 0
        self.assertEqual(len(registry()), 32)
        for model, spec in registry().items():
            if spec["access"] == "image":
                continue
            for condition in CONDITIONS:
                with self.subTest(model=model, condition=condition):
                    body = build_request(model, row, condition)
                    encoded = json.dumps(body)
                    self.assertNotIn("$SOURCE_TIKZ", encoded)
                    self.assertNotIn("$ORG_BASE64", encoded)
                    self.assertIn("Move the left node.", encoded) if condition in {"gold_text", "gold_text_visual"} else None
                    self.assertIn("<instruction>", encoded) if condition == "verbalize_first" else None
                    if model.endswith("-nothink") and spec["backend"] == "vllm":
                        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
                    count += 1
        self.assertEqual(count, 116)

    def test_model_options_are_unchanged_across_conditions(self) -> None:
        """言語化と両指示の追加でseed・sampling・token上限を変更しない。"""
        row = self.fixture()[0]
        for model, spec in registry().items():
            if spec["access"] == "image":
                continue
            options = [{k: v for k, v in build_request(model, row, c).items() if k not in {"input", "contents", "systemInstruction", "messages", "system"}} for c in CONDITIONS]
            self.assertTrue(all(o == options[0] for o in options))
        for model in ["qwen3.5-4b", "qwen3.5-35b-a3b", "gemma4-31b-it", "gemma4-e4b-it"]:
            body = build_request(model, row, "verbalize_first", profile="legacy_verbalization")
            self.assertEqual(body["seed"], 42)
            self.assertEqual(body["max_tokens"], 50000)

    def test_original_100_case_cohorts_are_not_mixed(self) -> None:
        """現行Figure9の64群で旧新のsubsetとprofileを保持する。"""
        self.fixture()
        groups = experiment_groups("verbalization")
        self.assertEqual(len(groups), 64)
        self.assertEqual(sum(g["cohort"] == "figure7" for g in groups), 48)
        self.assertEqual(sum(g["cohort"] == "legacy_verbalization" for g in groups), 16)
        self.assertEqual({g["cohort"] for g in experiment_groups("modality")}, {"figure6"})
        plan = prepare_experiment("verbalization", self.base / "run", self.data,
                                  models=["gemini-3.1-pro-preview", "gemma4-31b-it"], conditions=["direct"], limit=1)
        first, second = [read_jsonl(Path(p).parent / "metadata.jsonl")[0]["key"] for p in plan["jobs"]]
        self.assertNotEqual(first, second)

    def test_seed_omission_and_server_revision_options(self) -> None:
        """4B/9Bのseed省略と35B/旧実験のseed42を区別する。"""
        for model in ["qwen3.5-4b-nothink", "qwen3.5-9b-nothink"]:
            self.assertNotIn("--seed", server_command(model))
        self.assertIn("--seed", server_command("qwen3.5-35b-a3b-nothink"))
        command = server_command("qwen3.5-4b", profile="legacy_verbalization")
        self.assertEqual(command[command.index("--generation-config") + 1], "vllm")

    def test_def_sources_exist_and_match_their_backend(self) -> None:
        """両defの実在と全%files入力を検証し、旧パスへの依存を防ぐ。"""
        for name in ["editing", "evaluation_image-level"]:
            recipe = ROOT / name / "vllm/vllm.def"
            self.assertTrue(recipe.is_file())
            in_files = False
            sources = []
            for line in recipe.read_text().splitlines():
                if line.startswith("%"):
                    in_files = line == "%files"
                elif in_files and line.strip():
                    sources.append(line.split()[0])
            self.assertEqual(len(sources), 1)
            self.assertTrue((ROOT / sources[0]).is_file())
            self.assertIn(name + "/vllm/entrypoint.sh", sources[0])

    def test_supplement_checksum_and_traversal_rejection(self) -> None:
        """追加データの改変・欠落と外部パス参照を検出する。"""
        source = self.base / "supplement"
        source.mkdir()
        (source / "gold_text.jsonl").write_text("{}\n")
        save_json(source / "manifest.json", {"schema_version": 1, "files_sha256": {"gold_text.jsonl": digest(source / "gold_text.jsonl")}})
        self.assertEqual(install_supplement(source, self.base / "installed")["verified_files"], 1)
        (source / "gold_text.jsonl").write_text("changed")
        with self.assertRaises(ValueError):
            install_supplement(source, self.base / "installed2")
        save_json(source / "manifest.json", {"schema_version": 1, "files_sha256": {"../outside": "x"}})
        with self.assertRaises(ValueError):
            install_supplement(source, self.base / "installed3")

    def test_batch_submission_resumes_without_double_sending(self) -> None:
        """再実行は既存Batchを保持し、送信結果不明時には再送しない。"""
        job = self.one_job()
        with patch("viseditbench.providers.submit_one", return_value={"job_id": "mock-id"}) as send:
            submit_job(job, client=Mock())
            submit_job(job, client=Mock())
            self.assertEqual(send.call_count, 1)
        value = read_json(job)
        value["chunks"][0]["status"] = "submission_unknown"
        save_json(job, value)
        with patch("viseditbench.providers.submit_one") as send:
            with self.assertRaises(ValueError):
                submit_job(job, client=Mock())
            send.assert_not_called()

    def test_two_simultaneous_submitters_are_rejected(self) -> None:
        """別端末の処理が保持するロックを横取りしてAPIへ送信しない。"""
        job = self.one_job()
        with job.with_suffix(".lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch("viseditbench.providers.submit_one") as send:
                with self.assertRaises(RuntimeError):
                    submit_job(job, client=Mock())
                send.assert_not_called()

    def test_alternative_judge_uses_existing_outputs(self) -> None:
        """judge比較のために図を再生成する誤った実験計画を拒否する。"""
        self.fixture()
        self.assertTrue(all(r["stage"] == "judge_only" for r in experiment_groups("alternative_judge")))
        with self.assertRaisesRegex(ValueError, "do not regenerate"):
            prepare_experiment("alternative_judge", self.base / "run", self.data)

    def test_collect_separates_instruction_and_tex(self) -> None:
        """1回の応答から書き下し文とTikZを別保存し、元応答も保持する。"""
        job = self.one_job()
        value = read_json(job)
        value["chunks"][0].update(status="submitted", job_id="mock")
        save_json(job, value)
        key = value["expected_keys"][0]
        raw = self.google_response(key, "<instruction>Move the left node.</instruction>\n" + DOCUMENT)
        with patch("viseditbench.providers.get_job", return_value=(Mock(), "JOB_STATE_SUCCEEDED", True)), patch("viseditbench.providers.download_results", return_value=[raw]):
            collect_job(job, client=Mock())
        self.assertEqual((job.parent / "instructions" / (key + ".txt")).read_text(), "Move the left node.")
        self.assertEqual((job.parent / "tikz" / (key + ".tex")).read_text(), DOCUMENT)
        self.assertEqual(read_json(job.parent / "raw" / (key + ".json")), raw)

    def test_retry_only_failed_id_merges_into_original_run(self) -> None:
        """欠落応答1件だけを再試行し、親の成功評点へ対応付ける。"""
        job = self.one_job()
        value = read_json(job)
        key = value["expected_keys"][0]
        save_json(job.parent / "results" / (key + ".json"), {"key": key, "status": "api_error"})
        retry = Path(retry_job(job)["retry_job"])
        retry_value = read_json(retry)
        retry_value["chunks"][0].update(status="submitted", job_id="mock-retry")
        save_json(retry, retry_value)
        raw = self.google_response(key, "<instruction>Move.</instruction>\n" + DOCUMENT)
        with patch("viseditbench.providers.get_job", return_value=(Mock(), "JOB_STATE_SUCCEEDED", True)), patch("viseditbench.providers.download_results", return_value=[raw]):
            collect_job(retry, client=Mock())
        self.assertEqual(read_json(job.parent / "results" / (key + ".json"))["status"], "ok")
        self.assertTrue((job.parent / "tikz" / (key + ".tex")).is_file())

    def test_changed_input_is_rejected_before_submission(self) -> None:
        """準備後に変更された元コードで別の要求を評価しない。"""
        job = self.one_job()
        row = read_jsonl(job.parent / "metadata.jsonl")[0]
        Path(row["org_tikz_code_path"]).write_text("changed source")
        with patch("viseditbench.providers.submit_one") as send:
            with self.assertRaisesRegex(ValueError, "input changed"):
                submit_job(job, client=Mock())
            send.assert_not_called()

    def test_direct_google_image_backend_with_mocked_api(self) -> None:
        """直接画像編集のAPI応答を検証済みPNGと結果へ保存する。"""
        self.fixture()
        plan = prepare_experiment("direct_image", self.base / "run", self.data, models=["gemini-3-pro-image-preview"], limit=1)
        job = Path(plan["jobs"][0])
        buffer = BytesIO()
        Image.new("RGB", (16, 16), "white").save(buffer, format="PNG")
        response = Mock()
        response.json.return_value = {"candidates": [{"content": {"parts": [{"inlineData": {"data": base64.b64encode(buffer.getvalue()).decode()}}]}}]}
        post = Mock(return_value=response)
        self.assertEqual(run_images(job, clients={"post": post})["completed"], 1)
        self.assertEqual(run_images(job, clients={"post": post})["completed"], 0)
        self.assertEqual(post.call_count, 1)

    def test_statistics_and_ordinal_agreement(self) -> None:
        """既知の小例でFSR・標準偏差・ordinal alphaを確認する。"""
        rows = [dict.fromkeys(IMAGE_FIELDS, 5), dict.fromkeys(IMAGE_FIELDS, 1)]
        self.assertEqual(aggregate(rows)["FSR_percent"], 50)
        self.assertEqual(aggregate(rows)["instruction_adherence"], 3)
        self.assertEqual(ordinal_alpha([[1, 1], [5, 5]]), 1)
        self.assertAlmostEqual(ordinal_alpha([[1, 5], [1, 5]]), -0.5)
        self.assertIsNone(ordinal_alpha([[5, 5], [5, 5]]))

    def test_human_correlation_matches_by_question_id(self) -> None:
        """順序によらずquestion_idで結合し、同じ評点なら相関1となる。"""
        ratings = [{"question_id": str(i), "generated_model": "m", "rater": "a", **dict.fromkeys(IMAGE_FIELDS, i)} for i in [1, 2, 3]]
        auto = [{"question_id": str(i), "judge": "j", **dict.fromkeys(IMAGE_FIELDS, i)} for i in [3, 2, 1]]
        result = human_statistics(ratings, auto)
        self.assertTrue(all(row["spearman_r"] == 1 for row in result))
        self.assertTrue(all(row["n"] == 3 for row in result))

    def test_stability_requires_all_50_cases_and_5_repeats(self) -> None:
        """不完全な反復や異なる対象集合を平均して安定性としない。"""
        values = [{"question_id": str(i), "repeat": n, **dict.fromkeys(IMAGE_FIELDS, 4)} for i in range(50) for n in range(5)]
        self.assertTrue(all(row["mean_sample_std"] == 0 for row in stability_statistics(values)))
        with self.assertRaises(ValueError):
            stability_statistics(values[:-1])

    def test_stability_prepares_five_judgments_for_original_images(self) -> None:
        """元50出力を変更せず同じ250要求を準備・集計できる。"""
        row = self.fixture()[0]
        selection = self.base / "selection.jsonl"
        write_jsonl(selection, [{"question_id": f"question_{i:03d}", "source_key": row["key"],
                                "generated_model": "mock", "edited_image_path": row["org_image_path"]} for i in range(50)])
        run = self.base / "stability"
        result = prepare_stability(selection, run, self.data)
        self.assertEqual(len(result["judge_jobs"]), 5)
        for job in result["judge_jobs"]:
            path = Path(job)
            value = read_json(path)
            self.assertEqual(len(value["expected_keys"]), 50)
            for key in value["expected_keys"]:
                save_json(path.parent / "scores" / (key + ".json"), {"status": "ok", "scores": dict.fromkeys(IMAGE_FIELDS, 4)})
        summary = report_stability(run)
        self.assertTrue(all(r["mean_score"] == 4 and r["mean_sample_std"] == 0 for r in summary))

    @unittest.skipUnless(all(shutil.which(tool) for tool in ["pdflatex", "pdfcrop", "pdftoppm"]), "TeX/Poppler not installed")
    def test_real_render_judge_preparation_and_final_report(self) -> None:
        """言語化生成から実描画・judge要求・全件分母集計まで一通り検証する。"""
        job = self.one_job(limit=2)
        value = read_json(job)
        for index, key in enumerate(value["expected_keys"]):
            document = DOCUMENT if index == 0 else DOCUMENT.replace(r"\draw", r"\undefinedcommand")
            store_generation(job.parent, key, value, self.google_response(key, "<instruction>Move.</instruction>\n" + document))
        self.assertEqual(render_job(job)["rendered"], 1)
        judged = Path(prepare_judge(job)[0])
        self.assertEqual(len(read_jsonl(judged.parent / "input_000.jsonl")), 1)
        key = value["expected_keys"][0]
        save_json(judged.parent / "scores" / (key + ".json"), {"key": key, "status": "ok", "scores": dict.fromkeys(IMAGE_FIELDS, 5)})
        summary = report_job(judged)
        self.assertEqual(summary["overall"]["n"], 2)
        self.assertEqual(summary["overall"]["FSR_percent"], 50)
        self.assertEqual(summary["CFR_percent"], 50)
        self.assertEqual(len(prepare_judge(job, kind="code")), 1)

    def test_cli_verbalize_help_is_available(self) -> None:
        """公開したverbalize入口を実際のCLIで確認する。"""
        process = subprocess.run([sys.executable, str(ROOT / "edit.py"), "verbalize", "--help"], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("gold_text_visual", process.stdout)


if __name__ == "__main__":
    unittest.main()
