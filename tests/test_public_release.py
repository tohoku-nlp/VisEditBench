"""公開データだけの再現経路と、固定50件の安定性評価を検証する。"""

import copy
import csv
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from PIL import Image

import test_reproduction as fixtures
import viseditbench.data as data
from viseditbench.analysis import rank_statistics
from viseditbench.workflow import prepare_stability, read_json, save_json, select_stability_outputs


class PublicReleaseTests(unittest.TestCase):
    """外部通信なしで、内容照合・同梱データ・250要求の対応を検証する。"""

    setUp = fixtures.ReproductionTests.setUp

    def supplement(self):
        """3件の本物の画像hashとgold指示を持つ小さな追加データを作る。"""
        path = self.base / "supplement"
        path.mkdir()
        data.write_jsonl(path / "gold_text.jsonl", [{"id": row["id"], "text_instruction": "Move the node."} for row in self.rows])
        save_json(path / "manifest.json", {"schema_version": 1,
            "base_metadata_sha256": data.digest(self.metadata),
            "base_benchmark_sha256": data.benchmark_identity(self.rows, metadata=self.metadata, snapshot=self.snapshot),
            "files_sha256": {"gold_text.jsonl": data.digest(path / "gold_text.jsonl")}})
        return path

    def test_cli_uses_bundled_data_without_hf_supplement(self):
        """HF側に追加データがなくても、CLIが同梱goldを自動結合する。"""
        source = self.supplement()
        with patch.object(data, "BUNDLED_SUPPLEMENT", source), patch.object(sys, "argv", [
                "dataset.py", "prepare", "--snapshot", str(self.snapshot), "--data-dir", str(self.data)]):
            data.main()
        rows = data.read_jsonl(self.data / "metadata/edit_metadata.jsonl")
        self.assertTrue(all(row["text_instruction"] == "Move the node." for row in rows))
        self.assertTrue((self.data / "reproduction/manifest.json").is_file())

    def test_license_and_json_order_do_not_change_benchmark_identity(self):
        """license追加・JSONキー順・行順の変更だけなら同じベンチマークと認める。"""
        source = self.supplement()
        changed = [{**dict(reversed(list(row.items()))), "license": "CC-BY-SA-4.0"} for row in reversed(self.rows)]
        data.write_jsonl(self.metadata, changed)
        self.assertNotEqual(data.digest(self.metadata), read_json(source / "manifest.json")["base_metadata_sha256"])
        result = data.prepare(self.snapshot, self.data, supplement_source=source)
        self.assertEqual(result["text_instructions_present"], 3)

    def test_changed_tikz_is_rejected_before_outputs(self):
        """IDが同じでもコードを書き換えた別データへgoldを結合しない。"""
        source = self.supplement()
        changed = copy.deepcopy(self.rows)
        changed[0]["target_tikz"] += "\n% modified"
        data.write_jsonl(self.metadata, changed)
        with self.assertRaisesRegex(ValueError, "different benchmark content"):
            data.prepare(self.snapshot, self.data, supplement_source=source)
        self.assertFalse(self.data.exists())

    def test_changed_image_is_rejected_before_outputs(self):
        """JSONLが同一でもPNG bytesが変わったら結合を拒否する。"""
        source = self.supplement()
        Image.new("RGB", (32, 16), "red").save(self.metadata.parent / self.rows[0]["source_image"])
        with self.assertRaisesRegex(ValueError, "different benchmark content"):
            data.prepare(self.snapshot, self.data, supplement_source=source)
        self.assertFalse(self.data.exists())

    def test_supplement_can_be_added_after_base_only_preparation(self):
        """既存の準備済みデータにも内容hashを照合して追加できる。"""
        source = self.supplement()
        changed = [{**row, "license": "CC-BY-SA-3.0"} for row in self.rows]
        data.write_jsonl(self.metadata, changed)
        data.prepare(self.snapshot, self.data)
        result = data.add_supplement(source, self.data)
        self.assertEqual(result["gold_text_samples"], 3)

    def generation_fixture(self):
        """60件の合成した生成済み出力を作り、GPUやAPIに依存しない。"""
        data.prepare(self.snapshot, self.data)
        original = data.read_jsonl(self.data / "metadata/edit_metadata.jsonl")[0]
        rows = [{**original, "key": f"{i:08d}_ADD_1_edit_0", "id": f"{i:08d}_ADD_1_edit_0"} for i in range(1, 61)]
        data.write_jsonl(self.data / "metadata/edit_metadata.jsonl", rows)
        folder = self.base / "generation"
        folder.mkdir()
        data.write_jsonl(folder / "metadata.jsonl", rows)
        statuses = []
        for row in rows:
            save_json(folder / "results" / (row["key"] + ".json"), {"status": "ok"})
            statuses.append({"key": row["key"], "status": "ready", "image": row["org_image_path"]})
        data.write_jsonl(folder / "render_status.jsonl", statuses)
        save_json(folder / "job.json", {"stage": "generation", "model": "mock",
            "expected_keys": [row["key"] for row in rows], "metadata_sha256": data.digest(folder / "metadata.jsonl")})
        return folder / "job.json"

    def test_stability_selection_is_deterministic_and_order_independent(self):
        """同じseedは同じ50件を選び、記録順の変更で対象が変わらない。"""
        job = self.generation_fixture()
        first = select_stability_outputs(job, seed=42)
        statuses = data.read_jsonl(job.parent / "render_status.jsonl")
        data.write_jsonl(job.parent / "render_status.jsonl", list(reversed(statuses)))
        self.assertEqual(first, select_stability_outputs(job, seed=42))
        self.assertNotEqual(first, select_stability_outputs(job, seed=7))
        self.assertEqual(len(first), 50)
        (job.parent / "results" / (statuses[0]["key"] + ".json")).unlink()
        with self.assertRaisesRegex(ValueError, "Finish or retry"):
            select_stability_outputs(job)

    def test_stability_from_public_generation_repeats_same_images(self):
        """公開runから選んだ50件を5回要求へ固定し、歴史的対象とは称さない。"""
        job = self.generation_fixture()
        output = self.base / "stability"
        result = prepare_stability(None, output, self.data, generation_job=job, seed=42)
        self.assertEqual(len(result["judge_jobs"]), 5)
        frozen = read_json(output / "job.json")
        self.assertFalse(frozen["historical_selection_claimed"])
        self.assertEqual(frozen["selection_seed"], 42)
        selected = data.read_jsonl(output / "selection.jsonl")
        for row in selected:
            self.assertEqual(data.digest(Path(row["edited_image_path"])), data.digest(output / "images" / (row["question_id"] + ".png")))
        requests = []
        for filename in result["judge_jobs"]:
            path = Path(filename)
            judge = read_json(path)
            requests.append([record for chunk in judge["chunks"] for record in data.read_jsonl(path.parent / chunk["input"])])
        self.assertTrue(all(len(records) == 50 and records == requests[0] for records in requests))

    def test_released_data_hashes_and_updated_rank_correlations(self):
        """同梱データ全体のhashと、更新Table 6の4指標を実データで検算する。"""
        root = fixtures.ROOT / "reproduction"
        data.verify_supplement(root)
        update = root / "updates/20261107"
        tables = []
        for name in ["table2_corrected_printed.csv", "table5_updated_printed.csv", "table6_correlations.csv"]:
            with (update / name).open() as stream:
                tables.append(list(csv.DictReader(stream)))
        actual = rank_statistics(tables[0], tables[1], printed_values=True)
        expected = {row["metric"]: row for row in tables[2]}
        self.assertEqual(len(actual), 4)
        for row in actual:
            reference = expected[row["metric"]]
            self.assertEqual(row["n_models"], 29)
            self.assertAlmostEqual(row["spearman_r"], float(reference["spearman_rho"]), places=12)
            self.assertAlmostEqual(row["kendall_tau"], float(reference["kendall_tau_b"]), places=12)
            self.assertAlmostEqual(row["spearman_p"] / float(reference["spearman_p_two_sided"]), 1, places=9)
        with self.assertRaises(ValueError):
            rank_statistics(tables[0], tables[1][:-1], printed_values=True)


if __name__ == "__main__":
    unittest.main()
