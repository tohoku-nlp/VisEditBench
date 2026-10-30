"""著者用: 再現実験に必要な追加データをHF公開用フォルダへ書き出す。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viseditbench.backends import backend
from viseditbench.data import digest, read_jsonl, write_jsonl
from viseditbench.workflow import read_json, registry, save_json


def csv_rows(path: Path) -> list[dict]:
    """CSVを文字列として読み、元の丸め・表示値を保つ。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build(workspace: Path, output: Path, *, overwrite_unpublished: bool = False) -> dict:
    """画像データ本体を重複せず、gold・subset・匿名評点・掲載値を一式にする。"""
    destination = output / "reproduction"
    if (destination / "manifest.json").exists() and not overwrite_unpublished:
        raise FileExistsError("Release exists. Use a new output directory to keep published snapshots immutable.")
    destination.mkdir(parents=True, exist_ok=True)
    origins = {}
    catalog = {}
    metadata = read_jsonl(workspace / "data/metadata/edit_metadata_before_batch_api.jsonl")
    hf_metadata = read_jsonl(workspace / "dataset/VisEditBench/dataset/metadata.jsonl")
    keys = [row["id"] for row in hf_metadata]
    original = {row["key"]: row for row in metadata}
    if set(keys) != set(original) or any(not original[k].get("text_instruction") for k in keys):
        raise ValueError("Gold instructions must cover the exact 694 benchmark IDs")
    write_jsonl(destination / "gold_text.jsonl", [{"id": key, "text_instruction": original[key]["text_instruction"]} for key in keys])
    cohorts = {"main": keys}
    for cohort, relative in [("figure6", "data/metadata/ablation_100_eval_with_edited_paths.jsonl"),
                             ("figure7", "data/ablation_input_modality_100/metadata.jsonl"),
                             ("legacy_verbalization", "data/ablation_100/metadata.jsonl")]:
        source = workspace / relative
        selected = [row["key"] for row in read_jsonl(source)]
        if len(selected) != len(set(selected)) or len(selected) != 100 or not set(selected) <= set(keys):
            raise ValueError(f"Invalid cohort: {cohort}")
        cohorts[cohort] = selected
        origins[relative] = digest(source)
    save_json(destination / "cohorts.json", cohorts)

    def snapshot(name: str, relative: str, kind: str, description: str) -> None:
        """公開値の入力表をコピーし、元ファイルのハッシュと用途を記録する。"""
        source = workspace / relative
        if not source.is_file():
            catalog[name] = {"available": False, "reason": "Source file not present locally", "source": relative}
            return
        target = destination / "published" / (name + source.suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        catalog[name] = {"available": True, "file": str(target.relative_to(destination)), "kind": kind, "description": description,
                         "source": relative, "source_sha256": digest(source)}
        origins[relative] = digest(source)

    assets = [
        ("figure4", "paper/figure4_fsr_heatmap_20261020/figure4_values.csv", "figure", "One-operation edit-type heatmap"),
        ("figure5", "paper/figure5_operation_count_20261020/figure5_values_sorted_overall_fsr.csv", "figure", "FSR by operation count"),
        ("figure6", "paper/figures5_9_large_text_20261028/figure6_values.csv", "figure", "Input modality; original denominators retained"),
        ("figure7", "paper/figure7_text_visual_20261020/figure7_values_sorted_overall_fsr.csv", "figure", "Text/visual comparison"),
        ("figure8", "paper/figures5_9_large_text_20261028/figure8_values.csv", "figure", "Code/direct-image comparison"),
        ("figure9", "paper/figure40_no_qwen3vl_20261024/values.csv", "figure", "16 configurations x 4 conditions, each 100 examples"),
        ("table1_image_correlation", "paper/ablation_human_vs_auto/human_vs_auto_mainv2_correlations.csv", "table", "Human/image-judge correlations"),
        ("table2_main", "paper/ablation_gemini_judge_bias/gemini_judge_table2_recomputed.csv", "table", "Existing 29-model main table, source version retained"),
        ("table3_code_correlation", "paper/ablation_human_vs_auto/human_vs_evaluation_code_mainv2_correlations.csv", "table", "Human/code-judge correlations"),
        ("table5_alternative_judge", "paper/ablation_gemini_judge_bias/claude_opus_judge_table2.csv", "table", "Alternative judge, 29 models"),
        ("table6_judge_ranks", "paper/ablation_gemini_judge_bias/judge_rank_correlations_29models.csv", "table", "Judge rank correlation"),
        ("human_agreement_reference", "paper/human_evaluation/analysis_outputs/krippendorff_alpha_ordinal_excluded4workers_new.csv", "table", "Existing human agreement summary"),
    ]
    for arguments in assets:
        snapshot(*arguments)
    # Figure4のセル値を変えず、採用済みの全体FSR順へ行だけを並べ替える。
    ranking_source = workspace / "paper/figure4_fsr_heatmap_20261020/overall_fsr_ranking.csv"
    ranking = {row["model"]: index for index, row in enumerate(csv_rows(ranking_source))}
    figure4_path = destination / catalog["figure4"]["file"]
    figure4_rows = sorted(csv_rows(figure4_path), key=lambda row: (ranking[row["model"]], ["ADD", "REMOVE", "UPDATE", "LAYOUT", "STRUCTURE"].index(row["edit_type"])))
    with figure4_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(figure4_rows[0]))
        writer.writeheader()
        writer.writerows(figure4_rows)
    catalog["figure4"]["transformation"] = "Cell values unchanged; sorted by the adopted overall-FSR ranking"
    origins[str(ranking_source.relative_to(workspace))] = digest(ranking_source)
    # 元の表形式を保ち、後段でCSV/TeXへ変換する。
    snapshot("table4_standard_deviation", "paper/table3/modelwise_scores_std_table.txt", "text_table", "Sample standard deviations")

    model_aliases = {spec["table_slug"]: name for name, spec in registry().items()}
    model_aliases.update(qwen_image_edit_2511="qwen-image-edit", gpt_image_1_5="gpt-image-1.5", gemini_3_pro_image_preview="gemini-3-pro-image-preview")
    image_scores = []
    for source in sorted((workspace / "paper/table4/scores_latest").glob("*.jsonl")):
        if source.stem not in model_aliases:
            continue
        rows = read_jsonl(source)
        if len(rows) != 694 or {row["key"] for row in rows} != set(keys):
            raise ValueError(f"Incomplete archived score group: {source}")
        for row in rows:
            image_scores.append({"key": row["key"], "model": model_aliases[source.stem], "condition": "direct", "judge": "gemini-3.1-pro-preview",
                "edit_type": original[row["key"]]["edit_type"], "edit_operation": int(original[row["key"]]["num_edit_instruction"]),
                **{field: row[field] for field in ["instruction_adherence", "content_preservation", "diagram_readability"]},
                "source": "historical_main", "failure_recorded": bool(row.get("error"))})
        origins[str(source.relative_to(workspace))] = digest(source)
    # 最新のFigure4/5/7に採用された非Thinking4モデルを別スナップショットとして記録する。
    recent = workspace / "experiments/qwen35_nonthinking_gemini31pro_20261018/image_evaluation_gemini31pro/scores.csv"
    if recent.is_file():
        rows = csv_rows(recent)
        write_jsonl(destination / "scores/recent_qwen35_image.jsonl", rows)
        origins[str(recent.relative_to(workspace))] = digest(recent)
    write_jsonl(destination / "scores/main_image.jsonl", image_scores)
    code_scores, code_groups = [], []
    for name, spec in registry().items():
        if spec["access"] == "image":
            continue
        for tag, judge, provider in [("gemini31pro", "gemini-3.1-pro-preview", "google"), ("claude_opus_4_6", "claude-opus-4-6", "anthropic")]:
            folder = workspace / "batch_api_pipeline/evaluation_code/runs" / f"judge_{tag}__gen_{spec['table_slug']}__w_org_image_w_org_tikz"
            sources = sorted(folder.glob("output_*.jsonl"))
            if not sources:
                code_groups.append({"model": name, "judge": judge, "n": 0, "available": False})
                continue
            source = sources[0]
            parsed = backend("code").extract.extract_scores_from_jsonl(source, provider)
            seen = set()
            for row in parsed:
                match = re.match(r"^(\d{8}_(?:ADD|REMOVE|UPDATE|LAYOUT|STRUCTURE)_[123]_edit_\d+)(?:_|$)", row["custom_id"])
                if not match or match[1] not in original or match[1] in seen:
                    raise ValueError(f"Invalid/duplicate archived code score: {source}")
                key = match[1]
                seen.add(key)
                code_scores.append({"key": key, "model": name, "judge": judge, **{k: v for k, v in row.items() if k != "custom_id"}})
            code_groups.append({"model": name, "judge": judge, "n": len(seen), "available": True})
            origins[str(source.relative_to(workspace))] = digest(source)
    write_jsonl(destination / "scores/main_code.jsonl", code_scores)
    save_json(destination / "scores/main_code_coverage.json", code_groups)

    # Figure9は採用された64群だけを抽出。別subset/旧新設定を統合しない。
    values = csv_rows(workspace / "paper/figure40_no_qwen3vl_20261024/values.csv")
    figure9 = []
    old = read_jsonl(workspace / "paper/appendix_verbalize_first_20261020/data/scores.jsonl")
    new = read_jsonl(workspace / "figure7_four_conditions_13models_20261023/analysis/scores.jsonl")
    for value in values:
        model, condition = value["model"], value["condition"]
        if value["source"] == "original_figure40":
            rows = [r for r in old if r["model"] == model and r["condition"] == condition]
        else:
            rows = [r for r in new if r["model"] == model and r["condition"] == condition]
        if len(rows) != 100 or len({r["key"] for r in rows}) != 100:
            raise ValueError(f"Figure9 group is incomplete: {model}/{condition}")
        actual = sum(all(r[f] == 5 for f in ["instruction_adherence", "content_preservation", "diagram_readability"]) for r in rows)
        if actual != float(value["FSR_percent"]):
            raise ValueError(f"Figure9 count differs from published bar: {model}/{condition}")
        for row in rows:
            figure9.append({"key": row["key"], "model": model, "condition": condition, "source": value["source"],
                "judge": "gemini-3.6-flash", "label": value["label"],
                **{field: row[field] for field in ["instruction_adherence", "content_preservation", "diagram_readability"]},
                "status": row.get("status"), "score_assignment": row.get("score_assignment", row.get("status"))})
    write_jsonl(destination / "scores/figure9.jsonl", figure9)

    human_source = workspace / "paper/human_evaluation/human_score/Batch_5386866_batch_results_main_v2_all_excluded_conditional_4workers.csv"
    human_rows = [row for row in csv_rows(human_source) if row["AssignmentStatus"] in {"Approved", "Submitted"}]
    rater_ids = {identifier: f"rater_{index:03d}" for index, identifier in enumerate(sorted({r["WorkerId"] for r in human_rows}))}
    human = []
    for row in human_rows:
        path = row["Input.img_edited_url"].split("/")
        marker = path.index("main_v2")
        human.append({"question_id": row["Input.question_id"], "generated_model": path[marker + 1], "rater": rater_ids[row["WorkerId"]],
            "instruction_adherence": int(row["Answer.score_fidelity"]), "content_preservation": int(row["Answer.score_locality"]),
            "diagram_readability": int(row["Answer.score_layout"])})
    write_jsonl(destination / "human/ratings.jsonl", human)
    origins[str(human_source.relative_to(workspace))] = digest(human_source)
    judges = []
    for source in sorted((workspace / "paper/ablation_human_vs_auto/scores_mainv2").glob("*.jsonl")):
        for row in read_jsonl(source):
            scores = row.get("scores")
            if scores and not row.get("error"):
                judges.append({"question_id": row["question_id"], "judge": source.stem, **scores})
    write_jsonl(destination / "human/image_judges.jsonl", judges)
    code_judges = []
    for folder in (workspace / "batch_api_pipeline/evaluation_code/runs").iterdir():
        if "main_v2_300" not in folder.name or not folder.name.endswith("_qid_key") or not folder.is_dir():
            continue
        provider = "anthropic" if "claude" in folder.name else "openai" if "gpt54" in folder.name else "google"
        for source in sorted(folder.glob("output*.jsonl")):
            for row in backend("code").extract.extract_scores_from_jsonl(source, provider):
                code_judges.append({"question_id": row["custom_id"].split("__")[0], "judge": folder.name,
                                    **{k: v for k, v in row.items() if k != "custom_id"}})
    write_jsonl(destination / "human/code_judges.jsonl", code_judges)

    cases = []
    cases_root = workspace / "paper/ablation_layout_structure_failure_taxonomy/code_level_analysis_20261029"
    for slug in ["figure22_layout", "structure_wrong_destination_opus46_00004572"]:
        source = cases_root / slug / "case.json"
        record = read_json(source)
        case = {k: v for k, v in record.items() if k not in {"sources", "compilations", "comparison_files"}}
        case["files"] = {}
        for name, relative in record["sources"].items():
            asset = workspace / relative
            target = destination / "cases" / slug / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(asset, target)
            case["files"][name] = str(target.relative_to(destination))
        cases.append(case)
    save_json(destination / "cases/index.json", cases)
    save_json(destination / "dataset_construction.json", {"generated_validated": 2819, "passed_validation": 2148,
        "selected_for_visual_annotation": 750, "final_samples": 694, "source": "published manuscript construction statistics",
        "raw_validation_labels_available": False})
    save_json(destination / "catalog.json", catalog)
    save_json(destination / "provenance.json", {"input_sources_sha256": origins,
        "human_export": "Filtered ratings only; worker IDs replaced with independent anonymous labels. No comments, names, HIT/assignment IDs, timestamps, payment details, or URLs exported.",
        "figure9": {"groups": 64, "scores": len(figure9), "models": 16, "N_per_condition": 100},
        "cohort_overlap_figure7_legacy": len(set(cohorts["figure7"]) & set(cohorts["legacy_verbalization"]))})
    manifest = {"schema_version": 1, "base_dataset": "tohoku-nlp/VisEditBench", "base_metadata_sha256": digest(workspace / "dataset/VisEditBench/dataset/metadata.jsonl"),
        "samples": len(keys), "gold_text_samples": len(keys), "published_assets": catalog,
        "coverage_gaps": [name for name, item in catalog.items() if not item["available"]] + ["original_50_case_five_repeat_stability_assets", "raw_dataset_validation_labels"],
        "files_sha256": {str(path.relative_to(destination)): digest(path) for path in sorted(destination.rglob("*")) if path.is_file() and path != destination / "manifest.json"}}
    save_json(destination / "manifest.json", manifest)
    card = (workspace / "dataset/VisEditBench/README.md").read_text()
    pieces = card.split("---", 2)
    if len(pieces) != 3 or "configs:" in pieces[1]:
        raise ValueError("Review the dataset card configuration before preparing the updated card")
    # 再現用JSONLをbenchmarkの行へ混入させないよう、defaultの対象を明示する。
    yaml = pieces[1].rstrip() + "\nconfigs:\n  - config_name: default\n    data_files:\n      - split: train\n        path: dataset/metadata.jsonl\n"
    card = "---" + yaml + "---" + pieces[2]
    card += "\n\n## Reproduction supplement\n\nThe `reproduction/` directory contains gold instructions, fixed ablation cohorts, anonymized human ratings, archived model judgments, and published figure/table inputs. It is separate from the 694 benchmark examples. The default dataset configuration selects only `dataset/metadata.jsonl`. Download the full snapshot with `huggingface_hub.snapshot_download` or the VisEditBench code's `dataset.py download` command to obtain the supplement.\n"
    (output / "README.md").write_text(card, encoding="utf-8")
    (output / "UPLOAD.md").write_text("# Hugging Face追加公開用データ\n\n`reproduction/`と、defaultの対象を明示したREADME.mdを既存データセットへ追加します。コードのGitHubリポジトリには入れません。\n\n"
        "```bash\nhf upload tohoku-nlp/VisEditBench ./reproduction reproduction --repo-type dataset\nhf upload tohoku-nlp/VisEditBench ./README.md README.md --repo-type dataset\n```\n\n"
        "このコマンドはこのフォルダでユーザーが実行します。アップロード後のコミットを指定してコード側のdataset.py downloadを実行します。\n", encoding="utf-8")
    return {"output": str(output), "files": len(manifest["files_sha256"]), "gold_text": len(keys), "figure9_scores": len(figure9),
            "human_ratings": len(human), "main_image_scores": len(image_scores), "coverage_gaps": manifest["coverage_gaps"]}


def main() -> None:
    """転送を実行せず、著者workspaceからローカル公開用データを準備する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite-unpublished", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.workspace.resolve(), args.output.resolve(), overwrite_unpublished=args.overwrite_unpublished), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
