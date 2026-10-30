"""保存済みの評点・人手評価・掲載値から表、図、定性比較をオフライン再生成する。"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import difflib
import html
import json
import math
from pathlib import Path
import shutil
from statistics import mean, stdev
from typing import Any

from .data import digest, install_supplement, read_jsonl
from .workflow import IMAGE_FIELDS, read_json, save_json


def read_csv(path: Path) -> list[dict]:
    """掲載値のCSVを読み、丸め前の数値文字列を保持する。"""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_table(path: Path, rows: list[dict]) -> None:
    """同じ値のCSVとinput可能なLaTeX表を出力する。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def escape(value: Any) -> str:
        """LaTeXの特殊文字をエスケープし、浮動小数を読みやすく表示する。"""
        if isinstance(value, float):
            value = f"{value:.6g}"
        value = "" if value is None else str(value)
        replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
        return "".join(replacements.get(char, char) for char in value)
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline", " & ".join(map(escape, columns)) + r" \\", r"\hline"]
    lines += [" & ".join(escape(row.get(key)) for key in columns) + r" \\" for row in rows]
    lines += [r"\hline", r"\end{tabular}"]
    path.with_suffix(".tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(rows: list[dict], fields: tuple[str, ...] = IMAGE_FIELDS) -> dict:
    """整数評点から平均・標本標準偏差・全観点5点の割合を再計算する。"""
    if not rows:
        return {"n": 0}
    if any(type(row.get(field)) is not int or not 1 <= row[field] <= 5 for row in rows for field in fields):
        raise ValueError("Scores must be integers in [1,5]")
    values = {"n": len(rows)}
    for field in fields:
        values[field] = mean(row[field] for row in rows)
        values[field + "_std"] = stdev(row[field] for row in rows) if len(rows) > 1 else 0.0
    values["FSR_count"] = sum(all(row[f] == 5 for f in fields) for row in rows)
    values["FSR_percent"] = 100 * values["FSR_count"] / len(rows)
    return values


def ordinal_alpha(units: list[list[int]]) -> float | None:
    """欠損・異なる評価者数に対応したordinal Krippendorff alphaを計算する。"""
    counts = [Counter(unit) for unit in units if len(unit) >= 2]
    marginal = Counter()
    for count in counts:
        marginal.update(count)
    total = sum(marginal.values())
    if total < 2:
        return None
    categories = sorted(marginal)
    observed = expected = 0.0
    for first in categories:
        for second in categories:
            if first == second:
                continue
            low, high = sorted((first, second))
            distance = (sum(marginal[k] for k in categories if low <= k <= high) - (marginal[first] + marginal[second]) / 2) ** 2
            coincidence = sum(count[first] * count[second] / (sum(count.values()) - 1) for count in counts)
            observed += coincidence * distance
            expected += marginal[first] * marginal[second] / (total - 1) * distance
    return 1 - observed / expected if expected else None


def human_statistics(ratings: list[dict], auto: list[dict], *, code: bool = False) -> list[dict]:
    """人手平均と自動評点をquestion_idで結合し、相関と平均を計算する。"""
    from scipy.stats import kendalltau, spearmanr
    humans = defaultdict(list)
    for row in ratings:
        humans[row["question_id"]].append(row)
    human_mean = {key: {field: mean(row[field] for row in values) for field in IMAGE_FIELDS} for key, values in humans.items()}
    generated = {key: values[0]["generated_model"] for key, values in humans.items()}
    indexed = {}
    for row in auto:
        key = (row["judge"], row["question_id"])
        if key in indexed and any(indexed[key].get(field) != row.get(field) for field in (*IMAGE_FIELDS, "overall_alignment_score") if field in row):
            raise ValueError(f"Conflicting repeated judgment: {key}")
        indexed[key] = row
    result = []
    for judge in sorted({key[0] for key in indexed}):
        common = [key for key in human_mean if (judge, key) in indexed]
        groups = ["ALL_MODELS"] + sorted({generated[key] for key in common})
        for group in groups:
            keys = [key for key in common if group == "ALL_MODELS" or generated[key] == group]
            metrics = [("IA", "instruction_adherence"), ("CP", "content_preservation")]
            if not code:
                metrics.append(("DR", "diagram_readability"))
            metrics.append(("OES" if code else "OVERALL", None))
            for label, field in metrics:
                x = [human_mean[key][field] if field else mean(human_mean[key][f] for f in (IMAGE_FIELDS[:2] if code else IMAGE_FIELDS)) for key in keys]
                y = [indexed[judge, key][field] if field else indexed[judge, key]["overall_alignment_score"] if code else mean(indexed[judge, key][f] for f in IMAGE_FIELDS) for key in keys]
                defined = len(keys) > 1 and len(set(x)) > 1 and len(set(y)) > 1
                sp, kd = (spearmanr(x, y), kendalltau(x, y)) if defined else (None, None)
                result.append({"judge_model": judge, "group": group, "metric": label, "n": len(keys),
                    "human_average": mean(x) if x else None, "automatic_average": mean(y) if y else None,
                    "spearman_r": float(sp.statistic) if sp is not None and math.isfinite(sp.statistic) else None,
                    "spearman_p": float(sp.pvalue) if sp is not None and math.isfinite(sp.pvalue) else None,
                    "kendall_tau": float(kd.statistic) if kd is not None and math.isfinite(kd.statistic) else None,
                    "kendall_p": float(kd.pvalue) if kd is not None and math.isfinite(kd.pvalue) else None})
    return result


def rank_statistics(first: list[dict], second: list[dict]) -> list[dict]:
    """同じ29モデルを対応付け、百分率の誤差を正規化して順位相関を求める。"""
    from scipy.stats import kendalltau, spearmanr
    left, right = {r["Model"]: r for r in first}, {r["Model"]: r for r in second}
    models = sorted(set(left) & set(right))
    output = []
    for field in ["IA", "CP", "DR", "FSR", "IA_code", "CP_code", "OES_code"]:
        selected = [model for model in models if left[model].get(field) not in {None, ""} and right[model].get(field) not in {None, ""}]
        x = [round(float(left[m][field]) * 694 / 100) if field == "FSR" else round(float(left[m][field]), 12) for m in selected]
        y = [round(float(right[m][field]) * 694 / 100) if field == "FSR" else round(float(right[m][field]), 12) for m in selected]
        sp, kd = spearmanr(x, y), kendalltau(x, y)
        output.append({"metric": field, "n_models": len(selected), "spearman_r": float(sp.statistic), "spearman_p": float(sp.pvalue),
                       "kendall_tau": float(kd.statistic), "kendall_p": float(kd.pvalue)})
    return output


def stability_statistics(rows: list[dict]) -> list[dict]:
    """同じ50出力・5反復の揃った評点から、反復間の平均と標準偏差を求める。"""
    by_key = defaultdict(dict)
    for row in rows:
        if row["repeat"] in by_key[row["question_id"]]:
            raise ValueError("Duplicate stability observation")
        by_key[row["question_id"]][row["repeat"]] = row
    if len(by_key) != 50 or any(len(values) != 5 for values in by_key.values()):
        raise ValueError("Stability requires the same 50 outputs evaluated in all five repeats")
    return [{"metric": field, "n_examples": 50, "n_repeats": 5,
             "mean_score": mean(mean(row[field] for row in values.values()) for values in by_key.values()),
             "mean_sample_std": mean(stdev(row[field] for row in values.values()) for values in by_key.values())} for field in IMAGE_FIELDS]


def plot_published(name: str, rows: list[dict], destination: Path) -> None:
    """掲載された全値と系列を使って、読みやすいベクター図を再描画する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "serif", "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42})
    destination.parent.mkdir(parents=True, exist_ok=True)
    if name == "figure4":
        labels = list(dict.fromkeys(row["display_name"] for row in rows))
        edits = ["ADD", "REMOVE", "UPDATE", "LAYOUT", "STRUCTURE"]
        matrix = np.array([[float(next(row["FSR_percent"] for row in rows if row["display_name"] == label and row["edit_type"] == edit)) for label in labels] for edit in edits])
        figure, axis = plt.subplots(figsize=(12, 3.7))
        artist = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=10)
        axis.set_yticks(range(5), edits, fontsize=12)
        for row in rows:
            i, j = edits.index(row["edit_type"]), labels.index(row["display_name"])
            axis.text(j, i, row.get("FSR_display", f'{float(row["FSR_percent"]):.1f}'), ha="center", va="center", fontsize=8,
                      color="white" if matrix[i, j] < 22 or matrix[i, j] > 74 else "black")
        figure.colorbar(artist, ax=axis, fraction=0.018, pad=0.012, label="FSR (%)")
    elif name == "figure5":
        labels = list(dict.fromkeys(row["display_name"] for row in rows))
        figure, axis = plt.subplots(figsize=(8.2, 4.7))
        for i, label in enumerate(labels):
            selected = sorted((r for r in rows if r["display_name"] == label), key=lambda r: int(r["operation_count"]))
            axis.plot([int(r["operation_count"]) for r in selected], [float(r["FSR_percent"]) for r in selected],
                      marker=["o", "s", "D", "^", "v"][i % 5], linestyle=["-", "--", ":", "-."][i % 4], label=label)
        axis.set(xticks=[1, 2, 3], xlabel="Edit Operation", ylabel="FSR (%)", ylim=(0, 100))
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, fontsize=10)
    elif name == "figure6":
        figure, axis = plt.subplots(figsize=(9, 5.3))
        models = list(dict.fromkeys(r["display_name"] for r in rows))
        conditions = list(dict.fromkeys(r["condition"] for r in rows))
        x, width = np.arange(len(conditions)), 0.19
        for i, model in enumerate(models):
            values = [float(next(r["FSR_percent"] for r in rows if r["display_name"] == model and r["condition"] == condition)) for condition in conditions]
            axis.bar(x + (i - (len(models) - 1) / 2) * width, values, width, label=model,
                     color=["#6687cc", "#4d6eaa", "#8bcf80", "#659960"][i])
        modality_labels = {"visual_instruction_w_org_image_w_org_tikz": "$I_{vis}+C_{orig}+I_{orig}$", "visual_instruction_w_org_tikz": "$I_{vis}+C_{orig}$",
                           "visual_instruction_w_org_image": "$I_{vis}+I_{orig}$", "visual_instruction": "$I_{vis}$"}
        axis.set_xticks(x, [modality_labels.get(c, c) for c in conditions])
        axis.set(ylabel="FSR (%)", xlabel="Input Modality", ylim=(0, 100))
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=12)
    elif name == "figure8":
        figure, axis = plt.subplots(figsize=(8, 5))
        values = [float(r["FSR_percent"]) for r in rows]
        bars = axis.bar(range(len(rows)), values, color=["#6687cc" if i % 2 == 0 else "#8bcf80" for i in range(len(rows))])
        for i, bar in enumerate(bars):
            if i % 2:
                bar.set_hatch("/")
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{values[i]:.1f}", ha="center")
        axis.set_xticks(range(len(rows)), [r["display_name"] for r in rows], rotation=45, ha="right")
        axis.set(ylabel="FSR (%)", ylim=(0, 100))
    else:
        label_field = "label" if name == "figure9" else "display_name"
        labels = list(dict.fromkeys(row[label_field] for row in rows))
        conditions = list(dict.fromkeys(row["condition"] for row in rows))
        figure, axis = plt.subplots(figsize=((14.5 if name == "figure9" else 9), (7.3 if name == "figure9" else 5)))
        width, x = 0.82 / len(conditions), np.arange(len(labels))
        for i, condition in enumerate(conditions):
            values = [float(next(r["FSR_percent"] for r in rows if r[label_field] == label and r["condition"] == condition)) for label in labels]
            legend = {"direct": r"$I_{vis}$", "verbalize_first": r"$I_{vis}\rightarrow\widehat{I}_{text}$", "gold_text": r"$I_{text}$", "gold_text_visual": r"$I_{vis}+I_{text}$"}.get(condition, condition)
            colors = ["#6687cc", "#8bcf80"] if name == "figure7" else ["#6687cc", "#4d6eaa", "#8bcf80", "#659960"]
            axis.bar(x + (i - (len(conditions) - 1) / 2) * width, values, width, label=legend,
                     color=colors[i], edgecolor="white", linewidth=0.4,
                     hatch="/" if name == "figure7" and i == 1 else None)
        axis.set_xticks(x, labels, rotation=38 if name == "figure9" else 45, ha="right", fontsize=18 if name == "figure9" else 12)
        axis.set(ylabel="FSR (%)", ylim=(0, 100))
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4 if name == "figure9" else 2, fontsize=20 if name == "figure9" else 12)
        if name == "figure9":
            axis.tick_params(axis="y", labelsize=18)
            axis.set_ylabel("FSR (%)", fontsize=20)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)


def reproduce(data_dir: Path, output: Path) -> dict:
    """HFから取得した追加データだけで、保存済みの論文結果を再分析する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    supplement = data_dir / "reproduction"
    if not (supplement / "manifest.json").is_file():
        raise FileNotFoundError("HF reproduction supplement is missing. See docs/huggingface_release.md")
    manifest = read_json(supplement / "manifest.json")
    # ファイルをコピーし直さず、配布時の全ハッシュを検証する。
    install_supplement(supplement, supplement)
    output.mkdir(parents=True, exist_ok=True)
    catalog = read_json(supplement / "catalog.json")
    produced = []
    for name, asset in catalog.items():
        if not asset["available"]:
            continue
        source = supplement / asset["file"]
        if source.suffix == ".csv":
            rows = read_csv(source)
            write_table(output / "tables" / (name + ".csv"), rows)
            produced.append("tables/" + name + ".csv")
            if asset["kind"] == "figure":
                plot_published(name, rows, output / "figures" / (name + ".pdf"))
                produced.append("figures/" + name + ".pdf")
        elif asset["kind"] == "text_table":
            rows = []
            for line in source.read_text().splitlines()[1:]:
                parts = line.rsplit(None, 6)
                if len(parts) == 7:
                    rows.append(dict(zip(["Model", "IA_std", "CP_std", "DR_std", "IA_code_std", "CP_code_std", "OES_code_std"], parts)))
            write_table(output / "tables" / (name + ".csv"), rows)
            produced.append("tables/" + name + ".csv")
    metadata = read_jsonl(data_dir / "metadata/edit_metadata.jsonl")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, field in zip(axes, ["edit_operation", "edit_type"]):
        counts = Counter(str(row[field]) for row in metadata)
        axis.pie(counts.values(), labels=counts.keys(), autopct="%1.1f%%", wedgeprops={"width": 0.43, "edgecolor": "white"}, textprops={"fontsize": 12})
    figure.tight_layout()
    (output / "figures").mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "figures/figure3.pdf", bbox_inches="tight")
    plt.close(figure)
    produced.append("figures/figure3.pdf")
    main_scores = read_jsonl(supplement / "scores/main_image.jsonl")
    recomputed = [{"model": model, **aggregate([r for r in main_scores if r["model"] == model])} for model in sorted({r["model"] for r in main_scores})]
    write_table(output / "tables/main_image_recomputed.csv", recomputed)
    code_source = supplement / "scores/main_code.jsonl"
    if code_source.is_file():
        code_values = read_jsonl(code_source)
        from .workflow import registry
        names = registry()
        published = {row["Model"]: row for row in read_csv(supplement / catalog["table2_main"]["file"])}
        merged, deviations = [], []
        for image in recomputed:
            model = image["model"]
            if names[model]["access"] == "image":
                continue
            subset = [r for r in code_values if r["model"] == model and r["judge"] == "gemini-3.1-pro-preview"]
            if not subset:
                continue
            fields = ("instruction_adherence", "content_preservation", "overall_alignment_score")
            metrics = aggregate(subset, fields)
            label = names[model]["label"]
            reference = published.get(label, {})
            merged.append({"Model": label, "n_image": image["n"], "n_code": metrics["n"], "IA": image["instruction_adherence"],
                "CP": image["content_preservation"], "DR": image["diagram_readability"], "FSR": image["FSR_percent"],
                "CFR_published": reference.get("CFR"), "IA_code": metrics["instruction_adherence"], "CP_code": metrics["content_preservation"],
                "OES_code": metrics["overall_alignment_score"]})
            deviations.append({"Model": label, "IA_std": image["instruction_adherence_std"], "CP_std": image["content_preservation_std"],
                "DR_std": image["diagram_readability_std"], "IA_code_std": metrics["instruction_adherence_std"],
                "CP_code_std": metrics["content_preservation_std"], "OES_code_std": metrics["overall_alignment_score_std"]})
        write_table(output / "tables/table2_metrics_recomputed.csv", merged)
        write_table(output / "tables/table4_standard_deviation_recomputed.csv", deviations)
    direct = [r for r in main_scores if r["model"] in {"gemini-3-pro-image-preview", "gpt-image-1.5", "qwen-image-edit"}]
    for field, filename in [("edit_type", "table7_direct_by_type"), ("edit_operation", "table8_direct_by_operation")]:
        rows = [{"model": model, field: value, **aggregate([r for r in direct if r["model"] == model and r[field] == value])}
                for model in sorted({r["model"] for r in direct}) for value in sorted({r[field] for r in direct})]
        write_table(output / "tables" / (filename + ".csv"), rows)
        produced.append("tables/" + filename + ".csv")
    ratings = read_jsonl(supplement / "human/ratings.jsonl")
    human_rows = human_statistics(ratings, read_jsonl(supplement / "human/image_judges.jsonl"))
    write_table(output / "tables/table1_image_correlation_recomputed.csv", human_rows)
    code = read_jsonl(supplement / "human/code_judges.jsonl")
    if code:
        write_table(output / "tables/table3_code_correlation_recomputed.csv", human_statistics(ratings, code, code=True))
    by_question = defaultdict(list)
    for row in ratings:
        by_question[row["question_id"]].append(row)
    agreement = [{"metric": field, "ordinal_alpha": ordinal_alpha([[r[field] for r in rows] for rows in by_question.values()]),
                  "n_examples": len(by_question), "n_ratings": len(ratings)} for field in IMAGE_FIELDS]
    write_table(output / "tables/table10_human_agreement_recomputed.csv", agreement)
    generated = Counter(rows[0]["generated_model"] for rows in by_question.values())
    write_table(output / "tables/table9_human_sample_models.csv", [{"generated_model": model, "n_examples": n} for model, n in sorted(generated.items())])
    ranks = rank_statistics(read_csv(supplement / catalog["table2_main"]["file"]), read_csv(supplement / catalog["table5_alternative_judge"]["file"]))
    write_table(output / "tables/table6_judge_ranks_recomputed.csv", ranks)
    figure9 = read_csv(supplement / catalog["figure9"]["file"])
    groups = defaultdict(dict)
    for row in figure9:
        groups[row["series"]][row["condition"]] = float(row["FSR_percent"])
    contrasts = {"model_configurations": len(groups), "verbalization_better_than_visual": sum(r["verbalize_first"] > r["direct"] for r in groups.values()),
        "gold_better_than_verbalization": sum(r["gold_text"] > r["verbalize_first"] for r in groups.values()),
        "condition_means": {condition: mean(row[condition] for row in groups.values()) for condition in ["direct", "verbalize_first", "gold_text", "gold_text_visual"]}}
    save_json(output / "figure9_results.json", contrasts)
    if contrasts["model_configurations"] != 16 or contrasts["verbalization_better_than_visual"] != 15:
        raise ValueError("Published Figure9 reconstruction is inconsistent")
    for case in read_json(supplement / "cases/index.json"):
        target = output / "cases" / case["key"]
        target.mkdir(parents=True, exist_ok=True)
        for name, relative in case["files"].items():
            shutil.copy2(supplement / relative, target / name)
        original = (target / "original.tex").read_text()
        generated = (target / "generated_visual.tex").read_text()
        (target / "original_to_generated.diff").write_text("".join(difflib.unified_diff(original.splitlines(True), generated.splitlines(True), fromfile="original.tex", tofile="generated_visual.tex")))
        save_json(target / "case.json", case)
    gaps = list(manifest.get("coverage_gaps", []))
    stability = supplement / "human/stability_scores.jsonl"
    if stability.is_file():
        write_table(output / "tables/table11_stability_recomputed.csv", stability_statistics(read_jsonl(stability)))
        gaps = [gap for gap in gaps if gap != "original_50_case_five_repeat_stability_assets"]
    report = {"dataset_samples": len(metadata), "supplement_manifest_sha256": digest(supplement / "manifest.json"),
        "output": str(output), "figure9": contrasts, "main_models": len(recomputed), "human_examples": len(by_question),
        "human_ordinal_alpha": agreement, "code_judges_in_recomputed_correlation": len({r["judge"] for r in code}),
        "coverage_gaps": gaps, "all_paper_artifacts_verified": not gaps,
        "note": "Published CSV snapshots retain their recorded source versions. Recomputed tables are emitted separately; archived-score replay is distinct from a new API/GPU run."}
    save_json(output / "reproduction_report.json", report)
    links = [str(p.relative_to(output)) for p in sorted(output.rglob("*")) if p.is_file() and p.suffix in {".csv", ".pdf", ".json", ".diff"}]
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>VisEditBench reproduction</title><h1>VisEditBench reproduction</h1><p>Archived-score replay; API/GPU inference was not performed.</p>'
        + '<p>Coverage gaps: ' + html.escape(", ".join(gaps) or "none") + '</p><ul>'
        + "".join('<li><a href="' + html.escape(name) + '">' + html.escape(name) + '</a></li>' for name in links) + '</ul>', encoding="utf-8")
    return report
