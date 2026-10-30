"""生成失敗と採点未完了を区別し、全サンプルを分母にした指標を集計する。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from .data import read_jsonl, write_jsonl

FIELDS = ("instruction_adherence", "content_preservation", "diagram_readability")


def sample_key(row: dict[str, Any]) -> str:
    """短縮されたAPIのcustom_idからもサンプルキーを取り出す。"""
    value = row.get("key") or row.get("id") or row.get("custom_id")
    match = re.match(r"^(\d{8}_(?:ADD|REMOVE|UPDATE|LAYOUT|STRUCTURE)_[123]_edit_\d+)(?:_|$)", str(value))
    if not match:
        raise ValueError(f"Cannot resolve sample key from score row: {value}")
    return match[1]


def score_rows(path: Path) -> list[dict[str, Any]]:
    """評価済みのフラットなCSVまたはJSONLを読み、整数1〜5の範囲を検証する。"""
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for field in FIELDS:
                value = row.get(field, "")
                if not re.fullmatch(r"[1-5]", value):
                    raise ValueError(f"Invalid {field}: {value!r}")
                row[field] = int(value)
    else:
        rows = read_jsonl(path)
    for row in rows:
        if not all(type(row.get(field)) is int and 1 <= row[field] <= 5 for field in FIELDS):
            raise ValueError(f"Image evaluation requires integer IA/CP/DR scores between 1 and 5: {row}")
    return rows


def summarize(metadata: Path, scores: Path, model: str, input_type: str,
              *, allow_partial: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """未生成・画像化失敗は1点、画像あり未採点は未完了として全件FSRを計算する。"""
    inputs = read_jsonl(metadata)
    keys = [row["key"] for row in inputs]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Metadata must contain unique, nonempty sample keys")
    index = {}
    for row in score_rows(scores):
        key = sample_key(row)
        if key not in keys or key in index:
            raise ValueError(f"Unknown or duplicate score key: {key}")
        index[key] = row
    result, pending = [], []
    for row in inputs:
        key = row["key"]
        status = row.get("generation_status", {}).get(input_type, {}).get(model)
        if status not in {"ready", "compile_failed", "missing_generation"}:
            raise ValueError(f"Missing generation status for {key}. Run dataset.py attach first.")
        if status == "ready":
            if key not in index:
                pending.append(key)
                result.append({"key": key, "status": "unscored"})
                continue
            item = {field: index[key][field] for field in FIELDS}
            assignment = "judge"
        else:
            if key in index:
                raise ValueError(f"A score exists for an absent/failed image: {key}; attach outputs again")
            item = dict.fromkeys(FIELDS, 1)
            assignment = status
        result.append({"key": key, "edit_type": row["edit_type"], "edit_operation": int(row["edit_operation"]),
                       "status": assignment, **item, "FSR": int(all(item[f] == 5 for f in FIELDS))})
    if pending and not allow_partial:
        raise ValueError(f"{len(pending)} rendered outputs are unscored; finish/retry evaluation. First key: {pending[0]}")
    scored = [row for row in result if row["status"] != "unscored"]
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """指定群を分母にしてFSRと平均評点を計算する。"""
        return {"n": len(rows), "FSR_count": sum(row["FSR"] for row in rows),
                "FSR_percent": 100*sum(row["FSR"] for row in rows)/len(rows),
                **{field: sum(row[field] for row in rows)/len(rows) for field in FIELDS}}
    summary = {"model": model, "input_type": input_type, "expected": len(inputs), "scored_or_failed": len(scored),
        "unscored": len(pending), "complete": not pending,
        "missing_generation": sum(r["status"] == "missing_generation" for r in result),
        "compile_failed": sum(r["status"] == "compile_failed" for r in result),
        "overall": aggregate(scored) if scored and not pending else None,
        "by_edit_type": {edit: aggregate(group) for edit in sorted({r["edit_type"] for r in scored})
                         if (group := [r for r in scored if r["edit_type"] == edit])} if not pending else {},
        "by_edit_operation": {str(number): aggregate(group) for number in [1, 2, 3]
                              if (group := [r for r in scored if r["edit_operation"] == number])} if not pending else {},
        "partial_results": aggregate(scored) if pending and scored else None}
    return summary, result


def main() -> None:
    """画像評価CSVと生成状態を突合し、全体・編集タイプ・操作数別の結果を保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-type", default="w_org_image_w_org_tikz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    summary, rows = summarize(args.metadata, args.scores, args.model, args.input_type, allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_jsonl(args.output.with_suffix(".cases.jsonl"), rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
