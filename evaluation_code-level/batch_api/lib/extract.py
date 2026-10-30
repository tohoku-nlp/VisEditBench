from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCORE_KEYS = ("instruction_adherence", "content_preservation", "overall_alignment_score")


def _scores_from_object(obj: Any) -> dict[str, int] | None:
    """コード評価のmetricsとoverall_alignment_scoreをCSV用の三項目へ変換する。"""
    if not isinstance(obj, dict):
        return None
    try:
        if all(k in obj for k in SCORE_KEYS):
            scores = {k: obj[k] for k in SCORE_KEYS}
        else:
            metrics = obj['metrics']
            scores = {
                'instruction_adherence': metrics['Instruction Adherence']['score'],
                'content_preservation': metrics['Content Preservation']['score'],
                'overall_alignment_score': obj['overall_alignment_score'],
            }
        if all(type(value) is int and 1 <= value <= 5 for value in scores.values()):
            return scores
    except (KeyError, TypeError):
        pass
    return None


def _get_custom_id(record: dict[str, Any]) -> str | None:
    """全プロバイダ共通で custom_id / key を取得する。"""
    return record.get("custom_id") or record.get("key")


def _strip_thinking(text: str) -> str:
    """<think>...</think> ブロックを除去する（Qwen thinking モデル対応）。"""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_json_scores(text: str) -> dict[str, int] | None:
    """JSON文字列からスコアを抽出する。thinking タグがあれば除去して試みる。"""
    for t in (text.strip(), _strip_thinking(text)):
        try:
            obj = json.loads(t)
            scores = _scores_from_object(obj)
            if scores is not None:
                return scores
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


def extract_scores_openai(record: dict[str, Any]) -> dict[str, int] | None:
    """OpenAI chat/completions バッチ応答からスコアを抽出する。"""
    try:
        content = (
            record["response"]["body"]["choices"][0]["message"]["content"]
        )
        return _parse_json_scores(content)
    except (KeyError, IndexError, TypeError):
        return None


def extract_scores_gemini(record: dict[str, Any]) -> dict[str, int] | None:
    """Gemini バッチ応答からスコアを抽出する。"""
    try:
        parts = record["response"]["candidates"][0]["content"]["parts"]
        for part in reversed(parts):
            if "text" in part:
                scores = _parse_json_scores(part["text"])
                if scores:
                    return scores
    except (KeyError, IndexError, TypeError):
        pass
    return None


def extract_scores_claude(record: dict[str, Any]) -> dict[str, int] | None:
    """Claude バッチ応答からスコアを抽出する。"""
    try:
        content = record["result"]["message"]["content"]
        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "tool_use":
                input_obj = item.get("input")
                scores = _scores_from_object(input_obj)
                if scores is not None:
                    return scores

            if item.get("type") == "text":
                scores = _parse_json_scores(item.get("text", ""))
                if scores:
                    return scores
    except (KeyError, IndexError, TypeError):
        pass
    return None


def extract_scores_from_record(
    record: dict[str, Any], provider: str
) -> tuple[str | None, dict[str, int] | None]:
    """バッチ応答レコードから (custom_id, scores) を抽出する。"""
    custom_id = _get_custom_id(record)
    if provider == "openai":
        scores = extract_scores_openai(record)
    elif provider == "google":
        scores = extract_scores_gemini(record)
    elif provider == "anthropic":
        scores = extract_scores_claude(record)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return custom_id, scores


def extract_scores_from_jsonl(
    output_jsonl: Path, provider: str
) -> list[dict[str, Any]]:
    """バッチ出力 JSONL を読み込み、スコアのリストを返す。

    Returns:
        [{"custom_id": ..., "instruction_adherence": ..., ...}, ...]
    """
    results: list[dict[str, Any]] = []
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            custom_id, scores = extract_scores_from_record(record, provider)
            if custom_id is None or scores is None:
                continue
            results.append({"custom_id": custom_id, **scores})
    return results
