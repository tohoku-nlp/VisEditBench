"""論文の固定subsetと条件を使い、生成から評価までの状態をファイルへ記録する。"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import fcntl
from functools import wraps
import importlib.util
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any

from .backends import ROOT, backend
from .data import digest, read_jsonl, write_jsonl

CONDITIONS = ("direct", "verbalize_first", "gold_text", "gold_text_visual")
IMAGE_FIELDS = ("instruction_adherence", "content_preservation", "diagram_readability")
CODE_FIELDS = ("instruction_adherence", "content_preservation", "overall_alignment_score")


def job_lock(function: Any) -> Any:
    """同じrunの送信・回収・実行を排他し、別端末からの二重送信を防ぐ。"""
    @wraps(function)
    def guarded(path: Path, *args: Any, **kwargs: Any) -> Any:
        """プロセス終了時に解放されるファイルロックを保持して操作する。"""
        with Path(path).with_suffix(".lock").open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"This job is being modified by another process: {path}") from exc
            return function(path, *args, **kwargs)
    return guarded


def read_json(path: Path) -> Any:
    """JSONをUTF-8で読み込む。"""
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, content: Any) -> None:
    """状態ファイルを一時ファイルから置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(content: Any) -> str:
    """順序によらないJSON指紋を返す。"""
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def registry() -> dict:
    """公開リポジトリ内の全モデル設定を取得する。"""
    return read_json(ROOT / "configs/models.json")


def checked_job(path: Path) -> dict:
    """要求準備後にメタデータや元画像・コードが変わっていないことを確認する。"""
    job = read_json(path)
    if job.get("metadata_sha256") and digest(path.parent / "metadata.jsonl") != job["metadata_sha256"]:
        raise ValueError("Run metadata changed after request preparation")
    for filename, expected in job.get("input_files_sha256", {}).items():
        if not Path(filename).is_file() or digest(Path(filename)) != expected:
            raise ValueError(f"Benchmark input changed after request preparation: {filename}")
    return job


def blocks(body: dict, provider: str) -> list:
    """各社のuser入力ブロックを順序を変えず返す。"""
    if provider == "google":
        return body["contents"][0]["parts"]
    if provider == "openai":
        return body["input"][1]["content"]
    return body["messages"][-1]["content"]


def system(body: dict, provider: str, replacement: str | None = None) -> str:
    """system roleとGemmaのuser内systemを同じ操作で読み書きする。"""
    if provider == "google":
        container, field = body["systemInstruction"]["parts"][0], "text"
    elif provider == "openai":
        container, field = body["input"][0], "content"
    elif provider == "anthropic":
        container, field = body, "system"
    elif body["messages"][0]["role"] == "system":
        container, field = body["messages"][0], "content"
        if isinstance(container[field], list):
            container, field = container[field][0], "text"
    else:
        container, field = blocks(body, provider)[0], "text"
    previous = container[field]
    if replacement is not None:
        container[field] = replacement
    return previous


def build_request(model: str, row: dict, condition: str, *, profile: str = "main", modality: str = "full") -> dict:
    """固定した要求の画像・コードを埋め、条件差以外のsamplingを維持する。"""
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    spec = registry()[model]
    if spec["backend"] not in {"google", "openai", "anthropic", "vllm", "openrouter"}:
        raise ValueError("Direct image generation uses the direct-image backend")
    if condition in {"gold_text", "gold_text_visual"} and not row.get("text_instruction"):
        raise ValueError(f"Gold instruction missing: {row['key']}. Download the reproduction supplement first.")
    templates = read_json(ROOT / "configs/request_templates.json")[profile]
    base_condition = "gold_text" if condition in {"gold_text", "gold_text_visual"} else "direct"
    body = deepcopy(templates[model][base_condition])
    provider = spec["backend"]
    prompts = read_json(ROOT / "configs" / ("legacy_prompts.json" if profile == "legacy_verbalization" else "prompts.json"))
    if condition == "verbalize_first":
        previous = system(body, provider)
        anchor = "Apply the requested edits and produce"
        if previous.count(anchor) != 1:
            raise ValueError("Cannot identify original editing instruction")
        tail = "Solve this task in two explicit steps." + prompts[condition]["system"].split("Solve this task in two explicit steps.", 1)[1]
        if "\\n" in previous and "\n" not in previous:
            tail = tail.replace("\n", "\\n")
        if "{{standalone}}" in previous:
            tail = tail.replace("{standalone}", "{{standalone}}").replace("{document}", "{{document}}")
        system(body, provider, previous.split(anchor, 1)[0] + tail)
    elif condition == "gold_text_visual":
        system(body, provider, prompts[condition]["system"])
        blocks(body, provider).append(deepcopy(blocks(templates[model]["direct"], provider)[-1]))
    if modality != "full":
        if condition != "direct":
            raise ValueError("Modality ablation is defined only for visual instructions")
        types = {"image": "visual_instruction_w_org_image", "code": "visual_instruction_w_org_tikz", "visual": "visual_instruction"}
        if modality not in types:
            raise ValueError(modality)
        system(body, provider, backend("editing").generate.SYSTEM_PROMPTS[types[modality]])
        content = blocks(body, provider)
        content[:] = [block for block in content if not (modality in {"image", "visual"} and "$SOURCE_TIKZ" in json.dumps(block))
                      and not (modality in {"code", "visual"} and "$ORG_" in json.dumps(block))]
    replacements = {"SOURCE_TIKZ": Path(row["org_tikz_code_path"]).read_text(encoding="utf-8"), "GOLD_TEXT": row.get("text_instruction", "")}
    for prefix, field in [("ORG", "org_image_path"), ("VIS", "visual_instruction_image_path")]:
        encoded = base64.b64encode(Path(row[field]).read_bytes()).decode()
        replacements[prefix + "_BASE64"] = encoded
        replacements[prefix + "_DATA_URL"] = "data:image/png;base64," + encoded

    def fill(value: Any) -> Any:
        """プレースホルダーだけを1回置換し、挿入したコードを再解釈しない。"""
        if isinstance(value, str):
            return re.sub(r"\$(SOURCE_TIKZ|GOLD_TEXT|ORG_BASE64|VIS_BASE64|ORG_DATA_URL|VIS_DATA_URL)", lambda m: replacements[m[1]], value)
        if isinstance(value, list):
            return [fill(item) for item in value]
        if isinstance(value, dict):
            return {key: fill(item) for key, item in value.items()}
        return value
    return fill(body)


def select_rows(data_dir: Path, cohort: str) -> list[dict]:
    """公開したID順にサンプルを選び、任意の先頭100件への置換を防ぐ。"""
    rows = read_jsonl(data_dir / "metadata/edit_metadata.jsonl")
    if cohort == "main":
        return rows
    path = data_dir / "reproduction/cohorts.json"
    if not path.is_file():
        raise FileNotFoundError("Reproduction supplement missing: data/reproduction/cohorts.json. See docs/huggingface_release.md")
    keys = read_json(path)[cohort]
    index = {row["key"]: row for row in rows}
    if len(keys) != len(set(keys)) or any(key not in index for key in keys):
        raise ValueError(f"Invalid or incomplete cohort: {cohort}")
    return [index[key] for key in keys]


def experiment_groups(name: str) -> list[dict]:
    """論文のモデル・入力条件・評価者・subsetを実験別に列挙する。"""
    models = registry()
    groups = []
    if name == "alternative_judge":
        return [{"model": key, "stage": "judge_only", "source": "same generated outputs as main",
                 "judge": "claude-opus-4-6", "command": "paper.py judge-prepare --job <main/job.json> --judge-model claude-opus-4-6"}
                for key, spec in models.items() if spec["access"] != "image"]
    if name in {"main", "direct_image"}:
        selected = [key for key, spec in models.items() if (spec["access"] == "image") == (name == "direct_image")]
        groups = [dict(model=key, condition="direct", profile="main", cohort="main", modality="full") for key in selected]
    elif name == "modality":
        groups = [dict(model=key, condition="direct", profile="main", cohort="figure6", modality=modality)
            for key in ["gemini-3.1-pro-preview", "qwen3.5-397b-a17b", "claude-opus-4-6", "gpt-5.4"] for modality in ["full", "code", "image", "visual"]]
    elif name == "text_visual":
        groups = [dict(model=key, condition=condition, profile="main", cohort="figure7", modality="full")
            for key, spec in models.items() if spec.get("figure7") for condition in ["direct", "gold_text"]]
    elif name == "verbalization":
        groups = [dict(model=key, condition=condition, profile="main", cohort="figure7", modality="full")
            for key, spec in models.items() if spec.get("figure7") and key != "gemini-3.1-flash-lite-preview" for condition in CONDITIONS]
        groups += [dict(model=key, condition=condition, profile="legacy_verbalization", cohort="legacy_verbalization", modality="full")
            for key in ["qwen3.5-4b", "qwen3.5-35b-a3b", "gemma4-31b-it", "gemma4-e4b-it"] for condition in CONDITIONS]
    elif name == "stability":
        return [{"stage": "judge_only", "cohort": "original_50_human_outputs", "repeats": 5,
                 "command": "paper.py stability-prepare --selection <original_50_case_selection.jsonl> --output <run>"}]
    else:
        raise ValueError(name)
    for group in groups:
        group["judge"] = "gemini-3.6-flash" if name == "verbalization" else "claude-opus-4-6" if name == "alternative_judge" else "gemini-3.1-pro-preview"
    return groups


def wrap(provider: str, key: str, body: dict, *, endpoint: str = "/v1/responses") -> dict:
    """Batchの外側形式を付け、短いcanonical IDを維持する。"""
    if provider == "google":
        return {"key": key, "request": body}
    if provider == "anthropic":
        return {"custom_id": key, "params": body}
    if provider == "openai":
        return {"custom_id": key, "method": "POST", "url": endpoint, "body": body}
    return {"key": key, "body": body}


def prepare_experiment(name: str, output: Path, data_dir: Path, *, models: list[str] | None = None,
                       conditions: list[str] | None = None, limit: int | None = None) -> dict:
    """全入力を事前検証し、32MB以下の要求ファイルと固定計画を作る。"""
    if name == "alternative_judge":
        raise ValueError("Reuse each main job with judge-prepare --judge-model claude-opus-4-6; do not regenerate diagrams for judge comparison.")
    if name == "stability":
        raise ValueError("Use stability-prepare with the original 50 generated outputs; do not generate new diagrams for this study.")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if (output / "plan.json").exists():
        raise FileExistsError("Run exists. Use a new output directory; existing requests must not be replaced.")
    specs = registry()
    if models and set(models) - set(specs):
        raise ValueError(f"Unknown model ids: {set(models) - set(specs)}")
    groups = [g for g in experiment_groups(name) if (not models or g["model"] in models) and (not conditions or g["condition"] in conditions)]
    if not groups:
        raise ValueError("No matching experiment groups")
    inputs = {}
    for group in groups:
        selected = select_rows(data_dir, group["cohort"])
        if limit:
            selected = selected[:limit]
        if group["condition"] in {"gold_text", "gold_text_visual"} and any(not row.get("text_instruction") for row in selected):
            raise ValueError("Gold text is missing. Download the reproduction supplement before preparing this experiment.")
        inputs[group["cohort"]] = selected
    jobs = []
    input_hashes = {}
    for group in groups:
        identifier = "__".join(group[key] for key in ["model", "condition", "profile", "modality"])
        folder = output / identifier
        spec = specs[group["model"]]
        selected = inputs[group["cohort"]]
        assets = {}
        for row in selected:
            for field in ["org_image_path", "visual_instruction_image_path", "org_tikz_code_path", "reference_tikz_code_path"]:
                filename = row[field]
                if filename not in input_hashes:
                    input_hashes[filename] = digest(Path(filename))
                assets[filename] = input_hashes[filename]
        folder.mkdir(parents=True)
        write_jsonl(folder / "metadata.jsonl", selected)
        chunks, records, size = [], [], 0
        for row in selected:
            if spec["access"] == "image":
                from .image_edit import image_request
                body = image_request(spec, row)
            else:
                body = build_request(group["model"], row, group["condition"], profile=group["profile"], modality=group["modality"])
            request = wrap(spec["backend"], row["key"], body)
            length = len((json.dumps(request, ensure_ascii=False) + "\n").encode())
            if records and size + length > 30_000_000:
                path = folder / f"input_{len(chunks):03d}.jsonl"
                write_jsonl(path, records)
                chunks.append({"input": path.name, "sha256": digest(path), "status": "prepared"})
                records, size = [], 0
            records.append(request)
            size += length
        if records:
            path = folder / f"input_{len(chunks):03d}.jsonl"
            write_jsonl(path, records)
            chunks.append({"input": path.name, "sha256": digest(path), "status": "prepared"})
        job = {**group, "stage": "generation", "provider": spec["backend"], "api_model": spec["api_model"],
               "expected_keys": [row["key"] for row in selected], "chunks": chunks, "test_subset": bool(limit),
               "enabled": spec["enabled"], "metadata_sha256": digest(folder / "metadata.jsonl"), "input_files_sha256": assets}
        save_json(folder / "job.json", job)
        jobs.append(str((folder / "job.json").resolve()))
    plan = {"experiment": name, "groups": len(groups), "jobs": jobs, "test_subset": bool(limit),
            "request_templates_sha256": digest(ROOT / "configs/request_templates.json"), "models_sha256": digest(ROOT / "configs/models.json")}
    save_json(output / "plan.json", plan)
    return plan


def generation_text(provider: str, raw: dict) -> tuple[str, str | None, str | None]:
    """APIエラーを区別し、長さ上限に達した生成本文も元実装同様に残す。"""
    if raw.get("error"):
        return "", None, str(raw["error"])
    if provider in {"vllm", "openrouter"}:
        if not raw.get("choices"):
            return "", None, "Missing choices"
        choice = raw["choices"][0]
        return choice["message"].get("content") or "", choice.get("finish_reason"), None
    if provider == "google":
        candidates = raw.get("response", {}).get("candidates", [])
        if not candidates:
            return "", None, "Missing candidates"
        candidate = candidates[0]
        return "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", []) if not part.get("thought")), candidate.get("finishReason"), None
    if provider == "anthropic":
        result = raw.get("result", {})
        if result.get("type") != "succeeded":
            return "", None, str(result)
        body = result["message"]
        return "".join(p.get("text", "") for p in body.get("content", []) if p.get("type") == "text"), body.get("stop_reason"), None
    response = raw.get("response", {})
    if response.get("status_code") != 200:
        return "", None, str(response)
    body = response.get("body", {})
    if body.get("error"):
        return "", None, str(body["error"])
    parts = [part for item in body.get("output", []) if item.get("type") == "message" for part in item.get("content", [])]
    return "".join(p.get("text", "") for p in parts if p.get("type") == "output_text"), body.get("status"), None


def store_generation(folder: Path, key: str, job: dict, raw: dict) -> dict:
    """原始応答、言語化文、編集コードを分離し、採点未実施と生成失敗を区別する。"""
    text, finish, error = generation_text(job["provider"], raw)
    code = text.split("</think>", 1)[-1].strip()
    instruction = None
    if job["condition"] == "verbalize_first":
        match = re.match(r"<instruction>(.*?)</instruction>\s*", code, re.I | re.S)
        if match:
            instruction, code = match[1].strip(), code[match.end():].strip()
    result = {"key": key, "status": "api_error" if error else "ok" if code else "model_empty", "error": error,
              "finish_reason": finish, "raw_text": text, "instruction": instruction, "instruction_format_valid": bool(instruction)}
    save_json(folder / "raw" / (key + ".json"), raw)
    save_json(folder / "results" / (key + ".json"), result)
    if not error:
        code = backend("editing").extract.strip_code_fences(code)
        target = folder / "tikz" / (key + ".tex")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")
        if instruction is not None:
            target = folder / "instructions" / (key + ".txt")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(instruction, encoding="utf-8")
    return result


def online_client(provider: str) -> Any:
    """明示的な送信・回収コマンドの時だけ、このリポジトリの認証情報を読む。"""
    from dotenv import load_dotenv
    from .providers import create_client
    load_dotenv(ROOT / ".env", override=False)
    return create_client(provider)


def judgment_scores(stage: str, provider: str, row: dict) -> dict | None:
    """正常終了かつ整数1〜5の評点だけを、確定したjudge結果として受け入れる。"""
    if row.get("error"):
        return None
    if provider == "google":
        candidates = row.get("response", {}).get("candidates", [])
        if not candidates or candidates[0].get("finishReason") != "STOP":
            return None
    elif provider == "anthropic":
        result = row.get("result", {})
        if result.get("type") != "succeeded" or result.get("message", {}).get("stop_reason") not in {"end_turn", "tool_use"}:
            return None
    else:
        response = row.get("response", {})
        choices = response.get("body", {}).get("choices", [])
        if response.get("status_code") != 200 or not choices or choices[0].get("finish_reason") != "stop":
            return None
    return backend(stage).extract.extract_scores_from_record(row, provider)[1]


@job_lock
def import_responses(path: Path, responses: Path) -> dict:
    """取得済みの完全なAPI応答JSONLを、外部通信なしで同じ抽出処理へ渡す。"""
    job, folder = read_json(path), path.parent
    values = read_jsonl(responses)
    seen = set()
    expected = set(job["expected_keys"])
    for row in values:
        key = row.get("key") or row.get("custom_id")
        if key not in expected or key in seen:
            raise ValueError(f"Unknown/duplicate imported response key: {key}")
        seen.add(key)
    if seen != expected:
        raise ValueError("Imported responses must match the full expected ID set")
    for row in values:
        key = row.get("key") or row["custom_id"]
        previous = folder / ("results" if job["stage"] == "generation" else "scores") / (key + ".json")
        if previous.is_file() and read_json(previous)["status"] not in {"api_error", "judge_error"}:
            if job["stage"] == "generation":
                raw_path = folder / "raw" / (key + ".json")
                if not raw_path.is_file() or fingerprint(read_json(raw_path)) != fingerprint(row):
                    raise ValueError("Refusing to replace an existing successful generation")
            elif read_json(previous).get("scores") != judgment_scores(job["stage"], job["provider"], row):
                raise ValueError("Refusing to replace an existing successful judgment")
    for row in values:
        key = row.get("key") or row["custom_id"]
        if job["stage"] == "generation":
            store_generation(folder, key, job, row)
        else:
            scores = judgment_scores(job["stage"], job["provider"], row)
            save_json(folder / "scores" / (key + ".json"), {"key": key, "status": "ok" if scores else "judge_error", "scores": scores})
    write_jsonl(folder / "imported_responses.jsonl", values)
    for chunk in job["chunks"]:
        chunk["status"] = "collected"
        chunk["response_source"] = "offline_import"
    save_json(path, job)
    return {"imported": len(values), "external_calls": 0}


@job_lock
def submit_job(path: Path, *, client: Any = None, include_blocked: bool = False) -> dict:
    """送信前に状態を保存し、応答不明のBatchを自動で二重送信しない。"""
    from .providers import submit_one
    job, folder = checked_job(path), path.parent
    if not job.get("enabled", True) and not include_blocked:
        raise ValueError("Model is marked unavailable in the recorded settings. Explicitly review --include-blocked.")
    if job["provider"] not in {"openai", "google", "anthropic"}:
        raise ValueError("Use the chat or image-run command for this backend")
    if all(chunk["status"] == "collected" for chunk in job["chunks"]):
        return job
    client = client or online_client(job["provider"])
    for chunk in job["chunks"]:
        input_path = folder / chunk["input"]
        if digest(input_path) != chunk["sha256"]:
            raise ValueError("Input changed after preparation")
        if chunk["status"] in {"submitted", "collected"}:
            continue
        if chunk["status"] != "prepared":
            raise ValueError("Previous submission outcome unknown; recover the existing job ID before continuing")
        chunk["status"] = "submission_unknown"
        save_json(path, job)
        endpoint = read_jsonl(input_path)[0].get("url", "/v1/responses")
        chunk.update(submit_one(client, job["provider"], input_path, job["api_model"], endpoint=endpoint))
        chunk["status"] = "submitted"
        save_json(path, job)
    return job


@job_lock
def collect_job(path: Path, *, client: Any = None) -> dict:
    """Batchの成功・失敗行を回収し、expected IDの欠落・重複を検出する。"""
    from .providers import get_job, download_results
    job, folder = read_json(path), path.parent
    if all(chunk["status"] == "collected" for chunk in job["chunks"]):
        return job
    client = client or online_client(job["provider"])
    for index, chunk in enumerate(job["chunks"]):
        if chunk["status"] == "collected":
            continue
        if chunk["status"] != "submitted":
            raise ValueError("Submit or recover the job first")
        remote, state, terminal = get_job(client, job["provider"], chunk["job_id"])
        chunk["remote_state"] = state
        save_json(path, job)
        if not terminal:
            continue
        rows = download_results(client, job["provider"], remote)
        write_jsonl(folder / f"response_{index:03d}.jsonl", rows)
        expected = {row.get("key") or row["custom_id"] for row in read_jsonl(folder / chunk["input"])}
        seen = set()
        for row in rows:
            key = row.get("key") or row.get("custom_id")
            if key not in expected or key in seen:
                raise ValueError(f"Unknown/duplicate response key: {key}")
            seen.add(key)
            if job["stage"] == "generation":
                store_generation(folder, key, job, row)
            else:
                scores = judgment_scores(job["stage"], job["provider"], row)
                save_json(folder / "scores" / (key + ".json"), {"key": key, "status": "ok" if scores else "judge_error", "scores": scores})
        for key in expected - seen:
            save_json(folder / ("results" if job["stage"] == "generation" else "scores") / (key + ".json"),
                      {"key": key, "status": "api_error" if job["stage"] == "generation" else "judge_error", "error": "Missing batch response"})
        chunk["status"] = "collected"
        save_json(path, job)
    if job.get("parent_job") and all(c["status"] == "collected" for c in job["chunks"]):
        merge_retry(path)
    return job


@job_lock
def run_chat(path: Path, endpoint: str, *, retry_failed: bool = False, post: Any = None) -> dict:
    """vLLM/OpenRouterへ保存済み要求を送り、成功済みや送信不明の要求を再送しない。"""
    import os
    import requests
    from dotenv import load_dotenv
    job, folder = checked_job(path), path.parent
    if job["provider"] not in {"vllm", "openrouter"}:
        raise ValueError("chat supports vLLM/OpenRouter jobs")
    load_dotenv(ROOT / ".env", override=False)
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("OPENROUTER_API_KEY") if job["provider"] == "openrouter" else os.environ.get("VLLM_API_KEY")
    if token:
        headers["Authorization"] = "Bearer " + token
    post = post or requests.post
    for chunk in job["chunks"]:
        if digest(folder / chunk["input"]) != chunk["sha256"]:
            raise ValueError("Input changed after preparation")
        for row in read_jsonl(folder / chunk["input"]):
            key = row["key"]
            result_path = folder / "results" / (key + ".json")
            marker = folder / "attempts" / (key + ".json")
            if result_path.exists():
                previous = read_json(result_path)
                if previous["status"] != "api_error" or not retry_failed:
                    continue
            elif marker.exists():
                raise ValueError(f"Submission outcome unknown: {key}. Inspect the saved attempt before retrying.")
            save_json(marker, {"request_sha256": fingerprint(row["body"]), "endpoint": endpoint, "state": "submission_unknown"})
            try:
                response = post(endpoint.rstrip("/") + "/chat/completions", json=row["body"], headers=headers, timeout=1800)
                response.raise_for_status()
                raw = response.json()
            except Exception as exc:
                raw = {"error": {"type": type(exc).__name__, "message": str(exc)}}
            store_generation(folder, key, job, raw)
    if job.get("parent_job"):
        merge_retry(path)
    return job


def merge_retry(path: Path) -> dict:
    """同じ要求の再試行成功だけを親runへ戻し、原始応答を履歴へ保全する。"""
    import shutil
    retry, folder = read_json(path), path.parent
    parent_path = Path(retry["parent_job"])
    parent = read_json(parent_path)
    original = {r.get("key") or r["custom_id"]: r for c in parent["chunks"] for r in read_jsonl(parent_path.parent / c["input"])}
    requests = {r.get("key") or r["custom_id"]: r for c in retry["chunks"] for r in read_jsonl(folder / c["input"])}
    directory = "results" if retry["stage"] == "generation" else "scores"
    merged = 0
    for key in retry["expected_keys"]:
        if key not in original or fingerprint(original[key]) != fingerprint(requests[key]):
            raise ValueError("Retry request differs from the original run")
        result_path = folder / directory / (key + ".json")
        if not result_path.is_file() or read_json(result_path)["status"] in {"api_error", "judge_error"}:
            continue
        previous = parent_path.parent / directory / (key + ".json")
        if previous.is_file() and read_json(previous)["status"] not in {"api_error", "judge_error"}:
            if read_json(previous) != read_json(result_path):
                raise ValueError("Refusing to replace an existing successful observation")
            continue
        for subdirectory in [directory, "raw", "tikz", "instructions"]:
            suffix = ".tex" if subdirectory == "tikz" else ".txt" if subdirectory == "instructions" else ".json"
            source = folder / subdirectory / (key + suffix)
            target = parent_path.parent / subdirectory / (key + suffix)
            if source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file():
                    history = folder / "previous_parent" / subdirectory / target.name
                    history.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(target, history)
                shutil.copy2(source, target)
        merged += 1
    return {"merged": merged, "parent_job": str(parent_path)}


def render_job(path: Path, *, overwrite: bool = False, historical: bool = False) -> dict:
    """API完了後のコードだけを描画し、各サンプルの成功・失敗を記録する。"""
    job, folder = checked_job(path), path.parent
    module_path = ROOT / "editing/compile_tikz_batch.py"
    spec = importlib.util.spec_from_file_location("_viseditbench_compiler", module_path)
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    args = argparse.Namespace(env="apptainer", apptainer=None, bind=None, dpi=300, overwrite=overwrite,
                              historical_success_policy=historical)
    statuses = []
    for key in job["expected_keys"]:
        record = folder / "results" / (key + ".json")
        if not record.exists() or read_json(record)["status"] == "api_error":
            raise ValueError(f"Generation incomplete/API error: {key}. Finish or retry generation before rendering.")
        image = folder / "images" / (key + ".png")
        result = read_json(record)
        if job["provider"] in {"google_image", "openai_image", "diffusers"}:
            success = image.is_file()
        elif result["status"] == "model_empty":
            success = False
        else:
            success = compiler.compile_tex_to_png(folder / "tikz" / (key + ".tex"), folder / "images", args)
        statuses.append({"key": key, "status": "ready" if success else "render_failed", "image": str(image.resolve()) if success else None,
                         "compiler_policy": "historical_pdf_exists" if historical else "strict_exit_code"})
    write_jsonl(folder / "render_status.jsonl", statuses)
    return {"samples": len(statuses), "rendered": sum(row["status"] == "ready" for row in statuses)}


def prepare_judge(path: Path, *, kind: str = "image", judge_model: str | None = None, repeats: int = 1) -> list[str]:
    """画像/コードのjudge要求を作り、固定1点とAPI未採点を混同しない。"""
    if repeats < 1:
        raise ValueError("--repeats must be positive")
    job, folder = checked_job(path), path.parent
    if kind == "code" and job["provider"] in {"google_image", "openai_image", "diffusers"}:
        raise ValueError("Code-level evaluation is not defined for direct-image models")
    judge_model = judge_model or job["judge"]
    provider = "anthropic" if judge_model.startswith("claude") else "openai" if judge_model.startswith("gpt") else "google"
    source = backend(kind)
    config = source.config.EvalConfig(name="paper", judge_provider=provider, judge_model=judge_model,
        metadata_path=folder / "metadata.jsonl", save_path=Path("unused"), append=False,
        edited_model_type=job["model"], edited_input_type=job["condition"],
        **({"judge_input_type": "w_org_image"} if kind == "image" else {}))
    statuses = {row["key"]: row for row in read_jsonl(folder / "render_status.jsonl")} if kind == "image" else {}
    metadata = read_jsonl(folder / "metadata.jsonl")
    destinations = []
    for repetition in range(repeats):
        target = folder / "judges" / judge_model / kind / f"repeat_{repetition:02d}"
        if (target / "job.json").exists():
            raise FileExistsError(f"Judge run exists: {target}")
        requests, assigned = [], []
        for row in metadata:
            key = row["key"]
            generation_path = folder / "results" / (key + ".json")
            if not generation_path.exists() or read_json(generation_path)["status"] == "api_error":
                raise ValueError(f"Generation incomplete: {key}")
            failed = statuses[key]["status"] != "ready" if kind == "image" else read_json(generation_path)["status"] == "model_empty"
            if failed:
                assigned.append({"key": key, "status": "render_failed" if kind == "image" else "model_empty", "scores": dict.fromkeys(IMAGE_FIELDS if kind == "image" else CODE_FIELDS, 1)})
                continue
            original = Path(row["org_tikz_code_path"]).read_text(encoding="utf-8")
            generated_path = folder / "tikz" / (key + ".tex")
            generated = generated_path.read_text(encoding="utf-8") if generated_path.is_file() else ""
            if kind == "image":
                request = source.generate._build_json_line(config, key, Path(row["org_image_path"]),
                    Path(row["visual_instruction_image_path"]), original, folder / "images" / (key + ".png"), generated)
            else:
                reference = Path(row["reference_tikz_code_path"]).read_text(encoding="utf-8")
                request = source.generate._build_json_line(config, key, None, None, original, None, generated, reference_tikz_code=reference)
            requests.append(request)
        target.mkdir(parents=True)
        chunks, pending, size = [], [], 0
        for request in requests:
            length = len(json.dumps(request, ensure_ascii=False).encode()) + 1
            if pending and size + length > 30_000_000:
                output = target / f"input_{len(chunks):03d}.jsonl"
                write_jsonl(output, pending)
                chunks.append({"input": output.name, "sha256": digest(output), "status": "prepared"})
                pending, size = [], 0
            pending.append(request)
            size += length
        if pending:
            output = target / f"input_{len(chunks):03d}.jsonl"
            write_jsonl(output, pending)
            chunks.append({"input": output.name, "sha256": digest(output), "status": "prepared"})
        for row in assigned:
            save_json(target / "scores" / (row["key"] + ".json"), row)
        judge_job = {**{k: v for k, v in job.items() if k not in {"chunks", "metadata_sha256"}},
            "stage": kind, "provider": provider, "api_model": judge_model, "generation_job": str(path.resolve()),
            "generation_render_sha256": digest(folder / "render_status.jsonl") if kind == "image" else None,
            "chunks": chunks, "repeat": repetition, "enabled": True}
        save_json(target / "job.json", judge_job)
        destinations.append(str((target / "job.json").resolve()))
    return destinations


def report_job(path: Path, *, allow_partial: bool = False) -> dict:
    """全expected IDを分母にしてFSR・平均・標準偏差・CFRを保存する。"""
    from statistics import mean, stdev
    job, folder = read_json(path), path.parent
    if job["stage"] not in {"image", "code"}:
        raise ValueError("Pass an image/code judge job.json to report")
    generation = Path(job["generation_job"])
    checked_job(generation)
    if job.get("parent_job"):
        raise ValueError("Report the parent job after merging successful retries; a retry subset is not a full experiment")
    if job.get("generation_render_sha256") and digest(generation.parent / "render_status.jsonl") != job["generation_render_sha256"]:
        raise ValueError("Rendered outputs changed after judge preparation")
    metadata = {row["key"]: row for row in read_jsonl(generation.parent / "metadata.jsonl")}
    fields = IMAGE_FIELDS if job["stage"] == "image" else CODE_FIELDS
    cases, missing = [], []
    for key in job["expected_keys"]:
        record = folder / "scores" / (key + ".json")
        value = read_json(record) if record.is_file() else {}
        scores = value.get("scores")
        if not isinstance(scores, dict) or any(type(scores.get(f)) is not int or not 1 <= scores[f] <= 5 for f in fields):
            missing.append(key)
            continue
        cases.append({"key": key, "model": job["model"], "condition": job["condition"], "profile": job["profile"], "cohort": job["cohort"],
                      "judge": job["api_model"], "modality": job["modality"], "status": value["status"],
                      "edit_type": metadata[key]["edit_type"], "edit_operation": int(metadata[key]["edit_operation"]), **scores})
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} pending/invalid judgments. Finish or retry judging; these are not model failures.")

    def metrics(rows: list[dict]) -> dict:
        """標本標準偏差と全条件5点の率を同じ対象群から計算する。"""
        result = {"n": len(rows)}
        if not rows:
            return result
        for field in fields:
            result[field] = mean(row[field] for row in rows)
            result[field + "_std"] = stdev(row[field] for row in rows) if len(rows) > 1 else 0.0
        if job["stage"] == "image":
            result["FSR_count"] = sum(all(row[f] == 5 for f in fields) for row in rows)
            result["FSR_percent"] = 100 * result["FSR_count"] / len(rows)
        return result
    summary = {"model": job["model"], "condition": job["condition"], "cohort": job["cohort"], "profile": job["profile"],
        "judge": job["api_model"], "kind": job["stage"], "expected": len(job["expected_keys"]), "unscored": len(missing),
        "complete": not missing, "test_subset": job.get("test_subset", False), "overall": metrics(cases) if not missing else None,
        "partial": metrics(cases) if missing else None,
        "by_edit_type": {edit: metrics([r for r in cases if r["edit_type"] == edit]) for edit in sorted({r["edit_type"] for r in cases})} if not missing else {},
        "by_edit_operation": {str(count): metrics([r for r in cases if r["edit_operation"] == count]) for count in [1, 2, 3]} if not missing else {},
        "edit_type_one_operation": {edit: metrics([r for r in cases if r["edit_type"] == edit and r["edit_operation"] == 1]) for edit in sorted({r["edit_type"] for r in cases})} if not missing else {}}
    render_status = generation.parent / "render_status.jsonl"
    if render_status.is_file() and read_json(generation)["provider"] not in {"google_image", "openai_image", "diffusers"}:
        rendered = read_jsonl(render_status)
        summary["CFR_percent"] = 100 * sum(r["status"] != "ready" for r in rendered) / len(rendered)
    write_jsonl(folder / "cases.jsonl", cases)
    save_json(folder / "summary.json", summary)
    return summary


def server_command(model: str, *, profile: str = "main", port: int = 8000) -> list[str]:
    """固定revision・seed省略の区別を保ったvLLM起動コマンドを返す。"""
    spec = registry()[model]
    if spec["backend"] != "vllm":
        raise ValueError("This model is not configured for local vLLM")
    settings = dict(spec["server"])
    if profile == "legacy_verbalization":
        settings.update(dtype="bfloat16", seed=42, gpu_memory_utilization=0.95, generation_config="vllm", max_model_len=128000)
    args = ["vllm", "serve", spec["api_model"], "--port", str(port), "--trust-remote-code"]
    if spec.get("revision"):
        args.extend(["--revision", spec["revision"]])
    for key, value in settings.items():
        args.extend(["--" + key.replace("_", "-"), str(value)])
    args.extend(["--limit-mm-per-prompt", '{"image": 3}'])
    return args


def retry_job(path: Path, *, stage: str = "generation") -> dict:
    """失敗したIDだけを新しいjobへ分離し、成功要求の二重送信を防ぐ。"""
    job, folder = read_json(path), path.parent
    failed = set()
    directory = "results" if job["stage"] == "generation" else "scores"
    for key in job["expected_keys"]:
        record = folder / directory / (key + ".json")
        if record.is_file() and read_json(record)["status"] in {"api_error", "judge_error"}:
            failed.add(key)
    if not failed:
        raise ValueError("No recorded failed responses. Pending batches must be collected first.")
    index = 1
    while (folder / "retries" / f"attempt_{index:02d}").exists():
        index += 1
    target = folder / "retries" / f"attempt_{index:02d}"
    target.mkdir(parents=True)
    rows = [row for chunk in job["chunks"] for row in read_jsonl(folder / chunk["input"]) if (row.get("key") or row["custom_id"]) in failed]
    write_jsonl(target / "input.jsonl", rows)
    retry = {**job, "expected_keys": [key for key in job["expected_keys"] if key in failed], "parent_job": str(path.resolve()),
             "chunks": [{"input": "input.jsonl", "sha256": digest(target / "input.jsonl"), "status": "prepared"}]}
    if (folder / "metadata.jsonl").is_file():
        write_jsonl(target / "metadata.jsonl", [r for r in read_jsonl(folder / "metadata.jsonl") if r["key"] in failed])
        retry["metadata_sha256"] = digest(target / "metadata.jsonl")
    save_json(target / "job.json", retry)
    return {"retry_job": str((target / "job.json").resolve()), "samples": len(failed)}


def prepare_stability(selection: Path, output: Path, data_dir: Path) -> dict:
    """元の50出力を無加工で読み込み、同一画像を5回採点するrunを準備する。"""
    import shutil
    from PIL import Image
    selected = read_jsonl(selection)
    if len(selected) != 50 or len({row["question_id"] for row in selected}) != 50:
        raise ValueError("Provide exactly 50 unique original human-evaluation outputs")
    if (output / "job.json").exists():
        raise FileExistsError("Stability run exists; use a new output directory")
    index = {row["key"]: row for row in read_jsonl(data_dir / "metadata/edit_metadata.jsonl")}
    metadata, paths = [], []
    for row in selected:
        key = row["question_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key):
            raise ValueError(f"Invalid question ID: {key}")
        path = Path(row["edited_image_path"])
        if not path.is_absolute() and not path.is_file():
            path = selection.parent / path
        with Image.open(path) as image:
            image.verify()
        metadata.append({**index[row["source_key"]], "key": key, "source_key": row["source_key"], "generated_model": row["generated_model"]})
        paths.append(path)
    statuses = []
    for row, path in zip(metadata, paths):
        key = row["key"]
        image = output / "images" / (key + ".png")
        image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, image)
        save_json(output / "results" / (key + ".json"), {"key": key, "status": "ok", "imported_original_output": True, "sha256": digest(image)})
        statuses.append({"key": key, "status": "ready", "image": str(image.resolve()), "compiler_policy": "original_human_evaluation_output"})
    write_jsonl(output / "metadata.jsonl", metadata)
    write_jsonl(output / "render_status.jsonl", statuses)
    job = {"stage": "generation", "model": "mixed_original_human_outputs", "api_model": "archived", "provider": "archived",
        "condition": "direct", "profile": "original_stability", "cohort": "original_50_human_outputs", "modality": "full",
        "judge": "gemini-3.1-pro-preview", "chunks": [], "expected_keys": [r["key"] for r in metadata], "test_subset": False}
    save_json(output / "job.json", job)
    judges = prepare_judge(output / "job.json", repeats=5)
    return {"generation_job": str(output / "job.json"), "judge_jobs": judges, "n_examples": 50, "repeats": 5}


def report_stability(run: Path) -> list[dict]:
    """5反復の全250評点が揃ったときだけ、安定性の統計を保存する。"""
    from .analysis import stability_statistics, write_table
    values = []
    for repeat in range(5):
        folder = run / "judges/gemini-3.1-pro-preview/image" / f"repeat_{repeat:02d}"
        job = read_json(folder / "job.json")
        for key in job["expected_keys"]:
            record = read_json(folder / "scores" / (key + ".json"))
            if record.get("status") != "ok" or not record.get("scores"):
                raise ValueError("Complete/retry all five stability judgments before reporting")
            values.append({"question_id": key, "repeat": repeat, **record["scores"]})
    result = stability_statistics(values)
    write_jsonl(run / "stability_scores.jsonl", values)
    write_table(run / "table11_stability.csv", result)
    return result


def main() -> None:
    """オフライン準備とオンライン操作を別の明示的なコマンドで提供する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models")
    for name in ["plan", "prepare"]:
        command = commands.add_parser(name)
        command.add_argument("--experiment", choices=["main", "alternative_judge", "modality", "text_visual", "verbalization", "direct_image", "stability"], required=True)
        if name == "prepare":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--data-dir", type=Path, default=Path("data"))
            command.add_argument("--models", nargs="+")
            command.add_argument("--conditions", nargs="+", choices=CONDITIONS)
            command.add_argument("--limit", type=int)
    for name in ["submit", "collect", "chat", "render", "judge-prepare", "report", "retry", "recover", "image-run", "import-responses"]:
        command = commands.add_parser(name)
        command.add_argument("--job", type=Path, required=True)
        if name == "import-responses":
            command.add_argument("--responses", type=Path, required=True)
        if name == "submit":
            command.add_argument("--include-blocked", action="store_true")
        if name == "chat":
            command.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL ending in /v1")
            command.add_argument("--retry-failed", action="store_true")
        if name == "image-run":
            command.add_argument("--retry-failed", action="store_true")
        if name == "render":
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--historical", action="store_true", help="Match published PDF-exists compilation policy")
        if name == "judge-prepare":
            command.add_argument("--kind", choices=["image", "code"], default="image")
            command.add_argument("--judge-model")
            command.add_argument("--repeats", type=int, default=1)
        if name == "report":
            command.add_argument("--allow-partial", action="store_true")
        if name == "recover":
            command.add_argument("--chunk", type=int, required=True)
            command.add_argument("--job-id", required=True)
    command = commands.add_parser("server-command")
    command.add_argument("--model", required=True)
    command.add_argument("--profile", choices=["main", "legacy_verbalization"], default="main")
    command.add_argument("--port", type=int, default=8000)
    command = commands.add_parser("reproduce")
    command.add_argument("--data-dir", type=Path, default=Path("data"))
    command.add_argument("--output", type=Path, default=Path("runs/paper_reproduction"))
    command = commands.add_parser("stability-prepare")
    command.add_argument("--selection", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--data-dir", type=Path, default=Path("data"))
    command = commands.add_parser("stability-report")
    command.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "models":
        result = registry()
    elif args.command == "plan":
        result = experiment_groups(args.experiment)
    elif args.command == "prepare":
        result = prepare_experiment(args.experiment, args.output, args.data_dir, models=args.models, conditions=args.conditions, limit=args.limit)
    elif args.command == "submit":
        result = submit_job(args.job, include_blocked=args.include_blocked)
    elif args.command == "collect":
        result = collect_job(args.job)
    elif args.command == "import-responses":
        result = import_responses(args.job, args.responses)
    elif args.command == "chat":
        result = run_chat(args.job, args.endpoint, retry_failed=args.retry_failed)
    elif args.command == "render":
        result = render_job(args.job, overwrite=args.overwrite, historical=args.historical)
    elif args.command == "judge-prepare":
        result = prepare_judge(args.job, kind=args.kind, judge_model=args.judge_model, repeats=args.repeats)
    elif args.command == "report":
        result = report_job(args.job, allow_partial=args.allow_partial)
    elif args.command == "retry":
        result = retry_job(args.job)
    elif args.command == "recover":
        result = read_json(args.job)
        chunk = result["chunks"][args.chunk]
        if chunk["status"] != "submission_unknown":
            raise ValueError("Recovery applies only to submission_unknown")
        chunk.update(job_id=args.job_id, status="submitted")
        save_json(args.job, result)
    elif args.command == "image-run":
        from .image_edit import run_images
        result = run_images(args.job, retry_failed=args.retry_failed)
    elif args.command == "stability-prepare":
        result = prepare_stability(args.selection, args.output, args.data_dir)
    elif args.command == "stability-report":
        result = report_stability(args.run)
    elif args.command == "server-command":
        print(shlex.join(server_command(args.model, profile=args.profile, port=args.port)))
        return
    else:
        from .analysis import reproduce
        result = reproduce(args.data_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
