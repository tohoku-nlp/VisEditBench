"""著者用: 既存workspaceの要求から実データを除いた設定テンプレートを固定する。"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def freeze(workspace: Path, repository: Path) -> None:
    """元の要求オプションとpromptを保ち、画像・コード・指示をプレースホルダー化する。"""
    experiment = workspace / "figure7_four_conditions_13models_20261023"
    plan = json.loads((experiment / "config/experiment.json").read_text())
    cohort = workspace / "data/ablation_input_modality_100"
    metadata = {row["key"]: row for row in map(json.loads, (cohort / "metadata.jsonl").read_text().splitlines())}
    models, templates, provenance = {}, {}, []
    for spec in plan["models"].values():
        slug, backend = spec["slug"], spec["backend"]
        models[slug] = {k: spec[k] for k in ["label", "backend", "api_model", "enabled"]}
        models[slug].update(profile="main", figure7=True)
        if not spec["enabled"]:
            models[slug]["blocked_reason"] = spec["blocked_reason"]
        templates[slug] = {}
        for condition in ["direct", "gold_text"]:
            path = sorted((experiment / "templates" / slug / condition).glob("*.json"))[0]
            body, row = json.loads(path.read_text()), metadata[path.stem]
            code = (cohort / row["org_tikz_code_path"]).read_text()
            blocks = body["contents"][0]["parts"] if backend == "google" else body["input"][1]["content"] if backend == "openai" else body["messages"][-1]["content"]
            count = 0
            for block in blocks:
                if "text" in block:
                    if code not in block["text"]:
                        raise ValueError(f"Original code is not identifiable: {path}")
                    block["text"] = block["text"].replace(code, "$SOURCE_TIKZ").replace(row["text_instruction"], "$GOLD_TEXT")
                else:
                    image = "ORG" if count == 0 else "VIS"
                    count += 1
                    if "inlineData" in block:
                        block["inlineData"]["data"] = "$" + image + "_BASE64"
                    elif block.get("type") == "input_image":
                        block["image_url"] = "$" + image + "_DATA_URL"
                    elif block.get("type") == "image":
                        block["source"] = {"type": "base64", "media_type": "image/png", "data": "$" + image + "_BASE64"}
                    elif block.get("type") == "image_url":
                        block["image_url"]["url"] = "$" + image + "_DATA_URL"
                    else:
                        raise ValueError(f"Unknown image block in {path}")
            if count != (1 if condition == "gold_text" else 2):
                raise ValueError(f"Unexpected image count: {path}")
            templates[slug][condition] = body
            provenance.append({"file": "configs/request_templates.json", "source": str(path.relative_to(workspace)),
                "source_kind": "current_payload_skeleton", "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    previous = workspace / "experiments/matched_open_generation_20261013/bundle/config"
    old = json.loads((previous / "plan.json").read_text())
    revisions = dict(old["assets"]["revisions"])
    latest = workspace / "experiments/qwen35_nonthinking_noseed_main694_20261018/tsubame/config/plan.json"
    revisions.update(json.loads(latest.read_text())["assets"]["revisions"])
    prompts = json.loads((previous / "prompts.json").read_text())
    source = workspace / "paper/ablation_gemini_judge_bias/build_claude_opus_judge_table.py"
    assignment = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "MODEL_ORDER" for t in n.targets))
    aliases = ["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gemini-3.1-pro-preview", "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview", "claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"]
    for index, (table_slug, label, access) in enumerate(ast.literal_eval(assignment.value)):
        if index < 9:
            slug = aliases[index]
        elif table_slug.startswith("qwen3_5_"):
            slug = "qwen3.5-" + table_slug[8:].replace("_thinking", "").replace("_", "-").replace("397b", "397b-a17b")
            if not table_slug.endswith("thinking"):
                slug += "-nothink"
        elif table_slug.startswith("qwen3_vl_"):
            slug = table_slug.replace("qwen3_vl_", "qwen3-vl-").replace("_", "-")
        else:
            slug = table_slug.replace("gemma_4_", "gemma4-").replace("_", "-").lower()
        if slug not in models:
            if slug.startswith("qwen3.5-"):
                api = "Qwen/Qwen3.5-" + slug[8:].replace("-nothink", "").upper()
                thinking = not slug.endswith("-nothink")
            elif slug.startswith("qwen3-vl-"):
                size = slug.split("-")[2].upper()
                rest = "-A3B" if "a3b" in slug else "-A22B" if "a22b" in slug else ""
                api = "Qwen/Qwen3-VL-" + size + rest + ("-Thinking" if slug.endswith("thinking") else "-Instruct")
                thinking = None
            elif slug.startswith("gemma4-"):
                api, thinking = "google/gemma-4-" + ("E4B" if "e4b" in slug else "31B") + "-it", None
            else:
                api, thinking = "zai-org/" + label, None
            models[slug] = dict(label=label, backend="vllm", api_model=api, enabled=True, profile="submission", figure7=False)
            templates[slug] = chat_templates(api, prompts, {"seed": 42, "max_tokens": 128000}, thinking, not api.startswith("google/gemma"))
        spec = models[slug]
        spec.update(table_slug=table_slug, label=label, access=access)
        spec["revision"] = revisions.get(spec["api_model"])
        if spec["backend"] == "vllm":
            spec["server"] = dict(dtype="auto", max_model_len=262144 if spec["profile"] == "submission" else 128000, tensor_parallel_size=1,
                gpu_memory_utilization=0.7, generation_config="auto")
            if slug.endswith("-nothink"):
                if "35b-" in slug:
                    spec["server"].update(seed=42, tensor_parallel_size=2)
            else:
                spec["server"]["seed"] = 42
            if spec["profile"] == "submission":
                spec["runtime_evidence"] = "Submission fallback: 128000 output budget with a larger context. Validate native model context/GPU capacity; historical per-model server arguments were not fully preserved."
    legacy = {}
    # 397B Thinkingは保存済みの主実験TOMLのOpenRouter指定に合わせる。
    thinking_397 = models["qwen3.5-397b-a17b"]
    thinking_397.update(backend="openrouter", api_model="qwen/qwen3.5-397b-a17b", profile="historical_openrouter")
    thinking_397.pop("server", None)
    for condition in ["direct", "gold_text"]:
        payload = deepcopy(templates["qwen3.5-397b-a17b-nothink"][condition])
        payload.pop("reasoning", None)
        payload["enable_thinking"] = True
        templates["qwen3.5-397b-a17b"][condition] = payload
    for slug in ["qwen3.5-4b", "qwen3.5-35b-a3b", "gemma4-31b-it", "gemma4-e4b-it"]:
        setting = old["settings"][slug]
        legacy[slug] = chat_templates(old["models"][setting["group"]]["model"], prompts, old["sampling"],
                                     setting["enable_thinking"], setting["use_system_role"])
    for slug, label, backend, api in [("gemini-3-pro-image-preview", "Gemini 3 Pro Image", "google_image", "gemini-3-pro-image-preview"),
        ("gpt-image-1.5", "GPT Image 1.5", "openai_image", "gpt-image-1.5"),
        ("qwen-image-edit", "Qwen-Image-Edit", "diffusers", "Qwen/Qwen-Image-Edit-2511")]:
        models[slug] = dict(label=label, backend=backend, api_model=api, enabled=True, table_slug=slug, access="image", profile="direct_image")
    for filename, content in [("models.json", models), ("request_templates.json", {"main": templates, "legacy_verbalization": legacy}),
                              ("legacy_prompts.json", prompts)]:
        (repository / "configs" / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n")
    manifest_path = repository / "docs/implementation_sources.json"
    manifest = [r for r in json.loads(manifest_path.read_text()) if r["file"] != "configs/request_templates.json"]
    manifest_path.write_text(json.dumps(manifest + provenance, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen {len(models)} models, {len(templates)} main and {len(legacy)} legacy request pairs")


def chat_templates(api: str, prompts: dict, sampling: dict, thinking: bool | None, system_role: bool) -> dict:
    """画像とテキストを含まないOpenAI互換要求テンプレートを作る。"""
    result = {}
    for condition in ["direct", "gold_text"]:
        prompt = prompts[condition]
        text = prompt["user"].replace("{original_tikz_code}", "$SOURCE_TIKZ").replace("{text_instruction}", "$GOLD_TEXT")
        blocks = [{"type": "text", "text": text}] + [{"type": "image_url", "image_url": {"url": "$" + name + "_DATA_URL"}}
            for name in (["ORG"] if condition == "gold_text" else ["ORG", "VIS"])]
        messages = []
        if system_role:
            messages.append({"role": "system", "content": [{"type": "text", "text": prompt["system"]}]})
        else:
            blocks.insert(0, {"type": "text", "text": prompt["system"]})
        messages.append({"role": "user", "content": blocks})
        body = {"model": api, "messages": messages, **deepcopy(sampling)}
        if thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        result[condition] = body
    return result


def main() -> None:
    """著者のローカルworkspaceから設定だけを再生成する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.workspace.resolve(), Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
