"""Gemini・GPT・Qwenの直接画像編集を共通の出力記録へ接続する。"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

from .backends import ROOT
from .data import digest, read_jsonl
from .workflow import fingerprint, read_json, save_json, job_lock, checked_job


def image_request(spec: dict, row: dict) -> dict:
    """同じ元画像・指示画像を順に渡す直接画像編集の要求を作る。"""
    prompt = read_json(ROOT / "configs/image_prompts.json")["visual"]
    paths = [row["org_image_path"], row["visual_instruction_image_path"]]
    return {"model": spec["api_model"], "prompt": prompt, "image_paths": paths,
            "image_sha256": [digest(Path(path)) for path in paths], "backend": spec["backend"]}


@job_lock
def run_images(path: Path, *, retry_failed: bool = False, clients: dict[str, Any] | None = None) -> dict:
    """明示的な実行時だけ画像API/GPUを使い、元応答とPNG・実行状態を保存する。"""
    import os
    import requests
    from dotenv import load_dotenv
    from PIL import Image
    job, folder = checked_job(path), path.parent
    if job["provider"] not in {"google_image", "openai_image", "diffusers"}:
        raise ValueError("This job is not a direct-image experiment")
    load_dotenv(ROOT / ".env", override=False)
    clients = clients or {}
    pipeline, generator = None, None
    if job["provider"] == "diffusers":
        import torch
        from diffusers import QwenImageEditPlusPipeline
        pipeline = QwenImageEditPlusPipeline.from_pretrained(job["api_model"], torch_dtype=torch.bfloat16).to("cuda")
        generator = torch.Generator(device="cuda").manual_seed(0)
    completed, failed = 0, 0
    for chunk in job["chunks"]:
        if digest(folder / chunk["input"]) != chunk["sha256"]:
            raise ValueError("Image request changed after preparation")
        for request in read_jsonl(folder / chunk["input"]):
            key, body = request["key"], request["body"]
            result_path = folder / "results" / (key + ".json")
            if result_path.exists():
                if read_json(result_path)["status"] != "api_error" or not retry_failed:
                    continue
            marker = folder / "attempts" / (key + ".json")
            if marker.exists() and not result_path.exists():
                raise ValueError(f"Image submission outcome unknown: {key}")
            for image, expected in zip(body["image_paths"], body["image_sha256"]):
                if digest(Path(image)) != expected:
                    raise ValueError("Input image changed after preparation")
            save_json(marker, {"state": "submission_unknown", "request_sha256": fingerprint(body)})
            try:
                if job["provider"] == "google_image":
                    parts = [{"text": body["prompt"]}] + [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(Path(p).read_bytes()).decode()}} for p in body["image_paths"]]
                    token = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                    response = (clients.get("post") or requests.post)(
                        "https://generativelanguage.googleapis.com/v1beta/models/" + job["api_model"] + ":generateContent",
                        headers={"x-goog-api-key": token or ""}, json={"contents": [{"parts": parts}], "generationConfig": {"responseModalities": ["IMAGE"]}}, timeout=1800)
                    response.raise_for_status()
                    raw = response.json()
                    image_part = next(p["inlineData"] for c in raw.get("candidates", []) for p in c.get("content", {}).get("parts", []) if "inlineData" in p)
                    pixels = base64.b64decode(image_part["data"])
                    # 元応答には画像Base64も残す。APIキーは要求記録へ含めない。
                    save_json(folder / "raw" / (key + ".json"), raw)
                elif job["provider"] == "openai_image":
                    from contextlib import ExitStack
                    from openai import OpenAI
                    client = clients.get("openai") or OpenAI(timeout=1800, max_retries=0)
                    with ExitStack() as stack:
                        handles = [stack.enter_context(Path(p).open("rb")) for p in body["image_paths"]]
                        response = client.images.edit(model=job["api_model"], image=handles, prompt=body["prompt"])
                    raw = response.model_dump(mode="json")
                    item = raw["data"][0]
                    if item.get("b64_json"):
                        pixels = base64.b64decode(item["b64_json"])
                    else:
                        download = requests.get(item["url"], timeout=300)
                        download.raise_for_status()
                        pixels = download.content
                    save_json(folder / "raw" / (key + ".json"), raw)
                else:
                    images = [Image.open(p).convert("RGB") for p in body["image_paths"]]
                    result = pipeline(image=images, prompt=body["prompt"], generator=generator)
                    buffer = BytesIO()
                    result.images[0].save(buffer, format="PNG")
                    pixels = buffer.getvalue()
                    save_json(folder / "raw" / (key + ".json"), {"seed": 0, "model": job["api_model"], "sampling": "pipeline defaults; shared generator in metadata order"})
                target = folder / "images" / (key + ".png")
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(BytesIO(pixels)) as image:
                    image.load()
                    if image.format == "PNG":
                        target.write_bytes(pixels)
                    else:
                        image.save(target, format="PNG")
                save_json(result_path, {"key": key, "status": "ok", "image_sha256": digest(target), "request_sha256": fingerprint(body)})
                completed += 1
            except Exception as exc:
                save_json(result_path, {"key": key, "status": "api_error", "error": str(exc), "request_sha256": fingerprint(body)})
                failed += 1
    return {"completed": completed, "failed": failed}
