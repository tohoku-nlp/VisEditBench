from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI

logger = logging.getLogger(__name__)


def run_batch_openai(
    input_jsonl: Path, output_jsonl: Path, description: str, poll: int
) -> None:
    client = OpenAI()
    with open(input_jsonl, "rb") as f:
        upload_file = client.files.create(file=f, purpose="batch")
    logger.info(f"upload_file_id={upload_file.id}")

    batch = client.batches.create(
        input_file_id=upload_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": description},
    )
    logger.info(f"batch_id={batch.id}")

    while True:
        status_batch = client.batches.retrieve(batch.id)
        status = status_batch.status
        if status == "completed":
            logger.info("Batch completed")
            break
        if status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"Batch failed: {status}")
        time.sleep(poll)

    file_response = client.files.content(status_batch.output_file_id)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f_save:
        f_save.write(file_response.text)


def run_batch_gemini(
    input_jsonl: Path, output_jsonl: Path, model: str, poll: int
) -> None:
    client = genai.Client()
    with open(input_jsonl, "rb") as f:
        upload_file = client.files.upload(
            file=f,
            config=types.UploadFileConfig(
                display_name="gemini-request", mime_type="jsonl"
            ),
        )
    logger.info(f"upload_file_name={upload_file.name}")

    batch_job = client.batches.create(
        model=model,
        src=upload_file.name,
        config={"display_name": f"{model}-job"},
    )
    batch_job_name = batch_job.name
    logger.info(f"batch_job_name={batch_job_name}")

    completed_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    batch_job = client.batches.get(name=batch_job_name)
    while batch_job.state.name not in completed_states:
        logger.info(f"Current state: {batch_job.state.name}")
        time.sleep(poll)
        batch_job = client.batches.get(name=batch_job_name)

    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch failed: {batch_job.state.name}")

    if batch_job.dest and batch_job.dest.file_name:
        result_file_name = batch_job.dest.file_name
        logger.info(f"Results file: {result_file_name}")
        file_content = client.files.download(file=result_file_name)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f_save:
            f_save.write(file_content.decode("utf-8"))
    else:
        raise RuntimeError("No result file found")


def run_batch_claude(input_jsonl: Path, output_jsonl: Path, poll: int) -> None:
    client = Anthropic()
    requests: list[dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            requests.append(json.loads(line))

    message_batch = client.messages.batches.create(requests=requests)

    while True:
        message_batch = client.messages.batches.retrieve(message_batch.id)
        if message_batch.processing_status == "ended":
            break
        logger.info(f"Batch {message_batch.id} is still processing...")
        time.sleep(poll)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "a", encoding="utf-8") as f_save:
        for batch in client.messages.batches.results(message_batch.id):
            f_save.write(json.dumps(batch.model_dump(), ensure_ascii=False) + "\n")
