import argparse
import importlib
import json
import logging
import multiprocessing as mp
import queue
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional

try:
    from pydantic import BaseModel, Field  # type: ignore

    _HAVE_PYDANTIC = True
except Exception:  # pragma: no cover - fallback when pydantic not installed
    _HAVE_PYDANTIC = False

    class BaseModel:  # minimal stub
        pass

    def Field(*args, **kwargs):
        return ...


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.file_io import (
    count_jsonl_entries,
    enqueue_jsonl_records,
    filter_null_outputs,
    load_processed_indices,
)
from utils.logger import setup_logger, setup_queue_listener
from src.http_client import send_inference_request, wait_for_server_ready
from src.message import build_vision_messages
from src.schema import generate_response_format


class _NoOpTqdm:
    def __init__(self, total: int):
        self.total = total

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, n: int = 1) -> None:
        _ = n


def create_progress_bar(total: int):
    tqdm_module = importlib.util.find_spec("tqdm")
    if tqdm_module is None:
        return _NoOpTqdm(total)
    real_tqdm = importlib.import_module("tqdm").tqdm
    return real_tqdm(total=total)


MODEL_NAME_MAP = {
    "qwen3-vl-4b-instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3-vl-4b-thinking": "Qwen/Qwen3-VL-4B-Thinking",
    "qwen3-vl-8b-instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3-vl-8b-thinking": "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3-vl-32b-instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "qwen3-vl-32b-thinking": "Qwen/Qwen3-VL-32B-Thinking",
    "qwen3-vl-30b-a3b-instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3-vl-30b-a3b-thinking": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
}


class EvaluationResponse(BaseModel):
    """Response format for evaluation scores.

    Each score is expected to be an integer in [1, 5].
    """

    instruction_adherence: int = Field(..., ge=1, le=5)
    diagram_readability: int = Field(..., ge=1, le=5)
    content_preservation: int = Field(..., ge=1, le=5)


EVALUATION_SYSTEM_PROMPT = """\
You are an expert judge evaluating TikZ diagram editing quality.

You are given:
1. The ORIGINAL diagram image (before any edits)
2. An annotated version of the original diagram image containing visual edit instructions
3. The EDITED diagram image (after applying the visual edit instructions)

Use the annotated image to understand what changes were requested.
Compare the ORIGINAL and EDITED diagrams to evaluate quality.

Evaluate based on the three criteria below and return ONLY a JSON object with integer scores.

1. instruction_adherence (1-5)
   How well does the edited diagram satisfy the requested changes in the instructions?
   Focus on whether each requested change is correctly applied. Do not penalize for unintended changes in this criterion.
   Note: In this criterion, evaluate only whether the requested changes are correctly applied. Do not consider any changes that were not specified in the instructions; those should be evaluated under 3. content_preservation.
   5 (Perfect): All requested changes are correctly applied according to the instructions.
   4 (Good): Most requested changes are correctly applied, with only minor omissions or inaccuracies.
   3 (Fair): Some requested changes are correctly applied, but several important ones are missing or inaccurate.
   2 (Poor): Only a few requested changes are correctly applied, and many are missing or inaccurate.
   1 (Very Poor): Few or none of the requested changes are correctly applied.

2. diagram_readability (1-5)
   How easy is it to clearly see and understand the elements in the edited diagram?
   Focus only on visibility and legibility. Ignore whether the edits follow the instructions.
   Note: In this criterion, carefully check readability details as well, including any small cut-offs, overlaps, occlusions, or other visibility issues.
   5 (Perfect): All elements are clearly visible and easy to read, with no overlaps, occlusions, elements cut off, or size issues.
   4 (Good): Most elements are clearly visible, with only minor issues such as slight overlap, small elements, or elements partially cut off that do not affect overall readability.
   3 (Fair): Some elements are difficult to see, and this noticeably affects readability, but the diagram is still understandable overall.
   2 (Poor): Many elements are hard to see due to overlap, occlusion, small size, or being cut off, making the diagram difficult to interpret.
   1 (Very Poor): The diagram is largely unreadable, with severe visibility issues such as extensive overlap, occlusion, or elements being cut off.

3. content_preservation (1-5)
   To what extent does the edited diagram avoid unintended changes?
   Focus only on changes that were not requested. Do not consider whether the requested changes are correct.
   Evaluate any changes not specified in the instructions, including major layout modifications. Check carefully for subtle or minor unintended changes.
   5 (Perfect): No unintended changes are present; all non-requested parts of the diagram remain unchanged.
   4 (Good): Very few unintended changes are present, and they have minimal impact on the overall diagram.
   3 (Fair): Some unintended changes are present and somewhat affect the diagram.
   2 (Poor): Many unintended changes are introduced, affecting large parts of the diagram.
   1 (Very Poor): Extensive unintended changes are present, and much of the original content is not preserved.

Important Notes for Evaluation:
- Invalid Edited Output: If the edited output is not a diagram, please rate all three criteria as 1 (Very Poor). Examples include a completely blank (white) image and a text-only image.
- Whitespace/Margins: If the edited diagram is present but surrounded by whitespace or margins, ignore the whitespace and evaluate only the diagram content itself.
- Cropped or Cut-Off Diagrams: If the diagram appears cropped or cut off (missing elements at edges), evaluate this under the Diagram Readability criterion, as it directly affects visibility.

Return ONLY valid JSON (no markdown, no explanation):
{"instruction_adherence": <1-5>, "diagram_readability": <1-5>, "content_preservation": <1-5>}
"""


def strip_thinking(content: str) -> tuple[str, int]:
    end_marker = "</think>"
    end_idx = content.find(end_marker)
    if end_idx == -1:
        return content.strip(), 0
    start_marker = "<think>"
    start_idx = content.find(start_marker)
    if start_idx != -1 and start_idx < end_idx:
        thinking_chars = end_idx - (start_idx + len(start_marker))
    else:
        thinking_chars = end_idx
    return content[end_idx + len(end_marker) :].strip(), thinking_chars


def resolve_model_name(name: str) -> str:
    key = name.strip().lower()
    return MODEL_NAME_MAP.get(key, name)


def sanitize_filename_component(value: object) -> str:
    text = str(value).strip()
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "output"


SPECIAL_EDITED_W_ORG = {
    "gpt-image-1.5",
    "gemini-3-pro-image-preview",
    "qwen-image-edit-2511",
}


def resolve_image_paths(
    data: dict, model: Optional[str] = None
) -> tuple[str, str, str]:
    """Resolve original/annotated/edited image paths from input metadata.

    Looks up both top-level keys and `data.get("url", {})`.
    For edited image selection, if `model` is in SPECIAL_EDITED_W_ORG,
    prefer `w_org_image`; otherwise prefer `w_org_image_w_org_tikz`.
    Raises ValueError when required paths are missing.
    """
    url_map = data.get("url") if isinstance(data.get("url"), dict) else {}

    def lookup(*keys):
        for k in keys:
            # check top-level
            v = data.get(k)
            if v:
                return str(v)
            # check inside url map
            v = url_map.get(k)
            if v:
                return str(v)
        return None

    original_image_path = lookup(
        "org_image_path",
        "original_image_path",
        "org_img_path",
        "org_image",
        "original_image",
    )
    annotated_image_path = lookup(
        "visual_instruction_image_path",
        "annotated_image_path",
        "visual_instruction_path",
        "visual_instruction_image",
        "annotated_image",
    )

    # choose edited image key based on model
    model_key = (model or "").strip().lower()
    if model_key in SPECIAL_EDITED_W_ORG:
        edited_candidates = (
            "w_org_image",
            "edited_image_path",
            "edited_img_path",
            "edited_image",
        )
    else:
        edited_candidates = (
            "w_org_image_w_org_tikz",
            "w_org_image",
            "edited_image_path",
            "edited_img_path",
            "edited_image",
        )

    edited_image_path = lookup(*edited_candidates)

    if not original_image_path or not annotated_image_path or not edited_image_path:
        missing = []
        if not original_image_path:
            missing.append("original image")
        if not annotated_image_path:
            missing.append("annotated image")
        if not edited_image_path:
            missing.append("edited image")
        raise ValueError(f"Missing image paths: {', '.join(missing)}")

    return original_image_path, annotated_image_path, edited_image_path


def preprocess_visual_edit(data: dict, model: Optional[str] = None) -> list[dict]:
    """Build vLLM messages from one metadata row and model name.

    Messages include the system prompt and three user image placeholders:
    original, annotated, and edited.
    """
    original_image_path, annotated_image_path, edited_image_path = resolve_image_paths(
        data, model
    )
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url"},
                {"type": "image_url"},
                {"type": "image_url"},
            ],
        },
    ]
    return build_vision_messages(
        {
            "messages": messages,
            "image_paths": [
                original_image_path,
                annotated_image_path,
                edited_image_path,
            ],
        }
    )


def resolve_output_stem(data: dict) -> str:
    key = data.get("key")
    if key:
        return sanitize_filename_component(key)

    annotated_image_path = (
        data.get("visual_instruction_image_path")
        or data.get("annotated_image_path")
        or data.get("visual_instruction_path")
    )
    if annotated_image_path:
        return sanitize_filename_component(Path(str(annotated_image_path)).stem)

    index = data.get("__index__")
    if index is not None:
        return sanitize_filename_component(index)

    return "output"


def save_tikz_output(save_dir: Path, data: dict, tikz_code: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{resolve_output_stem(data)}.tex"
    save_path.write_text(tikz_code, encoding="utf-8")
    return save_path


def inference_worker(
    input_queue: mp.JoinableQueue,
    output_queue: mp.JoinableQueue,
    host: str,
    port: str,
    token: str,
    model: str,
    seed: int,
    generation_config: dict,
    log_queue: mp.Queue,
    preprocess_fn: Callable[[dict, Optional[str]], list[dict]] = preprocess_visual_edit,
    response_model: Optional[type] = None,
    save_tex_dir: Optional[Path] = None,
) -> None:
    logger = setup_logger(f"worker-{host}:{port}", logging.DEBUG, log_queue)

    base_url = f"http://{host}:{port}/"
    endpoint = base_url + "v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        if not wait_for_server_ready(
            base_url=base_url,
            headers=headers,
            max_retries=60,
            retry_interval=10,
            timeout=30,
            logger=logger,
        ):
            logger.error(
                f"Cannot start inference for {host}:{port} because server is not ready."
            )
            return

        while True:
            data = input_queue.get()
            if data is None:
                input_queue.task_done()
                logger.debug("Received termination signal.")
                break

            try:
                messages = preprocess_fn(data, model)
            except Exception as e:
                logger.warning(
                    f"Preprocess failed for sample_index={data.get('__index__')}: {e}"
                )
                result = None
                input_data = {k: v for k, v in data.items() if k != "__index__"}
                output = {
                    "__index__": data.get("__index__"),
                    "input": input_data,
                    "output": result,
                    "thinking_chars": 0,
                    "finish_reason": None,
                    "request_elapsed_seconds": None,
                }
                output_queue.put(output)
                input_queue.task_done()
                continue

            payload = {
                "model": model,
                "messages": messages,
                "seed": seed,
            }
            payload.update(generation_config)
            payload["temperature"] = 0.0
            if response_model is not None:
                payload["response_format"] = generate_response_format(response_model)

            request_start = time.monotonic()
            raw_result = send_inference_request(
                endpoint=endpoint,
                headers=headers,
                payload=payload,
                logger=logger,
            )
            request_elapsed = round(time.monotonic() - request_start, 3)

            thinking_chars = 0
            finish_reason = None
            try:
                if raw_result is not None:
                    raw_content, finish_reason = raw_result
                    if response_model is not None:
                        result = response_model.model_validate_json(
                            raw_content
                        ).model_dump()
                    else:
                        result, thinking_chars = strip_thinking(raw_content)
                else:
                    result = None
            except Exception as e:
                logger.warning(
                    f"Response parse failed for sample_index={data.get('__index__')}: {e}"
                )
                result = None

            if save_tex_dir is not None and isinstance(result, str):
                try:
                    save_path = save_tikz_output(save_tex_dir, data, result)
                    logger.debug(
                        f"Saved TikZ for sample_index={data.get('__index__')} to {save_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"TikZ save failed for sample_index={data.get('__index__')}: {e}"
                    )

            input_data = {k: v for k, v in data.items() if k != "__index__"}
            output = {
                "__index__": data.get("__index__"),
                "input": input_data,
                "output": result,
                "thinking_chars": thinking_chars,
                "finish_reason": finish_reason,
                "request_elapsed_seconds": request_elapsed,
            }
            output_queue.put(output)

            input_queue.task_done()
            logger.debug(f"Processed sample_index={data.get('__index__')}")
    except Exception as e:
        logger.error(f"Worker {host}:{port} crashed: {e}")
    finally:
        output_queue.put(None)


def collect_results(
    output_queue: mp.JoinableQueue,
    output_path: Path,
    num_tasks: int,
    num_processes: int,
    logger: logging.Logger,
) -> tuple[int, list[dict]]:
    num_finished = 0
    num_written = 0
    timed_out_inputs: list[dict] = []
    with (
        open(output_path, mode="a", encoding="utf-8") as f,
        create_progress_bar(total=num_tasks) as pbar,
    ):
        while num_finished < num_processes:
            output = output_queue.get()
            if output is None:
                output_queue.task_done()
                num_finished += 1
                continue

            record = {
                "__index__": output["__index__"],
                "input": output["input"],
                "output": output["output"],
                "thinking_chars": output.get("thinking_chars", 0),
                "finish_reason": output.get("finish_reason"),
                "request_elapsed_seconds": output.get("request_elapsed_seconds"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            num_written += 1

            if output["output"] is None:
                timed_out_inputs.append(output["input"])

            output_queue.task_done()
            pbar.update(1)
            logger.debug(f"Wrote result for sample_index={output['__index__']}")

    logger.info(f"Finished writing {num_written} results to {output_path}")
    return num_written, timed_out_inputs


def save_timed_out_inputs(
    timed_out_inputs: list[dict],
    output_path: Path,
    logger: logging.Logger,
) -> None:
    try:
        rel = output_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel = Path(output_path.name)
    timed_out_path = Path("data") / "timed_out" / rel
    timed_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(timed_out_path, mode="w", encoding="utf-8") as f:
        for entry in timed_out_inputs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(timed_out_inputs)} timed-out inputs to {timed_out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--hosts", type=str, nargs="+", required=True)
    parser.add_argument("--ports", type=str, nargs="+", required=True)
    parser.add_argument("--num_processes_per_server", type=int, default=1)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=128000)
    parser.add_argument("--tex_output_dir", type=Path)
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def main(
    preprocess_fn: Callable[[dict, Optional[str]], list[dict]] | None = None,
    response_model: Optional[type] = None,
    generation_config: Optional[dict] = None,
) -> None:
    args = parse_args()
    if preprocess_fn is None:
        preprocess_fn = preprocess_visual_edit
    if generation_config is None:
        generation_config = {
            "max_tokens": args.max_tokens,
        }
    if args.enable_thinking:
        generation_config["chat_template_kwargs"] = {"enable_thinking": True}

    # Ensure we use the evaluation response format by default
    if response_model is None:
        response_model = EvaluationResponse  # type: ignore

    log_queue: mp.Queue = mp.Queue()
    listener = setup_queue_listener(log_queue)
    listener.start()

    logger = setup_logger("main", logging.INFO, log_queue)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(args.hosts) != len(args.ports):
        raise ValueError(
            f"The number of hosts ({len(args.hosts)}) must match "
            f"the number of ports ({len(args.ports)})."
        )

    filter_null_outputs(args.output_path)
    processed_indices = load_processed_indices(args.output_path)
    if processed_indices:
        logger.info(
            f"Found {len(processed_indices)} already processed entries. Skipping."
        )

    num_tasks = count_jsonl_entries(args.input_path) - len(processed_indices)
    if num_tasks <= 0:
        logger.info("All entries already processed. Nothing to do.")
        listener.stop()
        return

    hosts = args.hosts * args.num_processes_per_server
    ports = args.ports * args.num_processes_per_server
    num_processes = len(hosts)

    tex_output_dir = args.tex_output_dir
    if tex_output_dir is None:
        tex_output_dir = Path("work") / "evaluation"
    tex_output_dir.mkdir(parents=True, exist_ok=True)

    input_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=num_processes)
    output_queue: mp.JoinableQueue = mp.JoinableQueue(maxsize=num_processes)

    processes = []
    for host, port in zip(hosts, ports):
        p = mp.Process(
            target=inference_worker,
            kwargs={
                "input_queue": input_queue,
                "output_queue": output_queue,
                "host": host,
                "port": port,
                "token": args.token,
                "model": args.model,
                "seed": args.seed,
                "generation_config": generation_config,
                "log_queue": log_queue,
                "preprocess_fn": preprocess_fn,
                "response_model": response_model,
                "save_tex_dir": tex_output_dir,
            },
        )
        p.start()
        processes.append(p)
        logger.info(f"Started inference process for {host}:{port}")

    dataloader = mp.Process(
        target=enqueue_jsonl_records,
        kwargs={
            "dataset_path": args.input_path,
            "input_queue": input_queue,
            "num_processes": num_processes,
            "log_queue": log_queue,
            "skip_indices": processed_indices,
        },
    )
    dataloader.start()
    logger.info("Started dataloader process.")
    num_written, timed_out_inputs = collect_results(
        output_queue=output_queue,
        output_path=args.output_path,
        num_tasks=num_tasks,
        num_processes=num_processes,
        logger=logger,
    )

    output_queue.join()
    logger.info("output_queue finished.")

    dataloader.join(timeout=0)
    while dataloader.is_alive():
        try:
            input_queue.get_nowait()
            input_queue.task_done()
        except queue.Empty:
            pass
        dataloader.join(timeout=0.1)
    logger.info("Dataloader process finished.")

    for p in processes:
        p.join()
    logger.info("All processes finished.")

    if timed_out_inputs:
        save_timed_out_inputs(timed_out_inputs, args.output_path, logger)
    logger.info(
        f"Done. {num_written}/{num_tasks} results written to {args.output_path}"
    )
    logger.info(f"TikZ files saved under {tex_output_dir}")
    listener.stop()


if __name__ == "__main__":
    main()
