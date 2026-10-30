import argparse
import importlib
import importlib.util
import json
import logging
import multiprocessing as mp
import queue
import re
import time
from pathlib import Path
from typing import Callable, Optional

from utils.file_io import (
    count_jsonl_entries,
    filter_null_outputs,
    jsonl_file_loader,
    load_processed_indices,
)
from utils.logger import setup_logger, setup_queue_listener
from vllm.http_client import send_inference_request, wait_for_server_ready
from vllm.message import build_vision_messages
from vllm.schema import generate_response_format


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
    "qwen3-vl-30b-a3b-instruct-fp8": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    "qwen3-vl-30b-a3b-thinking-fp8": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
}

VISUAL_EDIT_SYSTEM_PROMPT = r"""You are an expert at converting diagram images into LaTeX/TikZ.

You are given two images of the same diagram:
1. the original diagram image, before any edit instructions were added, and
2. an annotated version of that same diagram image containing written edit instructions.

Use both images together:
- Use the original image to understand the diagram clearly without overlaid markings.
- Use the annotated image to identify the requested edits.
- Use both the original image to capture the underlying content and the annotated image to understand the intended modifications.

Apply the requested edits and produce LaTeX/TikZ code that represents the final diagram after those edits have been applied.
Your output must describe the edited final state of the diagram itself, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

VISUAL_EDIT_W_ORG_TIKZ_SYSTEM_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1) the original TikZ code of a diagram,
2) the original diagram image, and
3) an annotated version of that diagram image containing visual edit instructions.

Use all inputs together:
- Use the original TikZ code and the original image to understand the structure and content of the diagram.
- Use the annotated image to identify the requested edits.
- Resolve ambiguities by referring to the original diagram (code + image) for the base structure and the annotated image for the intended modifications.

Apply the requested edits and produce LaTeX/TikZ code representing the final diagram after the edits have been applied.
Your output must describe only the final edited diagram, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

TEXT_EDIT_W_ORG_TIKZ_SYSTEM_PROMPT = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original diagram image,
2. the original TikZ code of a diagram, and
3. a text instruction that describes how to edit the diagram.

Use all inputs together:
- Use the original TikZ code and the original image to understand the structure and content of the diagram.
- Use the text instruction to identify the requested edits.

Apply the requested edits and produce LaTeX/TikZ code representing the final diagram after the edits have been applied.
Your output must describe only the final edited diagram, not the original diagram and not the editing process.

Return only a complete standalone LaTeX document that compiles the final edited diagram.
The output must:
- begin with \documentclass{standalone}
- include all required packages and TikZ libraries
- contain exactly one tikzpicture environment
- end with \end{document}

Do not include any explanation, reasoning, markdown, code fences, comments, or any text before or after the LaTeX document.
"""

USER_PROMPT_TEXT_W_ORG_TIKZ = r"""Original TikZ code:
{original_tikz_code}

Text instruction:
{text_instruction}
"""

USER_PROMPT_W_ORG_TIKZ = r"""Original TikZ code:
{original_tikz_code}
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


def resolve_original_image_path(data: dict) -> str:
    original_image_path = (
        data.get("org_image_path")
        or data.get("original_image_path")
        or data.get("org_img_path")
    )
    if not original_image_path:
        raise ValueError("Missing original image path: expected org_image_path")
    return str(original_image_path)


def resolve_annotated_image_path(data: dict) -> str:
    annotated_image_path = (
        data.get("visual_instruction_image_path")
        or data.get("annotated_image_path")
        or data.get("visual_instruction_path")
    )
    if not annotated_image_path:
        raise ValueError(
            "Missing annotated image path: expected visual_instruction_image_path"
        )
    return str(annotated_image_path)


def resolve_original_tikz_code(data: dict) -> str:
    original_tikz_code = (
        data.get("original_tikz_code")
        or data.get("org_tikz_code")
        or data.get("tikz_code")
        or data.get("original_tikz")
    )
    if original_tikz_code:
        return str(original_tikz_code)

    code_path = data.get("org_tikz_code_path")
    if code_path:
        return Path(str(code_path)).read_text(encoding="utf-8")

    raise ValueError(
        "Missing original tikz code: expected original_tikz_code or org_tikz_code_path"
    )


def resolve_text_instruction(data: dict) -> str:
    text_instruction = data.get("text_instruction") or data.get("instruction")
    if not text_instruction:
        raise ValueError("Missing text instruction: expected text_instruction")
    return str(text_instruction)


def preprocess_visual_edit(data: dict) -> list[dict]:
    original_image_path = resolve_original_image_path(data)
    annotated_image_path = resolve_annotated_image_path(data)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": VISUAL_EDIT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url"},
                {"type": "image_url"},
            ],
        },
    ]
    return build_vision_messages(
        {
            "messages": messages,
            "image_paths": [original_image_path, annotated_image_path],
        }
    )


def preprocess_visual_edit_w_org_tikz(data: dict) -> list[dict]:
    original_image_path = resolve_original_image_path(data)
    annotated_image_path = resolve_annotated_image_path(data)
    original_tikz_code = resolve_original_tikz_code(data)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": VISUAL_EDIT_W_ORG_TIKZ_SYSTEM_PROMPT}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_W_ORG_TIKZ.format(
                        original_tikz_code=original_tikz_code
                    ),
                },
                {"type": "image_url"},
                {"type": "image_url"},
            ],
        },
    ]
    return build_vision_messages(
        {
            "messages": messages,
            "image_paths": [original_image_path, annotated_image_path],
        }
    )


def preprocess_text_edit_w_org_tikz(data: dict) -> list[dict]:
    original_image_path = resolve_original_image_path(data)
    original_tikz_code = resolve_original_tikz_code(data)
    text_instruction = resolve_text_instruction(data)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": TEXT_EDIT_W_ORG_TIKZ_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": USER_PROMPT_TEXT_W_ORG_TIKZ.format(
                        original_tikz_code=original_tikz_code,
                        text_instruction=text_instruction,
                    ),
                },
                {"type": "image_url"},
            ],
        },
    ]
    return build_vision_messages(
        {
            "messages": messages,
            "image_paths": [original_image_path],
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
    preprocess_fn: Callable[[dict], list[dict]],
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
                messages = preprocess_fn(data)
            except Exception as e:
                logger.warning(
                    f"Preprocess failed for sample_index={data.get('__index__')}: {e}"
                )
                input_data = {k: v for k, v in data.items() if k != "__index__"}
                output_queue.put(
                    {
                        "__index__": data.get("__index__"),
                        "input": input_data,
                        "output": None,
                        "thinking_chars": 0,
                        "finish_reason": None,
                        "request_elapsed_seconds": None,
                    }
                )
                input_queue.task_done()
                continue

            payload = {
                "model": model,
                "messages": messages,
                "seed": seed,
            }
            payload.update(generation_config)
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


def default_tex_output_dir(mode: str, model_name: str) -> Path:
    if mode == "visual":
        return Path("work") / "editing" / "visual" / "w_org_image" / model_name / "tikz"
    if mode == "visual_w_org_tikz":
        return (
            Path("work")
            / "editing"
            / "visual"
            / "w_org_image_w_org_tikz"
            / model_name
            / "tikz"
        )
    if mode == "text_w_org_tikz":
        return (
            Path("work")
            / "editing"
            / "text"
            / "w_org_image_w_org_tikz"
            / model_name
            / "tikz"
        )
    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["visual", "visual_w_org_tikz", "text_w_org_tikz"],
    )
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
    preprocess_fn: Callable[[dict], list[dict]] | None = None,
    response_model: Optional[type] = None,
    generation_config: Optional[dict] = None,
) -> None:
    args = parse_args()

    preprocess_map: dict[str, Callable[[dict], list[dict]]] = {
        "visual": preprocess_visual_edit,
        "visual_w_org_tikz": preprocess_visual_edit_w_org_tikz,
        "text_w_org_tikz": preprocess_text_edit_w_org_tikz,
    }

    if preprocess_fn is None:
        preprocess_fn = preprocess_map[args.mode]

    if generation_config is None:
        generation_config = {"max_tokens": args.max_tokens}
    if args.enable_thinking:
        generation_config["chat_template_kwargs"] = {"enable_thinking": True}

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

    resolved_model_name = resolve_model_name(args.model)

    tex_output_dir = args.tex_output_dir
    if tex_output_dir is None:
        tex_output_dir = default_tex_output_dir(args.mode, args.model)
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
                "model": resolved_model_name,
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
        target=jsonl_file_loader,
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
