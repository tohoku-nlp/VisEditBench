from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator


def jsonl_file_loader(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "__index__" not in obj:
                obj["__index__"] = i
            yield obj


def enqueue_jsonl_records(
    dataset_path: Path,
    input_queue,
    num_processes: int,
    log_queue=None,
    skip_indices: set[int] | None = None,
) -> None:
    logger: logging.Logger | None = None
    if log_queue is not None:
        from .logger import setup_logger

        logger = setup_logger("dataloader", logging.INFO, log_queue)

    skipped = 0
    enqueued = 0
    skip = skip_indices or set()
    for obj in jsonl_file_loader(dataset_path):
        idx = obj.get("__index__")
        if isinstance(idx, int) and idx in skip:
            skipped += 1
            continue
        input_queue.put(obj)
        enqueued += 1

    for _ in range(num_processes):
        input_queue.put(None)

    if logger:
        logger.info(
            f"Dataloader finished: enqueued={enqueued}, skipped={skipped}, workers={num_processes}"
        )


def count_jsonl_entries(path: Path) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def load_processed_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    out: set[int] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = obj.get("__index__")
            if isinstance(idx, int):
                out.add(idx)
    return out


def filter_null_outputs(path: Path) -> None:
    if not path.exists():
        return
    kept: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("output") is None:
                continue
            kept.append(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(kept)
