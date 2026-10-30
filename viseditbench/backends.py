"""投稿時バックエンドの要求・評点パーサーを名前衝突なしで再利用する。"""

from __future__ import annotations

from functools import lru_cache
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=3)
def backend(kind: str) -> SimpleNamespace:
    """各libを別パッケージ名で読み、元のpromptとスコア検証を共有する。"""
    folders = {"editing": "editing/batch_api/lib", "image": "evaluation_image-level/batch_api/lib", "code": "evaluation_code-level/batch_api/lib"}
    path = ROOT / folders[kind]
    name = "_viseditbench_" + kind
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return SimpleNamespace(**{part: importlib.import_module(name + "." + part) for part in ["config", "generate", "extract"]})
