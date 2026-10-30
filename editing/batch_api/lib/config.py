from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass
class ConfigItem:
    name: str
    provider: str
    model: str
    system_instruction_type: str
    metadata_path: Path
    save_path: Path
    append: bool
    thinking: bool = False
    org_url_json: Path | None = None   # GCS URL JSON（org 画像）
    vi_url_json: Path | None = None    # GCS URL JSON（visual instruction 画像）
    image_detail: str = "auto"         # OpenAI detail パラメータ: "auto" | "high" | "low" | "original"


def load_config(path: Path) -> tuple[dict[str, str], list[ConfigItem]]:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    metadata_fields = data.get(
        "metadata_fields",
        {
            "key": "key",
            "org_image_path": "org_image_path",
            "visual_instruction_image_path": "visual_instruction_image_path",
            "org_tikz_code_path": "org_tikz_code_path",
        },
    )

    configs_raw = data.get("configs", [])
    config_items: list[ConfigItem] = []

    today = datetime.now().strftime("%Y-%m-%d")
    for cfg in configs_raw:
        name = cfg["name"]
        provider = cfg["provider"]
        model = cfg["model"]
        system_instruction_type = cfg["system_instruction_type"]
        metadata_path = Path(cfg["metadata_path"]).expanduser()

        append       = bool(cfg.get("append", False))
        thinking     = bool(cfg.get("thinking", False))
        org_url_json = cfg.get("org_url_json")
        vi_url_json  = cfg.get("vi_url_json")
        image_detail = cfg.get("image_detail", "auto")

        if "save_path" in cfg:
            save_path = Path(
                cfg["save_path"].format_map(
                    {"name": name, "date": today, "model": model, "detail": image_detail}
                )
            ).expanduser()
        else:
            base_dir = path.parent / "runs" / name
            save_path = base_dir / "input.jsonl"

        config_items.append(
            ConfigItem(
                name=name,
                provider=provider,
                model=model,
                system_instruction_type=system_instruction_type,
                metadata_path=metadata_path,
                save_path=save_path,
                append=append,
                thinking=thinking,
                org_url_json=Path(org_url_json).expanduser() if org_url_json else None,
                vi_url_json=Path(vi_url_json).expanduser()   if vi_url_json  else None,
                image_detail=image_detail,
            )
        )

    return metadata_fields, config_items


def resolve_metadata_fields(
    metadata_dict: dict[str, Any], mapping: dict[str, str]
) -> dict[str, Any]:
    resolved = {}
    for key, source_key in mapping.items():
        resolved[key] = metadata_dict.get(source_key)
    return resolved
