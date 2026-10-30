from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass
class EvalConfig:
    name: str
    judge_provider: str    # openai / google / anthropic
    judge_model: str       # gpt-5.4 / gemini-3.1-pro-preview / claude-opus-4.6
    judge_input_type: str  # w_org_image / w_org_tikz / w_org_image_w_org_tikz
    edited_model_type: str # gpt-5.4 / gemini-3.1-pro-preview / claude-opus-4.6
    edited_input_type: str # w_org_image / w_org_image_w_org_tikz
    metadata_path: Path
    save_path: Path
    append: bool
    org_url_json: Path | None = None      # GCS URL mapping JSON for org_images
    vi_url_json: Path | None = None       # GCS URL mapping JSON for visual_instruction images
    edited_url_json: Path | None = None   # GCS URL mapping JSON for edited images
    include_important_notes: bool = True  # Include "Important Notes" in prompt (backward compatibility)
    notes_variant: str = "default"        # "default" / "none" / "invalid_output"
    image_detail: str = "auto"            # OpenAI judge setting: "auto" / "original" / "high" / "low"
    temperature: float = 0.0             # Judge model temperature (default: 0)


def load_config(path: Path) -> tuple[dict[str, str], list[EvalConfig]]:
    """Load evaluation experiment settings from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)

    metadata_fields = data.get(
        "metadata_fields",
        {
            "key": "key",
            "org_image_path": "org_image_path",
            "visual_instruction_image_path": "visual_instruction_image_path",
            "org_tikz_code_path": "org_tikz_code_path",
            "text_instruction": "text_instruction",
            "visual_instruction_model_outputs": "visual_instruction_model_outputs",
        },
    )

    configs_raw = data.get("configs", [])
    config_items: list[EvalConfig] = []

    today = datetime.now().strftime("%Y-%m-%d")
    valid_providers = {"openai", "google", "anthropic"}
    valid_judge_input_types = {
        "w_org_image",
        "w_org_tikz",
        "w_org_image_w_org_tikz",
    }

    for cfg in configs_raw:
        name = cfg["name"]
        judge_provider = cfg["judge_provider"]
        judge_model = cfg["judge_model"]
        judge_input_type = cfg["judge_input_type"]
        edited_model_type = cfg["edited_model_type"]
        edited_input_type = cfg["edited_input_type"]
        metadata_path = Path(cfg["metadata_path"]).expanduser()
        append = bool(cfg.get("append", False))

        if judge_provider not in valid_providers:
            raise ValueError(
                f"Unsupported judge_provider in '{name}': {judge_provider}"
            )
        if judge_input_type not in valid_judge_input_types:
            raise ValueError(
                f"Unsupported judge_input_type in '{name}': {judge_input_type}"
            )

        def _opt_path(key: str) -> Path | None:
            v = cfg.get(key)
            return Path(v).expanduser() if v else None

        include_notes = bool(cfg.get("include_important_notes", True))
        raw_variant = cfg.get("notes_variant", "default" if include_notes else "none")
        image_detail = cfg.get("image_detail", "auto")
        temperature = float(cfg.get("temperature", 0.0))

        if "save_path" in cfg:
            save_path = Path(
                cfg["save_path"].format_map(
                    {"name": name, "date": today, "model": judge_model, "detail": image_detail}
                )
            ).expanduser()
        else:
            base_dir = path.parent / "runs" / name
            save_path = base_dir / "input.jsonl"

        config_items.append(
            EvalConfig(
                name=name,
                judge_provider=judge_provider,
                judge_model=judge_model,
                judge_input_type=judge_input_type,
                edited_model_type=edited_model_type,
                edited_input_type=edited_input_type,
                metadata_path=metadata_path,
                save_path=save_path,
                append=append,
                org_url_json=_opt_path("org_url_json"),
                vi_url_json=_opt_path("vi_url_json"),
                edited_url_json=_opt_path("edited_url_json"),
                include_important_notes=include_notes,
                notes_variant=raw_variant,
                image_detail=image_detail,
                temperature=temperature,
            )
        )

    return metadata_fields, config_items
