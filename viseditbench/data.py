"""Hugging Faceの公開データを既存の生成・評価バックエンドへ接続する。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any

REPO_ID = "tohoku-nlp/VisEditBench"
KEY_PATTERN = re.compile(r"^\d{8}_(ADD|REMOVE|UPDATE|LAYOUT|STRUCTURE)_([123])_edit_\d+$")
IMAGE_FIELDS = {
    "source_images": ("source_image_file_name", "source_image", "org_image_path"),
    "visual_instructions": ("visual_instruction_file_name", "visual_instruction", "visual_instruction_image_path"),
    "target_images": ("target_image_file_name", "target_image", "target_image_path"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONLを読み、重複キーや空行は呼び出し側で扱う。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """生成途中のJSONLを参照されないよう、一時ファイルから置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    """画像・コード・メタデータの内容をSHA-256で固定する。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_metadata(snapshot: Path) -> Path:
    """データセット直下とdataset/配下の公開形式を受け入れる。"""
    for relative in ["dataset/metadata.jsonl", "metadata.jsonl", "dataset/metadata_hf.jsonl", "metadata_hf.jsonl"]:
        if (snapshot / relative).is_file():
            return snapshot / relative
    raise FileNotFoundError(f"metadata.jsonl was not found in {snapshot} or its dataset/ directory")


def resolve_asset(row: dict[str, Any], aliases: tuple[str, ...], metadata: Path, snapshot: Path) -> Path:
    """公開メタデータの相対パスを解決し、URLや親ディレクトリ参照を拒否する。"""
    value = next((row.get(field) for field in aliases if row.get(field)), None)
    if not isinstance(value, str):
        raise ValueError(f"Missing image field: {aliases}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or ":" in value or "\\" in value:
        raise ValueError(f"Image paths must be dataset-relative: {value}")
    for base in [metadata.parent, snapshot]:
        path = base.joinpath(*relative.parts)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Image referenced by metadata does not exist: {value}")


def portable_path(path: Path, root: Path) -> str:
    """リポジトリ内は相対パス、外部の作業領域は絶対パスとして記録する。"""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def prepare(snapshot: Path, data_dir: Path, *, repo_id: str = REPO_ID, revision: str = "local",
            limit: int | None = None, force: bool = False,
            text_instructions: Path | None = None) -> dict[str, Any]:
    """画像を原寸コピーし、コードを展開して全バックエンド共通のメタデータを作る。"""
    from PIL import Image

    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    metadata = find_metadata(snapshot)
    supplement = next((p for p in [snapshot / "reproduction", metadata.parent / "reproduction"] if p.is_dir()), None)
    if supplement:
        supplemental_manifest = verify_supplement(supplement)
        if supplemental_manifest.get("base_metadata_sha256") and supplemental_manifest["base_metadata_sha256"] != digest(metadata):
            raise ValueError("The reproduction supplement targets a different benchmark metadata snapshot")
    if supplement and text_instructions is None and (supplement / "gold_text.jsonl").is_file():
        text_instructions = supplement / "gold_text.jsonl"
    rows = read_jsonl(metadata)
    seen = set()
    for row in rows:
        key = row.get("id") or row.get("key")
        match = KEY_PATTERN.fullmatch(str(key))
        if not match or key in seen:
            raise ValueError(f"Invalid or duplicate sample id: {key}")
        if row.get("edit_type") != match[1] or int(row.get("edit_operation", 0)) != int(match[2]):
            raise ValueError(f"Edit type/count disagrees with sample id: {key}")
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ["source_tikz", "target_tikz"]):
            raise ValueError(f"Inline source_tikz/target_tikz are required: {key}")
        seen.add(key)
    if not rows:
        raise ValueError("Dataset metadata is empty")
    if limit:
        rows = rows[:limit]
    destination = data_dir / "metadata/edit_metadata.jsonl"
    if destination.exists() and not force:
        raise FileExistsError(f"Prepared data exists: {destination}. Use --force to prepare it again.")
    extra = {}
    if text_instructions:
        for item in read_jsonl(text_instructions):
            key = item.get("id") or item.get("key")
            value = item.get("text_instruction")
            if key in extra or not isinstance(value, str) or not value.strip():
                raise ValueError(f"Invalid or duplicate text instruction: {key}")
            extra[key] = value
        if set(extra) - seen:
            raise ValueError("Supplemental text contains sample ids outside the dataset")
    # 変換前に全入力画像を検証し、不完全なダウンロードを検出する。
    assets = []
    for row in rows:
        images = {name: resolve_asset(row, fields, metadata, snapshot) for name, fields in IMAGE_FIELDS.items()}
        for path in images.values():
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError(f"Expected PNG image: {path}")
                image.verify()
        assets.append(images)
    root, normalized, hashes = Path.cwd(), [], {}
    for row, images in zip(rows, assets):
        key = row.get("id") or row["key"]
        item = dict(row, id=key, key=key, record_id=key.split("_")[0],
                    num_edit_instruction=str(row["edit_operation"]))
        for folder, source in images.items():
            target = data_dir / folder / (key + ".png")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
            hashes[portable_path(target, root)] = digest(target)
            field = {"source_images": "org_image_path", "visual_instructions": "visual_instruction_image_path",
                     "target_images": "target_image_path"}[folder]
            item[field] = portable_path(target, root)
        item["source_image"] = item["org_image_path"]
        item["visual_instruction"] = item["visual_instruction_image_path"]
        item["target_image"] = item["target_image_path"]
        for field, inline, folder in [("org_tikz_code_path", "source_tikz", "source_tikz"),
                                      ("reference_tikz_code_path", "target_tikz", "target_tikz")]:
            target = data_dir / folder / (key + ".tex")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(row[inline], encoding="utf-8")
            item[field] = portable_path(target, root)
            hashes[item[field]] = digest(target)
        if key in extra:
            item["text_instruction"] = extra[key]
        item.setdefault("edited_image_paths", {})
        item.setdefault("edited_tikz_code_paths", {})
        normalized.append(item)
    metadata_files = [destination, data_dir/"image_metadata.jsonl", data_dir/"code_metadata.jsonl"]
    for path in metadata_files:
        write_jsonl(path, normalized)
        hashes[portable_path(path, root)] = digest(path)
    manifest = {"repo_id": repo_id, "revision": revision, "samples": len(rows), "full_dataset_samples": len(seen),
        "metadata_sha256": digest(metadata), "text_instructions_present": sum(bool(r.get("text_instruction")) for r in normalized),
        "files_sha256": hashes}
    (data_dir/"dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if supplement:
        install_supplement(supplement, data_dir / "reproduction")
    return manifest


def verify_supplement(source: Path) -> dict[str, Any]:
    """追加データの全ファイルを配置前に検証し、改変や外部参照を拒否する。"""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files_sha256"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
            raise ValueError(f"Unsafe supplement path: {name}")
        path = source.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Supplement file missing or hash mismatch: {name}")
    return manifest


def install_supplement(source: Path, destination: Path) -> dict[str, Any]:
    """HFの再現用追加データを全件ハッシュ検証してからローカルへ配置する。"""
    manifest = verify_supplement(source)
    for name in manifest["files_sha256"]:
        relative = PurePosixPath(name)
        path = source.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() != path.resolve():
            shutil.copy2(path, target)
    destination.mkdir(parents=True, exist_ok=True)
    if (source / "manifest.json").resolve() != (destination / "manifest.json").resolve():
        shutil.copy2(source / "manifest.json", destination / "manifest.json")
    return {"verified_files": len(manifest["files_sha256"]), "schema_version": manifest["schema_version"]}


def add_supplement(source: Path, data_dir: Path) -> dict[str, Any]:
    """公開前の追加データを既存の準備済みデータへ結合し、ハッシュも更新する。"""
    supplemental = verify_supplement(source)
    manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if supplemental.get("base_metadata_sha256") and supplemental["base_metadata_sha256"] != manifest["metadata_sha256"]:
        raise ValueError("Supplement targets a different benchmark snapshot")
    texts = {}
    for row in read_jsonl(source / "gold_text.jsonl"):
        key, text = row["id"], row["text_instruction"]
        if key in texts or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid supplemental text: {key}")
        texts[key] = text
    metadata = read_jsonl(data_dir / "metadata/edit_metadata.jsonl")
    for row in metadata:
        if row["key"] not in texts:
            raise ValueError(f"Supplement is missing benchmark sample: {row['key']}")
        row["text_instruction"] = texts[row["key"]]
    result = install_supplement(source, data_dir / "reproduction")
    for path in [data_dir / "metadata/edit_metadata.jsonl", data_dir / "image_metadata.jsonl", data_dir / "code_metadata.jsonl"]:
        write_jsonl(path, metadata)
        manifest["files_sha256"][portable_path(path, Path.cwd())] = digest(path)
    manifest["text_instructions_present"] = len(metadata)
    manifest["reproduction_manifest_sha256"] = digest(data_dir / "reproduction/manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**result, "gold_text_samples": len(metadata)}


def download(repo_id: str, revision: str, data_dir: Path) -> tuple[Path, str]:
    """ユーザー実行時にHFのコミットを固定し、JSONLとPNGだけをダウンロードする。"""
    from huggingface_hub import HfApi, snapshot_download

    commit = HfApi().repo_info(repo_id, repo_type="dataset", revision=revision).sha
    local = data_dir / "raw_hf"
    result = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=commit,
        local_dir=str(local), allow_patterns=["*.jsonl", "dataset/*.jsonl", "source_images/*.png",
        "target_images/*.png", "visual_instructions/*.png", "dataset/source_images/*.png",
        "dataset/target_images/*.png", "dataset/visual_instructions/*.png", "reproduction/**", "dataset/reproduction/**"])
    return Path(result), commit


def attach(metadata: Path, output: Path, model: str, input_type: str,
           tex_dir: Path | None, image_dir: Path | None) -> dict[str, Any]:
    """生成コードと画像を全件へ対応付け、画像化失敗を分母から落とさず明示する。"""
    if not tex_dir and not image_dir:
        raise ValueError("Specify --tex-dir and/or --image-dir")
    rows = read_jsonl(metadata)
    seen, statuses = set(), {"ready": 0, "compile_failed": 0, "missing_generation": 0}
    for row in rows:
        key = row["key"]
        if not KEY_PATTERN.fullmatch(key) or key in seen:
            raise ValueError(f"Invalid or duplicate key: {key}")
        seen.add(key)
        tex = tex_dir/(key+".tex") if tex_dir else None
        image = image_dir/(key+".png") if image_dir else None
        has_tex, has_image = bool(tex and tex.is_file()), bool(image and image.is_file())
        if has_image:
            from PIL import Image
            with Image.open(image) as img:
                img.verify()
        status = "ready" if has_image else "compile_failed" if has_tex else "missing_generation"
        statuses[status] += 1
        # 存在しない出力も経路を記録する。既存の画像要求生成側は未存在画像を送信しない。
        if image:
            row.setdefault("edited_image_paths", {}).setdefault(input_type, {})[model] = portable_path(image, Path.cwd())
        if tex:
            row.setdefault("edited_tikz_code_paths", {}).setdefault(input_type, {})[model] = portable_path(tex, Path.cwd())
            row["generated_tikz_code_path"] = portable_path(tex, Path.cwd())
        row.setdefault("generation_status", {}).setdefault(input_type, {})[model] = status
    write_jsonl(output, rows)
    return {"samples": len(rows), "model": model, "input_type": input_type, "statuses": statuses}


def verify(metadata: Path) -> dict[str, Any]:
    """準備済みメタデータのキー・コード・画像ファイルをローカルで確認する。"""
    rows, seen = read_jsonl(metadata), set()
    for row in rows:
        key = row["key"]
        if not KEY_PATTERN.fullmatch(key) or key in seen:
            raise ValueError(f"Invalid or duplicate key: {key}")
        seen.add(key)
        for field in ["org_image_path", "visual_instruction_image_path", "target_image_path",
                      "org_tikz_code_path", "reference_tikz_code_path"]:
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
    return {"samples": len(rows), "verified": True}


def main() -> None:
    """ダウンロード、既存スナップショット変換、出力対応付け、検証を振り分ける。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ["download", "prepare"]:
        sub = commands.add_parser(command)
        sub.add_argument("--repo-id", default=REPO_ID)
        sub.add_argument("--revision", default="main" if command == "download" else "local")
        sub.add_argument("--data-dir", type=Path, default=Path("data"))
        sub.add_argument("--limit", type=int)
        sub.add_argument("--force", action="store_true")
        sub.add_argument("--text-instructions", type=Path)
        if command == "prepare":
            sub.add_argument("--snapshot", type=Path, required=True)
    sub = commands.add_parser("attach")
    sub.add_argument("--metadata", type=Path, default=Path("data/metadata/edit_metadata.jsonl"))
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--model", required=True)
    sub.add_argument("--input-type", default="w_org_image_w_org_tikz")
    sub.add_argument("--tex-dir", type=Path)
    sub.add_argument("--image-dir", type=Path)
    sub = commands.add_parser("verify")
    sub.add_argument("--metadata", type=Path, default=Path("data/metadata/edit_metadata.jsonl"))
    sub = commands.add_parser("supplement")
    sub.add_argument("--source", type=Path, required=True)
    sub.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.command in ["download", "prepare"]:
        if args.command == "download":
            if (args.data_dir/"metadata/edit_metadata.jsonl").exists() and not args.force:
                parser.error("Prepared data already exists. Use --force to prepare it again.")
            snapshot, revision = download(args.repo_id, args.revision, args.data_dir)
        else:
            snapshot, revision = args.snapshot, args.revision
        result = prepare(snapshot, args.data_dir, repo_id=args.repo_id, revision=revision,
                         limit=args.limit, force=args.force, text_instructions=args.text_instructions)
        result = {k: v for k, v in result.items() if k != "files_sha256"}
    elif args.command == "attach":
        result = attach(args.metadata, args.output, args.model, args.input_type, args.tex_dir, args.image_dir)
    elif args.command == "supplement":
        result = add_supplement(args.source, args.data_dir)
    else:
        result = verify(args.metadata)
    print(json.dumps(result, ensure_ascii=False, indent=2))
