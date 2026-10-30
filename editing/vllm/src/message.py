import base64
import copy
import mimetypes
from urllib.parse import urlparse


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_media_type(image_path: str) -> str:
    mime, _ = mimetypes.guess_type(image_path)
    return mime or "image/png"


def build_vision_messages(data: dict) -> list[dict]:
    messages = copy.deepcopy(data["messages"])
    images = data["image_paths"]
    image_idx = 0

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url" and "image_url" not in part:
                if image_idx >= len(images):
                    raise ValueError(
                        f"Not enough images: expected more than {image_idx}, "
                        f"got {len(images)}"
                    )
                path = str(images[image_idx])
                parsed = urlparse(path)
                if parsed.scheme in {"http", "https", "gs"}:
                    raise ValueError(
                        f"URL-based image input is not supported. Use local paths: {path}"
                    )
                media_type = detect_media_type(path)
                image_b64 = encode_image(path)
                part["image_url"] = {
                    "url": f"data:{media_type};base64,{image_b64}",
                }
                image_idx += 1

    if image_idx != len(images):
        raise ValueError(
            f"Image count mismatch: {len(images)} images provided, "
            f"but {image_idx} placeholders found"
        )

    return messages


def build_text_messages(data: dict) -> list[dict]:
    return data["messages"]
