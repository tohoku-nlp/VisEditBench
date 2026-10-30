from __future__ import annotations

import base64
import hashlib
import json
import io
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .config import ConfigItem, resolve_metadata_fields

EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION_W_ORG_IMAGE = r"""You are an expert at converting diagram images into LaTeX/TikZ.

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

EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION_W_ORG_IMAGE_W_ORG_TIKZ_CODE = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original TikZ code of a diagram,
2. the original diagram image, and
3. an annotated version of that diagram image containing visual edit instructions.

Use all inputs together:
- Use the original TikZ code and the original image to understand the structure and content of the diagram.
- Use the annotated image to identify the requested edits.
- Use both the original image to capture the underlying content and the annotated image to understand the intended modifications.

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

EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION_W_ORG_TIKZ_CODE = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
1. the original TikZ code of a diagram,
2. an annotated version of that diagram image containing visual edit instructions.

Use all inputs together:
- Use the original TikZ code to understand the structure and content of the diagram.
- Use the annotated image to identify the requested edits.

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

EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION = r"""You are an expert at editing LaTeX/TikZ diagrams.

You are given:
- an annotated version of the diagram image containing visual edit instructions.

Use input:
- Use the annotated image to understand the structure and content of the diagram, and to identify the requested edits.

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

EDITING_SYSTEM_PROMPT_TEXT_INSTRUCTION_W_ORG_IMAGE_W_ORG_TIKZ_CODE = r"""You are an expert at editing LaTeX/TikZ diagrams.

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

USER_PROMPT = r"""Original TikZ code:
{original_tikz_code}
"""

USER_PROMPT_WITH_TEXT_INSTRUCTION = r"""Original TikZ code:
{original_tikz_code}

Text instruction:
{text_instruction}
"""

SYSTEM_PROMPTS = {
    "visual_instruction_w_org_image": EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION_W_ORG_IMAGE,
    "visual_instruction_w_org_image_w_org_tikz": EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION_W_ORG_IMAGE_W_ORG_TIKZ_CODE,
    "visual_instruction_w_org_tikz": EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION_W_ORG_TIKZ_CODE,
    "visual_instruction": EDITING_SYSTEM_PROMPT_VISUAL_INSTRUCTION,
    "text_instruction_w_org_image_w_org_tikz": EDITING_SYSTEM_PROMPT_TEXT_INSTRUCTION_W_ORG_IMAGE_W_ORG_TIKZ_CODE,
}
CLAUDE_MAX_TOKENS = 8192
OPENROUTER_MAX_TOKENS = 128000


def encode_image_as_data_url(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    else:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def encode_image_as_b64data(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _resolve_image_url_from_metadata_or_map(
    metadata_dict: dict[str, Any],
    local_path: Path,
    direct_url_key: str,
    url_map: dict[str, str],
) -> str | None:
    direct_url = metadata_dict.get(direct_url_key)
    if isinstance(direct_url, str) and direct_url.strip():
        return direct_url
    return url_map.get(local_path.name)


def normalize_base64_data(data: str) -> str:
    # Claude/Gemini expects raw base64 without data URL prefix.
    if data.startswith("data:"):
        comma_idx = data.find(",")
        if comma_idx != -1:
            return data[comma_idx + 1 :]
    return data


def encode_image_base64_resized(image_path: Path, max_dim: int = 7900) -> str:
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))

    with Image.open(image_path) as img:
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > max_dim:
            scale = max_dim / long_edge
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _openai_image_block(image_url: str, detail: str) -> dict:
    block: dict = {"type": "input_image", "image_url": image_url}
    if detail != "auto":
        block["detail"] = detail
    return block


def build_request_openai_visual_instruction(
    custom_id: str,
    model: str,
    system_instruction: str,
    original_image_url: str,
    visual_instruction_image_url: str,
    image_detail: str = "auto",
) -> dict:
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    _openai_image_block(original_image_url, image_detail),
                    _openai_image_block(visual_instruction_image_url, image_detail),
                ],
            },
        ],
    }

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_request_openai_visual_instruction_with_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
    visual_instruction_image_url: str,
    image_detail: str = "auto",
) -> dict:
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    _openai_image_block(original_image_url, image_detail),
                    _openai_image_block(visual_instruction_image_url, image_detail),
                ],
            },
        ],
    }

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_request_openai_text_instruction_with_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
    image_detail: str = "auto",
) -> dict:
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    _openai_image_block(original_image_url, image_detail),
                ],
            },
        ],
    }

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_request_openai_visual_instruction_w_org_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    visual_instruction_image_url: str,
    image_detail: str = "auto",
) -> dict:
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    _openai_image_block(visual_instruction_image_url, image_detail),
                ],
            },
        ],
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_request_openai_visual_instruction_only(
    custom_id: str,
    model: str,
    system_instruction: str,
    visual_instruction_image_url: str,
    image_detail: str = "auto",
) -> dict:
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    _openai_image_block(visual_instruction_image_url, image_detail),
                ],
            },
        ],
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def build_request_gemini_visual_instruction(
    custom_id: str,
    system_instruction: str,
    original_image_data: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": original_image_data,
                        }
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": visual_instruction_image_data,
                        }
                    },
                ]
            }
        ],
    }

    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_by_url(
    custom_id: str,
    system_instruction: str,
    original_image_url: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": original_image_url,
                        }
                    },
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": visual_instruction_image_url,
                        }
                    },
                ]
            }
        ],
    }
    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_with_tikz(
    custom_id: str,
    system_instruction: str,
    user_prompt: str,
    original_image_data: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": original_image_data,
                        }
                    },
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": visual_instruction_image_data,
                        }
                    },
                ]
            }
        ],
    }

    return {"key": custom_id, "request": request}


def build_request_gemini_text_instruction_with_tikz(
    custom_id: str,
    system_instruction: str,
    user_prompt: str,
    original_image_data: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": original_image_data,
                        }
                    },
                ]
            }
        ],
    }

    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_with_tikz_by_url(
    custom_id: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": original_image_url,
                        }
                    },
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": visual_instruction_image_url,
                        }
                    },
                ]
            }
        ],
    }
    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_w_org_tikz(
    custom_id: str,
    system_instruction: str,
    user_prompt: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": visual_instruction_image_data,
                        }
                    },
                ]
            }
        ],
    }
    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_w_org_tikz_by_url(
    custom_id: str,
    system_instruction: str,
    user_prompt: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {"text": user_prompt},
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": visual_instruction_image_url,
                        }
                    },
                ]
            }
        ],
    }
    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_only(
    custom_id: str,
    system_instruction: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": visual_instruction_image_data,
                        }
                    },
                ]
            }
        ],
    }
    return {"key": custom_id, "request": request}


def build_request_gemini_visual_instruction_only_by_url(
    custom_id: str,
    system_instruction: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [
            {
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/png",
                            "fileUri": visual_instruction_image_url,
                        }
                    },
                ]
            }
        ],
    }
    return {"key": custom_id, "request": request}


def build_request_claude_visual_instruction(
    custom_id: str,
    model: str,
    system_instruction: str,
    original_image_data: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": original_image_data,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": visual_instruction_image_data,
                            },
                        },
                    ],
                }
            ],
        },
    }

    return request


def build_request_claude_visual_instruction_by_url(
    custom_id: str,
    model: str,
    system_instruction: str,
    original_image_url: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": original_image_url},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": visual_instruction_image_url,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_visual_instruction_with_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_data: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": original_image_data,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": visual_instruction_image_data,
                            },
                        },
                    ],
                }
            ],
        },
    }

    return request


def build_request_claude_text_instruction_with_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_data: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": original_image_data,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_text_instruction_with_tikz_by_url(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": original_image_url},
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_visual_instruction_with_tikz_by_url(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {"type": "url", "url": original_image_url},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": visual_instruction_image_url,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_visual_instruction_w_org_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": visual_instruction_image_data,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_visual_instruction_w_org_tikz_by_url(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": visual_instruction_image_url,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_visual_instruction_only(
    custom_id: str,
    model: str,
    system_instruction: str,
    visual_instruction_image_data: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": visual_instruction_image_data,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_claude_visual_instruction_only_by_url(
    custom_id: str,
    model: str,
    system_instruction: str,
    visual_instruction_image_url: str,
) -> dict:
    request = {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "system": system_instruction,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": visual_instruction_image_url,
                            },
                        },
                    ],
                }
            ],
        },
    }
    return request


def build_request_openrouter_visual_instruction(
    custom_id: str,
    model: str,
    system_instruction: str,
    original_image_url: str,
    visual_instruction_image_url: str,
    thinking: bool = False,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": original_image_url}},
                    {
                        "type": "image_url",
                        "image_url": {"url": visual_instruction_image_url},
                    },
                ],
            },
        ],
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    extra_body: dict = {"enable_thinking": True} if thinking else {}
    return {"custom_id": custom_id, "body": body, "extra_body": extra_body}


def build_request_openrouter_visual_instruction_with_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
    visual_instruction_image_url: str,
    thinking: bool = False,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": original_image_url}},
                    {
                        "type": "image_url",
                        "image_url": {"url": visual_instruction_image_url},
                    },
                ],
            },
        ],
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    extra_body: dict = {"enable_thinking": True} if thinking else {}
    return {"custom_id": custom_id, "body": body, "extra_body": extra_body}


def build_request_openrouter_text_instruction_with_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    original_image_url: str,
    thinking: bool = False,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": original_image_url}},
                ],
            },
        ],
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    extra_body: dict = {"enable_thinking": True} if thinking else {}
    return {"custom_id": custom_id, "body": body, "extra_body": extra_body}


def build_request_openrouter_visual_instruction_w_org_tikz(
    custom_id: str,
    model: str,
    system_instruction: str,
    user_prompt: str,
    visual_instruction_image_url: str,
    thinking: bool = False,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": visual_instruction_image_url},
                    },
                ],
            },
        ],
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    extra_body: dict = {"enable_thinking": True} if thinking else {}
    return {"custom_id": custom_id, "body": body, "extra_body": extra_body}


def build_request_openrouter_visual_instruction_only(
    custom_id: str,
    model: str,
    system_instruction: str,
    visual_instruction_image_url: str,
    thinking: bool = False,
) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": visual_instruction_image_url},
                    },
                ],
            },
        ],
        "max_tokens": OPENROUTER_MAX_TOKENS,
    }
    extra_body: dict = {"enable_thinking": True} if thinking else {}
    return {"custom_id": custom_id, "body": body, "extra_body": extra_body}


def generate_jsonl_for_experiment(
    exp: ConfigItem,
    metadata_fields: dict[str, str],
    limit: int | None,
    preview_first: bool,
) -> tuple[int, dict[str, Any] | None]:
    if exp.system_instruction_type not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unsupported system_instruction_type: {exp.system_instruction_type}"
        )

    system_instruction = SYSTEM_PROMPTS[exp.system_instruction_type]
    mode = "a" if exp.append else "w"

    # ディレクトリを作成
    save_path = Path(exp.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # GCS URL マップを読み込む（指定がある場合のみ）
    org_url_map: dict[str, str] = {}
    vi_url_map: dict[str, str] = {}
    if exp.org_url_json and exp.org_url_json.exists():
        org_url_map = json.loads(exp.org_url_json.read_text(encoding="utf-8"))
    if exp.vi_url_json and exp.vi_url_json.exists():
        vi_url_map = json.loads(exp.vi_url_json.read_text(encoding="utf-8"))

    count = 0
    preview_obj: dict[str, Any] | None = None
    with (
        (io.StringIO() if preview_first else open(save_path, mode, encoding="utf-8")) as f_out,
        open(exp.metadata_path, "r", encoding="utf-8") as f_metadata,
    ):
        for line in f_metadata:
            if not line.strip():
                continue
            metadata_dict = json.loads(line)
            resolved = resolve_metadata_fields(metadata_dict, metadata_fields)

            key = resolved["key"]
            stype = exp.system_instruction_type
            custom_id = f"{key}_{stype}"
            if exp.provider == "anthropic" and len(custom_id) > 64:
                digest = hashlib.sha1(custom_id.encode("utf-8")).hexdigest()[:16]
                prefix_len = 64 - 1 - len(digest)
                custom_id = f"{custom_id[:prefix_len]}_{digest}"
            vi_image_path = (
                Path(resolved["visual_instruction_image_path"])
                if stype
                in (
                    "visual_instruction_w_org_image",
                    "visual_instruction_w_org_image_w_org_tikz",
                    "visual_instruction_w_org_tikz",
                    "visual_instruction",
                )
                else None
            )
            org_image_path = (
                Path(resolved["org_image_path"])
                if stype
                in (
                    "visual_instruction_w_org_image",
                    "visual_instruction_w_org_image_w_org_tikz",
                    "text_instruction_w_org_image_w_org_tikz",
                )
                else None
            )
            user_prompt = None
            if stype in (
                "visual_instruction_w_org_image_w_org_tikz",
                "visual_instruction_w_org_tikz",
            ):
                org_tikz_code_path = resolved["org_tikz_code_path"]
                with open(org_tikz_code_path, "r", encoding="utf-8") as f_tikz:
                    original_tikz_code = f_tikz.read()
                user_prompt = USER_PROMPT.format(original_tikz_code=original_tikz_code)
            elif stype == "text_instruction_w_org_image_w_org_tikz":
                org_tikz_code_path = resolved["org_tikz_code_path"]
                with open(org_tikz_code_path, "r", encoding="utf-8") as f_tikz:
                    original_tikz_code = f_tikz.read()
                text_instruction = resolved["text_instruction"]
                if (
                    not isinstance(text_instruction, str)
                    or not text_instruction.strip()
                ):
                    raise ValueError(f"text_instruction is missing for key={key}")
                user_prompt = USER_PROMPT_WITH_TEXT_INSTRUCTION.format(
                    text_instruction=text_instruction,
                    original_tikz_code=original_tikz_code,
                )

            if exp.provider == "openai":
                # GCS URL があれば使用、なければ base64 データ URL にフォールバック
                visual_instruction_image_url = None
                if vi_image_path is not None:
                    visual_instruction_image_url = (
                        _resolve_image_url_from_metadata_or_map(
                            metadata_dict,
                            vi_image_path,
                            "visual_instruction_image_url",
                            vi_url_map,
                        )
                        or encode_image_as_data_url(vi_image_path)
                    )
                original_image_url = None
                if org_image_path is not None:
                    original_image_url = _resolve_image_url_from_metadata_or_map(
                        metadata_dict,
                        org_image_path,
                        "org_image_url",
                        org_url_map,
                    ) or encode_image_as_data_url(org_image_path)

                if stype == "visual_instruction_w_org_image":
                    json_line = build_request_openai_visual_instruction(
                        custom_id,
                        exp.model,
                        system_instruction,
                        original_image_url,
                        visual_instruction_image_url,
                        image_detail=exp.image_detail,
                    )
                elif stype == "visual_instruction_w_org_image_w_org_tikz":
                    json_line = build_request_openai_visual_instruction_with_tikz(
                        custom_id,
                        exp.model,
                        system_instruction,
                        user_prompt,
                        original_image_url,
                        visual_instruction_image_url,
                        image_detail=exp.image_detail,
                    )
                elif stype == "visual_instruction_w_org_tikz":
                    json_line = build_request_openai_visual_instruction_w_org_tikz(
                        custom_id,
                        exp.model,
                        system_instruction,
                        user_prompt,
                        visual_instruction_image_url,
                        image_detail=exp.image_detail,
                    )
                elif stype == "text_instruction_w_org_image_w_org_tikz":
                    json_line = build_request_openai_text_instruction_with_tikz(
                        custom_id,
                        exp.model,
                        system_instruction,
                        user_prompt,
                        original_image_url,
                        image_detail=exp.image_detail,
                    )
                else:
                    json_line = build_request_openai_visual_instruction_only(
                        custom_id,
                        exp.model,
                        system_instruction,
                        visual_instruction_image_url,
                        image_detail=exp.image_detail,
                    )

            elif exp.provider == "google":
                # Gemini は常にローカル画像を base64 で埋め込む（URL は使わない）
                visual_instruction_image_data = None
                if vi_image_path is not None:
                    visual_instruction_image_data = normalize_base64_data(
                        encode_image_as_b64data(vi_image_path)
                    )
                original_image_data = None
                if org_image_path is not None:
                    original_image_data = normalize_base64_data(
                        encode_image_as_b64data(org_image_path)
                    )

                if stype == "visual_instruction_w_org_image":
                    json_line = build_request_gemini_visual_instruction(
                        custom_id,
                        system_instruction,
                        original_image_data,
                        visual_instruction_image_data,
                    )
                elif stype == "visual_instruction_w_org_image_w_org_tikz":
                    json_line = build_request_gemini_visual_instruction_with_tikz(
                        custom_id,
                        system_instruction,
                        user_prompt,
                        original_image_data,
                        visual_instruction_image_data,
                    )
                elif stype == "visual_instruction_w_org_tikz":
                    json_line = build_request_gemini_visual_instruction_w_org_tikz(
                        custom_id,
                        system_instruction,
                        user_prompt,
                        visual_instruction_image_data,
                    )
                elif stype == "text_instruction_w_org_image_w_org_tikz":
                    json_line = build_request_gemini_text_instruction_with_tikz(
                        custom_id,
                        system_instruction,
                        user_prompt,
                        original_image_data,
                    )
                else:
                    json_line = build_request_gemini_visual_instruction_only(
                        custom_id,
                        system_instruction,
                        visual_instruction_image_data,
                    )

            elif exp.provider == "anthropic":
                visual_instruction_image_url = None
                if vi_image_path is not None:
                    visual_instruction_image_url = (
                        _resolve_image_url_from_metadata_or_map(
                            metadata_dict,
                            vi_image_path,
                            "visual_instruction_image_url",
                            vi_url_map,
                        )
                    )
                original_image_url = None
                if org_image_path is not None:
                    original_image_url = _resolve_image_url_from_metadata_or_map(
                        metadata_dict,
                        org_image_path,
                        "org_image_url",
                        org_url_map,
                    )
                visual_instruction_image_data = None
                original_image_data = None
                if vi_image_path is not None and visual_instruction_image_url is None:
                    visual_instruction_image_data = normalize_base64_data(
                        encode_image_as_b64data(vi_image_path)
                    )
                if org_image_path is not None and original_image_url is None:
                    original_image_data = normalize_base64_data(
                        encode_image_as_b64data(org_image_path)
                    )
                # text_instruction_w_org_image_w_org_tikz は常に org image を base64 で渡す
                if (
                    stype == "text_instruction_w_org_image_w_org_tikz"
                    and org_image_path is not None
                    and original_image_data is None
                ):
                    original_image_data = normalize_base64_data(
                        encode_image_as_b64data(org_image_path)
                    )

                if stype == "visual_instruction_w_org_image":
                    if original_image_url and visual_instruction_image_url:
                        json_line = build_request_claude_visual_instruction_by_url(
                            custom_id,
                            exp.model,
                            system_instruction,
                            original_image_url,
                            visual_instruction_image_url,
                        )
                    else:
                        json_line = build_request_claude_visual_instruction(
                            custom_id,
                            exp.model,
                            system_instruction,
                            original_image_data,
                            visual_instruction_image_data,
                        )
                elif stype == "visual_instruction_w_org_image_w_org_tikz":
                    if original_image_url and visual_instruction_image_url:
                        json_line = (
                            build_request_claude_visual_instruction_with_tikz_by_url(
                                custom_id,
                                exp.model,
                                system_instruction,
                                user_prompt,
                                original_image_url,
                                visual_instruction_image_url,
                            )
                        )
                    else:
                        json_line = build_request_claude_visual_instruction_with_tikz(
                            custom_id,
                            exp.model,
                            system_instruction,
                            user_prompt,
                            original_image_data,
                            visual_instruction_image_data,
                        )
                elif stype == "visual_instruction_w_org_tikz":
                    if visual_instruction_image_url:
                        json_line = (
                            build_request_claude_visual_instruction_w_org_tikz_by_url(
                                custom_id,
                                exp.model,
                                system_instruction,
                                user_prompt,
                                visual_instruction_image_url,
                            )
                        )
                    else:
                        json_line = build_request_claude_visual_instruction_w_org_tikz(
                            custom_id,
                            exp.model,
                            system_instruction,
                            user_prompt,
                            visual_instruction_image_data,
                        )
                elif stype == "text_instruction_w_org_image_w_org_tikz":
                    if original_image_url:
                        json_line = (
                            build_request_claude_text_instruction_with_tikz_by_url(
                                custom_id,
                                exp.model,
                                system_instruction,
                                user_prompt,
                                original_image_url,
                            )
                        )
                    else:
                        if original_image_data is None:
                            raise ValueError(
                                f"Failed to encode org image for key={key}"
                            )
                        json_line = build_request_claude_text_instruction_with_tikz(
                            custom_id,
                            exp.model,
                            system_instruction,
                            user_prompt,
                            original_image_data,
                        )
                else:
                    if visual_instruction_image_url:
                        json_line = build_request_claude_visual_instruction_only_by_url(
                            custom_id,
                            exp.model,
                            system_instruction,
                            visual_instruction_image_url,
                        )
                    else:
                        json_line = build_request_claude_visual_instruction_only(
                            custom_id,
                            exp.model,
                            system_instruction,
                            visual_instruction_image_data,
                        )
            elif exp.provider == "openrouter":
                visual_instruction_image_url = None
                if vi_image_path is not None:
                    visual_instruction_image_url = (
                        _resolve_image_url_from_metadata_or_map(
                            metadata_dict,
                            vi_image_path,
                            "visual_instruction_image_url",
                            vi_url_map,
                        )
                        or encode_image_as_data_url(vi_image_path)
                    )
                original_image_url = (
                    (
                        _resolve_image_url_from_metadata_or_map(
                            metadata_dict,
                            org_image_path,
                            "org_image_url",
                            org_url_map,
                        )
                        or encode_image_as_data_url(org_image_path)
                    )
                    if org_image_path is not None
                    else None
                )

                if stype == "visual_instruction_w_org_image":
                    json_line = build_request_openrouter_visual_instruction(
                        custom_id,
                        exp.model,
                        system_instruction,
                        original_image_url,
                        visual_instruction_image_url,
                        thinking=exp.thinking,
                    )
                elif stype == "visual_instruction_w_org_image_w_org_tikz":
                    json_line = build_request_openrouter_visual_instruction_with_tikz(
                        custom_id,
                        exp.model,
                        system_instruction,
                        user_prompt,
                        original_image_url,
                        visual_instruction_image_url,
                        thinking=exp.thinking,
                    )
                elif stype == "visual_instruction_w_org_tikz":
                    json_line = build_request_openrouter_visual_instruction_w_org_tikz(
                        custom_id,
                        exp.model,
                        system_instruction,
                        user_prompt,
                        visual_instruction_image_url,
                        thinking=exp.thinking,
                    )
                elif stype == "text_instruction_w_org_image_w_org_tikz":
                    json_line = build_request_openrouter_text_instruction_with_tikz(
                        custom_id,
                        exp.model,
                        system_instruction,
                        user_prompt,
                        original_image_url,
                        thinking=exp.thinking,
                    )
                else:
                    json_line = build_request_openrouter_visual_instruction_only(
                        custom_id,
                        exp.model,
                        system_instruction,
                        visual_instruction_image_url,
                        thinking=exp.thinking,
                    )

            else:
                raise ValueError(f"Unsupported provider: {exp.provider}")

            if preview_first:
                preview_obj = json_line
                return 0, preview_obj

            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break

    return count, preview_obj
