"""Build a searchable Mandol L0 unit from one ordered AML message."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from mandol_substrate.representation import MemoryUnit

from .render import RenderImage, render_text_content
from .representation import BASELINE, memory_text


def build_unit_uid(user_id: str, request_id: str, message_index: int) -> str:
    material = f"{user_id}\x00{request_id}\x00{message_index}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_memory_unit(
    *,
    user_id: str,
    request_id: str,
    session_id: str,
    message_index: int,
    role: str,
    timestamp_text: str | None,
    dialogue: str,
    content_parts: list[dict[str, Any]],
    images: list[RenderImage],
    extractor_model: str,
    prompt_version: str,
    schema_version: str,
    representation_policy: str = BASELINE,
    surrounding_text: str = "",
) -> MemoryUnit:
    image_info = []
    for image in images:
        extraction = asdict(image.extraction)
        extraction["media_id"] = image.media_id
        extraction["location"] = asdict(image.location) if image.location else None
        image_info.append(extraction)
    raw_data = {
        "text_content": memory_text(
            policy_id=representation_policy, dialogue=dialogue, timestamp_text=timestamp_text,
            images=images, surrounding_text=surrounding_text,
        ),
        "original_content": dialogue,
        "content_parts": [dict(part) for part in content_parts],
        "multimodal_info": {"images": image_info},
    }
    modality = "text_image" if dialogue.strip() and images else "image" if images else "text"
    return MemoryUnit(
        uid=build_unit_uid(user_id, request_id, message_index),
        raw_data=raw_data,
        metadata={
            "request_id": request_id,
            "message_index": message_index,
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "timestamp": timestamp_text,
            "modality": modality,
            "extractor_model": extractor_model,
            "extractor_prompt_version": prompt_version,
            "schema_version": schema_version,
        },
    )
