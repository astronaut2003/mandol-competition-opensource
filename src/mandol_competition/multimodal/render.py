"""Deterministic memory and query text with identical image field labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Location:
    value: str
    source: str


@dataclass(frozen=True)
class RenderImage:
    media_id: str
    extraction: Any
    location: Location | None = None


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _listed(items: Sequence[str]) -> str:
    return "; ".join(value for item in items if (value := _one_line(item)))


def _named(item: Any, extras: Sequence[str]) -> str:
    name = _one_line(item.name)
    details = _listed(extras)
    return f"{name} ({details})" if details else name


def _image_block(image: RenderImage) -> list[str]:
    extraction = image.extraction
    lines = [f"[Image {_one_line(image.media_id)}]"]
    description = _one_line(getattr(extraction, "description", "") or "")
    if description:
        lines.append(f"Visual description: {description}")

    entities = getattr(extraction, "entities", None) or []
    if entities:
        values = [_named(entity, [getattr(entity, "type", ""),
                                  *(getattr(entity, "attributes", None) or [])]) for entity in entities]
        if rendered := _listed(values):
            lines.append(f"Entities: {rendered}")

    objects = getattr(extraction, "objects", None) or []
    if objects:
        values = []
        for obj in objects:
            count = getattr(obj, "count", 1)
            extras = [f"x{count}"] if count and count > 1 else []
            extras.extend(getattr(obj, "attributes", None) or [])
            values.append(_named(obj, extras))
        if rendered := _listed(values):
            lines.append(f"Objects: {rendered}")

    ocr = _one_line(getattr(extraction, "ocr_text", "") or "")
    if ocr:
        lines.append(f"OCR: {ocr}")

    events = getattr(extraction, "events", None) or []
    if events:
        values = [_one_line(" ".join(part for part in
                  (event.subject, event.action, event.object) if part)) for event in events]
        if rendered := _listed(values):
            lines.append(f"Events: {rendered}")

    relations = getattr(extraction, "relations", None) or []
    if relations:
        values = [f"{relation.source} {relation.predicate} {relation.target}" for relation in relations]
        if rendered := _listed(values):
            lines.append(f"Relations: {rendered}")

    if tags := _listed(getattr(extraction, "tags", None) or []):
        lines.append(f"Tags: {tags}")
    if (image.location is not None and image.location.source in {"exif", "text"}
            and (value := _one_line(image.location.value))):
        lines.append(f"Location: {value} (source: {_one_line(image.location.source)})")
    return lines


def render_text_content(
    *, dialogue: str, timestamp_text: str | None, images: Sequence[RenderImage]
) -> str:
    """Keep source dialogue verbatim; only structured image fields are normalized."""
    lines: list[str] = []
    if dialogue:
        lines.append(f"[Dialogue] {dialogue}")
    if timestamp_text and (timestamp := _one_line(timestamp_text)):
        lines.append(f"[Timestamp] {timestamp}")
    for image in images:
        lines.extend(_image_block(image))
    return "\n".join(lines)


def render_query_text(
    *, query_text: str, images: Sequence[RenderImage], options: Sequence[str] | None = None
) -> str:
    """Use the memory image renderer so lexical labels never diverge."""
    lines: list[str] = []
    if query_text:
        lines.append(f"[Question] {query_text}")
    for image in images:
        lines.extend(_image_block(image))
    if options:
        lines.append("[Options]")
        lines.extend(options)
    return "\n".join(lines)
