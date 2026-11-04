"""Typed visual observations, independent of any network client."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ExtractionError(ValueError):
    """A response cannot be interpreted as visual extraction."""


@dataclass(frozen=True)
class Entity:
    name: str
    type: str = "other"
    attributes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VisualObject:
    name: str
    count: int = 1
    attributes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Event:
    subject: str
    action: str
    object: str = ""


@dataclass(frozen=True)
class Relation:
    source: str
    predicate: str
    target: str


@dataclass(frozen=True)
class Extraction:
    description: str = ""
    entities: list[Entity] = field(default_factory=list)
    objects: list[VisualObject] = field(default_factory=list)
    ocr_text: str = ""
    events: list[Event] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    location_hint: str | None = None
    model: str = ""
    prompt_version: str = ""
    schema_version: str = ""


@runtime_checkable
class VisionExtractor(Protocol):
    def extract(
        self, *, image_data_uri: str, context_text: str,
        image_position: int, image_count: int,
    ) -> Extraction: ...


def _as_str(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    return None


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := _as_str(item)) is not None]


def _parse_entities(value: Any) -> list[Entity]:
    if not isinstance(value, (list, tuple)):
        return []
    entities: list[Entity] = []
    for item in value:
        if isinstance(item, Mapping) and (name := _as_str(item.get("name"))):
            entities.append(Entity(name, _as_str(item.get("type")) or "other",
                                   _as_str_list(item.get("attributes"))))
    return entities


def _parse_objects(value: Any) -> list[VisualObject]:
    if not isinstance(value, (list, tuple)):
        return []
    objects: list[VisualObject] = []
    for item in value:
        if not isinstance(item, Mapping) or not (name := _as_str(item.get("name"))):
            continue
        try:
            count = max(int(item.get("count", 1)), 1)
        except (TypeError, ValueError, OverflowError):
            count = 1
        objects.append(VisualObject(name, count, _as_str_list(item.get("attributes"))))
    return objects


def _parse_events(value: Any) -> list[Event]:
    if not isinstance(value, (list, tuple)):
        return []
    events: list[Event] = []
    for item in value:
        if isinstance(item, Mapping):
            subject, action = _as_str(item.get("subject")), _as_str(item.get("action"))
            if subject and action:
                events.append(Event(subject, action, _as_str(item.get("object")) or ""))
    return events


def _parse_relations(value: Any) -> list[Relation]:
    if not isinstance(value, (list, tuple)):
        return []
    relations: list[Relation] = []
    for item in value:
        if isinstance(item, Mapping):
            source = _as_str(item.get("source"))
            predicate = _as_str(item.get("predicate"))
            target = _as_str(item.get("target"))
            if source and predicate and target:
                relations.append(Relation(source, predicate, target))
    return relations


def has_visual_evidence(extraction: Extraction) -> bool:
    return any((extraction.description, extraction.ocr_text, extraction.entities,
                extraction.objects, extraction.events, extraction.relations,
                extraction.tags, extraction.location_hint))


def parse_extraction(
    payload: object, *, model: str, prompt_version: str, schema_version: str
) -> Extraction:
    """Discard malformed list items, but reject unusable top-level text fields."""
    if not isinstance(payload, Mapping):
        raise ExtractionError("VLM response must be a JSON object")
    description = payload.get("description")
    ocr_text = payload.get("ocr_text")
    if description is not None and not isinstance(description, str):
        raise ExtractionError("'description' must be a string")
    if ocr_text is not None and not isinstance(ocr_text, str):
        raise ExtractionError("'ocr_text' must be a string")
    return Extraction(
        description=(description or "").strip(),
        entities=_parse_entities(payload.get("entities")),
        objects=_parse_objects(payload.get("objects")),
        ocr_text=(ocr_text or "").strip(),
        events=_parse_events(payload.get("events")),
        relations=_parse_relations(payload.get("relations")),
        tags=_as_str_list(payload.get("tags")),
        location_hint=_as_str(payload.get("location_hint")),
        model=model, prompt_version=prompt_version, schema_version=schema_version,
    )
