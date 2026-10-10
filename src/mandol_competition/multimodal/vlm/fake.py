"""Injected visual fixtures for deterministic tests; never calls a network."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .base import Extraction, ExtractionError, parse_extraction

_VISUAL_FIELDS = frozenset({
    "description", "entities", "objects", "ocr_text", "events", "relations",
    "tags", "location_hint",
})


def load_fixture(path: Path) -> dict[str, dict]:
    """Accept only image IDs mapped to visual fields, never evaluation labels."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture file must be a JSON object keyed by media_id")
    return _validate_mapping(payload)


def _validate_mapping(mapping: Mapping[str, dict]) -> dict[str, dict]:
    checked = dict(mapping)
    for media_id, payload in checked.items():
        if not isinstance(media_id, str) or not isinstance(payload, dict):
            raise ValueError("fixture entries must map media IDs to JSON objects")
        if forbidden := set(payload) - _VISUAL_FIELDS:
            raise ValueError(f"fixture entry {media_id!r} contains forbidden metadata: {sorted(forbidden)}")
    return checked


def _media_id(image_data_uri: str) -> str:
    header, separator, encoded = image_data_uri.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ExtractionError("image must be a Base64 image Data URI")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ExtractionError("invalid image Base64") from exc
    return "img_" + hashlib.sha256(raw).hexdigest()


class FakeVisionExtractor:
    """Return the caller's visual description for each content-addressed image."""

    def __init__(
        self, mapping: Mapping[str, dict], *, model: str, prompt_version: str,
        schema_version: str, strict: bool = False,
    ) -> None:
        self._mapping = _validate_mapping(mapping)
        self._model = model
        self._prompt_version = prompt_version
        self._schema_version = schema_version
        self._strict = strict
        self.calls = 0

    @classmethod
    def from_json_file(
        cls, path: Path, *, model: str, prompt_version: str,
        schema_version: str, strict: bool = False,
    ) -> FakeVisionExtractor:
        return cls(load_fixture(path), model=model, prompt_version=prompt_version,
                   schema_version=schema_version, strict=strict)

    @property
    def model_name(self) -> str:
        return self._model

    def extract(
        self, *, image_data_uri: str, context_text: str,
        image_position: int, image_count: int,
    ) -> Extraction:
        self.calls += 1
        media_id = _media_id(image_data_uri)
        payload = self._mapping.get(media_id)
        if payload is None:
            if self._strict:
                raise ExtractionError(f"no fixture entry for {media_id}")
            payload = {}
        return parse_extraction(payload, model=self._model,
                                prompt_version=self._prompt_version,
                                schema_version=self._schema_version)
