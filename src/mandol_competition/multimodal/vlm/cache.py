"""Per-user, media-sharded disk cache for structured visual observations."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .base import Entity, Event, Extraction, ExtractionError, Relation, VisualObject, VisionExtractor


def cache_key(
    media_id: str, context_hash: str, model: str, prompt_version: str, schema_version: str
) -> str:
    material = "\x00".join((media_id, context_hash, model, prompt_version, schema_version))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def context_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _media_id(image_data_uri: str) -> str:
    header, separator, encoded = image_data_uri.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ExtractionError("image must be a Base64 image Data URI")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ExtractionError("invalid image Base64") from exc
    return "img_" + hashlib.sha256(raw).hexdigest()


def _from_payload(payload: dict[str, Any]) -> Extraction:
    required = set(Extraction.__dataclass_fields__)
    if not required <= payload.keys():
        raise ValueError("incomplete cached extraction")
    return Extraction(
        description=payload["description"],
        entities=[Entity(**item) for item in payload["entities"]],
        objects=[VisualObject(**item) for item in payload["objects"]],
        ocr_text=payload["ocr_text"],
        events=[Event(**item) for item in payload["events"]],
        relations=[Relation(**item) for item in payload["relations"]],
        tags=list(payload["tags"]),
        location_hint=payload["location_hint"],
        model=payload["model"],
        prompt_version=payload["prompt_version"],
        schema_version=payload["schema_version"],
    )


@dataclass(frozen=True)
class CachedResult:
    extraction: Extraction
    usage: dict | None
    latency_ms: float | None


class ExtractionCache:
    """The caller supplies a distinct cache root for each user."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _shard(self, media_id: str) -> Path:
        digest = media_id.removeprefix("img_")
        if not re.fullmatch(r"[0-9a-f]+", digest):
            digest = hashlib.sha256(media_id.encode("utf-8")).hexdigest()
        return self.root / digest[:2] / digest[2:4]

    def _path_for(self, media_id: str, key: str) -> Path:
        filename = key if re.fullmatch(r"[\w-]+", key, flags=re.ASCII) else context_digest(key)
        return self._shard(media_id) / f"{filename}.json"

    def _read_record(self, media_id: str, key: str) -> dict | None:
        try:
            record = json.loads(self._path_for(media_id, key).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict) or record.get("media_id") != media_id:
            return None
        return record

    def _write_record(self, media_id: str, key: str, record: dict) -> None:
        path = self._path_for(media_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as output:
                temporary = output.name
                json.dump(record, output, ensure_ascii=False, sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def get_result(self, media_id: str, key: str) -> CachedResult | None:
        record = self._read_record(media_id, key)
        if record is None or not isinstance(record.get("extraction"), dict):
            return None
        try:
            extraction = _from_payload(record["extraction"])
        except (TypeError, ValueError, KeyError):
            return None
        return CachedResult(extraction, record.get("usage"), record.get("latency_ms"))

    def get(self, media_id: str, key: str) -> Extraction | None:
        result = self.get_result(media_id, key)
        return result.extraction if result else None

    def put(
        self, media_id: str, key: str, extraction: Extraction,
        *, usage: dict | None = None, latency_ms: float | None = None,
    ) -> None:
        self._write_record(media_id, key, {
            "media_id": media_id, "extraction": asdict(extraction),
            "usage": usage, "latency_ms": latency_ms,
        })

    def delete_for_media(self, media_id: str) -> int:
        shard = self._shard(media_id)
        if not shard.is_dir():
            return 0
        removed = 0
        for path in shard.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(record, dict) and record.get("media_id") == media_id:
                    path.unlink()
                    removed += 1
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
        return removed

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json")) if self.root.is_dir() else 0


class CachingExtractor:
    """Cache successful calls only, including every input used by the prompt."""

    def __init__(
        self, inner: VisionExtractor, cache: ExtractionCache, *, model: str,
        prompt_version: str, schema_version: str,
        provider_identity: str = "",
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model
        self._prompt_version = prompt_version
        self._schema_version = schema_version
        self._provider_identity = provider_identity
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self._model

    def cache_size(self) -> int:
        return len(self._cache)

    def _key(self, media_id: str, context_text: str, image_position: int, image_count: int) -> str:
        context = "\x00".join((context_digest(context_text), str(image_position), str(image_count)))
        if self._provider_identity:
            context += "\x00" + self._provider_identity
        return cache_key(media_id, context, self._model, self._prompt_version,
                         self._schema_version)

    def record_observation(
        self, image_data_uri: str, *, usage: dict | None, latency_ms: float | None,
        context_text: str = "", image_position: int = 0, image_count: int = 1,
    ) -> None:
        """Update only the matching context's cache entry, never a nearby image."""
        media_id = _media_id(image_data_uri)
        key = self._key(media_id, context_text, image_position, image_count)
        record = self._cache._read_record(media_id, key)
        if record is not None:
            record["usage"] = usage
            record["latency_ms"] = latency_ms
            self._cache._write_record(media_id, key, record)

    def extract(
        self, *, image_data_uri: str, context_text: str,
        image_position: int, image_count: int,
    ) -> Extraction:
        media_id = _media_id(image_data_uri)
        key = self._key(media_id, context_text, image_position, image_count)
        cached = self._cache.get(media_id, key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        extraction = self._inner.extract(
            image_data_uri=image_data_uri, context_text=context_text,
            image_position=image_position, image_count=image_count,
        )
        observations = self._inner.take_observations() if hasattr(self._inner, "take_observations") else []
        usage, latency = (observations[-1][1], observations[-1][2]) if observations else (None, None)
        self._cache.put(media_id, key, extraction, usage=usage, latency_ms=latency)
        return extraction
