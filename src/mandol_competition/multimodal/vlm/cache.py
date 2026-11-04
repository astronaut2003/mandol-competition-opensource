"""Per-user, media-sharded disk cache for structured visual observations."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .base import Entity, Event, Extraction, ExtractionError, Relation, VisualObject, VisionExtractor
from .output_guard import (MODEL_OUTPUT_LIMIT, validation_receipt_valid, extraction_fingerprint,
                           verified_ocr_coverage, INCOMPLETE_OCR_POLICY)


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
    output_validation: dict | None = None


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
        path = self._path_for(media_id, key)
        try:
            original_bytes = path.read_bytes()
        except OSError:
            original_bytes = None
        if original_bytes is not None:
            from ..ocr_cache_retirement import cache_record_retired
            if cache_record_retired(path, media_id, key, original_bytes):
                return None
        try:
            record = json.loads(original_bytes) if original_bytes is not None else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict) or record.get("media_id") != media_id:
            return None
        if record is None or not isinstance(record.get("extraction"), dict):
            return None
        receipt = record.get('output_validation')
        if isinstance(receipt, dict) and receipt.get('policy') == INCOMPLETE_OCR_POLICY:
            return None
        if receipt is not None and not validation_receipt_valid(receipt):
            return None
        if isinstance(receipt, dict) and receipt.get('media_id', media_id) != media_id:
            return None
        usage = record.get('usage')
        completion = usage.get('completion_tokens') if isinstance(usage, dict) else None
        if type(completion) is int and completion >= MODEL_OUTPUT_LIMIT:
            return None
        try:
            extraction = _from_payload(record["extraction"])
        except (TypeError, ValueError, KeyError):
            return None
        try:
            if isinstance(receipt, dict) and 'extraction_sha256' in receipt and receipt['extraction_sha256'] != extraction_fingerprint(extraction):
                return None
        except (TypeError, ValueError):
            return None
        return CachedResult(extraction, record.get("usage"), record.get("latency_ms"), deepcopy(receipt))

    def get(self, media_id: str, key: str) -> Extraction | None:
        result = self.get_result(media_id, key)
        return result.extraction if result else None

    def put(
        self, media_id: str, key: str, extraction: Extraction,
        *, usage: dict | None = None, latency_ms: float | None = None,
        output_validation: dict | None = None,
    ) -> None:
        if isinstance(output_validation, dict) and output_validation.get('policy') == INCOMPLETE_OCR_POLICY:
            raise ExtractionError('incomplete OCR belongs only to its request ledger, never the success cache')
        if output_validation is not None and not validation_receipt_valid(output_validation):
            raise ExtractionError('unverified output cannot enter the extraction cache')
        if isinstance(output_validation, dict) and 'ocr_coverage' in output_validation:
            if verified_ocr_coverage(output_validation, extraction, media_id=media_id,
                    context_sha256=output_validation.get('context_sha256'),
                    image_position=output_validation.get('image_position'),
                    image_count=output_validation.get('image_count')) is None:
                raise ExtractionError('unbound or inconsistent OCR coverage cannot enter the extraction cache')
        completion = usage.get('completion_tokens') if isinstance(usage, dict) else None
        if type(completion) is int and completion >= MODEL_OUTPUT_LIMIT:
            raise ExtractionError('output at the model limit cannot enter the extraction cache')
        record = {
            "media_id": media_id, "extraction": asdict(extraction),
            "usage": usage, "latency_ms": latency_ms,
        }
        if output_validation is not None:
            record['output_validation'] = dict(output_validation)
        self._write_record(media_id, key, record)

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
        require_evidence: bool = False,
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model
        self._prompt_version = prompt_version
        self._schema_version = schema_version
        self._provider_identity = provider_identity
        self.require_evidence = require_evidence
        self.hits = 0
        self.misses = 0

    @property
    def model_name(self) -> str:
        return self._model

    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def supports_resumable_ocr(self):
        return bool(getattr(self._inner, 'supports_resumable_ocr', False))

    def get_cached_result(self, *, image_data_uri, context_text, image_position, image_count):
        media_id = _media_id(image_data_uri)
        key = self._key(media_id, context_text, image_position, image_count)
        reader = getattr(self._cache, 'get_result', None)
        if callable(reader):
            result = reader(media_id, key)
        else:
            extraction = self._cache.get(media_id, key)
            result = CachedResult(extraction, None, None) if extraction is not None else None
        if result is None:
            return None
        receipt = getattr(result, 'output_validation', None)
        if isinstance(receipt, dict) and receipt.get('policy') == INCOMPLETE_OCR_POLICY:
            return None
        if isinstance(receipt, dict):
            identities = dict(media_id=media_id, context_sha256=context_digest(context_text),
                image_position=image_position, image_count=image_count)
            if any(k in receipt and (receipt[k] != v or k in {'image_position', 'image_count'}
                    and type(receipt[k]) is not int) for k, v in identities.items()):
                return None
            if 'ocr_coverage' in receipt and verified_ocr_coverage(receipt, result.extraction, **identities) is None:
                return None
        return result

    def extract_resumable(self, *, resume=None, checkpoint=None, binding=None, legacy=False,
                          image_data_uri, context_text, image_position, image_count):
        if not self.supports_resumable_ocr:
            return self.extract(image_data_uri=image_data_uri, context_text=context_text,
                image_position=image_position, image_count=image_count)
        from .base import has_visual_evidence, ExtractionError
        media_id = _media_id(image_data_uri)
        key = self._key(media_id, context_text, image_position, image_count)
        record = self.get_cached_result(image_data_uri=image_data_uri, context_text=context_text,
            image_position=image_position, image_count=image_count)
        cached = record.extraction if record is not None else None
        if cached is not None and (not self.require_evidence or has_visual_evidence(cached)):
            self.hits += 1
            return cached
        self.misses += 1
        extraction = self._inner.extract_resumable(resume=resume, checkpoint=checkpoint,
            binding=binding, legacy=legacy, image_data_uri=image_data_uri,
            context_text=context_text, image_position=image_position, image_count=image_count)
        if self.require_evidence and not has_visual_evidence(extraction):
            raise ExtractionError('model returned no observable visual evidence')
        observations = self._inner.take_observations()
        receipts = self._inner.take_output_receipts()
        usage, latency = (observations[-1][1], observations[-1][2]) if observations else (None, None)
        # Only the fully verified result reaches the success cache. Scene and
        # partial crop checkpoints live solely in the request preparation ledger.
        receipt = receipts[-1] if receipts else None
        if not isinstance(receipt, dict) or receipt.get('policy') != INCOMPLETE_OCR_POLICY:
            self._cache.put(media_id, key, extraction, usage=usage, latency_ms=latency,
                output_validation=receipt)
        return extraction

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
        record = self.get_cached_result(image_data_uri=image_data_uri, context_text=context_text,
            image_position=image_position, image_count=image_count)
        cached = record.extraction if record is not None else None
        from .base import has_visual_evidence, ExtractionError
        if cached is not None and (not self.require_evidence or has_visual_evidence(cached)):
            self.hits += 1
            return cached
        self.misses += 1
        extraction = self._inner.extract(
            image_data_uri=image_data_uri, context_text=context_text,
            image_position=image_position, image_count=image_count,
        )
        if self.require_evidence and not has_visual_evidence(extraction):
            raise ExtractionError("model returned no observable visual evidence")
        observations = self._inner.take_observations() if hasattr(self._inner, "take_observations") else []
        usage, latency = (observations[-1][1], observations[-1][2]) if observations else (None, None)
        receipts = self._inner.take_output_receipts() if hasattr(self._inner, 'take_output_receipts') else []
        self._cache.put(media_id, key, extraction, usage=usage, latency_ms=latency,
                        output_validation=receipts[-1] if receipts else None)
        return extraction
