"""AML Add and Search orchestration over durable Mandol memory."""

from __future__ import annotations

import base64
from dataclasses import asdict
from mandol_substrate.state.ledger import RequestLedger
from .vlm.cache import _from_payload
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mandol_substrate.state.models import Conflict
from mandol_substrate.state.models import digest as canonical_body_hash

from mandol_substrate.representation import MemoryUnit

from .config import MultimodalConfig
from .media_store import UserMediaStore, compute_media_id
from .protocol import (
    AddRequest,
    ContentPart,
    ImagePart,
    Message,
    SearchRequest,
    TextPart,
    join_text_parts,
)
from .render import Location, RenderImage, render_query_text
from .unit_builder import build_memory_unit
from .validation import (
    DecodedImage,
    PayloadError,
    decode_message_images,
    enforce_request_image_budget,
)
from .vlm.base import Extraction, ExtractionError, VisionExtractor
from .vlm.cache import CachingExtractor, ExtractionCache, context_digest
from .profiling import timed
from .representation import BASELINE, query_text as render_profiled_query

decode_message_images = timed(decode_message_images, "image_decode")
build_memory_unit = timed(build_memory_unit, "render_memory")
render_query_text = timed(render_query_text, "render_query")
render_profiled_query = timed(render_profiled_query, "render_query")
encode_response_image = timed(base64.b64encode, "base64_encoding")

if TYPE_CHECKING:
    from .backend import MultimodalBackend
    from .persistence import CanonicalStore
    from .store import UserGraphRegistry


class ConflictError(Conflict):
    """A request ID was reused with a different payload."""


def selector_candidate_text(unit: MemoryUnit) -> str:
    """Exactly the response's text view, before its aggregate media budget.

    The score harness consumes wire evidence. Image memories return ordered raw
    text plus the rendered representation; a selector must see that same view.
    This changes only explicitly opted-in selection, never baseline responses.
    """
    raw=unit.raw_data
    rendered=raw.get('text_content','')
    parts=raw.get('content_parts',[])
    if not any(p.get('type')=='image' for p in parts):return rendered
    texts=[p['text'] for p in parts if p.get('type')=='text']
    if not texts or rendered!=raw.get('original_content'):texts.append(rendered)
    return '\n'.join(texts)


def _timestamp(message: Message, images: dict[int, DecodedImage]) -> str | None:
    if message.timestamp is not None:
        try:
            return datetime.fromtimestamp(message.timestamp / 1000, UTC).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (OverflowError, OSError, ValueError):
            pass
    return next((image.exif.timestamp for image in images.values() if image.exif.timestamp), None)


def _location(image: DecodedImage) -> Location | None:
    if image.exif.gps is None:
        return None
    latitude, longitude = image.exif.gps
    return Location(f"{latitude},{longitude}", "exif")


def _cached_extractor(
    config: MultimodalConfig, user_id: str, extractor: VisionExtractor
) -> CachingExtractor:
    return CachingExtractor(
        extractor,
        ExtractionCache(config.user_root(user_id) / "vlm_cache"),
        model=getattr(extractor, "model_name", config.vlm_model),
        prompt_version=config.prompt_version,
        schema_version=config.schema_version,
        provider_identity=(context_digest(f"{config.vlm_base_url}\x00{config.vlm_request_model}")
                           if config.vlm_request_model else ""),
    )


class AddService:
    def __init__(
        self,
        config: MultimodalConfig,
        *,
        extractor: VisionExtractor,
        registry: UserGraphRegistry | None = None,
        store: CanonicalStore | None = None,
        backend: MultimodalBackend | None = None,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.backend = backend
        self.registry = backend or registry
        self.store = backend.state if backend else store
    def _decode(self, request: AddRequest) -> list[tuple[Message, list[ContentPart], dict[int, DecodedImage]]]:
        decoded_messages = []
        all_images = []
        for message in request.messages:
            parts = message.parts()
            images = decode_message_images(parts, self.config)
            decoded_messages.append((message, parts, images))
            all_images.extend(images.values())
        enforce_request_image_budget(all_images, self.config)
        return decoded_messages

    def prepare_units(
        self,
        request: AddRequest,
        decoded_messages: list[tuple[Message, list[ContentPart], dict[int, DecodedImage]]] | None = None,
        preparation=None,
    ) -> list[MemoryUnit]:
        decoded_messages = decoded_messages if decoded_messages is not None else self._decode(request)
        media = UserMediaStore(self.config, request.user_id)
        vision = _cached_extractor(self.config, request.user_id, self.extractor)
        units = []
        representation_policy = (self.backend.profile.options.get("representation_policy", BASELINE)
                                 if self.backend is not None else BASELINE)
        for message_index, (message, parts, decoded) in enumerate(decoded_messages):
            dialogue = join_text_parts(parts)
            image_count = len(decoded)
            content_parts: list[dict[str, Any]] = []
            images: list[RenderImage] = []
            degraded = False
            for part_index, part in enumerate(parts):
                if isinstance(part, TextPart):
                    content_parts.append({"type": "text", "text": part.text})
                    continue
                assert isinstance(part, ImagePart)
                image = decoded[part_index]
                record = media.put(image)
                content_parts.append({"type": "image", "media_id": record.media_id})
                try:
                    extraction = (preparation.extract if preparation is not None else vision.extract)(
                        **({"key": f"{message_index}:{part_index}", "vision": vision} if preparation is not None else {}),
                        image_data_uri=part.image_url.url,
                        context_text=dialogue,
                        image_position=len(images),
                        image_count=image_count,
                    )
                except ExtractionError:
                    degraded = True
                    extraction = Extraction(
                        model=vision.model_name,
                        prompt_version=self.config.prompt_version,
                        schema_version=self.config.schema_version,
                    )
                images.append(RenderImage(record.media_id, extraction, _location(image)))
            unit = build_memory_unit(
                user_id=request.user_id,
                request_id=request.request_id,
                session_id=request.session_id,
                message_index=message_index,
                role=message.role,
                timestamp_text=_timestamp(message, decoded),
                dialogue=dialogue,
                content_parts=content_parts,
                images=images,
                extractor_model=vision.model_name,
                prompt_version=self.config.prompt_version,
                schema_version=self.config.schema_version,
                representation_policy=representation_policy,
                surrounding_text="\n".join(f"{other.role}: {join_text_parts(other_parts)}"
                    for i, (other, other_parts, _) in enumerate(decoded_messages) if i != message_index)
                    if representation_policy == "mm-representation-r6-v1" else "",
            )
            if degraded:
                unit.metadata["vlm_degraded"] = True
            units.append(unit)
        return units

    def handle(self, request: AddRequest) -> dict[str, Any]:
        with self.registry.user_operation(request.user_id):
            return self._handle(request)

    def _handle(self, request: AddRequest) -> dict[str, Any]:
        if self.backend is not None and len(request.messages) > self.backend.coordinator.max_mutation_memories:
            raise PayloadError("request exceeds bounded publication budget", status_code=413)
        decoded_messages = self._decode(request)
        body_hash = canonical_body_hash(request.model_dump())
        if self.backend is not None:
            result = self.backend.replay(request.user_id, request.request_id, body_hash)
            if result is not None:
                return result
            preparation = PreparationLedger(self.backend, request, body_hash)
            if "units" in preparation.value:
                units = [MemoryUnit(item["uid"], item["raw_data"], item["metadata"]) for item in preparation.value["units"]]
            else:
                units = self.prepare_units(request, decoded_messages, preparation)
                preparation.value["units"] = [{"uid": unit.uid, "raw_data": unit.raw_data, "metadata": unit.metadata} for unit in units]
                preparation.save()
            response = {"success": True, "request_id": request.request_id,
                        "user_id": request.user_id, "session_id": request.session_id}
            return self.backend.publish(request, body_hash, units, response)
        from .persistence import BeginOutcome

        with self.store.hold(request.request_id, body_hash, request.user_id) as begin:
            if begin.outcome is BeginOutcome.CONFLICT:
                raise ConflictError("request_id already belongs to another payload")
            if begin.outcome is BeginOutcome.REPLAY:
                assert begin.response is not None
                return begin.response
            try:
                units = self.prepare_units(request, decoded_messages)
            except Exception:
                self.store.fail(request.request_id, body_hash, "preparation failed")
                raise
            self.store.stage(request.request_id, request.user_id, body_hash, units)
            response = {
                "success": True,
                "request_id": request.request_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
            }
            with self.registry.acquire(request.user_id) as session, session.transaction():
                session.add_units(units)
                self.store.mark_succeeded(request.request_id, body_hash, response)
            return response


class SearchService:
    def __init__(
        self,
        config: MultimodalConfig,
        *,
        extractor: VisionExtractor,
        registry: UserGraphRegistry | None = None,
        store: CanonicalStore | None = None,
        backend: MultimodalBackend | None = None,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.backend = backend
        self.registry = backend or registry
        self.store = backend.state if backend else store

        self.evidence_selector = None
        policy = backend.profile.options.get("answer_utility_policy") if backend else None
        if policy:
            from .answer_utility import validate_policy
            validate_policy(policy)
            if policy.get("selector", "none") != "none":
                import os
                from mandol_competition.inference.evidence_selector import EvidenceSelector
                self.evidence_selector = EvidenceSelector(config.vlm_base_url,
                    api_key=os.environ.get(config.vlm_api_key_env), timeout=90)

    def close(self):
        if self.evidence_selector is not None:
            self.evidence_selector.close()

    def prepare_query(self, request: SearchRequest) -> str:
        parts = request.parts()
        question = join_text_parts(parts)
        decoded = decode_message_images(parts, self.config)
        vision = _cached_extractor(self.config, request.user_id, self.extractor)
        images: list[RenderImage] = []
        for part_index, part in enumerate(parts):
            if not isinstance(part, ImagePart):
                continue
            image = decoded[part_index]
            try:
                extraction = vision.extract(
                    image_data_uri=part.image_url.url,
                    context_text=question,
                    image_position=len(images),
                    image_count=len(decoded),
                )
            except ExtractionError:
                extraction = Extraction(
                    model=vision.model_name,
                    prompt_version=self.config.prompt_version,
                    schema_version=self.config.schema_version,
                )
            images.append(RenderImage(compute_media_id(image.data), extraction, _location(image)))
        options = self.backend.profile.options if self.backend is not None else {}
        return render_profiled_query(policy_id=options.get("query_policy", "mm-query-v1"),
            query_text=question, images=images, options=request.options,
            use_options=options.get("query_use_options", True))

    def handle(self, request: SearchRequest) -> dict[str, Any]:
        # Text-only Search reads canonical evidence and restores media under a
        # read lease. Add/purge still wait for those readers. Image queries keep
        # the original exclusive lease because they can write extraction caches
        # and must preserve one paid attempt sequence for an identical query.
        operation = (self.backend.user_operation(request.user_id, write=False)
                     if self.backend is not None and not any(isinstance(p, ImagePart) for p in request.parts())
                     else self.registry.user_operation(request.user_id))
        with operation:
            return self._handle(request)

    def _handle(self, request: SearchRequest) -> dict[str, Any]:
        if self.backend is not None and request.top_k > self.backend.projection.policy.physical_search_budget:
            raise PayloadError("top_k exceeds bounded retrieval budget")
        if self.backend is not None:
            self.backend.assert_searchable(request.user_id)
        query = self.prepare_query(request)
        if not query.strip():
            return {"data": []}
        if self.backend is not None:
            policy = self.backend.profile.options.get("answer_utility_policy", {})
            candidate_k = max(request.top_k, policy.get("candidate_k", request.top_k))
            hits = self.backend.search(request.user_id, query, top_k=candidate_k)
        else:
            with self.registry.acquire(request.user_id) as session:
                hits = session.search(
                    query, top_k=request.top_k, rerank_method=self.config.rerank_method
                )
        media = UserMediaStore(self.config, request.user_id)
        if self.backend is not None and policy:
            from .answer_utility import output_limit
            output_k = output_limit(policy,join_text_parts(request.parts()),
                any(isinstance(p,ImagePart) for p in request.parts()),request.top_k)
            if self.evidence_selector is not None:
                candidates = []
                for unit, _ in hits[:policy.get("candidate_k", 30)]:
                    candidate = {"text": selector_candidate_text(unit), "images": []}
                    if policy.get("selector") == "visual":
                        for part in unit.raw_data.get("content_parts", []):
                            if part.get("type") == "image":
                                loaded = media.get(part["media_id"])
                                if loaded:
                                    data, mime = loaded
                                    candidate["images"].append(f"data:{mime};base64," +
                                        encode_response_image(data).decode("ascii"))
                    candidates.append(candidate)
                selected = self.evidence_selector.select(query=join_text_parts(request.parts()),
                    options=request.options, candidates=candidates,
                    limit=min(output_k, len(candidates)), visual=policy.get("selector") == "visual",
                    query_images=[p.image_url.url for p in request.parts() if isinstance(p, ImagePart)])
                if selected.status == "selected":
                    hits = [(hits[i][0], 1. / (rank + 1)) for rank, i in enumerate(selected.indices)]
                # A failed selector preserves the complete baseline result floor.
            else:
                hits = hits[:output_k]
        image_count = 0
        image_bytes = 0
        reuse_media = bool(self.backend and self.backend.profile.options.get('reuse_media_within_response', False))
        # Request-local and inside the user's operation lease. Cache only images
        # actually admitted to this response; memory stays bounded by its budget.
        restored_media = {}
        results: list[dict[str, Any]] = []
        for unit, score in hits[: request.top_k]:
            text = unit.raw_data.get("text_content")
            if not isinstance(text, str) or not text:
                continue
            content: list[dict[str, Any]] = [] if self.backend is not None else [{"type": "text", "text": text}]
            for part in unit.raw_data.get("content_parts", []):
                if part.get("type") == "text":
                    if self.backend is not None:
                        content.append({"type": "text", "text": part["text"]})
                    continue
                if part.get("type") != "image" or not isinstance(part.get("media_id"), str):
                    continue
                if image_count >= self.config.max_images:
                    continue
                cached = restored_media.get(part['media_id']) if reuse_media else None
                if cached is not None:
                    size, uri = cached
                else:
                    loaded = media.get(part["media_id"])
                    if loaded is None:
                        continue
                    data, mime = loaded
                    size = len(data)
                if (size > self.config.max_image_bytes
                        or image_bytes + size > self.config.image_byte_budget):
                    continue
                if cached is None:
                    uri = f'data:{mime};base64,' + encode_response_image(data).decode('ascii')
                    if reuse_media:
                        restored_media[part['media_id']] = (size, uri)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": uri},
                })
                image_bytes += size
                image_count += 1
            if self.backend is not None and not any(part.get("type") == "image" for part in unit.raw_data.get("content_parts", [])):
                content = []
            if self.backend is not None and (not content or text != unit.raw_data.get("original_content")):
                content.append({"type": "text", "text": text})
            results.append({"id": unit.uid, "content": content, "score": float(score)})
        return {"data": results}


class PreparationLedger:
    """At most one VLM attempt sequence per logical image preparation.

    A process death after dispatch is indeterminate. A retry uses durable cache
    if present, otherwise the existing degraded VLM policy; it does not issue
    another potentially billable unknown-outcome call.
    """
    def __init__(self, backend, request, body_hash):
        self.ledger = RequestLedger(backend.state)
        self.scope, self.request_id, self.body_hash = backend.scope(request.user_id), request.request_id, body_hash
        self.value = self.ledger.reserve(self.scope, self.request_id, body_hash)

    def save(self):
        self.ledger.checkpoint(self.scope, self.request_id, self.body_hash, self.value)

    def extract(self, *, key, vision, **inputs):
        entries = self.value.setdefault("vision", {})
        prior = entries.get(key)
        if prior:
            if prior["status"] == "complete":
                return _from_payload(prior["extraction"])
            if prior["status"] == "started":
                from .vlm.cache import _media_id
                media_id = _media_id(inputs["image_data_uri"])
                cache_key = vision._key(media_id, inputs["context_text"], inputs["image_position"], inputs["image_count"])
                cached = vision._cache.get(media_id, cache_key)
                if cached is not None:
                    entries[key] = {"status": "complete", "extraction": asdict(cached)}
                    self.save()
                    return cached
            entries[key] = {"status": "degraded"}
            self.save()
            raise ExtractionError("previous VLM preparation unavailable or outcome unknown")
        entries[key] = {"status": "started"}
        self.save()
        try:
            extraction = vision.extract(**inputs)
        except ExtractionError:
            entries[key] = {"status": "degraded"}
            self.save()
            raise
        entries[key] = {"status": "complete", "extraction": asdict(extraction)}
        self.save()
        return extraction
