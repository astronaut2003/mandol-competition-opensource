"""AML Add and Search orchestration over durable Mandol memory."""

from __future__ import annotations

import base64
from copy import deepcopy
import math
import time
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
    result = '\n'.join(texts)
    from .ocr_provenance import incomplete_ocr_note
    note = incomplete_ocr_note(unit)
    from .source_fallback import delivery_note
    source_note = delivery_note(unit)
    return result + ('\n' + note if note else '') + ('\n' + source_note if source_note else '')


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
        require_evidence=config.vision_failure_policy == "retry-required-v1",
        provider_identity=(context_digest(f"{config.vlm_base_url}\x00{config.vlm_request_model}")
                           if config.vlm_request_model else ""),
    )


def _cached_vision_result(vision, inputs):
    getter = getattr(vision, 'get_cached_result', None)
    if callable(getter):
        return getter(**inputs)
    # Older/fake adapters retain their extraction-only cache interface. They
    # cannot supply optional readability provenance they never recorded.
    from .vlm.cache import CachedResult, _media_id
    if not hasattr(vision, '_cache') or not hasattr(vision, '_key'):
        return None
    media_id = _media_id(inputs['image_data_uri'])
    key = vision._key(media_id, inputs['context_text'], inputs['image_position'], inputs['image_count'])
    cached = vision._cache.get(media_id, key)
    return CachedResult(cached, None, None) if cached is not None else None


def _vision_result_receipt(vision, extraction, inputs, receipt=None):
    from .vlm.cache import _media_id
    from .vlm.output_guard import bound_validation_receipt_valid, extraction_fingerprint
    identity = dict(media_id=_media_id(inputs['image_data_uri']), context_sha256=context_digest(inputs['context_text']),
        image_position=inputs['image_position'], image_count=inputs['image_count'])
    if bound_validation_receipt_valid(receipt, extraction, **identity):
        return deepcopy(receipt)
    cached = _cached_vision_result(vision, inputs)
    if cached is None or extraction_fingerprint(cached.extraction) != extraction_fingerprint(extraction):
        return None
    receipt = getattr(cached, 'output_validation', None)
    return deepcopy(receipt) if bound_validation_receipt_valid(receipt, extraction, **identity) else None


def _entry_with_vision_receipt(entry, vision, extraction, inputs):
    from .vlm.cache import _media_id
    from .vlm.output_guard import verified_ocr_coverage
    receipt = _vision_result_receipt(vision, extraction, inputs, entry.get('output_validation'))
    result = dict(entry)
    result.pop('ocr_coverage', None)
    result.pop('output_validation', None)
    if receipt is not None:
        result['output_validation'] = receipt
        coverage = verified_ocr_coverage(receipt, extraction, media_id=_media_id(inputs['image_data_uri']),
            context_sha256=context_digest(inputs['context_text']), image_position=inputs['image_position'],
            image_count=inputs['image_count'])
        if coverage is not None:
            result['ocr_coverage'] = coverage
    return result


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
            ocr_readability = []
            source_only = []
            quality_statuses = []
            degraded = False
            for part_index, part in enumerate(parts):
                if isinstance(part, TextPart):
                    content_parts.append({"type": "text", "text": part.text})
                    continue
                assert isinstance(part, ImagePart)
                image = decoded[part_index]
                record = media.put(image)
                content_parts.append({"type": "image", "media_id": record.media_id})
                inputs = dict(image_data_uri=part.image_url.url,context_text=dialogue,
                    image_position=len(images),image_count=image_count)
                from .source_fallback import restore as restore_source
                from .vlm.openai_compat import VisionExtractionUnavailable
                try:
                    extraction = restore_source(self.config,preparation,f'{message_index}:{part_index}',vision,inputs)
                    if extraction is None:
                        extraction = (preparation.extract if preparation is not None else vision.extract)(
                            **({"key": f"{message_index}:{part_index}", "vision": vision} if preparation is not None else {}),
                            image_data_uri=part.image_url.url,
                            context_text=dialogue,
                            image_position=len(images),
                            image_count=image_count,
                        )
                except VisionExtractionUnavailable as exc:
                    extraction = restore_source(self.config,preparation,f'{message_index}:{part_index}',vision,inputs,error=exc)
                    if extraction is None:raise
                except ExtractionError as exc:
                    if self.config.vision_failure_policy == "retry-required-v1":
                        from .vlm.openai_compat import VisionExtractionUnavailable
                        raise VisionExtractionUnavailable(retry_safe=True) from exc
                    degraded = True
                    extraction = Extraction(
                        model=vision.model_name,
                        prompt_version=self.config.prompt_version,
                        schema_version=self.config.schema_version,
                    )
                images.append(RenderImage(record.media_id, extraction, _location(image)))
                if preparation is not None:
                    retained=preparation.value.get('source_only',{}).get(f'{message_index}:{part_index}')
                    if retained is not None:source_only.append(deepcopy(retained['receipt']))
                from .vlm.cache import _media_id
                from .vlm.output_guard import verified_ocr_coverage
                entry = (preparation.value.get('vision', {}).get(f'{message_index}:{part_index}', {})
                         if preparation is not None else {})
                inputs = dict(image_data_uri=part.image_url.url, context_text=dialogue,
                    image_position=len(images)-1, image_count=image_count)
                receipt = _vision_result_receipt(vision, extraction, inputs,
                    entry.get('output_validation') if isinstance(entry, dict) else None)
                coverage = verified_ocr_coverage(receipt, extraction, media_id=_media_id(inputs['image_data_uri']),
                    context_sha256=context_digest(dialogue), image_position=len(images)-1, image_count=image_count)
                if coverage is not None:
                    ocr_readability.append(dict(image_position=len(images)-1, **coverage))
                    if coverage['status'] == 'incomplete':
                        degraded = True
                retained=(preparation.value.get('source_only',{}).get(f'{message_index}:{part_index}')
                          if preparation is not None else None)
                quality_statuses.append('source_only' if retained is not None else
                    'incomplete_ocr' if coverage is not None and coverage['status']=='incomplete' else
                    'complete' if receipt is not None else 'unverified')
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
            if ocr_readability:
                unit.metadata['visual_ocr_readability'] = ocr_readability
            from .source_fallback import annotate
            annotate(unit,source_only)
            from .add_quality import annotate as annotate_quality
            annotate_quality(unit,quality_statuses)
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
            strict_vision = self.config.vision_failure_policy == "retry-required-v1"
            preparation = PreparationLedger(self.backend, request, body_hash, strict=strict_vision)
            from .raw_intake import RawIntakeStore
            originals = RawIntakeStore(self.config, request.user_id)
            saved_intake = preparation.value.get('raw_intake')
            if saved_intake is None:
                saved_intake = originals.save(request, decoded_messages,
                    scope_key=preparation.scope.key, body_hash=body_hash)
                preparation.value['raw_intake'] = saved_intake
                preparation.save()
            else:
                originals.load(saved_intake, scope_key=preparation.scope.key,
                    request_id=request.request_id, body_hash=body_hash)
            if "units" in preparation.value:
                units = [MemoryUnit(item["uid"], item["raw_data"], item["metadata"]) for item in preparation.value["units"]]
                if not self.config.vision_source_fallback and any(unit.metadata.get('visual_source_only') for unit in units):
                    from .vlm.openai_compat import VisionExtractionUnavailable
                    raise VisionExtractionUnavailable(retry_safe=False,unknown_outcome=False)
                if strict_vision and any(unit.metadata.get("vlm_degraded") for unit in units):
                    # Rebuild from the input-bound extraction ledger rather than
                    # accepting the old marker or rejecting verified incomplete OCR.
                    # Completed stage results need no further model dispatch.
                    units = self.prepare_units(request, decoded_messages, preparation)
                    preparation.value["units"] = [{"uid": unit.uid, "raw_data": unit.raw_data,
                        "metadata": unit.metadata} for unit in units]
                    preparation.save()
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
                from mandol_substrate.state.models import Scope
                from .raw_intake import RawIntakeStore
                RawIntakeStore(self.config, request.user_id).save(request, decoded_messages,
                    scope_key=Scope(self.config.namespace, request.user_id).key, body_hash=body_hash)
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
        if config.evidence_policy in {"source-grounded-v1", "source-grounded-atomic-v2", "source-grounded-delivery-v3", "source-grounded-recall-v4"}:
            from .evidence import create_selector
            self.evidence_selector = create_selector(config)
        policy = backend.profile.options.get("answer_utility_policy") if backend else None
        if policy and config.evidence_policy == "baseline":
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
            except ExtractionError as exc:
                if self.config.vision_failure_policy == "retry-required-v1":
                    from .vlm.openai_compat import VisionExtractionUnavailable
                    raise VisionExtractionUnavailable(retry_safe=True) from exc
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
        from .ocr_provenance import ocr_source_provenance
        for unit, _score in hits:
            ocr_source_provenance(unit, namespace=self.config.namespace)
        media = UserMediaStore(self.config, request.user_id)
        if self.config.evidence_policy == "source-grounded-recall-v4":
            from .evidence_review import review_sources
            from .grounded_delivery import deliver
            from .source_context import collect_context
            contexts = (collect_context(self.backend, request.user_id, hits, join_text_parts(request.parts()))
                        if self.backend is not None else ())
            for context in contexts:
                ocr_source_provenance(context.unit, namespace=self.config.namespace)
            plan = review_sources(self.evidence_selector, request, hits, media, contexts)
            return deliver(self.config, plan, media)
        grounded = self.config.evidence_policy in {"source-grounded-v1", "source-grounded-atomic-v2", "source-grounded-delivery-v3"}
        if self.config.evidence_policy in {"source-grounded-atomic-v2", "source-grounded-delivery-v3"}:
            from .atomic_guard import guard_sources
            hits = guard_sources(self.evidence_selector, request, hits, media)
        elif grounded:
            from .evidence import select_sources
            hits = select_sources(self.evidence_selector, request, hits, media)
        elif self.backend is not None and policy:
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
        delivered_images = {}
        delivery = None
        if self.config.evidence_policy == "source-grounded-delivery-v3":
            from .evidence_delivery import ResponseDelivery
            delivery = ResponseDelivery()
        results: list[dict[str, Any]] = []
        for unit, score in hits[: request.top_k]:
            from .ocr_provenance import ocr_delivery_note
            ocr_field_note = ocr_delivery_note(unit, namespace=self.config.namespace)
            text = unit.raw_data.get("text_content")
            if not isinstance(text, str) or not text:
                continue
            source_delivery = delivery.source(unit) if delivery is not None else None
            content: list[dict[str, Any]] = [] if self.backend is not None else [{"type": "text", "text": text}]
            for part in unit.raw_data.get("content_parts", []):
                if part.get("type") == "text":
                    if self.backend is not None:
                        content.append({"type": "text", "text": part["text"]})
                    continue
                if part.get("type") != "image" or not isinstance(part.get("media_id"), str):
                    continue
                if grounded and part["media_id"] in delivered_images:
                    import json
                    content.append({"type": "text", "text": "[Shared original image]\n" + json.dumps({
                        "media_id": part["media_id"], "image_in_memory": delivered_images[part["media_id"]]})})
                    if source_delivery is not None:
                        source_delivery.record(part["media_id"], "shared_in_response")
                    continue
                if image_count >= self.config.max_images:
                    if source_delivery is not None:
                        source_delivery.record(part["media_id"], "image_count_budget")
                    continue
                cached = restored_media.get(part['media_id']) if reuse_media else None
                if cached is not None:
                    size, uri = cached
                else:
                    loaded = media.get(part["media_id"])
                    if loaded is None:
                        if source_delivery is not None:
                            source_delivery.record(part["media_id"], "media_unavailable")
                        continue
                    data, mime = loaded
                    size = len(data)
                if (size > self.config.max_image_bytes
                        or image_bytes + size > self.config.image_byte_budget):
                    if source_delivery is not None:
                        source_delivery.record(part["media_id"], "single_image_byte_budget"
                            if size > self.config.max_image_bytes else "response_image_byte_budget")
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
                if grounded:
                    delivered_images[part["media_id"]] = unit.uid
                if source_delivery is not None:
                    source_delivery.record(part["media_id"], "attached_here")
            if self.backend is not None and not any(part.get("type") == "image" for part in unit.raw_data.get("content_parts", [])):
                content = []
            if self.backend is not None and (not content or text != unit.raw_data.get("original_content")):
                content.append({"type": "text", "text": text})
            if grounded:
                from .evidence import source_text
                content.insert(0, {"type": "text", "text": source_text(unit)})
            if ocr_field_note is not None:
                content.append({"type": "text", "text": ocr_field_note})
            from .source_fallback import delivery_note
            if note := delivery_note(unit):
                content.append({'type':'text','text':note})
            results.append({"id": unit.uid, "content": content, "score": float(score)})
        if delivery is not None:
            delivery.attach(results)
        return {"data": results}


class PreparationLedger:
    """Durable, bounded recovery of interrupted visual preparation.

    Strict Add first reuses complete results or a valid scoped cache. If the
    previous call's outcome remains unknown, a later Add retry may run at most
    three recovery sequences per image in one minute. The durable cooldown can
    expire, so a provider outage never permanently poisons an unpublished Add.
    Reserve that allowance before dispatch; process loss cannot reset it.
    Publication remains the authority for exactly one visible Add result.
    """
    MAX_RECOVERY_ATTEMPTS = 3
    RECOVERY_WINDOW_SECONDS = 60.
    def __init__(self, backend, request, body_hash, *, strict=False):
        self.strict = strict
        self.ledger = RequestLedger(backend.state)
        self.scope, self.request_id, self.body_hash = backend.scope(request.user_id), request.request_id, body_hash
        self.value = self.ledger.reserve(self.scope, self.request_id, body_hash)

    def save(self):
        self.ledger.checkpoint(self.scope, self.request_id, self.body_hash, self.value)

    def _recovery_now(self):
        return time.time()

    def extract(self, *, key, vision, **inputs):
        from mandol_competition.inference.errors import InferenceAdmissionDeferred, InferenceDeadlineExceeded, InferenceRateLimited
        from mandol_competition.inference.http import remaining_budget
        from .vlm.openai_compat import VisionExtractionUnavailable, VisionRecoveryExhausted

        entries = self.value.setdefault("vision", {})
        prior = entries.get(key)
        if self.strict and getattr(vision, 'supports_resumable_ocr', False):
            return self._extract_staged(key, vision, prior, inputs)
        recovering = False
        recovery_attempts = 0
        recovery_window_started_at = None
        if prior:
            if prior["status"] == "complete":
                completed = _from_payload(prior["extraction"])
                from .vlm.base import has_visual_evidence
                if not self.strict or has_visual_evidence(completed):
                    return completed
                # A completed but empty response is known, not indeterminate.
                entries.pop(key)
                self.save()
                prior = None
            if prior and (prior["status"] == "started" or self.strict):
                from .vlm.cache import _media_id
                media_id = _media_id(inputs["image_data_uri"])
                cache_key = vision._key(media_id, inputs["context_text"], inputs["image_position"], inputs["image_count"])
                cached = vision._cache.get(media_id, cache_key)
                if cached is not None:
                    from .vlm.base import has_visual_evidence
                    if not self.strict or has_visual_evidence(cached):
                        entries[key] = {"status": "complete", "extraction": asdict(cached)}
                        self.save()
                        return cached
            if prior and self.strict:
                if remaining_budget(1.) <= 0:
                    raise InferenceDeadlineExceeded()
                recovery_attempts = prior.get("recovery_attempts", 0)
                if (prior["status"] not in {"started", "unknown", "failed", "degraded"}
                        or type(recovery_attempts) is not int
                        or not 0 <= recovery_attempts <= self.MAX_RECOVERY_ATTEMPTS):
                    raise VisionRecoveryExhausted()
                recovery_window_started_at = prior.get("recovery_window_started_at")
                if recovery_window_started_at is not None and (
                        type(recovery_window_started_at) not in (int, float)
                        or not math.isfinite(recovery_window_started_at)
                        or recovery_window_started_at < 0):
                    raise VisionRecoveryExhausted()
                now = self._recovery_now()
                if recovery_window_started_at is None:
                    recovery_window_started_at = now
                    if recovery_attempts == self.MAX_RECOVERY_ATTEMPTS:
                        # Old exhausted checkpoints get one durable cooldown,
                        # never an immediate reset on every new Add or process.
                        entries[key] = dict(prior, recovery_window_started_at=now)
                        self.save()
                elif now >= recovery_window_started_at + self.RECOVERY_WINDOW_SECONDS:
                    recovery_attempts = 0
                    recovery_window_started_at = now
                if recovery_attempts == self.MAX_RECOVERY_ATTEMPTS:
                    delay = recovery_window_started_at + self.RECOVERY_WINDOW_SECONDS - now
                    raise InferenceAdmissionDeferred("vision_recovery_cooldown",
                        retry_after=max(1, math.ceil(min(self.RECOVERY_WINDOW_SECONDS, delay))))
                recovering = True
            elif prior:
                entries[key] = {"status": "degraded"}
                self.save()
                raise ExtractionError("previous VLM preparation unavailable or outcome unknown")
        if remaining_budget(1.) <= 0:
            raise InferenceDeadlineExceeded()
        entries[key] = {"status": "started"}
        if self.strict:
            recovery_attempts += int(recovering)
            entries[key]["recovery_attempts"] = recovery_attempts
            if recovering:
                entries[key]["recovery_window_started_at"] = recovery_window_started_at
        self.save()
        try:
            extraction = vision.extract(**inputs)
        except (InferenceDeadlineExceeded, InferenceRateLimited) as exc:
            # An explicit rejection or pre-dispatch deadline is not an
            # indeterminate paid call. Preserve earlier completed images but
            # allow this image to retry instead of permanently degrading it.
            if getattr(exc, 'vision_paid_output_failure', False):
                # A later explicit rejection cannot refund an earlier paid,
                # truncated output within this same extraction sequence.
                entries[key] = dict(entries[key], status='failed')
            elif recovering:
                entries[key] = prior
            else:
                entries.pop(key, None)
            self.save()
            raise
        except Exception as exc:
            if isinstance(exc, VisionExtractionUnavailable):
                if exc.retry_safe:
                    if recovering:
                        entries[key] = dict(entries[key], status="failed")
                    else:
                        entries.pop(key, None)
                else:
                    entries[key] = dict(entries[key], status="unknown" if exc.unknown_outcome else "failed")
                self.save()
                raise
            if not isinstance(exc, ExtractionError):
                # Leave 'started' durable for any unclassified dispatch outcome.
                raise
            if self.strict:
                if recovering:
                    entries[key] = dict(entries[key], status="failed")
                else:
                    entries.pop(key, None)
                self.save()
                raise VisionExtractionUnavailable(retry_safe=True) from exc
            entries[key] = {"status": "degraded"}
            self.save()
            raise
        entries[key] = {"status": "complete", "extraction": asdict(extraction)}
        self.save()
        return extraction

    def _extract_staged(self, key, vision, prior, inputs):
        """Lifetime paid-stage limits; the old network cooldown grants no new OCR budget."""
        from mandol_competition.inference.errors import InferenceDeadlineExceeded, InferenceRateLimited
        from mandol_competition.inference.http import remaining_budget
        from .vlm.openai_compat import VisionExtractionUnavailable, VisionRecoveryExhausted
        from .vlm.base import has_visual_evidence
        entries = self.value['vision']
        if prior is not None:
            if not isinstance(prior, dict):
                raise VisionRecoveryExhausted()
            if prior.get('status') == 'complete':
                completed = _from_payload(prior['extraction'])
                if has_visual_evidence(completed):
                    return completed
            # A matching verified cache remains usable even after a failed
            # workflow consumed its allowance; no model dispatch is required.
            cached_result = _cached_vision_result(vision, inputs)
            cached = cached_result.extraction if cached_result is not None else None
            if cached is not None and has_visual_evidence(cached):
                entries[key] = _entry_with_vision_receipt(dict(status='complete', extraction=asdict(cached)),
                    vision, cached, inputs)
                self.save()
                return cached
            count = prior.get('recovery_attempts', 0)
            window = prior.get('recovery_window_started_at')
            if (prior.get('status') not in {'started', 'unknown', 'failed', 'degraded', 'complete'}
                    or type(count) is not int or not 0 <= count <= self.MAX_RECOVERY_ATTEMPTS
                    or window is not None and (type(window) not in (int, float) or not math.isfinite(window) or window < 0)):
                raise VisionRecoveryExhausted()
        if remaining_budget(1.) <= 0:
            raise InferenceDeadlineExceeded()
        binding = dict(scope_sha256=self.scope.key,
            request_sha256=canonical_body_hash([self.request_id, self.body_hash]), part=key)
        resume = deepcopy(prior.get('stage_checkpoint')) if prior is not None else None
        entry = deepcopy(prior) if prior is not None else {}
        entry['status'] = 'started'
        entries[key] = entry

        def checkpoint(value):
            # The closure belongs to this exact request/part; neither shared
            # extractor nor HTTP client receives mutable per-request fields.
            if value.get('binding', {}).get('parent') != binding:
                raise VisionRecoveryExhausted()
            entries[key] = dict(entries[key], stage_checkpoint=deepcopy(value))
            self.save()

        try:
            result = vision.extract_resumable(resume=resume, checkpoint=checkpoint,
                binding=binding, legacy=prior is not None and resume is None, **inputs)
        except (InferenceDeadlineExceeded, InferenceRateLimited):
            if entries[key].get('stage_checkpoint') is not None:
                entries[key] = dict(entries[key], status='failed')
                self.save()
            elif prior is not None:
                entries[key] = prior
            else:
                entries.pop(key, None)
            raise
        except VisionExtractionUnavailable as exc:
            entries[key] = dict(entries[key], status='unknown' if exc.unknown_outcome else 'failed')
            self.save()
            raise
        # Incomplete OCR is never cached, so bind its explicit request-only
        # receipt directly to this successfully validated stage session.
        from .vlm.output_guard import INCOMPLETE_OCR_POLICY, validation_receipt_valid, extraction_fingerprint
        from .vlm.base import parse_extraction
        stage_value = entries[key].get('stage_checkpoint', {})
        receipt = stage_value.get('complete_receipt')
        if isinstance(receipt, dict) and receipt.get('policy') == INCOMPLETE_OCR_POLICY:
            expected = parse_extraction(stage_value['final_payload'], model=result.model,
                prompt_version=result.prompt_version, schema_version=result.schema_version)
            if expected != result or not validation_receipt_valid(receipt):
                raise VisionRecoveryExhausted()
            from .vlm.cache import _media_id
            entries[key]['output_validation'] = dict(receipt, media_id=_media_id(inputs['image_data_uri']),
                context_sha256=context_digest(inputs['context_text']), image_position=inputs['image_position'],
                image_count=inputs['image_count'], extraction_sha256=extraction_fingerprint(result))
        entries[key] = _entry_with_vision_receipt(dict(entries[key], status='complete', extraction=asdict(result)),
            vision, result, inputs)
        self.save()
        return result
