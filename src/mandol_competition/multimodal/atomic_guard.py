"""An abstention gate that preserves the baseline pool, order and scores.

Only complete, twice-rejected evidence may be suppressed. A useful source or
an incomplete review preserves every original hit; no partial selection prunes
an atomic fact out of an otherwise answerable request.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os

from mandol_competition.inference.evidence_selector import EvidenceSelector, SYSTEM_PROMPT
from mandol_competition.inference.errors import InferenceSchemaError
from .evidence import selector_preview, source_text
from .protocol import ImagePart, join_text_parts

POLICY = "source-grounded-atomic-v2"
MAX_SOURCES = 100
MAX_REVIEW_IMAGES = 16
REVIEW_URI_BYTES = 32 * 1024 * 1024
REVIEW_TEXT_BYTES = 64 * 1024

PRIMARY_PROMPT = SYSTEM_PROMPT + """
This is an evidence-presence check, not answer generation. Select ONE candidate
with concrete useful evidence, including partial or bridge evidence. You need
not find a complete answer or prove every premise before selecting a source.
Observable image attributes, readable OCR, an explicit positive OR negative
statement, a dated correction, and distinct links across sessions can all be
useful evidence. Do not require a speaker to restate the question's exact words.
Question premises and answer options are not remembered facts. Topic overlap
or visual resemblance alone does not prove a person's identity, ownership,
event, date or exact location. Return [] only when no useful evidence is present.
Some previews may be incomplete; do not infer that a hidden detail is absent.
"""

VERIFY_PROMPT = SYSTEM_PROMPT + """
Audit a proposed absence of evidence. Select ONE source if it contains any
concrete direct, atomic, negative, corrective, visual, OCR or necessary bridge
fact useful for the requested information. The source need not answer the whole
question. Preserve explicit denials as evidence when the question asks whether
something happened. A visible color, count, object category or readable label
is sufficient evidence for a question asking for that observable attribute;
do not additionally require a name, location, ownership or exact date unless
the question asks for it. Check distinct sessions and roles without confusing
assistant speculation with a user's statement or with observable image facts.
Do not select a source solely for topic overlap. Never accept the question's
premise or an answer option as proof. If concrete partial support exists, select
it instead of concluding absence. Return [] only if all sources lack useful
evidence. Output indices only; never an answer, option choice or invented fact.
"""


@dataclass(frozen=True)
class Review:
    candidates: list[dict]
    query_images: tuple[str, ...]
    complete: bool
    incomplete_reasons: tuple[str, ...]


class AtomicRecallGuard:
    def __init__(self, config):
        options = dict(api_key=os.environ.get(config.vlm_api_key_env),
                       timeout=config.vlm_timeout_seconds, strict=True)
        self.primary = EvidenceSelector(config.vlm_base_url, system_prompt=PRIMARY_PROMPT, **options)
        try:
            self.verifier = EvidenceSelector(config.vlm_base_url, system_prompt=VERIFY_PROMPT, **options)
        except BaseException:
            self.primary.close()
            raise

    def select(self, **inputs):
        return self.primary.select(**inputs)

    def confirm_absence(self, **inputs):
        return self.verifier.select(**inputs)

    def close(self):
        try:
            self.primary.close()
        finally:
            self.verifier.close()


def prepare_review(request, pool, media) -> Review:
    from .service import selector_candidate_text

    examined = pool[:MAX_SOURCES]
    if not examined:
        return Review([], (), not pool, ())
    reasons = set()
    if len(examined) != len(pool):
        reasons.add("candidate_limit")
    candidates, query_images = [], []
    image_owners, omitted_images = {}, set()
    image_count, image_bytes = 0, 0
    for part in request.parts():
        if not isinstance(part, ImagePart):
            continue
        uri = part.image_url.url
        if image_count >= MAX_REVIEW_IMAGES or image_bytes + len(uri) > REVIEW_URI_BYTES:
            reasons.add("query_image_not_reviewed")
            continue
        query_images.append(uri)
        image_count += 1
        image_bytes += len(uri)
    for index, (unit, _) in enumerate(examined):
        candidate = {"text": source_text(unit) + selector_candidate_text(unit), "images": []}
        references = []
        for part in unit.raw_data.get("content_parts", []):
            media_id = part.get("media_id") if part.get("type") == "image" else None
            if not isinstance(media_id, str):
                continue
            if media_id in image_owners:
                references.append({"media_id": media_id, "image_in_candidate": image_owners[media_id]})
                continue
            if media_id in omitted_images or image_count >= MAX_REVIEW_IMAGES:
                reasons.add("source_image_not_reviewed")
                omitted_images.add(media_id)
                continue
            loaded = media.get(media_id)
            if loaded is None:
                reasons.add("source_image_missing")
                omitted_images.add(media_id)
                continue
            data, mime = loaded
            encoded_size = 4 * ((len(data) + 2) // 3) + len(f"data:{mime};base64,")
            if len(data) > 10 * 1024 * 1024 or image_bytes + encoded_size > REVIEW_URI_BYTES:
                reasons.add("source_image_not_reviewed")
                omitted_images.add(media_id)
                continue
            candidate["images"].append(f"data:{mime};base64," + base64.b64encode(data).decode("ascii"))
            image_owners[media_id] = index
            image_count += 1
            image_bytes += encoded_size
        if references:
            candidate["text"] = "[Shared original images]\n" + json.dumps(references) + "\n" + candidate["text"]
        original_view = candidate["text"]
        candidate["text"] = selector_preview(original_view, REVIEW_TEXT_BYTES // len(examined))
        # The shared selector also has a 2400-character preview bound. Compare
        # actual text, rather than trusting the presence/absence of a marker.
        if candidate["text"][:2400] != original_view:
            reasons.add("source_text_not_fully_reviewed")
        candidates.append(candidate)
    return Review(candidates, tuple(query_images), not reasons, tuple(sorted(reasons)))


def _has_evidence(selection, source_count):
    indices = getattr(selection, "indices", None)
    status = getattr(selection, "status", None)
    if (status not in {"selected", "empty"}
            or not isinstance(indices, (tuple, list)) or len(indices) > 1
            or (status == "empty" and indices)
            or any(type(i) is not int or not 0 <= i < source_count for i in indices)):
        raise InferenceSchemaError("atomic_guard_invalid_selection")
    return bool(indices)


def guard_sources(guard, request, hits, media):
    baseline = list(hits[:request.top_k])
    if not baseline:
        return []
    review = prepare_review(request, baseline, media)
    inputs = dict(query=join_text_parts(request.parts()), options=request.options,
        candidates=review.candidates, limit=1, visual=True, query_images=review.query_images)
    if _has_evidence(guard.select(**inputs), len(review.candidates)):
        return baseline
    # Budget/preview omissions are uncertainty, never proof of absent evidence.
    if not review.complete:
        return baseline
    if _has_evidence(guard.confirm_absence(**inputs), len(review.candidates)):
        return baseline
    return []
