"""Bounded, complete-fragment review of existing sources, with an atomic floor.

Positive or incomplete review retains every baseline ID, rank and score. An
absence verdict requires two full passes over every planned fragment. Model
errors propagate; neither a bad reply nor a budget omission means no evidence.
Only request-local selection and media priorities are produced here.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
import os

from mandol_substrate.deadlines import remaining_budget
from mandol_competition.inference.evidence_selector import EvidenceSelector, SYSTEM_PROMPT
from mandol_competition.inference.errors import InferenceSchemaError
from .protocol import ImagePart, join_text_parts

POLICY = 'source-grounded-recall-v4'
MAX_SOURCES = 108
MAX_FRAGMENTS = 256
MAX_BATCHES = 8
MAX_MODEL_CALLS = 2 * MAX_BATCHES
MAX_BATCH_IMAGES = 16
MAX_BATCH_TEXT_BYTES = 32 * 1024
MAX_URI_BYTES = 32 * 1024 * 1024
MAX_QUERY_BYTES = 16 * 1024
MAX_OPTIONS_BYTES = 8 * 1024
MAX_SELECTED = 8
MAX_FRAGMENT_CHARS = 2200
OVERLAP_CHARS = 160

PRIMARY_PROMPT = SYSTEM_PROMPT + """
Check for existing evidence; do not generate an answer. Each candidate is a
fragment of a real source, not a new source. Fragment/source metadata identifies
provenance. Select useful direct, atomic, negative, corrective, visual, OCR or
bridge facts, including partial support. A source need not answer the entire
question. Retain distinct dated or speaker-specific facts needed together.
An observable attribute suffices for a question about that attribute; do not
add requirements for identity, ownership, date or place that were not asked.
A shared text anchor or nearby message is a retrieval hint, not proof of a
shared person, event, place or cause. Topic overlap and a question's premises
or options alone are not evidence. Return up to the requested limit of useful
candidate indices; [] only when this batch contains no useful evidence.
Return only the outer candidate_index integers. source_group labels are
provenance across fragments, never output indices.
"""
VERIFY_PROMPT = SYSTEM_PROMPT + """
Independently audit a proposed absence in this batch of original-source
fragments. Find any concrete direct, partial, atomic, visual, OCR, negative,
dated correction or necessary bridge evidence. Do not demand a complete answer
from one fragment. Distinguish user facts, assistant speculation and visible
image attributes. An explicit denial is evidence for a question asking whether
something occurred. Do not accept an option, question premise, shared name,
visual resemblance or nearby message as proof of an unobserved relationship.
Return useful candidate indices, or [] only if every fragment here is unrelated
to the requested information. Never output an answer or invented assertion.
Return only outer candidate_index integers, never source_group labels.
"""


class PartitionedEvidenceGuard:
    def __init__(self, config):
        options = dict(api_key=os.environ.get(config.vlm_api_key_env), strict=True,
                       timeout=min(90., config.vlm_timeout_seconds))
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


def _bounded(text, budget):
    encoded = text.encode('utf-8')
    return (text, True) if len(encoded) <= budget else (encoded[:budget].decode('utf-8', 'ignore'), False)


def evidence_text(unit):
    """Remove only a provably redundant verbatim dialogue, never a fact."""
    from .service import selector_candidate_text
    rendered = unit.raw_data.get('text_content', '')
    original = unit.raw_data.get('original_content', '')
    if original and (rendered == '[Dialogue] ' + original
                     or rendered.startswith('[Dialogue] ' + original + '\n')):
        return rendered
    return selector_candidate_text(unit)


def media_ids(unit):
    return tuple(dict.fromkeys(p['media_id'] for p in unit.raw_data.get('content_parts', [])
        if p.get('type') == 'image' and isinstance(p.get('media_id'), str)))


@dataclass(frozen=True)
class ContextSource:
    parent_uid: str
    unit: object
    relation: str


@dataclass(frozen=True)
class EvidencePlan:
    hits: list
    preferred_sources: tuple = ()
    contexts: tuple[ContextSource, ...] = ()
    complete_absence_review: bool = False
    model_calls: int = 0


@dataclass
class Scan:
    sources: list
    request: object
    media: object
    complete: bool = True
    query: str = ''
    options: list = field(default_factory=list)
    query_images: list = field(default_factory=list)

    def __post_init__(self):
        self.query, intact = _bounded(join_text_parts(self.request.parts()), MAX_QUERY_BYTES)
        self.complete &= intact
        left = MAX_OPTIONS_BYTES
        for option in self.request.options or []:
            value, intact = _bounded(option, max(0, left))
            if value:
                self.options.append(value)
            self.complete &= intact
            left -= len(value.encode('utf-8'))
        size = 0
        for part in self.request.parts():
            if not isinstance(part, ImagePart):
                continue
            uri = part.image_url.url
            if len(self.query_images) >= MAX_BATCH_IMAGES or size + len(uri) > MAX_URI_BYTES:
                self.complete = False
                continue
            self.query_images.append(uri)
            size += len(uri)

    def fragments(self):
        total = 0
        if len(self.sources) > MAX_SOURCES:
            self.complete = False
        for source_index, unit in enumerate(self.sources[:MAX_SOURCES]):
            metadata = {k: unit.metadata[k] for k in ('role', 'timestamp', 'session_id', 'message_index')
                        if k in unit.metadata}
            preview, intact = _bounded(json.dumps(metadata, ensure_ascii=False), 512)
            self.complete &= intact
            header = '[Source fragment metadata]\n' + json.dumps(dict(
                source_group='source-' + str(source_index),
                provenance=metadata if intact else dict(incomplete_metadata_preview=preview)), ensure_ascii=False) + '\n'
            # Source metadata can contain escaping/control characters. Bound the
            # actual characters the shared selector sees after JSON quoting.
            if len(header) >= MAX_FRAGMENT_CHARS - OVERLAP_CHARS - 1:
                self.complete = False
                header = json.dumps(dict(source_group='source-' + str(source_index))) + '\n'
            width = MAX_FRAGMENT_CHARS - len(header)
            body, ids = evidence_text(unit), media_ids(unit)
            start, image_index = 0, 0
            first = True
            while first or start < len(body) or image_index < len(ids):
                first = False
                if total >= MAX_FRAGMENTS:
                    self.complete = False
                    return
                total += 1
                end = min(len(body), start + width)
                # A newline/sentence boundary plus overlap keeps local atomic
                # relationships together while still covering every character.
                if end < len(body):
                    cut = max(body.rfind('\n', start + width // 2, end),
                              body.rfind('. ', start + width // 2, end),
                              body.rfind('。', start + width // 2, end))
                    if cut > start:
                        end = cut + 1
                text = header + body[start:end]
                image_id = ids[image_index] if image_index < len(ids) else None
                image_index += bool(image_id)
                yield source_index, text, image_id
                if end < len(body):
                    start = max(start + 1, end - OVERLAP_CHARS)
                else:
                    start = len(body)

    def batches(self):
        candidates, mapping, owners = [], [], {}
        text_bytes, uri_bytes = 0, sum(map(len, self.query_images))
        image_count = len(self.query_images)
        for index, text, image_id in self.fragments():
            size = len(json.dumps(text, ensure_ascii=False).encode('utf-8')) + 128
            if candidates and (len(candidates) == 100 or text_bytes + size > MAX_BATCH_TEXT_BYTES
                               or image_id not in owners and image_id is not None
                               and image_count >= MAX_BATCH_IMAGES):
                yield candidates, mapping
                candidates, mapping, owners = [], [], {}
                text_bytes, uri_bytes = 0, sum(map(len, self.query_images))
                image_count = len(self.query_images)
            images = []
            if image_id in owners:
                text += '\n[Shared original image in this batch] ' + str(owners[image_id])
            elif image_id is not None:
                loaded = self.media.get(image_id)
                if loaded is None:
                    self.complete = False
                else:
                    data, mime = loaded
                    encoded_size = 4 * ((len(data) + 2) // 3) + len(f'data:{mime};base64,')
                    if candidates and uri_bytes + encoded_size > MAX_URI_BYTES:
                        yield candidates, mapping
                        candidates, mapping, owners = [], [], {}
                        text_bytes, uri_bytes = 0, sum(map(len, self.query_images))
                        image_count = len(self.query_images)
                    if (len(data) > 10 * 1024 * 1024 or uri_bytes + encoded_size > MAX_URI_BYTES
                            or image_count >= MAX_BATCH_IMAGES):
                        self.complete = False
                    else:
                        images = [f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')]
                        owners[image_id] = len(candidates)
                        image_count += 1
                        uri_bytes += encoded_size
            if len(text) > 2400 or size > MAX_BATCH_TEXT_BYTES:
                # This is uncertainty, not permission to silently drop a tail.
                self.complete = False
            candidates.append(dict(text=text, images=images))
            mapping.append(index)
            text_bytes += size
        if candidates:
            yield candidates, mapping


def _indices(selection, count, limit):
    values = getattr(selection, 'indices', None)
    status = getattr(selection, 'status', None)
    if (status not in {'selected', 'empty'} or not isinstance(values, (tuple, list))
            or len(values) > limit
            or status == 'empty' and values
            or any(type(i) is not int or not 0 <= i < count for i in values)
            or len(set(values)) != len(values)):
        raise InferenceSchemaError('partitioned_guard_invalid_selection')
    return tuple(values)


def review_sources(guard, request, hits, media, contexts=()):
    baseline = list(hits[:request.top_k])
    if not baseline:
        return EvidencePlan([])
    known = {unit.uid for unit, _ in baseline}
    by_parent = {}
    for context in contexts:
        if context.parent_uid in known and context.unit.uid not in known:
            known.add(context.unit.uid)
            by_parent.setdefault(context.parent_uid, []).append(context)
    sources, added = [], {}
    for unit, _ in baseline:
        sources.append(unit)
        for context in by_parent.get(unit.uid, ()):
            added[context.unit.uid] = context
            sources.append(context.unit)
    calls, batch_maps = 0, []
    scan = Scan(sources, request, media)
    for phase in ('primary', 'verification'):
        if phase == 'verification':
            if not scan.complete:
                return EvidencePlan(baseline, model_calls=calls)
            scan = Scan(sources, request, media)
        current_maps = []
        for batch_index, (candidates, mapping) in enumerate(scan.batches()):
            if batch_index >= MAX_BATCHES or remaining_budget(90.) < 2.:
                scan.complete = False
                break
            current_maps.append(tuple(mapping))
            if phase == 'verification' and (batch_index >= len(batch_maps)
                                            or tuple(mapping) != batch_maps[batch_index]):
                scan.complete = False
                break
            inputs = dict(query=scan.query, options=scan.options, candidates=candidates,
                          limit=min(MAX_SELECTED, len(candidates)), visual=True,
                          query_images=tuple(scan.query_images))
            select = guard.select if phase == 'primary' else guard.confirm_absence
            selected = _indices(select(**inputs), len(candidates), inputs['limit'])
            calls += 1
            if selected:
                units = tuple(dict.fromkeys(sources[mapping[i]].uid for i in selected))
                preferred = tuple(unit for unit in sources if unit.uid in units)
                return EvidencePlan(baseline, preferred,
                    tuple(added[uid] for uid in units if uid in added), model_calls=calls)
        if phase == 'primary':
            batch_maps = current_maps
        elif current_maps != batch_maps:
            scan.complete = False
    if scan.complete:
        return EvidencePlan([], complete_absence_review=True, model_calls=calls)
    return EvidencePlan(baseline, model_calls=calls)
