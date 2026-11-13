"""Evidence-aware media allocation with complete, attributed source text."""
import base64
import json
from itertools import chain

from .evidence import source_text
from .evidence_delivery import ResponseDelivery
from .evidence_review import media_ids


def _round_robin(units):
    groups = [media_ids(unit) for unit in units]
    for offset in range(max(map(len, groups), default=0)):
        for group in groups:
            if offset < len(group):
                yield group[offset]


def _allocate(config, units, preferred, media):
    admitted, reasons, seen = {}, {}, set()
    size = 0
    for image_id in chain(_round_robin(preferred), _round_robin(units)):
        if image_id in seen:
            continue
        seen.add(image_id)
        if len(admitted) >= config.max_images:
            reasons[image_id] = 'image_count_budget'
            continue
        loaded = media.get(image_id)
        if loaded is None:
            reasons[image_id] = 'media_unavailable'
            continue
        data, mime = loaded
        if len(data) > config.max_image_bytes:
            reasons[image_id] = 'single_image_byte_budget'
            continue
        if size + len(data) > config.image_byte_budget:
            reasons[image_id] = 'response_image_byte_budget'
            continue
        size += len(data)
        admitted[image_id] = (data, mime)
        if len(admitted) == config.max_images:
            break
    return admitted, reasons


def _rendered_extra(unit):
    rendered = unit.raw_data.get('text_content', '')
    original = unit.raw_data.get('original_content', '')
    prefix = '[Dialogue] ' + original
    original_parts = '\n'.join(p['text'] for p in unit.raw_data.get('content_parts', []) if p.get('type') == 'text')
    if original and original_parts == original and (rendered == prefix or rendered.startswith(prefix + '\n')):
        return rendered[len(prefix):].lstrip('\n')
    return rendered if rendered != original or original_parts != original else ''


def deliver(config, plan, media):
    contexts = {}
    for source in plan.contexts:
        contexts.setdefault(source.parent_uid, []).append(source)
    units = []
    for unit, _ in plan.hits:
        units.append(unit)
        units.extend(source.unit for source in contexts.get(unit.uid, ()))
    admitted, reasons = _allocate(config, units, plan.preferred_sources, media)
    notes, attached, uris = ResponseDelivery(), {}, {}

    def content(unit, owner):
        from .ocr_provenance import ocr_delivery_note
        parts = [{'type': 'text', 'text': source_text(unit)}]
        record = notes.source(unit)
        source_parts = unit.raw_data.get('content_parts', [])
        has_image = any(p.get('type') == 'image' for p in source_parts)
        if not has_image:
            parts.append({'type': 'text', 'text': unit.raw_data['text_content']})
        else:
            for part in source_parts:
                if part.get('type') == 'text':
                    parts.append({'type': 'text', 'text': part['text']})
                    continue
                image_id = part.get('media_id') if part.get('type') == 'image' else None
                if not isinstance(image_id, str):
                    continue
                if image_id in attached:
                    parts.append({'type': 'text', 'text': '[Shared original image]\n' + json.dumps(dict(
                        media_id=image_id, image_in_memory=attached[image_id]))})
                    record.record(image_id, 'shared_in_response')
                elif image_id in admitted:
                    if image_id not in uris:
                        data, mime = admitted[image_id]
                        uris[image_id] = f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')
                    parts.append({'type': 'image_url', 'image_url': {'url': uris[image_id]}})
                    attached[image_id] = owner
                    record.record(image_id, 'attached_here')
                else:
                    record.record(image_id, reasons.get(image_id, 'image_count_budget'))
            # Only an exactly redundant dialogue prefix is removed. Canonical
            # content is never mutated; every original text part stays intact.
            if extra := _rendered_extra(unit):
                parts.append({'type': 'text', 'text': extra})
        if note := ocr_delivery_note(unit, namespace=config.namespace):
            parts.append({'type': 'text', 'text': note})
        from .source_fallback import delivery_note
        if note := delivery_note(unit):
            parts.append({'type':'text','text':note})
        return parts

    results = []
    for unit, score in plan.hits:
        parts = content(unit, unit.uid)
        for source in contexts.get(unit.uid, ()):
            parts.append({'type': 'text', 'text': '[Additional source context]\n' + json.dumps(dict(
                parent_memory_id=unit.uid, source_memory_id=source.unit.uid,
                retrieval_hint=source.relation), ensure_ascii=False)})
            parts.extend(content(source.unit, unit.uid))
        results.append(dict(id=unit.uid, score=float(score), content=parts))
    notes.attach(results)
    if plan.review_incomplete_reason and results:
        results[0]['content'].append({'type': 'text', 'text':
            '[Evidence review status]\nReview incomplete: the selector response was invalid. '
            'Returned original sources retain their retrieval order; evidence selection is unverified.'})
    return {'data': results}
