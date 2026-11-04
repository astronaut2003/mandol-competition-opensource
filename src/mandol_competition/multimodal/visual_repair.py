"""Offline proposals only. This module never calls a model, DB or publisher.

The expected source fingerprint is an optimistic guard for a later, separately
reviewed publication transaction. Existing IDs, source text and image references
are retained. Proposals do not authorize applying or reindexing production data.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json

from mandol_substrate.representation import MemoryUnit
from .render import Location, RenderImage
from .representation import POLICIES
from .unit_builder import build_memory_unit
from .vlm.base import Extraction, has_visual_evidence
from .vlm.cache import _from_payload, context_digest
from .vlm.output_guard import extraction_fingerprint, validation_receipt_valid


class VisualRepairRejected(ValueError):
    pass


def source_fingerprint(unit):
    material = {'uid': unit.uid, 'raw_data': unit.raw_data, 'metadata': unit.metadata}
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class VisualRepairProposal:
    expected_source_sha256: str
    replacement: MemoryUnit
    image_position: int
    output_validation: dict
    requires_atomic_reindex: bool = True
    applied: bool = False


def propose_visual_repair(unit, image_position, replacement: Extraction, receipt, *,
                          expected_source_sha256, representation_policy,
                          review_unflagged_empty=False, surrounding_text=None):
    if source_fingerprint(unit) != expected_source_sha256:
        raise VisualRepairRejected('source changed since the audit')
    if representation_policy not in POLICIES:
        raise VisualRepairRejected('the existing representation policy is required')
    if POLICIES[representation_policy].surrounding and surrounding_text is None:
        raise VisualRepairRejected('the exact surrounding context is required for this policy')
    raw, metadata = deepcopy(unit.raw_data), deepcopy(unit.metadata)
    values = raw.get('multimodal_info', {}).get('images')
    if not isinstance(values, list) or type(image_position) is not int or not 0 <= image_position < len(values):
        raise VisualRepairRejected('valid existing image position required')
    prior = _from_payload(values[image_position])
    if has_visual_evidence(prior):
        raise VisualRepairRejected('healthy visual observations cannot be replaced by this repair')
    if not metadata.get('vlm_degraded') and not review_unflagged_empty:
        raise VisualRepairRejected('unflagged empty observations require separate review')
    if not has_visual_evidence(replacement):
        raise VisualRepairRejected('empty replacement is not a repair')
    if (prior.model, prior.prompt_version, prior.schema_version) != (
            replacement.model, replacement.prompt_version, replacement.schema_version):
        raise VisualRepairRejected('model and extraction identities must remain unchanged')
    if (not validation_receipt_valid(receipt) or type(receipt.get('completion_tokens')) is not int
            or receipt.get('media_id') != values[image_position].get('media_id')
            or receipt.get('context_sha256') != context_digest(raw.get('original_content', ''))
            or receipt.get('image_position') != image_position or receipt.get('image_count') != len(values)
            or receipt.get('extraction_sha256') != extraction_fingerprint(replacement)):
        raise VisualRepairRejected('a complete, input-bound output receipt is required')
    replacement_value = asdict(replacement)
    for key in ('media_id', 'location'):
        replacement_value[key] = values[image_position].get(key)
    values[image_position] = replacement_value
    rendered = []
    for value in values:
        location = Location(**value['location']) if value.get('location') is not None else None
        rendered.append(RenderImage(value['media_id'], _from_payload(value), location))
    rebuilt = build_memory_unit(user_id=metadata['user_id'], request_id=metadata['request_id'],
        session_id=metadata['session_id'], message_index=metadata['message_index'], role=metadata['role'],
        timestamp_text=metadata.get('timestamp'), dialogue=raw['original_content'],
        content_parts=raw['content_parts'], images=rendered, extractor_model=metadata['extractor_model'],
        prompt_version=metadata['extractor_prompt_version'], schema_version=metadata['schema_version'],
        representation_policy=representation_policy, surrounding_text=surrounding_text or '')
    if rebuilt.uid != unit.uid:
        raise VisualRepairRejected('the original memory ID cannot be reconstructed exactly')
    raw['multimodal_info']['images'] = values
    raw['text_content'] = rebuilt.raw_data['text_content']
    if all(has_visual_evidence(_from_payload(value)) for value in values):
        metadata.pop('vlm_degraded', None)
    proposed = MemoryUnit(unit.uid, raw, metadata)
    return VisualRepairProposal(expected_source_sha256, proposed, image_position, deepcopy(receipt))
