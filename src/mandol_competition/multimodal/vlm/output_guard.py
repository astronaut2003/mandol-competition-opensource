"""Output framing and cost bounds; no visual facts are truncated or fabricated."""
from dataclasses import asdict
from copy import deepcopy
import hashlib
import json
from .base import ExtractionError

OUTPUT_POLICY = 'bounded-json-v1'
COMPOSITE_POLICY = 'bounded-scene-ocr-v2'
REGION_COMPOSITE_POLICY = 'bounded-scene-localized-ocr-v3'
REGION_REVIEW_POLICY = 'bounded-scene-reviewed-ocr-v4'
INCOMPLETE_OCR_POLICY = 'retained-scene-incomplete-ocr-v1'
MODEL_OUTPUT_LIMIT = 16384
VISION_REQUEST_OUTPUT_LIMIT = 1024
OCR_COVERAGE_POLICY = 'localized-physical-ocr-recovery-v1'


def ocr_coverage_valid(value, *, completed_crops=None):
    if isinstance(value, dict) and value.get('status') == 'incomplete':
        return (set(value) == {'policy', 'status', 'unreadable_regions', 'completed_crops', 'planned_crops'}
            and value['policy'] == OCR_COVERAGE_POLICY
            and type(value['planned_crops']) is int and 0 <= value['planned_crops'] <= 12
            and type(value['completed_crops']) is int and 0 <= value['completed_crops'] <= value['planned_crops']
            and type(value['unreadable_regions']) is int
            and 0 <= value['unreadable_regions'] <= value['completed_crops'] * 32)
    if not isinstance(value, dict) or set(value) != {'policy', 'status', 'unreadable_regions', 'completed_crops'}:
        return False
    status, unreadable, crops = value['status'], value['unreadable_regions'], value['completed_crops']
    return (value['policy'] == OCR_COVERAGE_POLICY
        and isinstance(status, str) and status in {'readable', 'no_text', 'unreadable', 'partially_unreadable'}
        and type(crops) is int and 1 <= crops <= 12 and (completed_crops is None or crops == completed_crops)
        and type(unreadable) is int and 0 <= unreadable <= crops * 32
        and (unreadable > 0 if status in {'unreadable', 'partially_unreadable'} else unreadable == 0))


def bound_validation_receipt_valid(value, extraction, *, media_id, context_sha256, image_position, image_count):
    """Full receipt identity is mandatory before carrying optional readability."""
    if (not validation_receipt_valid(value) or not isinstance(media_id, str)
            or not isinstance(context_sha256, str) or len(context_sha256) != 64
            or any(c not in '0123456789abcdef' for c in context_sha256)
            or type(image_position) is not int or type(image_count) is not int
            or not 0 <= image_position < image_count):
        return False
    try:
        return (value.get('media_id') == media_id and value.get('context_sha256') == context_sha256
            and type(value.get('image_position')) is int and value['image_position'] == image_position
            and type(value.get('image_count')) is int and value['image_count'] == image_count
            and value.get('extraction_sha256') == extraction_fingerprint(extraction))
    except (TypeError, ValueError):
        return False


def verified_ocr_coverage(value, extraction, **identity):
    if not bound_validation_receipt_valid(value, extraction, **identity) or 'ocr_coverage' not in value:
        return None
    coverage = value['ocr_coverage']
    text = getattr(extraction, 'ocr_text', None)
    if not isinstance(text, str):
        return None
    if coverage['status'] != 'incomplete' and bool(text.strip()) != (coverage['status'] in {'readable', 'partially_unreadable'}):
        return None
    return deepcopy(coverage)


def extraction_fingerprint(value):
    return hashlib.sha256(json.dumps(asdict(value), sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def validate_output_budgets(initial, maximum):
    if (type(initial) is not int or type(maximum) is not int
            or not 1 <= initial <= maximum <= MODEL_OUTPUT_LIMIT):
        raise ValueError('vision output budgets must be ordered positive integers within the model limit')


def validation_receipt_valid(value):
    if isinstance(value, dict) and value.get('policy') == INCOMPLETE_OCR_POLICY:
        coverage = value.get('ocr_coverage')
        return (value.get('finish_reason') == 'incomplete' and value.get('reason') == 'recovery_exhausted'
            and value.get('scene_retained') is True and value.get('completion_tokens') is None
            and value.get('output_token_budget') == MODEL_OUTPUT_LIMIT
            and isinstance(coverage, dict) and coverage.get('status') == 'incomplete'
            and ocr_coverage_valid(coverage))
    if (not isinstance(value, dict) or not isinstance(value.get('policy'), str)
            or value['policy'] not in {OUTPUT_POLICY, COMPOSITE_POLICY, REGION_COMPOSITE_POLICY, REGION_REVIEW_POLICY}
            or value.get('finish_reason') != 'stop'):
        return False
    if 'ocr_coverage' in value and (isinstance(value['ocr_coverage'], dict) and value['ocr_coverage'].get('status') == 'incomplete'
          or value['policy'] not in {REGION_COMPOSITE_POLICY, REGION_REVIEW_POLICY}
            or not ocr_coverage_valid(value['ocr_coverage'])):
        return False
    budget = value.get('output_token_budget')
    completion = value.get('completion_tokens')
    valid = (type(budget) is int and 1 <= budget <= MODEL_OUTPUT_LIMIT
             and (completion is None or type(completion) is int and 0 <= completion < budget))
    if not valid or value['policy'] == OUTPUT_POLICY:
        return valid
    stages = value.get('stages')
    if value['policy'] in {REGION_COMPOSITE_POLICY, REGION_REVIEW_POLICY}:
        if not isinstance(stages, list) or not 3 <= len(stages) <= 14:
            return False
        if 'ocr_coverage' in value and not ocr_coverage_valid(value['ocr_coverage'], completed_crops=len(stages)-2):
            return False
        used, known = [], []
        for index, stage in enumerate(stages):
            component = ('scene' if index == 0 else
                         ('ocr_review' if value['policy'] == REGION_REVIEW_POLICY else 'ocr_locator')
                         if index == 1 else 'ocr_crop')
            if (not isinstance(stage, dict) or stage.get('component') != component
                    or stage.get('complete') is not True
                    or stage.get('finish_reason') not in ({'stop', 'length'} if index == 0 else {'stop'})):
                return False
            limit, tokens = stage.get('output_token_budget'), stage.get('completion_tokens')
            if (type(limit) is not int or not 1 <= limit <= MODEL_OUTPUT_LIMIT
                    or tokens is not None and (type(tokens) is not int or not 0 <= tokens <= limit
                        or index > 0 and tokens >= limit)):
                return False
            used.append(tokens if tokens is not None else limit)
            known.append(tokens)
        crops = [stage.get('crop_id') for stage in stages[2:]]
        return (all(isinstance(v, str) and v for v in crops) and len(set(crops)) == len(crops)
            and sum(used) <= budget and (completion == sum(known)
                if all(type(v) is int for v in known) else completion is None))
    if not isinstance(stages,list) or len(stages) != 2:
        return False
    used = []
    for index, stage in enumerate(stages):
        if (not isinstance(stage,dict) or stage.get('component') != ('scene' if index == 0 else 'ocr')
                or stage.get('complete') is not True
                or stage.get('finish_reason') not in ({'stop','length'} if index == 0 else {'stop'})):
            return False
        limit, tokens = stage.get('output_token_budget'), stage.get('completion_tokens')
        if type(limit) is not int or not 1 <= limit <= MODEL_OUTPUT_LIMIT:
            return False
        if tokens is not None and (type(tokens) is not int or not 0 <= tokens <= limit
                or index == 1 and tokens >= limit):
            return False
        used.append(tokens)
    primary = used[0] if used[0] is not None else stages[0]['output_token_budget']
    if primary + stages[1]['output_token_budget'] > budget:
        return False
    return completion == sum(used) if all(n is not None for n in used) else completion is None


class VisionOutputError(ExtractionError):
    def __init__(self, error_class, *, finish_reason=None, completion_tokens=None,
                 output_token_budget=None, retryable=False):
        super().__init__('vision output did not satisfy the bounded JSON contract')
        self.error_class = error_class
        self.retryable = retryable
        self.unknown_outcome = False
        self.vlm_finish_reason = finish_reason if isinstance(finish_reason, str) and finish_reason in {
            'stop', 'length', 'content_filter', 'tool_calls', 'function_call'} else 'unverified'
        self.vlm_completion_tokens = completion_tokens
        self.vlm_output_token_budget = output_token_budget


class VisionOutputTruncated(VisionOutputError):
    def __init__(self, **metadata):
        super().__init__('vision_output_truncated', retryable=True, **metadata)
