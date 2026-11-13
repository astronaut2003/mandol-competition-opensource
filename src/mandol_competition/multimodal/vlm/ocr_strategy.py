"""Reuse a proven concise OCR recipe; no additional calls or retry allowance."""
from copy import deepcopy
import hashlib
import json
from . import ocr_concise
from .ocr_recovery import CropTask,ValidatedCrop,crop_response_format

VERSION='ocr_switched_v1'

def policy():
    material=[ocr_concise.SYSTEM,ocr_concise.PROMPT,crop_response_format()]
    return VERSION+':'+hashlib.sha256(json.dumps(material,sort_keys=True).encode()).hexdigest()[:24]

def stage(task):
    return VERSION+':'+hashlib.sha256(json.dumps([policy(),task.to_payload()],sort_keys=True).encode()).hexdigest()[:24]

def is_stage(value):
    return isinstance(value,str) and value.startswith(VERSION+':')

def validate(value,calls):
    if value is None:return
    if not isinstance(value,dict) or set(value)!={'policy','task','validated','metadata','usage'} or value['policy']!=policy():
        raise ValueError('OCR strategy shape or recipe changed')
    task=CropTask.from_payload(value['task'])
    ValidatedCrop.from_payload(task,value['validated'])
    if not task.disjoint_input or task.source_box!=task.ownership_box:
        raise ValueError('OCR strategy needs an owned disjoint source')
    history=[c for c in calls if not c['known_rejection'] and c['stage']==ocr_concise.stage(task)]
    if not history or history[-1]['metadata']!=value['metadata'] or history[-1]['usage']!=value['usage']:
        raise ValueError('OCR strategy lacks the successful concise paid receipt')
    meta=value['metadata'];tokens=meta.get('completion_tokens');limit=meta.get('output_token_budget')
    if (meta.get('finish_reason')!='stop' or type(limit) is not int or not 0<limit<=1024
            or tokens is not None and (type(tokens) is not int or not 0<=tokens<limit)):
        raise ValueError('OCR strategy trigger was incomplete')

def activate(task,checked,metadata,usage):
    return dict(policy=policy(),task=task.to_payload(),validated=checked.to_payload(),
        metadata=deepcopy(metadata),usage=deepcopy(usage))

def authorized(state,requested):
    if state['ocr_strategy'] is None:return False
    validate(state['ocr_strategy'],state['calls'])
    return any(t.get('disjoint_input') is True and t['source_box']==t['ownership_box']
        and stage(CropTask.from_payload(t))==requested for t in state['tasks'])
