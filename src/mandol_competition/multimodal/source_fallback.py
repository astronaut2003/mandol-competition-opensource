"""Opt-in original-evidence Add publication; never a successful VLM cache entry."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

from mandol_substrate.state.models import StateUnavailable
from .vlm.base import parse_extraction
from .vlm.stage_state import (StageState, RecoveryBudgetExhausted, MAX_OUTPUT_TOKENS,
                              MAX_PAID_CALLS, MAX_CONTENT_FAILURES)
from .vlm.staged_ocr import binding_for, digest
from .vlm.ocr_lines import FORMAT_POLICY
from .vlm.prompt import build_extraction_prompt
from .vlm.cache import _media_id, context_digest

FORMAT = 'mm-original-evidence-only-v1'


def _fail():
    raise StateUnavailable('original evidence fallback verification failed')


def _material(config, preparation, key, vision, inputs):
    raw=preparation.value.get('raw_intake',{})
    import hashlib
    if (not isinstance(raw,dict) or raw.get('format')!='mm-raw-intake-receipt-v1' or raw.get('scope_key')!=preparation.scope.key
            or raw.get('body_hash')!=preparation.body_hash or raw.get('request_id_sha256')!=
            hashlib.sha256(preparation.request_id.encode()).hexdigest()):
        _fail()
    entry=preparation.value.get('vision',{}).get(key,{})
    checkpoint=entry.get('stage_checkpoint')
    if not isinstance(checkpoint,dict) or checkpoint.get('reservation') is not None:
        _fail()
    inner=getattr(vision,'_inner',None)
    client=getattr(inner,'_client',None)
    if client is None:
        # Reconstruct identity without requesting a model or reading credentials.
        client=SimpleNamespace(model=vision.model_name,request_model=config.vlm_request_model or vision.model_name,
            output_tokens=config.vlm_output_tokens,max_output_tokens=config.vlm_max_output_tokens,
            wire_format=FORMAT_POLICY,_http=SimpleNamespace(route=SimpleNamespace(endpoint=config.vlm_base_url or '')))
    parent=dict(scope_sha256=preparation.scope.key,
        request_sha256=digest([preparation.request_id,preparation.body_hash]),part=key)
    binding=binding_for(client,build_extraction_prompt(inputs['context_text'],inputs['image_position'],
        inputs['image_count'],version=config.prompt_version),inputs['image_data_uri'],inputs['context_text'],
        inputs['image_position'],inputs['image_count'],parent,config.schema_version)
    state=StageState(binding,resume=checkpoint).state
    if state['scene'] is not None or state['final_payload'] is not None or state['paid_calls']<1:
        _fail()
    global_exhausted=(state['paid_calls']>=MAX_PAID_CALLS or state['content_failures']>=MAX_CONTENT_FAILURES
        or state['spent']+min(config.vlm_max_output_tokens,1024)>MAX_OUTPUT_TOKENS)
    from .vlm import scene_recovery as fields
    progress=state['scene_recovery']
    planned=fields.next_page(progress) if progress is not None else None
    fields_exhausted=(progress is not None and (planned is None or not fields.authorized(
        progress,state['calls'],fields.stage(*planned))))
    if not (global_exhausted or fields_exhausted):
        _fail()
    payload=dict(description='',entities=[],objects=[],events=[],relations=[],tags=[],location_hint=None,ocr_text='')
    verified=[]
    if progress:
        for page in progress['pages']:
            group=page['group']
            if group=='overview':
                payload.update(deepcopy(page['wire']))
                verified.extend(('description','tags','location_hint'))
            else:
                payload[group].extend(deepcopy(page['wire']['items']))
                if group not in verified:verified.append(group)
    result=parse_extraction(payload,model=vision.model_name,prompt_version=config.prompt_version,
        schema_version=config.schema_version)
    receipt=dict(format=FORMAT,status='source_only',reason='scene_recovery_exhausted',
        binding_sha256=digest(binding),checkpoint_sha256=digest(checkpoint),
        media_id=_media_id(inputs['image_data_uri']),context_sha256=context_digest(inputs['context_text']),
        image_position=inputs['image_position'],image_count=inputs['image_count'],
        extraction_sha256=digest(asdict(result)),verified_field_groups=verified,
        spent=state['spent'],paid_calls=state['paid_calls'])
    return result,receipt


def restore(config, preparation, key, vision, inputs, *, error=None):
    """Only terminal, validated output-budget failure can authorize this path."""
    if not config.vision_source_fallback or preparation is None:
        return None
    if error is not None:
        from mandol_competition.inference.errors import InferenceIdentityError
        cause=error;seen=set();blocked=None
        while cause is not None and id(cause) not in seen and len(seen)<12:
            seen.add(id(cause))
            if isinstance(cause,InferenceIdentityError):blocked='identity'
            if getattr(cause,'status_code',None) in {400,401,403,404,405,413,415,422}:blocked='permanent_input_or_identity'
            cause=cause.__cause__ or cause.__context__
        if blocked:
            preparation.value.setdefault('source_fallback_blocked',{})[key]=blocked
            preparation.save()
    if key in preparation.value.get('source_fallback_blocked',{}):
        return None
    stored=preparation.value.get('source_only',{}).get(key)
    if stored is None:
        if (error is None or getattr(error,'error_class',None)!='vision_recovery_exhausted'
                or getattr(error,'unknown_outcome',True) is not False
                or not isinstance(error.__cause__,RecoveryBudgetExhausted)):
            return None
        # An exhausted allowance caused by unknown transport outcomes is not
        # confirmed model-content failure and cannot authorize successful Add.
        checkpoint=preparation.value.get('vision',{}).get(key,{}).get('stage_checkpoint') or {}
        if any(c.get('metadata',{}).get('error_class') in {
                'unverified_paid_outcome','dispatch_outcome_unknown_after_resume'}
                for c in checkpoint.get('calls',[])):
            return None
    result,receipt=_material(config,preparation,key,vision,inputs)
    record=dict(extraction=asdict(result),receipt=receipt)
    if stored is not None:
        if stored!=record:_fail()
    else:
        preparation.value.setdefault('source_only',{})[key]=record
        preparation.save()  # Publication cannot precede the bound durable decision.
    return result


def annotate(unit, records):
    if not records:return
    unit.metadata['visual_extraction_status']='source_only'
    unit.metadata['visual_source_only']=deepcopy(records)


def delivery_note(unit):
    rows=unit.metadata.get('visual_source_only')
    if not isinstance(rows,list) or not rows:return None
    values=[]
    from .vlm.base import Extraction
    try:
        images=unit.raw_data['multimodal_info']['images']
        seen=set()
        for row in rows:
            position=row['image_position']
            if (row.get('format')!=FORMAT or row.get('status')!='source_only'
                    or row.get('reason')!='scene_recovery_exhausted' or type(position) is not int
                    or not 0<=position<len(images) or position in seen
                    or images[position]['media_id']!=row['media_id']
                    or digest({key:images[position][key] for key in Extraction.__dataclass_fields__})!=row['extraction_sha256']):
                _fail()
            seen.add(position)
            values.append({key:row[key] for key in ('media_id','image_position','status','reason','verified_field_groups')})
    except (TypeError,ValueError,KeyError,IndexError):
        _fail()
    return '[Visual extraction status]\n'+json.dumps(dict(original_images_retained=True,
        extraction_complete=False,images=values),ensure_ascii=False,allow_nan=False,sort_keys=True)
