"""Reviewed OCR-only canonical preparation. No provider or serving configuration changes."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import re

from mandol_substrate.state import CanonicalMemory, Conflict, Scope
from mandol_substrate.state.models import digest
from .representation import POLICIES, memory_text
from .render import Location, RenderImage
from .unit_builder import build_unit_uid
from .visual_repair import source_fingerprint
from .vlm.cache import _from_payload
from mandol_substrate.representation import MemoryUnit

POLICY="reviewed-ocr-field-repair-v1"
PROVENANCE_POLICY="mixed-ocr-field-provenance-v1"
PLAN_FORMAT="reviewed-ocr-field-maintenance-plan-v1"
PREPARED_FORMAT="prepared-ocr-field-target-v1"
HISTORY_KEY="visual_ocr_field_repair_history"
FACTS=("description","ocr_text","entities","objects","events","relations","tags","location_hint")
IDENTITY=("model","prompt_version","schema_version")

def text_sha(value):
    if not isinstance(value,str):raise Conflict("OCR/source text must be a string")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def sha_ok(value):
    return isinstance(value,str) and re.fullmatch("[a-f0-9]{64}",value) is not None

def require(ok,message):
    if not ok:raise Conflict(message)

def retained_image_hashes(image):
    return {k:digest(v) for k,v in sorted(image.items()) if k!="ocr_text"}

def _has_full_receipt(value):
    if isinstance(value,dict):
        for key,child in value.items():
            if ("receipt" in key.casefold() or key.casefold() in {"output_validation","validation_receipt"}
                    or re.search(r"(?:extraction|output).*(?:sha|hash|digest|fingerprint)",key,re.I)
                    or _has_full_receipt(child)):
                return True
    elif isinstance(value,list):
        return any(_has_full_receipt(child) for child in value)
    return False

def _render(raw,metadata,policy,surrounding):
    images=[RenderImage(v["media_id"],_from_payload(v),
        Location(**v["location"]) if v.get("location") is not None else None)
        for v in raw["multimodal_info"]["images"]]
    return memory_text(policy_id=policy,dialogue=raw["original_content"],
        timestamp_text=metadata.get("timestamp"),images=images,surrounding_text=surrounding or "")

def _producer(payload):
    value=payload["provenance"]
    # Preserve an evidence hash, but do not place private diagnostic paths in
    # canonical records or public response attribution.
    names=("provider","engine","models","software_versions","model_sha256","candidate_source_sha256",
        "recipe_sha256","model","prompt_version","model_context_policy","source_context_sha256",
        "model_context_sha256","request_prompt_sha256","native_receipt_sha256",
        "algorithm_recipe_ref","execution_recipe_sha256","text_policy","not_gpt_full_extraction")
    out={k:deepcopy(value[k]) for k in names if k in value}
    if "algorithm_recipe_ref" in out:
        out["algorithm_recipe_sha256"]=out.pop("algorithm_recipe_ref")["sha256"]
    out["original_provenance_sha256"]=digest(value)
    return out

def prepare_ocr_repair(memory,scope,payload,*,payload_sha256,profile_hash,representation_policy,
                       cache_records=(),surrounding_text=None):
    require(isinstance(memory,CanonicalMemory) and isinstance(scope,Scope),"canonical source and scope required")
    require(payload.get("schema")=="ocr-field-repair-proposal-v1" and payload.get("field")=="ocr_text"
        and set(payload.get("replacement",{}))=={"ocr_text"},"only the reviewed OCR field proposal schema is accepted")
    require(sha_ok(payload_sha256) and digest(payload)==payload_sha256,"reviewed payload SHA mismatch")
    require(payload.get("guards",{}).get("whole_extraction_replacement") is False
        and payload["guards"].get("automatic_replacement") is False
        and payload["guards"].get("production_apply_authorized") is False,"frozen field-only candidate guards required")
    p=payload["preconditions"];images=memory.raw_data["multimodal_info"]["images"]
    position,count=p["image_position"],p["image_count"]
    require(scope.key==p["scope_key"] and asdict(scope)==p["scope"],"source scope mismatch")
    require(memory.logical_uid==p["source_uid"] and memory.version==p["source_version"]
        and memory.commit_seq==p["source_commit_seq"],"source UID/version/commit drift")
    require(memory.active and memory.lifecycle=="active" and memory.memory_level=="L0_Observation"
        and memory.memory_type=="observation","active canonical observation required")
    require(memory.metadata.get("user_id")==scope.user_id and
        build_unit_uid(scope.user_id,memory.metadata["request_id"],memory.metadata["message_index"])==memory.logical_uid,
        "source Add identity cannot be reconstructed")
    require(memory.text_content==memory.raw_data.get("text_content")
        and digest(asdict(memory))==p["canonical_snapshot_sha256"]
        and source_fingerprint(MemoryUnit(memory.logical_uid,memory.raw_data,memory.metadata))==p["source_snapshot_sha256"],
        "complete source canonical/fingerprint drift")
    require(profile_hash==p["profile_hash"] and sha_ok(profile_hash) and representation_policy in POLICIES,
        "source profile/representation mismatch")
    require(not POLICIES[representation_policy].surrounding or surrounding_text is not None,
        "exact original surrounding text required")
    require(type(position) is int and type(count) is int and 0<=position<count==len(images),
        "source image position/count drift")
    old=images[position]
    require(old["media_id"]==p["media_id"]=="img_"+p["image_sha256"]
        and digest(old)==p["image_payload_sha256"],"source image payload drift")
    require(text_sha(memory.raw_data["original_content"])==p["source_context_sha256"]
        and text_sha(old["ocr_text"])==p["old_ocr_sha256"]
        and digest({k:old.get(k) for k in FACTS})==p["old_semantic_sha256"],"source context/OCR/semantic drift")
    require(all(sha_ok(p[k]) for k in ("raw_data_sha256","metadata_sha256")),"exact SQL JSON SHA guards required")
    require(_render(memory.raw_data,memory.metadata,representation_policy,surrounding_text)==memory.text_content,
        "old representation cannot be reconstructed exactly")
    new=payload["replacement"]["ocr_text"]
    require(isinstance(new,str) and bool(new.strip()) and text_sha(new)==payload["replacement_ocr_sha256"]
        and text_sha(new)!=p["old_ocr_sha256"],"new OCR must be exact nonempty changed reviewed text")
    # Old full-result receipts must not be silently left attached to a hybrid.
    require(not _has_full_receipt(old),"old full-extraction receipt migration requires separate review")
    raw,metadata=deepcopy(memory.raw_data),deepcopy(memory.metadata)
    raw["multimodal_info"]["images"][position]["ocr_text"]=new
    require(retained_image_hashes(raw["multimodal_info"]["images"][position])==retained_image_hashes(old),
        "non-OCR image fields changed")
    component=dict(field="ocr_text",value_sha256=text_sha(new),old_value_sha256=p["old_ocr_sha256"],
        media_id=old["media_id"],image_position=position,image_count=count,
        source_context_sha256=p["source_context_sha256"],payload_sha256=payload_sha256,
        producer=_producer(payload),evidence_sha256={k:v["sha256"] for k,v in payload["evidence_refs"].items()})
    event=dict(policy=POLICY,provenance_policy=PROVENANCE_POLICY,source_uid=memory.logical_uid,
        source_scope=asdict(scope),source_scope_key=scope.key,source_version=memory.version,source_commit_seq=memory.commit_seq,
        source_fingerprint=p["source_snapshot_sha256"],source_canonical_sha256=p["canonical_snapshot_sha256"],
        old_image_sha256=digest(old),retained_image_field_sha256=retained_image_hashes(old),
        historical_identity={k:old[k] for k in IDENTITY},component=component,component_sha256=digest(component),
        full_extraction_receipt_validates_hybrid=False)
    history=metadata.setdefault(HISTORY_KEY,[])
    require(isinstance(history,list),"malformed OCR field history")
    history.append(event)
    raw["text_content"]=_render(raw,metadata,representation_policy,surrounding_text)
    target=replace(memory,raw_data=raw,metadata=metadata,text_content=raw["text_content"],version=memory.version+1)
    records=[]
    for value in cache_records:
        r=deepcopy(value);path=Path(r["path"])
        require(path.is_absolute() and path.suffix==".json" and sha_ok(r["sha256"]),"exact historical cache file required")
        require(r.get("media_id",old["media_id"])==old["media_id"],"cache media differs from source")
        r.update(media_id=old["media_id"],key=path.stem,marker_path=str(path)+".ocr-field-retired",
                 scope_key=scope.key,source_context_sha256=p["source_context_sha256"])
        require(sha_ok(r["key"]),"historical cache key must be exact SHA")
        records.append(r)
    require(records and len({r["path"] for r in records})==len(records),"nonempty unique exact cache refs required")
    value=dict(format=PREPARED_FORMAT,scope=asdict(scope),payload_sha256=payload_sha256,
        reviewed_payload=deepcopy(payload),source_preconditions=deepcopy(p),source_memory=asdict(memory),
        replacement=asdict(target),profile_hash=profile_hash,representation_policy=representation_policy,
        surrounding_text=surrounding_text,cache_records=records,component=component)
    value["prepared_sha256"]=digest(value)
    return value

def canonical_from_payload(value):
    value=deepcopy(value);value["spaces"]=tuple(value["spaces"])
    return CanonicalMemory(**value)

def prepared_from_payload(value):
    require(isinstance(value,dict) and value.get("format")==PREPARED_FORMAT,"prepared OCR target required")
    result=deepcopy(value);expected=result.pop("prepared_sha256",None)
    require(sha_ok(expected) and digest(result)==expected,"prepared target SHA mismatch")
    source=canonical_from_payload(result["source_memory"])
    rebuilt=prepare_ocr_repair(source,Scope(**result["scope"]),result["reviewed_payload"],
        payload_sha256=result["payload_sha256"],profile_hash=result["profile_hash"],
        representation_policy=result["representation_policy"],cache_records=result["cache_records"],
        surrounding_text=result["surrounding_text"])
    require(digest(rebuilt)==digest(value),"prepared OCR target differs from exact reviewed inputs")
    return deepcopy(value)

def prepare_ocr_plan(backend,materials,*,expected_payload_shas):
    expected=tuple(sorted(expected_payload_shas));materials=tuple(materials)
    require(len(expected)==len(set(expected))==len(materials)==5,"this maintenance plan requires exactly five reviewed payloads")
    require({m["payload_sha256"] for m in materials}==set(expected),"maintenance material whitelist mismatch")
    targets=[];seen=set()
    for material in materials:
        payload=material["payload"];p=payload["preconditions"];scope=Scope(**p["scope"])
        require(scope.namespace==backend.config.namespace,"maintenance namespace mismatch")
        source=backend.state.get_memories(scope,[p["source_uid"]]).get(p["source_uid"])
        require(source is not None,"selected source missing")
        rows=backend.state._rows("SELECT raw_data,metadata FROM memories WHERE scope=? AND logical_uid=?",(scope.key,source.logical_uid))
        require(len(rows)==1 and text_sha(rows[0]["raw_data"])==p["raw_data_sha256"]
            and text_sha(rows[0]["metadata"])==p["metadata_sha256"],"selected exact SQL JSON source drift")
        key=(scope.key,source.logical_uid)
        require(key not in seen,"duplicate source targets must be grouped");seen.add(key)
        prepared=prepare_ocr_repair(source,scope,payload,payload_sha256=material["payload_sha256"],
            profile_hash=backend.state.profile_hash,
            representation_policy=backend.profile.options.get("representation_policy","mm-representation-v1"),
            surrounding_text=material.get("surrounding_text"),cache_records=material["cache_records"])
        targets.append(prepared)
    return dict(format=PLAN_FORMAT,policy=POLICY,allowed_fields=["ocr_text"],count=5,
        payload_sha256s=list(expected),targets=sorted(targets,key=lambda p:(digest(p["scope"]),p["source_preconditions"]["source_uid"])))

