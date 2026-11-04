"""Owned SQLite/Qdrant OCR field maintenance; no real provider or official data."""
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timedelta,timezone
import hashlib,json,os
from pathlib import Path
import pytest

from mandol_substrate.state import Conflict,Scope,StateUnavailable
from mandol_substrate.state.models import digest,canonical_json
from mandol_substrate.representation import MemoryUnit
from mandol_competition.multimodal.service import AddService,SearchService,_cached_extractor
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor
from mandol_competition.multimodal.vlm.cache import _media_id
from mandol_competition.multimodal.visual_repair import source_fingerprint
from mandol_competition.multimodal.ocr_field_repair import (
    POLICY,HISTORY_KEY,FACTS,prepare_ocr_repair,prepared_from_payload,text_sha)
from mandol_competition.multimodal.ocr_field_maintenance import (
    HISTORICAL_REPLAY_HEAD,PERMIT_FORMAT,REQUIRED_GATES,prepare_ocr_plan,
    load_ocr_maintenance_capability,OcrFieldMaintenancePublisher,verify_prepared_current)
from mandol_competition.multimodal.ocr_cache_retirement import marker_path_for_cache_file
from mandol_competition.multimodal.ocr_provenance import ocr_source_provenance
from .backend_support import request,query
from .test_sqlite_qdrant import image_uri
from mandol_competition.multimodal.unit_builder import build_unit_uid

def payload_for(backend,memory,new_text):
    scope=backend.scope(memory.metadata["user_id"]);image=memory.raw_data["multimodal_info"]["images"][0]
    row=backend.state._rows("SELECT raw_data,metadata FROM memories WHERE scope=? AND logical_uid=?",(scope.key,memory.logical_uid))[0]
    p=dict(scope=asdict(scope),scope_key=scope.key,source_uid=memory.logical_uid,
        source_version=memory.version,source_commit_seq=memory.commit_seq,
        raw_data_sha256=text_sha(row["raw_data"]),metadata_sha256=text_sha(row["metadata"]),
        canonical_snapshot_sha256=digest(asdict(memory)),
        source_snapshot_sha256=source_fingerprint(MemoryUnit(memory.logical_uid,memory.raw_data,memory.metadata)),
        media_id=image["media_id"],image_sha256=image["media_id"][4:],image_position=0,image_count=1,
        image_payload_sha256=digest(image),source_context_sha256=text_sha(memory.raw_data["original_content"]),
        old_ocr_sha256=text_sha(image["ocr_text"]),old_semantic_sha256=digest({k:image[k] for k in FACTS}),
        profile_hash=backend.state.profile_hash)
    return dict(schema="ocr-field-repair-proposal-v1",field="ocr_text",preconditions=p,
        replacement={"ocr_text":new_text},replacement_ocr_sha256=text_sha(new_text),
        provenance=dict(provider="local-cpu-offline",engine="RapidOCR",models="owned synthetic OCR",
            software_versions={"fixture":"1"},model_sha256={"det":"a"*64},recipe_sha256="b"*64,
            text_policy="exact-lines",not_gpt_full_extraction=True),
        evidence_refs={"visual_review":{"sha256":"c"*64},"evidence":{"sha256":"d"*64}},
        guards=dict(whole_extraction_replacement=False,automatic_replacement=False,production_apply_authorized=False),
        quality_gate_passed=False,production_ready=False)

@pytest.fixture
def owned(backend,tmp_path):
    uri=image_uri();media=_media_id(uri)
    vision=FakeVisionExtractor({media:dict(description="Owned blue card",ocr_text="Old literal label",
        objects=[dict(name="card",count=1,attributes=["blue"])])},
        model="owned-mini",prompt_version=backend.config.prompt_version,schema_version="1")
    add=AddService(backend.config,extractor=vision,backend=backend)
    originals=[];materials=[];bodies=[];original_results=[]
    for i in range(5):
        context="Owned source context "+str(i)
        body=request(request_id="original-"+str(i),content=[
            {"type":"text","text":context},{"type":"image_url","image_url":{"url":uri}}])
        original_results.append(add.handle(body));bodies.append(body)
        uid=build_unit_uid("alice","original-"+str(i),0)
        memory=backend.state.get_memories(backend.scope("alice"),[uid])[uid]
        originals.append(memory)
        payload=payload_for(backend,memory,"UNIQUE_OCR_"+str(i)+"\nLiteral second line")
        wrapper=_cached_extractor(backend.config,"alice",vision)
        key=wrapper._key(media,context,0,1);path=wrapper._cache._path_for(media,key)
        cache=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),media_id=media)
        materials.append(dict(payload=payload,payload_sha256=digest(payload),cache_records=[cache]))
    plan=prepare_ocr_plan(backend,materials,expected_payload_shas=[m["payload_sha256"] for m in materials])
    now=datetime.now(timezone.utc)
    permit=dict(format=PERMIT_FORMAT,policy=POLICY,allowed_fields=["ocr_text"],live_head="e"*40,
        historical_replay_head=HISTORICAL_REPLAY_HEAD,plan_sha256=digest(plan),tested_candidate_sha256="f"*64,
        reader_deployment_manifest_sha256="a"*64,startup_guard_sha256="c"*64,db_uuid=backend.state.db_uuid,profile_hash=backend.state.profile_hash,
        qdrant_collection=backend.config.qdrant_collection,data_dir=str(backend.config.data_dir.resolve()),
        namespace=backend.config.namespace,operator_stop_confirmed_at_utc=(now-timedelta(seconds=2)).isoformat(),
        issued_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(minutes=5)).isoformat(),
        allowed_targets=[dict(scope_key=t["source_preconditions"]["scope_key"],
            logical_uid=t["source_preconditions"]["source_uid"],payload_sha256=t["payload_sha256"],
            prepared_sha256=t["prepared_sha256"]) for t in plan["targets"]],allow_compensation=True)
    permit_path=tmp_path/"permit.json";permit_path.write_text(json.dumps(permit));permit_path.chmod(0o600)
    phases=[];gate={"idle":True}
    def validator(backend,value,*,phase):
        phases.append(phase)
        evidence=dict(**{k:True for k in REQUIRED_GATES},
            mm_main_pid=0,official_gate_age_seconds=.01,cancelled=False,db_uuid=backend.state.db_uuid,
            profile_hash=backend.state.profile_hash,qdrant_collection=backend.config.qdrant_collection,
            data_dir=str(backend.config.data_dir.resolve()),backup_manifest_sha256="b"*64,
            ocr_reader_installed=True,ocr_reader_manifest_sha256=permit["reader_deployment_manifest_sha256"],
            ocr_startup_guard_verified=True,ocr_startup_guard_sha256=permit["startup_guard_sha256"])
        evidence["official_three_track_idle"]=gate["idle"]
        return evidence
    cap=load_ocr_maintenance_capability(permit_path,plan_sha256=digest(plan),tested_candidate_sha256="f"*64,
        backend=backend,validate_external_gates=validator)
    return dict(backend=backend,vision=vision,add=add,bodies=bodies,materials=materials,
        originals=originals,original_results=original_results,plan=plan,permit=permit,permit_path=permit_path,cap=cap,
        publisher=OcrFieldMaintenancePublisher(backend,cap),phases=phases,gate=gate)

def target(owned,index=0):
    return next(p for p in owned["plan"]["targets"] if
        p["source_memory"]["request_id"]=="original-"+str(index))

def no_start(*a,**kw):pytest.fail("offline maintenance must never call backend.start()")

def test_preparation_preserves_original_identity_unknown_fields_and_all_other_facts(owned):
    p=target(owned);old=p["source_memory"];new=p["replacement"]
    assert set(new["metadata"])-set(old["metadata"])=={HISTORY_KEY}
    assert new["raw_data"]["original_content"]==old["raw_data"]["original_content"]
    assert new["raw_data"]["content_parts"]==old["raw_data"]["content_parts"]
    before=deepcopy(old["raw_data"]["multimodal_info"]["images"])
    after=deepcopy(new["raw_data"]["multimodal_info"]["images"])
    after[0]["ocr_text"]=before[0]["ocr_text"];assert before==after
    assert new["logical_uid"]==old["logical_uid"] and new["request_id"]==old["request_id"]
    assert "UNIQUE_OCR_0" in new["text_content"]
    serialized=json.loads(canonical_json(p));assert digest(prepared_from_payload(serialized))==digest(p)

def test_apply_replay_original_add_and_cache_retirement_are_targeted_and_no_start(owned,monkeypatch):
    o=owned;p=target(o);backend=o["backend"];before_calls=o["vision"].calls
    old_cache=Path(p["cache_records"][0]["path"]);original_bytes=old_cache.read_bytes()
    before_version=p["source_memory"]["version"]
    with monkeypatch.context() as m:
        m.setattr(backend,"start",no_start)
        result=o["publisher"].apply(p)
        assert o["publisher"].resume_pending_only(p)==result
    current=backend.state.get_memories(Scope(**p["scope"]),[p["source_preconditions"]["source_uid"]])[p["source_preconditions"]["source_uid"]]
    assert current.version==before_version+1 and result["markers_finalized"]
    assert old_cache.read_bytes()==original_bytes and marker_path_for_cache_file(old_cache).is_file()
    cache=_cached_extractor(backend.config,"alice",o["vision"])._cache
    assert cache.get(p["cache_records"][0]["media_id"],p["cache_records"][0]["key"]) is None
    assert len(cache)==5 # retirement markers never count as cache JSON records
    assert o["add"].handle(o["bodies"][0])==o["original_results"][0]
    assert o["vision"].calls==before_calls
    verified=verify_prepared_current(backend,p,journal_root=o["cap"].journal_root)
    assert verified==result and "before_t1" in o["phases"]
    result=SearchService(backend.config,extractor=o["vision"],backend=backend).handle(query(text="UNIQUE_OCR_0"))
    hit=next(v for v in result["data"] if v["id"]==current.logical_uid)
    assert any("[OCR field provenance]" in v.get("text","") for v in hit["content"])

@pytest.mark.parametrize("phase",["after_T1","before_T2","after_T2_before_http_return","after_ocr_cache_marker","after_ocr_cache_finalize"])
def test_committed_intent_or_t2_only_marker_recovery_never_creates_another_t1(owned,monkeypatch,phase):
    o=owned;p=target(o);fired=[]
    def fault(point):
        if point==phase and not fired:
            fired.append(point);raise RuntimeError("owned interrupted maintenance")
    o["backend"].coordinator.fault=fault;o["backend"].fault=fault
    count=lambda:len(o["backend"].state._rows("SELECT * FROM mutations"))
    before=count()
    with monkeypatch.context() as m:
        m.setattr(o["backend"],"start",no_start)
        with pytest.raises(RuntimeError):o["publisher"].apply(p)
        assert count()==before+1
        o["gate"]["idle"]=False # only an already committed intent may recover
        result=o["publisher"].resume_pending_only(p)
        assert result["markers_finalized"] and result["status"]=="complete"
    assert count()==before+1

def test_resume_cannot_create_an_uncommitted_new_target(owned):
    o=owned;o["gate"]["idle"]=False
    with pytest.raises(StateUnavailable):o["publisher"].resume_pending_only(target(o))
    assert o["backend"].state.request(Scope(**target(o)["scope"]),
        "ocr-field-repair:"+target(o)["prepared_sha256"]) is None

def test_new_scope_add_does_not_invalidate_unchanged_selected_source(owned):
    o=owned;p=target(o)
    o["add"].handle(request(request_id="later-add",content="A later legitimate source"))
    result=o["publisher"].apply(p)
    assert result["markers_finalized"]
    assert any(m.request_id=="later-add" for batch in o["backend"].state.iter_memories(o["backend"].scope("alice")) for m in batch)

def test_exact_raw_json_cas_catches_drift_after_before_prepare(owned,monkeypatch):
    o=owned;p=target(o);state=o["backend"].state;original=state.prepare
    def race(scope,request_id,payload_digest,mutation,result,**kwargs):
        with state.transaction() as db:
            db.execute("UPDATE memories SET raw_data=raw_data||' ' WHERE scope=? AND logical_uid=?",
                (scope.key,p["source_preconditions"]["source_uid"]))
        return original(scope,request_id,payload_digest,mutation,result,**kwargs)
    monkeypatch.setattr(state,"prepare",race)
    before=len(state._rows("SELECT * FROM mutations"))
    with pytest.raises(Conflict):o["publisher"].apply(p)
    assert len(state._rows("SELECT * FROM mutations"))==before
    assert not marker_path_for_cache_file(p["cache_records"][0]["path"]).exists()

def test_successful_compensation_preserves_later_adds_and_never_calls_start(owned,monkeypatch):
    o=owned;p=target(o);result=o["publisher"].apply(p)
    o["add"].handle(request(request_id="new-unrelated",content="Preserve this newer Add"))
    frontier=o["backend"].state.scope_state(Scope(**p["scope"]))["published_seq"]
    with monkeypatch.context() as m:
        m.setattr(o["backend"],"start",no_start)
        restored=o["publisher"].compensate(p,expected_current_sha256=result["canonical_sha256"],expected_published_seq=frontier)
    assert restored["status"]=="compensated"
    current=o["backend"].state.get_memories(Scope(**p["scope"]),[p["source_preconditions"]["source_uid"]])[p["source_preconditions"]["source_uid"]]
    assert current.raw_data==p["source_memory"]["raw_data"] and current.metadata==p["source_memory"]["metadata"]
    assert current.version==p["source_memory"]["version"]+2
    assert not marker_path_for_cache_file(p["cache_records"][0]["path"]).exists()
    assert any(m.request_id=="new-unrelated" for batch in o["backend"].state.iter_memories(Scope(**p["scope"])) for m in batch)

def test_later_fresh_same_key_cache_is_never_overwritten_or_retired(owned):
    o=owned;p=target(o);r=p["cache_records"][0];result=o["publisher"].apply(p)
    cache=_cached_extractor(o["backend"].config,"alice",o["vision"])._cache
    old=cache._read_record(r["media_id"],r["key"])
    from mandol_competition.multimodal.vlm.cache import _from_payload
    fresh=replace(_from_payload(old["extraction"]),ocr_text="NEW_LATER_CACHE")
    cache.put(r["media_id"],r["key"],fresh);fresh_bytes=Path(r["path"]).read_bytes()
    assert cache.get(r["media_id"],r["key"]).ocr_text=="NEW_LATER_CACHE"
    scope=Scope(**p["scope"])
    o["publisher"].compensate(p,expected_current_sha256=result["canonical_sha256"],
        expected_published_seq=o["backend"].state.scope_state(scope)["published_seq"])
    assert Path(r["path"]).read_bytes()==fresh_bytes
    assert cache.get(r["media_id"],r["key"]).ocr_text=="NEW_LATER_CACHE"

def test_corrupted_or_foreign_cache_marker_fails_closed_and_prevents_compensation_t1(owned):
    o=owned;p=target(o);result=o["publisher"].apply(p);r=p["cache_records"][0]
    marker=marker_path_for_cache_file(r["path"]);marker.write_text("{}");marker.chmod(0o600)
    cache=_cached_extractor(o["backend"].config,"alice",o["vision"])._cache
    with pytest.raises(StateUnavailable):cache.get_result(r["media_id"],r["key"])
    before=len(o["backend"].state._rows("SELECT * FROM mutations"))
    with pytest.raises(StateUnavailable):
        o["publisher"].compensate(p,expected_current_sha256=result["canonical_sha256"],
            expected_published_seq=o["backend"].state.scope_state(Scope(**p["scope"]))["published_seq"])
    assert len(o["backend"].state._rows("SELECT * FROM mutations"))==before

@pytest.mark.parametrize("change",["ocr","description","producer","context","identity","scope"])
def test_mixed_provenance_rejects_binding_and_retained_field_tamper(owned,change):
    p=target(owned);value=deepcopy(p["replacement"]);images=value["raw_data"]["multimodal_info"]["images"]
    event=value["metadata"][HISTORY_KEY][-1]
    if change=="ocr":images[0]["ocr_text"]+="x"
    if change=="description":images[0]["description"]+="x"
    if change=="producer":event["component"]["producer"]["engine"]="unreviewed"
    if change=="context":value["raw_data"]["original_content"]+="x"
    if change=="identity":images[0]["model"]="different"
    if change=="scope":event["source_scope"]["namespace"]="foreign"
    with pytest.raises(StateUnavailable):
        ocr_source_provenance(MemoryUnit(value["logical_uid"],value["raw_data"],value["metadata"]),namespace="multimodal")

def test_unapproved_or_partial_other_field_payloads_cannot_be_prepared(owned):
    o=owned;material=deepcopy(o["materials"][0])
    material["payload"]["replacement"]["description"]="not authorized"
    with pytest.raises(Conflict):
        prepare_ocr_plan(o["backend"],[material]+o["materials"][1:],
            expected_payload_shas=[m["payload_sha256"] for m in o["materials"]])

@pytest.mark.parametrize("kind",["file","broken_symlink"])
def test_pending_startup_guard_precedes_any_provider_recovery_schema_work(backend,monkeypatch,tmp_path,kind):
    from types import SimpleNamespace
    marker=backend.config.data_dir/".ocr-field-maintenance.pending"
    if kind=="file":
        marker.write_text("owned incomplete maintenance");marker.chmod(0o600)
    else:marker.symlink_to(tmp_path/"does-not-exist")
    backend._started=False
    monkeypatch.setattr(backend.encoders,"dense",SimpleNamespace(validate_profile=no_start),raising=False)
    monkeypatch.setattr(backend.coordinator,"recover",no_start)
    monkeypatch.setattr(backend.projection,"ensure_schema",no_start)
    with pytest.raises(StateUnavailable):backend.start()
    with backend.maintenance():
        with backend._operations.lease(backend.scope("alice"),write=True):
            assert not backend._started
    marker.unlink()

def test_absent_startup_guard_preserves_normal_start_sequence(backend,monkeypatch):
    from types import SimpleNamespace
    from mandol_competition.multimodal import backend as module
    seen=[];backend._started=False
    monkeypatch.setattr(backend.encoders,"dense",SimpleNamespace(validate_profile=lambda:seen.append("validate")),raising=False)
    monkeypatch.setattr(backend.coordinator,"recover",lambda:seen.append("recover"))
    monkeypatch.setattr(backend.projection,"ensure_schema",lambda *a,**kw:seen.append("schema"))
    monkeypatch.setattr(module,"verify_projection",lambda *a,**kw:seen.append("verify") or {"servable":True})
    backend.start()
    assert seen==["validate","recover","schema","verify"] and backend._started

def test_offline_maintenance_still_operates_with_startup_guard_present(owned,monkeypatch):
    o=owned;p=target(o);marker=o["backend"].config.data_dir/".ocr-field-maintenance.pending"
    marker.write_text("owned root startup barrier");marker.chmod(0o600)
    with monkeypatch.context() as m:
        m.setattr(o["backend"],"start",no_start)
        assert o["publisher"].apply(p)["markers_finalized"]
        assert o["publisher"].resume_pending_only(p)["markers_finalized"]
    with pytest.raises(StateUnavailable):o["backend"].start()
    marker.unlink()

@pytest.mark.parametrize("change",["no_callback","active","old_get","nan_age","reader_missing","guard_missing","resource_wrong"])
def test_capability_requires_fresh_stopped_reader_guard_resource_and_backup(owned,change):
    o=owned;backend=o["backend"]
    def validator(backend,value,*,phase):
        e=dict(**{k:True for k in REQUIRED_GATES},mm_main_pid=0,official_gate_age_seconds=.1,cancelled=False,
            db_uuid=backend.state.db_uuid,profile_hash=backend.state.profile_hash,
            data_dir=str(backend.config.data_dir.resolve()),qdrant_collection=backend.config.qdrant_collection,
            backup_manifest_sha256="b"*64,ocr_reader_installed=True,
            ocr_reader_manifest_sha256=o["permit"]["reader_deployment_manifest_sha256"],
            ocr_startup_guard_verified=True,ocr_startup_guard_sha256=o["permit"]["startup_guard_sha256"])
        if change=="active":e["official_three_track_idle"]=False
        if change=="old_get":e["official_gate_age_seconds"]=11
        if change=="nan_age":e["official_gate_age_seconds"]=float("nan")
        if change=="reader_missing":e["ocr_reader_installed"]=False
        if change=="guard_missing":e["ocr_startup_guard_verified"]=False
        if change=="resource_wrong":e["db_uuid"]="other"
        return e
    with pytest.raises(StateUnavailable):
        load_ocr_maintenance_capability(o["permit_path"],plan_sha256=digest(o["plan"]),
            tested_candidate_sha256="f"*64,backend=backend,
            validate_external_gates=None if change=="no_callback" else validator)

def test_retained_unknown_field_and_original_receipt_migration_guard(owned):
    o=owned;source=deepcopy(o["originals"][0]);image=source.raw_data["multimodal_info"]["images"][0]
    image["unknown_retained_metadata"]={"label":"kept"}
    # Make an owned immutable source fixture with a matching complete source SHA.
    payload=payload_for(o["backend"],source,"Reviewed owned new OCR")
    cache=o["materials"][0]["cache_records"]
    prepared=prepare_ocr_repair(source,o["backend"].scope("alice"),payload,payload_sha256=digest(payload),
        profile_hash=o["backend"].state.profile_hash,representation_policy="mm-representation-v1",cache_records=cache)
    assert prepared["replacement"]["raw_data"]["multimodal_info"]["images"][0]["unknown_retained_metadata"]=={"label":"kept"}
    image["output_validation"]={"claimed_full_receipt":True}
    payload=payload_for(o["backend"],source,"Reviewed owned new OCR")
    with pytest.raises(Conflict):
        prepare_ocr_repair(source,o["backend"].scope("alice"),payload,payload_sha256=digest(payload),
            profile_hash=o["backend"].state.profile_hash,representation_policy="mm-representation-v1",cache_records=cache)
    image.pop("output_validation")
    image["nested_old_metadata"]={"full_extraction_receipt":{"old":True}}
    payload=payload_for(o["backend"],source,"Reviewed owned new OCR")
    with pytest.raises(Conflict):
        prepare_ocr_repair(source,o["backend"].scope("alice"),payload,payload_sha256=digest(payload),
            profile_hash=o["backend"].state.profile_hash,representation_policy="mm-representation-v1",cache_records=cache)


