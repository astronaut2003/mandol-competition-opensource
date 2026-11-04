"""Scoped offline OCR maintenance. A root controller owns stop, idle, install and backups."""
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib,json,math,os,re
from pathlib import Path

from mandol_substrate.state import Conflict,Mutation,Scope,StateUnavailable
from mandol_substrate.state.models import canonical_json,digest
from mandol_substrate.vector.codec import point_id
from mandol_substrate.representation import MemoryUnit
from .ocr_field_repair import (POLICY,PLAN_FORMAT,HISTORY_KEY,canonical_from_payload,prepared_from_payload,
    prepare_ocr_plan,sha_ok,text_sha)
from .ocr_cache_retirement import (atomic_private_bytes,marker_path_for_cache_file,marker_value,
    persist_marker,read_marker,remove_exact_marker)
from .ocr_provenance import ocr_source_provenance

_prepare_plan = prepare_ocr_plan

HISTORICAL_REPLAY_HEAD="d46f62e4160711797800a9d8b8571eb743749229"
PERMIT_FORMAT="mandol-ocr-field-maintenance-permit-v1"
REQUIRED_GATES=("operator_stop_verified","official_three_track_idle","mm_service_inactive",
    "protected_source_config_match","backup_manifest_verified")
_CAPABILITY=object()

def _need(ok,message):
    if not ok:raise StateUnavailable(message)

def _private(path,directory=False):
    path=Path(path).absolute()
    _need(not any(p.is_symlink() for p in (path,*path.parents)),"maintenance path contains symlink")
    s=path.stat()
    _need(s.st_uid==os.geteuid() and not s.st_mode&0o077 and
        (path.is_dir() if directory else path.is_file()),"private owned maintenance artifact required")
    return path

def _utc(value):
    try:result=datetime.fromisoformat(value.replace("Z","+00:00"))
    except (ValueError,TypeError,AttributeError):raise StateUnavailable("explicit UTC maintenance time required") from None
    _need(result.tzinfo is not None and result.utcoffset().total_seconds()==0,"UTC maintenance time required")
    return result

def _permit(path,expected=None,recovery_only=False):
    path=_private(path);_private(path.parent,directory=True)
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();value=json.loads(raw)
    _need(expected is None or sha==expected,"OCR maintenance permit changed")
    _need(value.get("format")==PERMIT_FORMAT and value.get("policy")==POLICY and
        value.get("allowed_fields")==["ocr_text"] and value.get("historical_replay_head")==HISTORICAL_REPLAY_HEAD and
        isinstance(value.get("live_head"),str) and re.fullmatch("[a-f0-9]{40}",value["live_head"]) and
        value["live_head"]!=HISTORICAL_REPLAY_HEAD and sha_ok(value.get("reader_deployment_manifest_sha256")) and
        sha_ok(value.get("startup_guard_sha256")),
        "a distinct installed-reader OCR maintenance permit is required")
    now=datetime.now(timezone.utc);issued=_utc(value.get("issued_at_utc"));expires=_utc(value.get("expires_at_utc"))
    stopped=_utc(value.get("operator_stop_confirmed_at_utc"))
    _need(stopped<=issued<=now and issued<expires and (expires-issued).total_seconds()<=900 and
        (recovery_only or now<expires),"OCR maintenance authorization expired or invalid")
    return path,sha,value

def _external(backend,validator,permit,*,phase,recovery_only=False):
    e=validator(backend,permit,phase=phase)
    required=tuple(k for k in REQUIRED_GATES if not(recovery_only and k=="official_three_track_idle"))
    _need(isinstance(e,dict) and all(e.get(k) is True for k in required) and
        type(e.get("mm_main_pid")) is int and e["mm_main_pid"]==0 and
        e.get("ocr_reader_installed") is True and e.get("ocr_reader_manifest_sha256")==permit["reader_deployment_manifest_sha256"] and
        e.get("ocr_startup_guard_verified") is True and e.get("ocr_startup_guard_sha256")==permit["startup_guard_sha256"],
        "fresh stopped-resource, installed-reader and backup gates required")
    if not recovery_only:
        age=e.get("official_gate_age_seconds")
        _need(type(age) in (int,float) and math.isfinite(age) and 0<=age<=10 and e.get("cancelled") is False,
            "a fresh official GET idle gate is required")
    _need(e.get("db_uuid")==backend.state.db_uuid and e.get("profile_hash")==backend.state.profile_hash and
        e.get("qdrant_collection")==backend.config.qdrant_collection and
        Path(e.get("data_dir","")).resolve()==backend.config.data_dir.resolve() and
        sha_ok(e.get("backup_manifest_sha256")),"maintenance resource/backup mismatch")
    return e

@dataclass(frozen=True)
class OcrMaintenanceCapability:
    permit_path:Path
    permit_sha256:str
    journal_root:Path
    plan_sha256:str
    tested_candidate_sha256:str
    db_uuid:str
    profile_hash:str
    collection:str
    data_dir:Path
    namespace:str
    targets:dict
    validator:object
    capability:object
    def __reduce__(self):raise TypeError("OCR maintenance capabilities cannot be serialized")

def load_ocr_maintenance_capability(permit_path,*,plan_sha256,tested_candidate_sha256,backend,validate_external_gates):
    _need(callable(validate_external_gates),"root external maintenance validator required")
    path,sha,value=_permit(permit_path)
    _need(sha_ok(plan_sha256) and sha_ok(tested_candidate_sha256) and value.get("plan_sha256")==plan_sha256 and
        value.get("tested_candidate_sha256")==tested_candidate_sha256,"tested OCR plan/source mismatch")
    _need(value.get("db_uuid")==backend.state.db_uuid and value.get("profile_hash")==backend.state.profile_hash and
        value.get("qdrant_collection")==backend.config.qdrant_collection and value.get("namespace")==backend.config.namespace and
        Path(value.get("data_dir","")).resolve()==backend.config.data_dir.resolve(),"OCR permit resource mismatch")
    targets={}
    for t in value.get("allowed_targets",[]):
        _need(sha_ok(t.get("scope_key")) and isinstance(t.get("logical_uid"),str) and bool(t["logical_uid"]) and
            sha_ok(t.get("payload_sha256")) and sha_ok(t.get("prepared_sha256")),"exact reviewed OCR target required")
        key=t["scope_key"]+":"+t["logical_uid"]
        _need(key not in targets,"duplicate OCR authorized target");targets[key]=t
    _need(len(targets)==5,"this OCR authorization is limited to the exact five reviewed targets")
    _external(backend,validate_external_gates,value,phase="capability_issue")
    journal=path.parent/"ocr-field-preimages";journal.mkdir(mode=0o700,exist_ok=True);_private(journal,True)
    return OcrMaintenanceCapability(path,sha,journal,plan_sha256,tested_candidate_sha256,backend.state.db_uuid,
        backend.state.profile_hash,backend.config.qdrant_collection,backend.config.data_dir.resolve(),
        backend.config.namespace,targets,validate_external_gates,_CAPABILITY)

def _validate(backend,cap,p,*,phase="authority_check",recovery_only=False):
    _need(isinstance(cap,OcrMaintenanceCapability) and cap.capability is _CAPABILITY,"explicit OCR root capability required")
    _need(backend.state.db_uuid==cap.db_uuid and backend.state.profile_hash==cap.profile_hash and
        backend.config.data_dir.resolve()==cap.data_dir and backend.config.qdrant_collection==cap.collection and
        backend.config.namespace==cap.namespace,"OCR capability resource drift")
    _private(cap.journal_root,True);_,_,permit=_permit(cap.permit_path,cap.permit_sha256,recovery_only)
    _need(permit["plan_sha256"]==cap.plan_sha256 and permit["tested_candidate_sha256"]==cap.tested_candidate_sha256,
        "OCR permit plan drift")
    _need(permit.get("db_uuid")==cap.db_uuid and permit.get("profile_hash")==cap.profile_hash and
        permit.get("qdrant_collection")==cap.collection and permit.get("namespace")==cap.namespace and
        Path(permit.get("data_dir","")).resolve()==cap.data_dir,"OCR permit resource drift")
    scope=Scope(**p["scope"]);key=scope.key+":"+p["source_preconditions"]["source_uid"]
    expected=dict(scope_key=scope.key,logical_uid=p["source_preconditions"]["source_uid"],
        payload_sha256=p["payload_sha256"],prepared_sha256=p["prepared_sha256"])
    _need(cap.targets.get(key)==expected and expected in permit.get("allowed_targets",[]),
        "OCR target is outside explicit five-source authority")
    _external(backend,cap.validator,permit,phase=phase,recovery_only=recovery_only)
    return permit

def _persist(path,payload):
    payload=json.loads(canonical_json(payload));value=dict(sha256=digest(payload),payload=payload)
    atomic_private_bytes(path,canonical_json(value).encode())
    return value["sha256"]

def _saved(path,p,backend):
    value=json.loads(_private(path).read_text());body=value["payload"]
    _need(digest(body)==value["sha256"] and body.get("prepared_sha256")==p["prepared_sha256"] and
        body.get("db_uuid")==backend.state.db_uuid and body.get("profile_hash")==backend.state.profile_hash and
        body.get("scope")==p["scope"] and body.get("canonical_sha256")==p["source_preconditions"]["canonical_snapshot_sha256"] and
        digest(body.get("prepared_payload"))==digest(p),"OCR preimage binding mismatch")
    return body

def _sql_hashes(state,scope,uid):
    rows=state._rows("SELECT raw_data,metadata FROM memories WHERE scope=? AND logical_uid=?",(scope.key,uid))
    _need(len(rows)==1,"selected OCR canonical row missing")
    return text_sha(rows[0]["raw_data"]),text_sha(rows[0]["metadata"])

def _source_check(backend,p):
    scope=Scope(**p["scope"]);before=p["source_preconditions"]
    current=backend.state.get_memories(scope,[before["source_uid"]]).get(before["source_uid"])
    _need(current is not None and digest(asdict(current))==before["canonical_snapshot_sha256"] and
        _sql_hashes(backend.state,scope,current.logical_uid)==(before["raw_data_sha256"],before["metadata_sha256"]),
        "selected OCR source changed since independent review")
    frontier=backend.state.scope_state(scope)
    _need(frontier["canonical_seq"]==frontier["lexical_seq"]==frontier["published_seq"],"selected scope has unpublished work")
    return scope,current,frontier

def _cache_preflight(backend,p):
    scope=Scope(**p["scope"]);allowed=backend.config.user_root(scope.user_id)/"vlm_cache"
    original=p["source_memory"]["raw_data"]["multimodal_info"]["images"][p["source_preconditions"]["image_position"]]
    for r in p["cache_records"]:
        path=Path(r["path"])
        _need(path.resolve().is_relative_to(allowed.resolve()) and r["scope_key"]==scope.key and
            r["media_id"]==p["source_preconditions"]["media_id"] and r["key"]==path.stem and
            r["marker_path"]==str(marker_path_for_cache_file(path)),"cache retirement scope/path mismatch")
        body=_private(path).read_bytes()
        _need(hashlib.sha256(body).hexdigest()==r["sha256"],"historical cache bytes changed before maintenance")
        record=json.loads(body)
        _need(record.get("media_id")==r["media_id"] and
            all(record["extraction"].get(k)==original.get(k) for k in
                ("description","ocr_text","entities","objects","events","relations","tags","location_hint","model","prompt_version","schema_version")),
            "historical cache is not the exact retained source extraction")
        _need(not marker_path_for_cache_file(path).exists() and not marker_path_for_cache_file(path).is_symlink(),
            "an earlier cache retirement needs separate review")

def _request_id(p):return "ocr-field-repair:"+p["prepared_sha256"]

def prepare_ocr_plan(backend,materials,*,expected_payload_shas):
    plan=_prepare_plan(backend,materials,expected_payload_shas=expected_payload_shas)
    for p in plan["targets"]:
        _source_check(backend,p)
        _cache_preflight(backend,p)
    return plan

def _after_memory(backend,p,request,*,replacement=None):
    scope=Scope(**p["scope"]);seq=request["mutation_seq"]
    rows=backend.state._rows("SELECT created_at_ms FROM mutations WHERE scope=? AND mutation_seq=?",(scope.key,seq))
    _need(len(rows)==1,"OCR publication mutation record missing")
    target=canonical_from_payload(p["replacement"]) if replacement is None else replacement
    return replace(target,commit_seq=seq,updated_at_ms=rows[0]["created_at_ms"],
        created_at_ms=p["source_memory"]["created_at_ms"])

def verify_prepared_current(backend,prepared,*,journal_root=None,require_cache_finalized=True):
    p=prepared_from_payload(prepared);scope=Scope(**p["scope"]);uid=p["source_preconditions"]["source_uid"]
    request=backend.state.request(scope,_request_id(p))
    _need(request is not None and request["payload_digest"]==p["prepared_sha256"] and request["status"]=="success",
        "OCR target has not completed its exact T2")
    expected=_after_memory(backend,p,request);current=backend.state.get_memories(scope,[uid]).get(uid)
    _need(current is not None and digest(asdict(current))==digest(asdict(expected)),"OCR postimage changed or differs from reviewed field-only replacement")
    raw_sha,meta_sha=_sql_hashes(backend.state,scope,uid)
    _need(raw_sha==text_sha(canonical_json(expected.raw_data)) and meta_sha==text_sha(canonical_json(expected.metadata)),
        "OCR postimage SQL bytes differ from prepared replacement")
    row=backend.projection.get_projection_metadata(scope,[uid],vectors=True).get(uid)
    _need(row is not None and backend.projection.codec.matches(row,backend.projection.codec.metadata(scope,current)),
        "OCR dense/sparse published revision mismatch")
    ocr_source_provenance(MemoryUnit(current.logical_uid,current.raw_data,current.metadata),namespace=scope.namespace)
    markers=[]
    after_sha=digest(asdict(current))
    for r in p["cache_records"]:
        marker=read_marker(r["path"],r["media_id"],r["key"])
        expected_marker=marker_value(r,prepared_sha256=p["prepared_sha256"],payload_sha256=p["payload_sha256"],
            after_canonical_sha256=after_sha)
        if require_cache_finalized:_need(marker==expected_marker,"OCR cache retirement finalization incomplete")
        if marker is not None:
            _need(marker==expected_marker,"OCR retirement marker belongs to another postimage")
            mp=marker_path_for_cache_file(r["path"]);markers.append(dict(path=str(mp),sha256=hashlib.sha256(mp.read_bytes()).hexdigest()))
    result=dict(status="complete" if len(markers)==len(p["cache_records"]) else "cache_finalize_pending",
        markers_finalized=len(markers)==len(p["cache_records"]),policy=POLICY,logical_uid=uid,
        payload_sha256=p["payload_sha256"],prepared_sha256=p["prepared_sha256"],mutation_seq=request["mutation_seq"],
        canonical_sha256=after_sha,raw_data_sha256=raw_sha,metadata_sha256=meta_sha,cache_markers=markers)
    if journal_root is not None and require_cache_finalized:
        final=json.loads(_private(Path(journal_root)/(p["prepared_sha256"]+".finalized")).read_text())
        _need(digest(final["payload"])==final["sha256"] and final["payload"]==result,"OCR finalization journal differs")
    return result

class OcrFieldMaintenancePublisher:
    def __init__(self,backend,cap):
        _need(isinstance(cap,OcrMaintenanceCapability) and cap.capability is _CAPABILITY,"OCR maintenance capability required")
        self.backend,self.cap=backend,cap

    @contextmanager
    def _operation(self,scope):
        # user_operation() calls backend.start(), which globally recovers other
        # Add intents. Offline maintenance must never take that serving path.
        with self.backend.maintenance():
            with self.backend._operations.lease(scope,write=True):
                self.backend.state.assert_not_purging(scope)
                yield

    def _path(self,p):return self.cap.journal_root/(p["prepared_sha256"]+".json")

    def apply(self,prepared):
        p=prepared_from_payload(prepared);_validate(self.backend,self.cap,p)
        scope=Scope(**p["scope"]);uid=p["source_preconditions"]["source_uid"]
        existing=self.backend.state.request(scope,_request_id(p))
        if existing is not None:return self.resume_pending_only(p)
        _,_,frontier=_source_check(self.backend,p)
        guard=dict(published_seq=frontier["published_seq"],profile_hash=p["profile_hash"],
            memories={uid:dict(version=p["source_preconditions"]["source_version"],
                commit_seq=p["source_preconditions"]["source_commit_seq"],
                canonical_sha256=p["source_preconditions"]["canonical_snapshot_sha256"],
                raw_data_sha256=p["source_preconditions"]["raw_data_sha256"],
                metadata_sha256=p["source_preconditions"]["metadata_sha256"])})
        target=canonical_from_payload(p["replacement"])
        def before():
            _source_check(self.backend,p);_cache_preflight(self.backend,p)
            saved=snapshot_record(self.backend.state,self.backend.projection,scope,uid,replacement=target)
            saved.update(prepared_sha256=p["prepared_sha256"],prepared_payload=p,
                exact_cache_preimages=p["cache_records"],restoration="guarded compensating publication only")
            _persist(self._path(p),saved)
            _validate(self.backend,self.cap,p,phase="before_t1")
        result=dict(policy=POLICY,logical_uid=uid,prepared_sha256=p["prepared_sha256"],
            prior_version=p["source_preconditions"]["source_version"],version=target.version)
        with self._operation(scope):
            self.backend.coordinator.apply(scope,_request_id(p),p["prepared_sha256"],
                Mutation(memories=(target,),kind=POLICY),result,expected_publication=guard,before_prepare=before)
        return self.finalize_cache_retirements(p)

    def resume_pending_only(self,prepared):
        p=prepared_from_payload(prepared);_validate(self.backend,self.cap,p,recovery_only=True,phase="pending_t2_recovery")
        scope=Scope(**p["scope"]);request=self.backend.state.request(scope,_request_id(p))
        _need(request is not None and request["payload_digest"]==p["prepared_sha256"],"resume cannot create a new OCR intent")
        _saved(self._path(p),p,self.backend)
        with self._operation(scope):
            self.backend.coordinator.apply(scope,_request_id(p),p["prepared_sha256"],Mutation(),{})
        return self.finalize_cache_retirements(p)

    def finalize_cache_retirements(self,prepared):
        p=prepared_from_payload(prepared);_validate(self.backend,self.cap,p,recovery_only=True,phase="ocr_cache_finalize")
        _saved(self._path(p),p,self.backend)
        status=verify_prepared_current(self.backend,p,require_cache_finalized=False)
        for r in p["cache_records"]:
            persist_marker(r,prepared_sha256=p["prepared_sha256"],payload_sha256=p["payload_sha256"],
                after_canonical_sha256=status["canonical_sha256"])
            self.backend.fault("after_ocr_cache_marker")
        result=verify_prepared_current(self.backend,p)
        _persist(self.cap.journal_root/(p["prepared_sha256"]+".finalized"),result)
        self.backend.fault("after_ocr_cache_finalize")
        return verify_prepared_current(self.backend,p,journal_root=self.cap.journal_root)

    def compensate(self,prepared,*,expected_current_sha256,expected_published_seq):
        p=prepared_from_payload(prepared)
        permit=_validate(self.backend,self.cap,p,phase="compensation_authority")
        _need(permit.get("allow_compensation") is True,"explicit guarded compensation authority required")
        scope=Scope(**p["scope"]);uid=p["source_preconditions"]["source_uid"]
        saved=_saved(self._path(p),p,self.backend)
        identity=digest(["ocr-field-compensation-v1",p["prepared_sha256"],expected_current_sha256])
        rid="ocr-field-compensation:"+identity
        request=self.backend.state.request(scope,rid)
        original=canonical_from_payload(saved["canonical"])
        forward=self.backend.state.request(scope,_request_id(p))
        _need(forward is not None and forward["status"]=="success","compensation requires completed original OCR publication")
        forward_after=_after_memory(self.backend,p,forward)
        restoration=replace(original,version=forward_after.version+1,commit_seq=forward_after.commit_seq)
        final_path=self.cap.journal_root/(p["prepared_sha256"]+".finalized")
        final=json.loads(_private(final_path).read_text())
        _need(digest(final["payload"])==final["sha256"] and final["payload"]["canonical_sha256"]==expected_current_sha256,
            "compensation must bind this exact finalized postimage")
        if request is None:
            for r,marker in zip(p["cache_records"],final["payload"]["cache_markers"],strict=True):
                mp=marker_path_for_cache_file(r["path"])
                _need(str(mp)==marker["path"] and hashlib.sha256(_private(mp).read_bytes()).hexdigest()==marker["sha256"],
                    "compensation cache marker changed before T1")
            current=self.backend.state.get_memories(scope,[uid]).get(uid)
            _need(current is not None and digest(asdict(current))==expected_current_sha256==digest(asdict(forward_after)),
                "later target edits cannot be rolled back")
            raw_sha,meta_sha=_sql_hashes(self.backend.state,scope,uid)
            frontier=self.backend.state.scope_state(scope)
            _need(frontier["published_seq"]==expected_published_seq and frontier["canonical_seq"]==frontier["lexical_seq"]==expected_published_seq,
                "compensation publication frontier changed")
            guard=dict(published_seq=expected_published_seq,profile_hash=p["profile_hash"],memories={uid:dict(
                version=current.version,commit_seq=current.commit_seq,canonical_sha256=expected_current_sha256,
                raw_data_sha256=raw_sha,metadata_sha256=meta_sha)})
            def before():
                actual=self.backend.state.get_memories(scope,[uid]).get(uid)
                _need(actual is not None and digest(asdict(actual))==expected_current_sha256,"compensation target changed")
                snapshot=snapshot_record(self.backend.state,self.backend.projection,scope,uid,replacement=restoration)
                snapshot.update(compensation_identity=identity,prepared_sha256=p["prepared_sha256"])
                _persist(self.cap.journal_root/(identity+".compensation"),snapshot)
                _validate(self.backend,self.cap,p,phase="before_compensation_t1")
            with self._operation(scope):
                self.backend.coordinator.apply(scope,rid,identity,Mutation(memories=(restoration,),kind="ocr-field-compensation-v1"),
                    dict(compensated=True,prepared_sha256=p["prepared_sha256"]),expected_publication=guard,before_prepare=before)
        else:
            _need(request["payload_digest"]==identity,"compensation request identity mismatch")
            with self._operation(scope):
                self.backend.coordinator.apply(scope,rid,identity,Mutation(),{})
        request=self.backend.state.request(scope,rid)
        expected=_after_memory(self.backend,p,request,replacement=restoration)
        current=self.backend.state.get_memories(scope,[uid]).get(uid)
        _need(current is not None and digest(asdict(current))==digest(asdict(expected)),"compensating target postimage differs")
        for r,marker in zip(p["cache_records"],final["payload"]["cache_markers"],strict=True):
            _need(str(marker_path_for_cache_file(r["path"]))==marker["path"],"compensation marker path mismatch")
            remove_exact_marker(r,marker["sha256"])
        result=dict(status="compensated",prepared_sha256=p["prepared_sha256"],canonical_sha256=digest(asdict(current)),
            mutation_seq=request["mutation_seq"],database_restored=False,other_records_untouched=True)
        _persist(self.cap.journal_root/(identity+".compensated"),result)
        return result



def snapshot_record(state, projection, scope, uid, *, replacement=None):
    """Read a targeted preimage under the caller's publication write lease.

    SQLite stores current canonical rows, not a canonical revision history.
    This captures the old row plus its lexical/projection preimage and related
    intent metadata. Restoration must publish a compensating new version; the
    SQL rows here are evidence and must never be INSERTed over a live database.
    """
    current = state.scope_state(scope)
    if not current['canonical_seq'] == current['lexical_seq'] == current['published_seq']:
        raise StateUnavailable('backup requires completely published canonical/lexical state')
    memory = state.get_memories(scope, [uid]).get(uid)
    if memory is None:
        raise Conflict('backup target disappeared')
    projection_rows = projection.get_projection_metadata(scope, [uid], vectors=True)
    row = projection_rows.get(uid)
    if not projection.codec.matches(row, projection.codec.metadata(scope, memory)):
        raise StateUnavailable('backup requires the exact verified dense and sparse revision')
    tables = {}
    for table in ('memories', 'memberships', 'lexical_documents', 'lexical_postings', 'tombstones'):
        tables[table] = state._rows(f'SELECT * FROM {table} WHERE scope=? AND logical_uid=?',
                                   (scope.key, uid))
    tables['graph_nodes'] = state._rows('SELECT * FROM graph_nodes WHERE scope=? AND memory_uid=?',
                                       (scope.key, uid))
    tables['graph_edges'] = state._rows('SELECT * FROM graph_edges WHERE scope=? AND '
        '(source_uid=? OR target_uid=?)', (scope.key, uid, uid))
    # Include old AND new term DF rows: a new repair term might already occur in
    # another document. Corpus stats must never be restored as an old snapshot.
    for table in ('lexical_corpus_stats', 'projection_frontiers',
                  'scope_attributes', 'scope_purges', 'memory_spaces', 'space_edges'):
        tables[table] = state._rows(f'SELECT * FROM {table} WHERE scope=?', (scope.key,))
    terms = {r['term'] for r in tables['lexical_postings']}
    if replacement is not None:
        terms.update(term for term, _ in state.lexical.prepare_document(replacement).terms)
    affected_terms = sorted(terms)
    tables['lexical_term_stats'] = []
    for offset in range(0, len(affected_terms), 512):
        batch = affected_terms[offset:offset + 512]
        marks = ','.join('?' for _ in batch)
        tables['lexical_term_stats'] += state._rows(
            f'SELECT * FROM lexical_term_stats WHERE scope=? AND term IN ({marks})',
            (scope.key, *batch))
    jobs = state._rows('SELECT * FROM build_jobs WHERE scope=?', (scope.key,))
    tables['build_jobs'] = [j for j in jobs if uid in json.loads(j['input_versions'])
                          or uid in json.loads(j['output_uids'])]
    tables['outbox_targets'] = state._rows('SELECT * FROM outbox_targets WHERE scope=? AND logical_uid=?',
                                         (scope.key, uid))
    sequences = [r['mutation_seq'] for r in tables['outbox_targets']]
    tables['mutations'], tables['requests'], tables['request_preparations'] = [], [], []
    for seq in sequences:
        tables['mutations'] += state._rows('SELECT * FROM mutations WHERE scope=? AND mutation_seq=?',
                                          (scope.key, seq))
        requests = state._rows('SELECT * FROM requests WHERE scope=? AND mutation_seq=?', (scope.key, seq))
        tables['requests'] += requests
        for request in requests:
            tables['request_preparations'] += state._rows('SELECT * FROM request_preparations '
                'WHERE scope=? AND request_id=?', (scope.key, request['request_id']))
    return dict(format='targeted-visual-preimage-v1', db_uuid=state.db_uuid,
        scope=asdict(scope), scope_state=current, profile_hash=state.profile_hash,
        manifest=state._rows('SELECT * FROM manifest ORDER BY key'), canonical=asdict(memory),
        canonical_sha256=digest(asdict(memory)), tables=tables, affected_lexical_terms=affected_terms,
        projection=dict(collection=projection.collection_name,
            policy_hash=projection.codec.projection_policy_hash,
            point_id=point_id(row['physical_id']), current_revision=row),
        recovery='roll-forward pending outbox; otherwise publish a guarded compensating version')
