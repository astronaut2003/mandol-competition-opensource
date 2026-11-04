"""Original intake durability before visual work; artificial local fixtures only."""
import base64
from contextlib import nullcontext
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from types import SimpleNamespace

from PIL import Image
import pytest

from mandol_substrate.state import Conflict, Scope
from mandol_substrate.state.models import digest
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.media_store import UserMediaStore, compute_media_id
from mandol_competition.multimodal.protocol import AddRequest
from mandol_competition.multimodal.raw_intake import RawIntakeError, RawIntakeStore
from mandol_competition.multimodal.service import AddService
from mandol_competition.multimodal.vlm.openai_compat import VisionExtractionUnavailable


def image_part(color, *, noncanonical=False):
    out=BytesIO()
    with Image.new('RGB',(17,19),color) as image:image.save(out,format='PNG')
    raw=out.getvalue()
    if noncanonical:
        # PNG permits trailing bytes. Ensure padding exists, then change only
        # unused Base64 pad bits: decoded original bytes remain identical.
        while len(raw)%3==0:raw+=b' '
    encoded=base64.b64encode(raw).decode()
    header='data:image/PNG;base64' if noncanonical else 'data:image/png;base64'
    if noncanonical:
        alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
        index=-3 if encoded.endswith('==') else -2
        encoded=encoded[:index]+alphabet[alphabet.index(encoded[index])+1]+encoded[index+1:]
        assert base64.b64decode(encoded,validate=True)==raw
    return dict(type='image_url',image_url=dict(url=header+','+encoded))


def request(user='alice',rid='intake-one',*,noncanonical=False):
    return AddRequest.model_validate(dict(user_id=user,request_id=rid,session_id='session',messages=[
        dict(role='user',timestamp=-1234567890123,content=[
            dict(type='text',text='  原始文字\nkeep spaces  '),image_part('red',noncanonical=noncanonical),
            dict(type='text',text='literal data:image/ is source text, not an inline payload'),image_part('blue')]),
        dict(role='assistant',timestamp=None,content='  string-form message\n中文 café  '),
        dict(role='user',timestamp=9223372036854775807,content=[image_part('green'),
            dict(type='text',text='last image caption')])]))


class ForbiddenVision:
    model_name='artificial-model'
    def __init__(self,observer=None):self.calls=0;self.observer=observer
    def extract(self,**inputs):
        self.calls+=1
        if self.observer is not None:self.observer()
        raise VisionExtractionUnavailable(retry_safe=False,unknown_outcome=True)


def decoded(config,body):
    return AddService(config,extractor=ForbiddenVision())._decode(body)


def scope(config,body):return Scope(config.namespace,body.user_id).key


def save(config,body):
    return RawIntakeStore(config,body.user_id).save(body,decoded(config,body),
        scope_key=scope(config,body),body_hash=digest(body.model_dump()))


def load(config,body,receipt):
    return RawIntakeStore(config,body.user_id).load(receipt,scope_key=scope(config,body),
        request_id=body.request_id,body_hash=digest(body.model_dump()))


def manifest_path(config,body,receipt):
    return config.user_root(body.user_id)/receipt['manifest_relpath']


@pytest.mark.parametrize('noncanonical',[False,True])
def test_process_reconstruction_preserves_full_original_request_and_uri_representation(tmp_path,noncanonical):
    config=MultimodalConfig(data_dir=tmp_path);body=request(noncanonical=noncanonical)
    receipt=save(config,body)
    assert load(config,body,receipt)==body.model_dump()
    assert AddRequest.model_validate(load(config,body,receipt)).model_dump()==body.model_dump()
    assert set(receipt)=={'format','manifest_relpath','manifest_sha256','scope_key','request_id_sha256','body_hash'}
    assert receipt['format']=='mm-raw-intake-receipt-v1' and receipt['scope_key']==scope(config,body)
    assert receipt['body_hash']==digest(body.model_dump())
    assert receipt['request_id_sha256']==hashlib.sha256(body.request_id.encode()).hexdigest()
    path=PurePosixPath(receipt['manifest_relpath'])
    assert len(path.parts)==2 and path.parts[0]=='raw-intake' and path.suffix=='.json'
    assert len(path.stem)==64 and all(c in '0123456789abcdef' for c in path.stem)
    assert hashlib.sha256(manifest_path(config,body,receipt).read_bytes()).hexdigest()==receipt['manifest_sha256']
    assert stat.S_IMODE(manifest_path(config,body,receipt).stat().st_mode)==0o600
    assert not stat.S_IMODE(manifest_path(config,body,receipt).parent.stat().st_mode)&0o077


def test_manifest_has_only_bound_media_references_and_receipt_has_no_source_text(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    envelope=json.loads(manifest_path(config,body,receipt).read_bytes())
    assert set(envelope)=={'format','binding','request'}
    assert envelope['binding']['scope_key']==scope(config,body)
    assert envelope['binding']['body_hash']==digest(body.model_dump())
    assert envelope['request']['messages'][1]==body.model_dump()['messages'][1]
    originals=[part for message in body.messages for part in message.parts() if part.type=='image_url']
    refs=[part['image_url']['url'] for message in envelope['request']['messages']
          if isinstance(message['content'],list) for part in message['content'] if part['type']=='image_url']
    assert len(refs)==len(originals)==3
    for original,ref in zip(originals,refs,strict=True):
        assert isinstance(ref,dict) and set(ref)=={'media_id','mime_type','byte_size','sha256',
            'uri_header','base64_prefix_length','base64_tail','uri_sha256'}
        assert ref['uri_sha256']==hashlib.sha256(original.image_url.url.encode()).hexdigest()
        assert ref['media_id']=='img_'+ref['sha256']
        assert original.image_url.url not in manifest_path(config,body,receipt).read_text()
    encoded=json.dumps(receipt,ensure_ascii=False)
    assert 'data:image/' not in encoded and '原始文字' not in encoded and body.user_id not in encoded


def test_same_request_replay_is_idempotent_but_different_body_conflicts(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    assert save(config,body)==receipt
    changed=body.model_dump();changed['messages'][0]['content'][0]['text']='changed original text'
    changed=AddRequest.model_validate(changed)
    with pytest.raises(Conflict):save(config,changed)
    assert load(config,body,receipt)==body.model_dump()


@pytest.mark.parametrize('field,value',[('scope_key','0'*64),('request_id','other-request'),('body_hash','0'*64)])
def test_load_rejects_receipt_identity_drift_before_restoring_text(tmp_path,field,value):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    arguments=dict(scope_key=scope(config,body),request_id=body.request_id,body_hash=digest(body.model_dump()))
    arguments[field]=value
    with pytest.raises(RawIntakeError) as caught:RawIntakeStore(config,body.user_id).load(receipt,**arguments)
    assert '原始文字' not in str(caught.value)


def test_other_user_cannot_reuse_or_repoint_an_intake_receipt(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);alice=request();receipt=save(config,alice)
    bob=request(user='bob');save(config,bob)
    with pytest.raises(RawIntakeError):load(config,bob,receipt)
    forged=deepcopy(receipt);forged['scope_key']=scope(config,bob)
    with pytest.raises(RawIntakeError):load(config,bob,forged)
    assert load(config,alice,receipt)==alice.model_dump()


def test_save_cannot_misbind_caller_user_scope_or_original_body_hash(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);body=request();messages=decoded(config,body)
    with pytest.raises(RawIntakeError):RawIntakeStore(config,'bob').save(body,messages,
        scope_key=scope(config,body),body_hash=digest(body.model_dump()))
    with pytest.raises(RawIntakeError):RawIntakeStore(config,body.user_id).save(body,messages,
        scope_key=Scope('other-namespace',body.user_id).key,body_hash=digest(body.model_dump()))
    with pytest.raises(RawIntakeError):RawIntakeStore(config,body.user_id).save(body,messages,
        scope_key=scope(config,body),body_hash='0'*64)


@pytest.mark.parametrize('relative',['../foreign.json','/tmp/foreign.json','raw-intake/../foreign.json'])
def test_receipt_cannot_escape_same_user_root(tmp_path,relative):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    forged=deepcopy(receipt);forged['manifest_relpath']=relative
    with pytest.raises(RawIntakeError):load(config,body,forged)


def test_manifest_and_original_media_tampering_fail_closed(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    path=manifest_path(config,body,receipt);original=path.read_bytes();path.write_bytes(original+b'\n')
    with pytest.raises(RawIntakeError):load(config,body,receipt)
    path.write_bytes(original)
    image=next(image for _,_,images in decoded(config,body) for image in images.values())
    media=UserMediaStore(config,body.user_id).path_for(compute_media_id(image.data))
    media.write_bytes(b'artificial corrupt image')
    with pytest.raises(RawIntakeError):load(config,body,receipt)


def test_symlink_manifest_cannot_substitute_an_external_file_even_with_same_bytes(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    path=manifest_path(config,body,receipt);external=tmp_path/'foreign-manifest.json';external.write_bytes(path.read_bytes())
    path.unlink();path.symlink_to(external)
    with pytest.raises(RawIntakeError):load(config,body,receipt)


def test_symlink_media_cannot_escape_the_users_original_store_even_with_same_bytes(tmp_path):
    config=MultimodalConfig(data_dir=tmp_path);body=request();receipt=save(config,body)
    image=next(image for _,_,images in decoded(config,body) for image in images.values())
    media=UserMediaStore(config,body.user_id).path_for(compute_media_id(image.data))
    external=tmp_path/'foreign-image.bin';external.write_bytes(media.read_bytes())
    media.unlink();media.symlink_to(external)
    with pytest.raises(RawIntakeError):load(config,body,receipt)


class MemoryRequestLedger:
    def __init__(self,state):self.state=state
    def reserve(self,scope,rid,body_hash):
        key=scope.key,rid
        if key not in self.state:self.state[key]=dict(body_hash=body_hash,value={})
        if self.state[key]['body_hash']!=body_hash:raise Conflict('artificial body conflict')
        return deepcopy(self.state[key]['value'])
    def checkpoint(self,scope,rid,body_hash,value):
        assert self.state[scope.key,rid]['body_hash']==body_hash
        # The ledger may carry only the compact intake receipt, never original
        # media/source text from the sidecar envelope.
        assert 'data:image/' not in json.dumps(value)
        self.state[scope.key,rid]['value']=deepcopy(value)


def service_for(config,monkeypatch,observer=None):
    import mandol_competition.multimodal.service as service
    monkeypatch.setattr(service,'RequestLedger',MemoryRequestLedger)
    backend=SimpleNamespace(state={},profile=SimpleNamespace(options={}),
        scope=lambda user:Scope(config.namespace,user),coordinator=SimpleNamespace(max_mutation_memories=100),
        replay=lambda *args:None,user_operation=lambda user:nullcontext())
    backend.publish=lambda *args:pytest.fail('a failed raw/vision preparation must never publish')
    vision=ForbiddenVision(observer)
    return AddService(config,extractor=vision,backend=backend),backend,vision


def test_all_images_and_original_text_are_durable_before_first_image_extraction_fails(tmp_path,monkeypatch):
    config=MultimodalConfig(data_dir=tmp_path,vision_failure_policy='retry-required-v1');body=request()
    images=[image for _,_,batch in decoded(config,body) for image in batch.values()]
    observed=[]
    def before_failure():
        for image in images:
            assert UserMediaStore(config,body.user_id).get(compute_media_id(image.data))==(image.data,image.mime_type)
        value=backend.state[scope(config,body),body.request_id]['value']
        raw=value['raw_intake']
        assert load(config,body,raw)==body.model_dump()
        observed.append(raw)
    add,backend,vision=service_for(config,monkeypatch,before_failure)
    with pytest.raises(VisionExtractionUnavailable):add.handle(body)
    assert vision.calls==1 and len(observed)==1
    assert load(config,body,observed[0])==body.model_dump()
    assert 'units' not in backend.state[scope(config,body),body.request_id]['value']


def test_intake_write_failure_stops_before_any_model_dispatch_or_success(tmp_path,monkeypatch):
    config=MultimodalConfig(data_dir=tmp_path,vision_failure_policy='retry-required-v1');body=request()
    add,backend,vision=service_for(config,monkeypatch)
    def failed_save(*args,**kwargs):raise OSError('artificial disk full')
    monkeypatch.setattr(RawIntakeStore,'save',failed_save)
    with pytest.raises((OSError,RawIntakeError)):add.handle(body)
    assert vision.calls==0 and 'units' not in backend.state[scope(config,body),body.request_id]['value']


def test_raw_receipt_checkpoint_failure_dispatches_nothing_and_same_intake_is_reusable(tmp_path,monkeypatch):
    config=MultimodalConfig(data_dir=tmp_path,vision_failure_policy='retry-required-v1');body=request()
    add,backend,vision=service_for(config,monkeypatch)
    import mandol_competition.multimodal.service as service
    failed=[];saved=[];original_save=RawIntakeStore.save
    class FailOneRawCheckpoint(MemoryRequestLedger):
        def checkpoint(self,scope,rid,body_hash,value):
            if 'raw_intake' in value and not failed:
                failed.append(True)
                raise OSError('artificial receipt checkpoint unavailable')
            return super().checkpoint(scope,rid,body_hash,value)
    def capture_save(self,*args,**kwargs):
        result=original_save(self,*args,**kwargs);saved.append(deepcopy(result));return result
    monkeypatch.setattr(service,'RequestLedger',FailOneRawCheckpoint)
    monkeypatch.setattr(RawIntakeStore,'save',capture_save)
    with pytest.raises((OSError,RawIntakeError)):add.handle(body)
    assert failed==[True] and vision.calls==0 and len(saved)==1
    assert load(config,body,saved[0])==body.model_dump()
    assert save(config,body)==saved[0]
    assert len(list((config.user_root(body.user_id)/'raw-intake').glob('*.json')))==1
    with pytest.raises(VisionExtractionUnavailable):add.handle(body)
    assert vision.calls==1 and all(value==saved[0] for value in saved)
    assert backend.state[scope(config,body),body.request_id]['value']['raw_intake']==saved[0]


def test_user_purge_removes_intake_media_and_loadability_without_removing_other_user(backend):
    config=backend.config;alice=request();bob=request(user='bob')
    alice_receipt=save(config,alice);bob_receipt=save(config,bob)
    assert config.user_root('alice').is_dir() and config.user_root('bob').is_dir()
    backend.purge('alice')
    assert not config.user_root('alice').exists()
    with pytest.raises(RawIntakeError):load(config,alice,alice_receipt)
    assert not config.user_root('alice').exists()
    assert load(config,bob,bob_receipt)==bob.model_dump()


"""Append-only test fragment for tests/multimodal/test_raw_intake.py.

Uses that module's existing artificial image/request/service helpers. Keep this
outside the frozen candidate until the parent integrates its focused run.
"""
import errno as _raw_sync_errno
import os as _raw_sync_os


def _raw_sync_fault(monkeypatch, directory, published_file):
    """Fail only a real directory fsync after its file has been published."""
    original = _raw_sync_os.fsync
    state = dict(enabled=True, failures=0, recovered_file_fences=set())

    def fsync(descriptor):
        info = _raw_sync_os.fstat(descriptor)
        if stat.S_ISDIR(info.st_mode) and published_file.is_file() and directory.is_dir():
            current = directory.stat()
            if ((info.st_dev, info.st_ino) == (current.st_dev, current.st_ino)
                    and state['enabled']):
                state['failures'] += 1
                raise OSError(_raw_sync_errno.EIO, 'artificial post-publication directory fsync failure')
        result = original(descriptor)
        if not state['enabled'] and stat.S_ISREG(info.st_mode):
            state['recovered_file_fences'].add((info.st_dev, info.st_ino))
        return result

    monkeypatch.setattr(_raw_sync_os, 'fsync', fsync)
    return state


def _raw_original_file_snapshot(root):
    # Compare identity, bytes and mtime: an equal-content replacement still
    # violates the immutable-file retry contract and must fail this assertion.
    return {str(path.relative_to(root)): (path.stat().st_dev, path.stat().st_ino,
            path.stat().st_mtime_ns, path.read_bytes())
            for path in root.rglob('*')
            if path.is_file() and path.suffix in {'.bin', '.mime', '.json'}}


def _raw_assert_recovered_fences(state, snapshot):
    assert {(value[0], value[1]) for value in snapshot.values()} <= state['recovered_file_fences']


def test_manifest_link_then_directory_sync_failure_requires_durable_retry_before_model(tmp_path, monkeypatch):
    config = MultimodalConfig(data_dir=tmp_path, vision_failure_policy='retry-required-v1')
    body = request(rid='manifest-directory-fsync')
    add, backend, vision = service_for(config, monkeypatch)
    root = config.user_root(body.user_id)
    manifest = root / 'raw-intake' / (hashlib.sha256(body.request_id.encode()).hexdigest() + '.json')
    fault = _raw_sync_fault(monkeypatch, manifest.parent, manifest)

    # The first failure is after atomic link, not before any bytes exist.
    with pytest.raises(RawIntakeError):
        add.handle(body)
    assert fault['failures'] == 1 and manifest.is_file() and vision.calls == 0
    snapshot = _raw_original_file_snapshot(root)
    assert len([name for name in snapshot if name.endswith('.bin')]) == 3
    assert 'raw_intake' not in backend.state[scope(config, body), body.request_id]['value']

    # The existing-manifest branch must repeat the actual durability fence.
    with pytest.raises(RawIntakeError):
        add.handle(body)
    assert fault['failures'] == 2 and vision.calls == 0
    assert _raw_original_file_snapshot(root) == snapshot

    fault['enabled'] = False
    with pytest.raises(VisionExtractionUnavailable):
        add.handle(body)
    assert vision.calls == 1
    assert _raw_original_file_snapshot(root) == snapshot
    receipt = backend.state[scope(config, body), body.request_id]['value']['raw_intake']
    assert receipt['manifest_sha256'] == hashlib.sha256(snapshot[str(manifest.relative_to(root))][3]).hexdigest()
    assert load(config, body, receipt) == body.model_dump()
    _raw_assert_recovered_fences(fault, snapshot)


def test_media_replace_then_directory_sync_failure_is_not_skipped_by_identical_file_retry(tmp_path, monkeypatch):
    config = MultimodalConfig(data_dir=tmp_path, vision_failure_policy='retry-required-v1')
    body = request(rid='media-directory-fsync')
    add, backend, vision = service_for(config, monkeypatch)
    first = next(image for _, _, images in decoded(config, body) for image in images.values())
    sha = compute_media_id(first.data)[4:]
    root = config.user_root(body.user_id)
    original_image = root / 'media' / sha[:2] / sha[2:4] / (sha + '.bin')
    fault = _raw_sync_fault(monkeypatch, original_image.parent, original_image)

    # MIME is already written, then the first original image is os.replace'd.
    # Its following directory fsync fails, so no receipt or model is allowed.
    with pytest.raises(RawIntakeError):
        add.handle(body)
    assert fault['failures'] == 1 and original_image.is_file() and vision.calls == 0
    first_identity = (original_image.stat().st_ino, original_image.stat().st_mtime_ns, original_image.read_bytes())
    assert not (root / 'raw-intake').exists()

    # MediaStore.put skips identical bytes on retry. The intake's final fence
    # must nevertheless revisit that leaf directory and reject the retry.
    with pytest.raises(RawIntakeError):
        add.handle(body)
    assert fault['failures'] == 2 and vision.calls == 0
    assert (original_image.stat().st_ino, original_image.stat().st_mtime_ns, original_image.read_bytes()) == first_identity
    snapshot = _raw_original_file_snapshot(root)
    assert len([name for name in snapshot if name.startswith('raw-intake/')]) == 1
    assert 'raw_intake' not in backend.state[scope(config, body), body.request_id]['value']

    fault['enabled'] = False
    with pytest.raises(VisionExtractionUnavailable):
        add.handle(body)
    assert vision.calls == 1 and _raw_original_file_snapshot(root) == snapshot
    receipt = backend.state[scope(config, body), body.request_id]['value']['raw_intake']
    assert load(config, body, receipt) == body.model_dump()
    _raw_assert_recovered_fences(fault, snapshot)
