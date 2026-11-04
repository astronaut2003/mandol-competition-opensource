"""Exact historical-cache retirement markers; original GPT cache bytes stay intact."""
import hashlib
import json
import os
from pathlib import Path
import stat
import secrets

from mandol_substrate.state import Conflict, StateUnavailable
from mandol_substrate.state.models import canonical_json, digest

FORMAT="ocr-field-cache-retirement-v1"
SUFFIX=".ocr-field-retired"
POLICY="reviewed-ocr-field-repair-v1"

def marker_path_for_cache_file(path):
    return Path(str(path)+SUFFIX)

def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)

def _path_sha(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()

def _private_file(path):
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise StateUnavailable("cache retirement paths cannot contain symlinks")
    s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_mode&0o077 or s.st_uid!=os.geteuid():
        raise StateUnavailable("cache retirement artifact must be private and owned")
    return path

def marker_value(record,*,prepared_sha256,payload_sha256,after_canonical_sha256):
    path=Path(record["path"])
    return dict(format=FORMAT,policy=POLICY,scope_key=record["scope_key"],media_id=record["media_id"],
        cache_key=record["key"],cache_path_sha256=_path_sha(path),
        retired_record_sha256=record["sha256"],source_context_sha256=record["source_context_sha256"],
        prepared_sha256=prepared_sha256,payload_sha256=payload_sha256,
        expected_after_canonical_sha256=after_canonical_sha256,
        full_extraction_not_modified=True)

def atomic_private_bytes(path,data):
    """Publish fsynced complete bytes exclusively; process death leaves no partial target."""
    path=Path(path)
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise StateUnavailable("private artifact path contains a symlink")
    tmp=path.parent/("."+path.name+".tmp-"+secrets.token_hex(12))
    fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        with os.fdopen(fd,"wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,path,follow_symlinks=False)
        except FileExistsError:
            _private_file(path)
            if path.read_bytes()!=data:raise Conflict("existing private artifact differs")
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)

def read_marker(cache_path,media_id,key):
    path=marker_path_for_cache_file(cache_path)
    if not path.exists() and not path.is_symlink():return None
    _private_file(path)
    try:
        if path.stat().st_size>8192:raise ValueError()
        value=json.loads(path.read_text())
        if (value.get("format")!=FORMAT or value.get("policy")!=POLICY
                or value.get("media_id")!=media_id or value.get("cache_key")!=key
                or value.get("cache_path_sha256")!=_path_sha(cache_path)
                or value.get("full_extraction_not_modified") is not True
                or any(not _sha(value.get(k)) for k in ("scope_key","retired_record_sha256","source_context_sha256",
                    "prepared_sha256","payload_sha256","expected_after_canonical_sha256"))):
            raise ValueError()
        return value
    except (ValueError,TypeError,KeyError):
        raise StateUnavailable("historical cache retirement marker is inconsistent") from None

def cache_record_retired(path,media_id,key,raw_bytes):
    marker=read_marker(path,media_id,key)
    # A later fresh cache result under this key has different bytes and is not
    # retired by an older exact-SHA marker. Never overwrite it.
    return marker is not None and hashlib.sha256(raw_bytes).hexdigest()==marker["retired_record_sha256"]

def persist_marker(record,**binding):
    path=Path(record["path"]);marker=marker_path_for_cache_file(path)
    expected=marker_value(record,**binding);data=canonical_json(expected).encode()
    # The marker is still useful if the old file was replaced later: its SHA
    # only disables those original bytes, never the new cache result.
    if path.exists():
        _private_file(path)
    atomic_private_bytes(marker,data)
    if read_marker(path,record["media_id"],record["key"])!=expected:
        raise StateUnavailable("retirement marker verification failed")
    return dict(path=str(marker),sha256=hashlib.sha256(data).hexdigest())

def remove_exact_marker(record,expected_marker_sha256):
    marker=marker_path_for_cache_file(record["path"])
    if not marker.exists() and not marker.is_symlink():return
    _private_file(marker)
    if hashlib.sha256(marker.read_bytes()).hexdigest()!=expected_marker_sha256:
        raise Conflict("cache retirement marker changed after this maintenance")
    marker.unlink()
    fd=os.open(marker.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

