import importlib.util
import os
from pathlib import Path
import subprocess
import sys


def module():
    spec = importlib.util.spec_from_file_location('owned_service', Path(__file__).parents[2] / 'deployment/cluster/service.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_pid_reuse_executable_and_boot_mismatch_are_never_adopted():
    svc = module()
    actual = svc.identity(os.getpid())
    assert svc.owned(actual)
    for field in ('start_ticks', 'executable', 'boot_id'):
        assert not svc.owned(actual | {field: 'different'})


def test_process_title_is_diagnostic_and_empty_exec_metadata_is_rejected(monkeypatch):
    svc = module()
    record = svc.identity(os.getpid())
    assert svc.owned(record | {'argv_sha256': 'vLLM title changed'})
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda p: b'' if p.name == 'cmdline' else original(p))
    assert svc.identity(os.getpid()) is None
    assert not svc.owned(record)


def test_exited_child_cannot_be_signalled_as_owned():
    svc = module()
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(1)'])
    try:
        record = svc.identity(child.pid)
        assert record is not None
        child.wait(timeout=10)
        assert not svc.owned(record)
    finally:
        if child.poll() is None:
            child.terminate()
