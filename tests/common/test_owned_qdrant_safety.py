import json
import signal
import socket
import subprocess
import sys
import uuid

import pytest

from mandol_substrate.qdrant.service import (
    QdrantServiceManager,
    owns_loopback_listener,
    process_identity,
    signal_owned_process,
)
from mandol_substrate.state import StateUnavailable


def test_wrong_start_ticks_cannot_signal_current_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = process_identity(child.pid)
        with pytest.raises(StateUnavailable):
            signal_owned_process(child.pid, dict(identity, start_ticks=identity["start_ticks"] + 1), signal.SIGKILL)
        assert child.poll() is None
        signal_owned_process(child.pid, identity, signal.SIGTERM)
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_wrong_manifest_or_symlink_refuses_service(tmp_path):
    storage_id = uuid.uuid4().hex
    (tmp_path / "manifest.json").write_text(json.dumps({"storage_id": storage_id}))
    with pytest.raises(ValueError, match="identity"):
        QdrantServiceManager(tmp_path, "foreign", mode="docker", image_id="sha256:test", allow_scratch=True)
    linked = tmp_path / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        QdrantServiceManager(linked, storage_id, mode="docker", image_id="sha256:test", allow_scratch=True)


def test_listener_in_same_namespace_is_not_owned_by_foreign_process():
    import os

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            port = sock.getsockname()[1]
            assert owns_loopback_listener(os.getpid(), port)
            assert not owns_loopback_listener(child.pid, port)
    finally:
        child.terminate()
        child.wait(timeout=5)
