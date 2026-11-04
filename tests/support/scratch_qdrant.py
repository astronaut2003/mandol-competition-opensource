"""Own a new scratch Qdrant process; never discover or adopt an existing server."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import time

import httpx
import pytest


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@contextmanager
def formal_stack_qdrant(root):
    executable = os.getenv('MANDOL_TEST_QDRANT_BINARY')
    if not executable:
        pytest.skip('Owned per-process Qdrant binary is not configured')
    binary = Path(executable).resolve(strict=True)
    root = Path(root).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    port, grpc = free_port(), free_port()
    while grpc == port:
        grpc = free_port()
    config = root / 'qdrant.yaml'
    config.write_text(f'storage:\n  storage_path: {root}/storage\n'
        f'  snapshots_path: {root}/snapshots\n  performance:\n    max_search_threads: 2\n'
        f'service:\n  host: 127.0.0.1\n  http_port: {port}\n  grpc_port: {grpc}\n'
        'telemetry_disabled: true\nlog_level: WARN\n')
    endpoint = f'http://127.0.0.1:{port}'
    with (root / 'qdrant.log').open('w') as log:
        process = subprocess.Popen([str(binary), '--config-path', str(config)],
            cwd=root, stdout=log, stderr=subprocess.STDOUT)
        try:
            with httpx.Client(trust_env=False, timeout=1) as client:
                until = time.monotonic() + 30
                while time.monotonic() < until:
                    if process.poll() is not None:
                        raise RuntimeError('Owned scratch Qdrant exited')
                    try:
                        response = client.get(endpoint + '/')
                        if response.status_code == 200:
                            assert response.json()['version'] == '1.19.1'
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(.05)
                else:
                    raise RuntimeError('Owned scratch Qdrant startup timed out')
            yield endpoint
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
