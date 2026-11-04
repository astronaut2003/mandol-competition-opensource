"""Foreground synthetic API stack with an explicit child environment."""
from contextlib import contextmanager
import os
import subprocess
import sys
import time

import httpx

TRACKS = ('textual', 'code', 'multimodal')


@contextmanager
def running(root, config, ports, fixture_env, baseline=None, mode='local'):
    if baseline is not None:
        raise ValueError('Public tests run only the current checkout')
    allowed = {'PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT',
               'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(fixture_env)
    env.update(MANDOL_CONFIG=str(config), PYTHONDONTWRITEBYTECODE='1')
    with (root / 'owned-api.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-m', 'mandol_competition.runtime',
            '--profile', mode, '--config', str(config), 'run'], env=env,
            stdout=log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        urls = {track: f'http://127.0.0.1:{port}' for track, port in zip(TRACKS, ports, strict=True)}
        try:
            with httpx.Client(timeout=1, trust_env=False) as client:
                until = time.monotonic() + 45
                while time.monotonic() < until:
                    if process.poll() is not None:
                        raise AssertionError('Owned API exited; inspect isolated test log')
                    try:
                        if all(client.get(url + '/health').status_code == 200 for url in urls.values()):
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(.05)
                else:
                    raise AssertionError('Owned API startup timed out')
            yield urls, time.monotonic() - started
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
