"""Foreground three-API runner; model/Qdrant processes remain separately owned."""
import argparse
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import httpx

from mandol_competition.config import load_config
from mandol_competition.config.unified import TRACKS


def profile_path(name):
    return Path(__file__).parents[1] / 'config' / 'profiles' / (name + '.toml')


def describe(settings):
    """No credential values, credential mount paths or arbitrary environment dump."""
    from mandol_competition.textual.core.evidence_profiles import resolve_evidence_profile
    return {'mode': settings.runtime.mode, 'api_host': settings.runtime.host,
        'api_ports': dict(zip(TRACKS, settings.runtime.ports)),
        'limits': settings.runtime.limits,
        'credentials_configured': {name: bool(track.memory_system_key) for name, track in settings.tracks.items()},
        'tracks': {name: {'root': str(t.root), 'namespace': t.namespace, 'collection': t.collection,
            'qdrant': t.qdrant_endpoint, 'inference_contract': t.inference_contract,
            'inference': t.inference, 'reranker_pool': t.reranker_pool,
            'reranker_backend': t.reranker_backend, 'retrieval_policy': t.retrieval_policy,
            **({'evidence_profile': resolve_evidence_profile(t.evidence_profile,
                official=t.inference_contract == 'aml-official-eligible-v1',
                experimental=bool(t.quality_candidate))} if name == 'textual' else {}),
            **({'quality_candidate': t.quality_candidate} if t.quality_candidate else {})}
            for name, t in settings.tracks.items()},
        'serving_identity': settings.runtime.bindings,
        'http_retry': 'bounded existing client policy, at most3 attempts; whole-search failover at most pool size',
        'model_qualification': 'configuration only; no inference probe, weight/dtype attestation or accuracy claim'}


def child_environment(settings, path):
    mode_prefix = 'MANDOL_' + settings.runtime.mode.upper() + '_'
    credential_names = {n + suffix for n in (settings.runtime.credential_env,
        settings.runtime.vlm_credential_env, settings.runtime.qdrant_credential_env) for suffix in ('', '_FILE')}
    from mandol_competition.inference.dashscope_rerank import BACKEND
    credential_names.update(t.reranker_credential_env + suffix for t in settings.tracks.values()
                            if t.reranker_backend == BACKEND for suffix in ('', '_FILE'))
    # Grouped Local/Cluster launches require a mode-scoped opt-in. Ambient
    # production routing/credentials must not silently enter a Local run.
    generation = os.getenv(mode_prefix + 'GENERATION_PROVIDER', 'configured')
    if generation not in {'configured', 'openrouter', 'closeai'}:
        raise ValueError('unknown mode-scoped generation provider')
    if generation == 'openrouter':
        credential_names.update({'OPENROUTER_API_KEY', 'OPENROUTER_API_KEY_FILE'})
    elif generation == 'closeai':
        credential_names.difference_update({'OPENROUTER_API_KEY', 'OPENROUTER_API_KEY_FILE'})
        credential_names.update({'CLOSEAI_API_KEY', 'CLOSEAI_API_KEY_FILE'})
    # A new Local process must not inherit a live deployment's config, traces,
    # bindings or unrelated paid-provider credentials.
    basic = {'HOME','USER','LOGNAME','PATH','LANG','LC_ALL','LC_CTYPE','TMPDIR','TZ',
             'SYSTEMROOT','WINDIR','TEMP','TMP','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}
    result = {k: v for k, v in os.environ.items()
              if k in basic or k.startswith(mode_prefix) or k in credential_names}
    result.update(settings.runtime.environment())
    if generation == 'openrouter':
        result['MANDOL_GENERATION_PROVIDER'] = generation
        result['MANDOL_OPENROUTER_PROXY'] = os.getenv(mode_prefix + 'OPENROUTER_PROXY',
                                                      'http://127.0.0.1:17897')
    elif generation == 'closeai':
        result['MANDOL_GENERATION_PROVIDER'] = generation
        result['MANDOL_CLOSEAI_BASE_URL'] = os.getenv(mode_prefix + 'CLOSEAI_BASE_URL',
                                                    'https://api.openai-proxy.org/v1')
    result['MANDOL_CONFIG'] = str(path)
    result['PYTHONUNBUFFERED'] = '1'
    result['MANDOL_INTERNAL_ENDPOINTS_LOOPBACK_ONLY'] = '1'
    return result


def health(settings):
    address = ipaddress.ip_address(settings.runtime.host)
    host = ('127.0.0.1' if address.version == 4 else '::1') if address.is_unspecified else str(address)
    host = '[' + host + ']' if ':' in host else host
    rows = []
    with httpx.Client(trust_env=False, timeout=2) as client:
        for track, port in zip(TRACKS, settings.runtime.ports):
            try:
                r = client.get(f'http://{host}:{port}/health')
                ok = r.status_code == 200 and r.json() == {'status': 'ok'}
            except (httpx.RequestError, ValueError):
                ok = False
            rows.append({'track': track, 'ready': ok})
    return rows


def run(settings, path):
    if any(not t.memory_system_key and not t.allow_unauthenticated_dev for t in settings.tracks.values()):
        raise ValueError('configure your own API credential before startup')
    for port in settings.runtime.ports:
        with socket.socket(socket.AF_INET6 if ':' in settings.runtime.host else socket.AF_INET) as check:
            check.bind((settings.runtime.host, port))
    children = []
    stopped = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stopped.set()) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        env = child_environment(settings, path)
        for track, port in zip(TRACKS, settings.runtime.ports):
            children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn',
                'mandol_competition.runtime.apps:' + track, '--factory',
                '--host', settings.runtime.host, '--port', str(port), '--workers', '1',
                '--no-access-log', '--timeout-keep-alive', '30',
                '--timeout-graceful-shutdown', '1810'], env=env, start_new_session=True))
        until = time.monotonic() + settings.runtime.startup_timeout
        while not stopped.is_set():
            if any(p.poll() is not None for p in children):
                raise RuntimeError('an owned API exited; other owned children will drain')
            rows = health(settings)
            if all(row['ready'] for row in rows):
                print(json.dumps({'status': 'READY', 'mode': settings.runtime.mode, 'apis': rows}), flush=True)
                break
            if time.monotonic() >= until:
                raise RuntimeError('bounded API startup deadline exceeded')
            stopped.wait(.2)
        while not stopped.wait(.5):
            if any(p.poll() is not None for p in children):
                raise RuntimeError('an owned API exited; other owned children will drain')
    finally:
        # Only subprocess objects created here are signaled, never PID/name
        # discovery, existing listeners, models, databases or production services.
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            try:
                p.wait(timeout=1820)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait(timeout=5)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, choices=('local', 'cluster'))
    parser.add_argument('--config', type=Path, help='explicit custom profile; no hostname/IP inference')
    parser.add_argument('--runtime-root', type=Path, help='mode-scoped override, never touches other modes')
    parser.add_argument('action', choices=('example', 'check', 'health', 'run'))
    args = parser.parse_args()
    path = (args.config or profile_path(args.profile)).resolve()
    if args.action == 'example':
        print(path.read_text(), end=''); return
    if args.runtime_root:
        os.environ['MANDOL_' + args.profile.upper() + '_RUNTIME_ROOT'] = str(args.runtime_root.resolve())
    settings = load_config(path)
    if settings.runtime.mode != args.profile:
        raise ValueError('selected profile and explicit configuration mode differ')
    if args.action == 'check':
        print(json.dumps(describe(settings), indent=2))
    elif args.action == 'health':
        rows = health(settings); print(json.dumps(rows)); return 0 if all(r['ready'] for r in rows) else 1
    else:
        run(settings, path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
