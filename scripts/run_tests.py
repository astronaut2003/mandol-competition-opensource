#!/usr/bin/env python3
"""Run isolated regression against an owned, empty, loopback-only Qdrant."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qdrant', required=True)
    p.add_argument('--container', required=True)
    p.add_argument('--qdrant-binary', type=Path,
                   help='Optional Qdrant 1.19.1 binary for additional owned-process lifecycle tests')
    p.add_argument('--output', type=Path, default=ROOT/'.test-results')
    a = p.parse_args()
    url = urlsplit(a.qdrant)
    if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.username or url.password or url.path not in ('','/'):
        raise SystemExit('Disposable loopback Qdrant required')
    info = json.loads(subprocess.check_output(['docker','inspect',a.container]))[0]
    labels = info['Config'].get('Labels',{})
    if labels.get('mandol.test') != 'opensource' and labels.get('mandol.audit') != 'opensource-20261010':
        raise SystemExit('Container lacks explicit open-source-test ownership label')
    ports = info['NetworkSettings']['Ports']['6333/tcp']
    if not any(v['HostIp']=='127.0.0.1' and int(v['HostPort'])==url.port for v in ports):
        raise SystemExit('Endpoint does not belong to owned container')
    import httpx
    with httpx.Client(trust_env=False, timeout=5) as http:
        response = http.get(a.qdrant+'/collections');response.raise_for_status()
        if response.json()['result']['collections']:
            raise SystemExit('Qdrant is nonempty; refuse to adopt existing collections')
    env = {k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LC_ALL','TMPDIR','SYSTEMROOT')}
    env.update(MANDOL_TEST_QDRANT=a.qdrant, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    if a.qdrant_binary is not None:
        env['MANDOL_TEST_QDRANT_BINARY'] = str(a.qdrant_binary.resolve(strict=True))
    a.output.mkdir(parents=True, exist_ok=True)
    command=[sys.executable,'-m','pytest','tests','-q',
        '--ignore=tests/aml_contract/test_real_wire.py',
        '--junitxml='+str(a.output/'pytest.xml'), '--basetemp='+str(a.output/'tmp')]
    result = subprocess.run(command,cwd=ROOT,env=env)
    (a.output/'execution.json').write_text(json.dumps({'returncode':result.returncode,
        'inference':'deterministic fixtures / no real-model claim',
        'storage':'real isolated SQLite and scratch Qdrant'},indent=2)+'\n')
    raise SystemExit(result.returncode)

if __name__=='__main__':main()
