#!/usr/bin/env python3
"""Check public runtime selections and resource bytes without network access."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'configs/frozen-pipelines.json').read_text())
    if manifest.get('schema') != 'mandol-public-runtime-integrity-v2':
        raise ValueError('Unsupported public integrity manifest')
    results = {}
    if set(manifest['selections']) != {'textual', 'code', 'multimodal'}:
        raise ValueError('Incomplete public track selections')
    for track, expected in manifest['selections'].items():
        value = json.loads((root / 'configs/profiles/submission' / (track + '.json')).read_text())
        actual = hashlib.sha256(json.dumps(value['profile'], sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        results['selection/' + track] = actual == expected == value['profile_sha256']
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe public integrity path')
        path = root / name
        results[name] = (path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(root)
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected)
    return {'status': 'PASS' if all(results.values()) else 'FROZEN_PIPELINE_CONFLICT',
            'checks': results, 'snapshot': manifest['snapshot'],
            'scope': 'public source integrity; no production or accuracy attestation'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
