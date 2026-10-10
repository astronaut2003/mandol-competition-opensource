#!/usr/bin/env python3
"""Regression oracle for the reviewed frozen accuracy artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = {
    'textual': '914da2cc3457ea94ccfa6715bcf792ab33e938865d26dadc7a0bba87ade021e6',
    'code': '1ba2bba5aab6f9264ab228c47c1bc2032c2265e8f220fc1e2712cb3d3a430faa',
    'multimodal': '4af0e6ef21e4ba95caa8cfae55cce8b4bc27754d8e13a10c8834232ff2156f8a',
}
FILES = {
    'src/mandol_competition/textual/profiles/aml_official_v4_v1/profile.json': '851113b4480cd00b8c922e20cb0ac977750881ac7440701fb21f10eeb0ed58df',
    'src/mandol_competition/textual/profiles/aml_official_v4_v1/quality-policies.json': 'ddbbfc8c39a97aea3f598d57c10c61accb07335319c791bfdc29440d9324cbd0',
    'src/mandol_competition/textual/profiles/aml_official_v4_v1/quality-profile.json': 'fdc4bb47fcab086346e2f91c9adf3c48f8e3618ae78d7e88e7ec181995802c66',
    'src/mandol_competition/code/api/config.py': 'f85bf9e32e1b7d960e2a3a30bc8e098fbe1d823a3d4c07f41c08ea624d65e462',
    'src/mandol_competition/code/software/experience_retrieval.py': 'bdcb02eb8016f260053cfaaac839f7532257f958cd661fd9f449d8c99612b973',
    'configs/profiles/aml-official-eligible/multimodal-v1.json': 'abeb011ee2cebdb50253f727bd242e525a1ab8f85e3686e475a635634f8d9d84',
    'src/mandol_competition/multimodal/vlm/prompt.py': '36001196f07d8005a1ca1df9a40ab2e75352ad90925e340983f1682281fe57e6',
}


def verify(root=ROOT):
    results = {}
    for track, expected in SELECTIONS.items():
        value = json.loads((root / 'configs/profiles/submission' / (track + '.json')).read_text())
        actual = hashlib.sha256(json.dumps(value['profile'], sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        results['selection/' + track] = actual == expected == value['profile_sha256']
    for name, expected in FILES.items():
        results[name] = hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    return {'status': 'PASS' if all(results.values()) else 'FROZEN_PIPELINE_CONFLICT',
            'checks': results, 'base_commit': 'cd0e92b4506cd32bbefb956a50792e208c678592'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
