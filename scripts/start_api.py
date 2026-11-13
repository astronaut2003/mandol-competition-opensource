#!/usr/bin/env python3
"""Start one reviewed current-workflow API against configured external services."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]

def validate_frozen_environment():
    """Refuse ambient research/quality overrides; never silently rewrite them."""
    forbidden = (
        'AML_EXPERIMENTAL_HLM_POLICY', 'MANDOL_TEXTUAL_QUALITY_CANDIDATE',
        'MANDOL_CODE_QUALITY_CANDIDATE', 'AML_CODING_RETRIEVAL_MODE',
        'AML_CODING_CODE_AWARE', 'AML_CODING_ROUTER', 'AML_CODING_EVIDENCE_NEEDS',
        'AML_CODING_ANCHOR_WEIGHT', 'AML_CODING_OUTCOME_WEIGHT',
        'AML_CODING_ROUTE_WEIGHT', 'AML_CODING_MAX_CARDS',
        'AML_CODING_EVIDENCE_CHARS', 'AML_CODING_PER_CARD_CHARS',
        'MANDOL_MM_RETRIEVAL_METHODS', 'MANDOL_MM_RERANK_METHOD',
        'MANDOL_MM_EMBEDDING_MODEL', 'MANDOL_MM_EMBEDDING_DIM',
    )
    if any(os.getenv(name) for name in forbidden):
        raise ValueError('Unset research and quality overrides for frozen startup')
    fixed = {
        'MANDOL_MM_VLM_BACKEND': 'openai_compat',
        'MANDOL_MM_VLM_MODEL': 'gpt-4o-mini-closeai',
        'MANDOL_MM_VLM_REQUEST_MODEL': 'gpt-4o-mini',
        'MANDOL_MM_PROMPT_VERSION': 'multimodal_sgm_v3_ocr_lines',
        'MANDOL_MM_SCHEMA_VERSION': '1',
        'MANDOL_MM_EVIDENCE_POLICY': 'source-grounded-recall-v4',
        'MANDOL_MM_VISION_FAILURE_POLICY': 'retry-required-v1',
        'MANDOL_MM_VISION_SOURCE_FALLBACK': '1',
        'MANDOL_MM_FULL_DATABASE_AUDIT': '0',
        'MANDOL_MM_VLM_OUTPUT_TOKENS': '1024',
        'MANDOL_MM_VLM_MAX_OUTPUT_TOKENS': '1024',
    }
    if any(os.getenv(name, expected) != expected for name, expected in fixed.items()):
        raise ValueError('Frozen VLM model/prompt/schema override refused')
    profile = os.getenv('MANDOL_MM_RETRIEVAL_PROFILE')
    if profile:
        p = Path(profile)
        frozen = ROOT/'configs/profiles/aml-official-eligible/multimodal-v1.json'
        if p.is_symlink() or p.read_bytes() != frozen.read_bytes():
            raise ValueError('Frozen multimodal profile override refused')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('track', choices=('textual', 'code', 'multimodal'))
    p.add_argument('--config', type=Path, default=ROOT/'configs/deployment/current-cloud.example.toml')
    p.add_argument('--port', type=int)
    p.add_argument('--initialize', action='store_true',
        help='Initialize a fresh Multimodal store with a full initial audit; later starts omit this flag')
    a = p.parse_args()
    if a.initialize and a.track != 'multimodal':
        p.error('--initialize applies only to a fresh Multimodal store')
    validate_frozen_environment()
    frozen = runpy.run_path(str(ROOT/'deployment/cluster/frozen.py'))['verify'](ROOT)
    if frozen['status'] != 'PASS':
        raise SystemExit('Frozen artifact mismatch; refusing startup')
    os.environ['MANDOL_CONFIG'] = str(a.config.resolve())
    os.environ.setdefault('MANDOL_MM_RETRIEVAL_PROFILE', str(ROOT/'configs/profiles/aml-official-eligible/multimodal-v1.json'))
    os.environ.setdefault('MANDOL_MM_MODEL_DEVICE', 'external-http')
    os.environ.setdefault('MANDOL_MM_VLM_BACKEND', 'openai_compat')
    os.environ.setdefault('MANDOL_MM_VLM_MODEL', 'gpt-4o-mini-closeai')
    os.environ.setdefault('MANDOL_MM_VLM_REQUEST_MODEL', 'gpt-4o-mini')
    os.environ.setdefault('MANDOL_MM_VLM_API_KEY_ENV', 'CLOSEAI_API_KEY')
    os.environ.setdefault('MANDOL_MM_PROMPT_VERSION', 'multimodal_sgm_v3_ocr_lines')
    os.environ.setdefault('MANDOL_MM_SCHEMA_VERSION', '1')
    os.environ.setdefault('MANDOL_MM_VLM_OUTPUT_TOKENS', '1024')
    os.environ.setdefault('MANDOL_MM_VLM_MAX_OUTPUT_TOKENS', '1024')
    os.environ.setdefault('MANDOL_MM_EVIDENCE_POLICY', 'source-grounded-recall-v4')
    os.environ.setdefault('MANDOL_MM_VISION_FAILURE_POLICY', 'retry-required-v1')
    os.environ.setdefault('MANDOL_MM_VISION_SOURCE_FALLBACK', '1')
    os.environ.setdefault('MANDOL_MM_FULL_DATABASE_AUDIT', '0')
    os.environ.setdefault('MANDOL_INTERNAL_ENDPOINTS_LOOPBACK_ONLY', '1')
    # Original deployment deadlines/admission capacities, independent of quality.
    for key,value in {'AML_ADD_TIMEOUT_SECONDS':'1740','AML_SEARCH_TIMEOUT_SECONDS':'1740',
      'MANDOL_MM_ADD_TIMEOUT_SECONDS':'1740','MANDOL_MM_SEARCH_TIMEOUT_SECONDS':'1740',
      'MANDOL_API_WORKER_CAPACITY':'8','MANDOL_API_MAX_QUEUE':'8',
      'MANDOL_API_QUEUE_TIMEOUT_SECONDS':'5'}.items():os.environ.setdefault(key,value)
    from mandol_competition.config import load_config
    settings = load_config()
    if a.initialize:
        root = settings.track('multimodal').root
        if any((root/name).exists() or (root/name).is_symlink()
               for name in ('state.sqlite3', 'canonical.sqlite3')):
            raise SystemExit('--initialize requires fresh storage; existing state needs its explicit recovery procedure')
        os.environ['MANDOL_MM_FULL_DATABASE_AUDIT'] = '1'
    if settings.track(a.track).inference_contract != 'aml-official-eligible-v1':
        raise SystemExit('Formal launcher requires the native v4 contract')
    port = a.port or {'textual':18600,'code':18601,'multimodal':18602}[a.track]
    os.chdir(ROOT)
    import uvicorn
    uvicorn.run('mandol_competition.runtime.apps:'+a.track, factory=True,
        host='127.0.0.1', port=port, workers=1, access_log=False,
        timeout_keep_alive=30, limit_concurrency=64, timeout_graceful_shutdown=1820)

if __name__ == '__main__':main()
