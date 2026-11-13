"""The public launcher must reject accidentally inherited research policy."""
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name,value', [
    ('MANDOL_TEXTUAL_QUALITY_CANDIDATE', '/tmp/research.json'),
    ('MANDOL_CODE_QUALITY_CANDIDATE', '/tmp/research.json'),
    ('AML_CODING_ANCHOR_WEIGHT', '0.5'),
    ('MANDOL_MM_VLM_MODEL', 'unapproved-model'),
    ('MANDOL_MM_VLM_BACKEND', 'fake'),
    ('MANDOL_MM_PROMPT_VERSION', 'different-prompt'),
    ('MANDOL_MM_EVIDENCE_POLICY', 'baseline'),
    ('MANDOL_MM_EVIDENCE_POLICY', 'source-grounded-delivery-v3'),
    ('MANDOL_MM_VISION_FAILURE_POLICY', 'degrade'),
    ('MANDOL_MM_VISION_SOURCE_FALLBACK', '0'),
    ('MANDOL_MM_FULL_DATABASE_AUDIT', '1'),
    ('MANDOL_MM_VLM_MAX_OUTPUT_TOKENS', '16384'),
])
def test_launcher_refuses_quality_changes(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    validate = runpy.run_path(str(ROOT/'scripts/start_api.py'))['validate_frozen_environment']
    with pytest.raises(ValueError):
        validate()


def test_launcher_checks_relocated_profile_bytes(monkeypatch, tmp_path):
    validate = runpy.run_path(str(ROOT/'scripts/start_api.py'))['validate_frozen_environment']
    profile = tmp_path/'profile.json'
    profile.write_bytes((ROOT/'configs/profiles/aml-official-eligible/multimodal-v1.json').read_bytes())
    monkeypatch.setenv('MANDOL_MM_RETRIEVAL_PROFILE', str(profile))
    validate()
    profile.write_text('{}')
    with pytest.raises(ValueError):
        validate()
