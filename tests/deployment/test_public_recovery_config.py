"""Public entrypoints propagate the selected recovery recipe without providers."""
from dataclasses import replace
from pathlib import Path
import os
import runpy
import sys
from types import SimpleNamespace

import pytest

from mandol_competition.config import load_config
from mandol_competition.multimodal.config import ConfigurationError, get_config
from mandol_competition.runtime import apps

ROOT = Path(__file__).resolve().parents[2]


def capture_config(monkeypatch, mode):
    from mandol_competition.multimodal import api
    cfg = load_config(ROOT/'configs/deployment/current-cloud.example.toml').track('multimodal')
    cfg = replace(cfg, runtime=replace(cfg.runtime, mode=mode))
    monkeypatch.setattr(apps, '_settings', lambda track: cfg)
    monkeypatch.setattr(api, 'create_app', lambda value: value)
    get_config.cache_clear()
    return cfg


@pytest.mark.parametrize('mode', ['legacy', 'local', 'cluster'])
@pytest.mark.parametrize('source,audit', [('1','0'), ('0','1')])
def test_factory_keeps_explicit_recovery_flags(monkeypatch, mode, source, audit):
    capture_config(monkeypatch, mode)
    monkeypatch.setenv('MANDOL_MM_VISION_FAILURE_POLICY', 'retry-required-v1')
    monkeypatch.setenv('MANDOL_MM_VISION_SOURCE_FALLBACK', source)
    monkeypatch.setenv('MANDOL_MM_FULL_DATABASE_AUDIT', audit)
    try:
        actual = apps.multimodal()
        assert actual.vision_source_fallback is (source == '1')
        assert actual.full_database_audit is (audit == '1')
        assert actual.vision_failure_policy == 'retry-required-v1'
    finally:
        get_config.cache_clear()


@pytest.mark.parametrize('existing', ['state.sqlite3', 'canonical.sqlite3', 'dangling_link'])
def test_initialize_refuses_existing_or_linked_storage(monkeypatch, tmp_path, existing):
    from mandol_competition import config
    import uvicorn
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    monkeypatch.delenv('MANDOL_MM_FULL_DATABASE_AUDIT', raising=False)
    if existing == 'dangling_link':
        (tmp_path/'state.sqlite3').symlink_to(tmp_path/'absent')
    else:
        (tmp_path/existing).write_bytes(b'synthetic existing state')
    monkeypatch.setattr(config, 'load_config', lambda: SimpleNamespace(
        track=lambda name: SimpleNamespace(root=tmp_path)))
    monkeypatch.setattr(uvicorn, 'run', lambda *a, **k: pytest.fail('Refused state reached API startup'))
    monkeypatch.setattr(sys, 'argv', ['start_api.py', 'multimodal', '--initialize'])
    with pytest.raises(SystemExit, match='fresh storage'):
        runpy.run_path(str(ROOT/'scripts/start_api.py'))['main']()


@pytest.mark.parametrize('name', ['MANDOL_MM_VISION_SOURCE_FALLBACK', 'MANDOL_MM_FULL_DATABASE_AUDIT'])
def test_local_factory_rejects_invalid_flag(monkeypatch, name):
    capture_config(monkeypatch, 'local')
    monkeypatch.setenv(name, 'sometimes')
    with pytest.raises(ConfigurationError):
        apps.multimodal()


def test_public_launcher_sets_effective_recovery_recipe(monkeypatch):
    import uvicorn
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    capture_config(monkeypatch, 'legacy')
    for name in ('MANDOL_MM_VISION_SOURCE_FALLBACK', 'MANDOL_MM_FULL_DATABASE_AUDIT',
                 'MANDOL_MM_VISION_FAILURE_POLICY'):
        monkeypatch.delenv(name, raising=False)
    captured = []
    monkeypatch.setattr(uvicorn, 'run', lambda *args, **kwargs: captured.append(apps.multimodal()))
    monkeypatch.setattr(sys, 'argv', ['start_api.py', 'multimodal'])
    try:
        runpy.run_path(str(ROOT/'scripts/start_api.py'))['main']()
        assert len(captured) == 1
        cfg = captured[0]
        assert cfg.vision_source_fallback is True and cfg.full_database_audit is False
        assert cfg.vlm_output_tokens == cfg.vlm_max_output_tokens == 1024
        assert cfg.evidence_policy == 'source-grounded-recall-v4'
    finally:
        get_config.cache_clear()
