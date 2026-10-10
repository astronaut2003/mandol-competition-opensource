"""Pinned QA source loading fails before executing unverified external code."""

import pytest

from mandol_competition.textual.core import aml_protocol_sources as sources
from mandol_competition.textual.core.evaluation_protocol import (
    EvaluationProtocol,
    LEGACY_EVALUATION_PROTOCOL,
    OFFICIAL_REVISION,
)


def test_missing_sources_fail_before_network_or_model_calls(tmp_path, monkeypatch):
    monkeypatch.setenv(sources.SOURCE_DIRECTORY_ENV, str(tmp_path))
    monkeypatch.setattr(sources, "urlopen", lambda *a, **kw: pytest.fail("implicit download"))
    with pytest.raises(FileNotFoundError, match="fetch_aml_protocol_sources.py"):
        EvaluationProtocol("locomo")
    assert EvaluationProtocol("locomo", LEGACY_EVALUATION_PROTOCOL).manifest["status"] == "DIAGNOSTIC_ONLY"


def test_changed_source_fails_before_execution(tmp_path, monkeypatch):
    monkeypatch.setenv(sources.SOURCE_DIRECTORY_ENV, str(tmp_path))
    (tmp_path / "locomo.py").write_text("raise RuntimeError('must never execute')\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        EvaluationProtocol("locomo")


def test_verify_only_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "urlopen", lambda *a, **kw: pytest.fail("network during verify-only"))
    with pytest.raises(FileNotFoundError, match="source is missing"):
        sources.fetch_sources(tmp_path, verify_only=True)


def test_fetch_refuses_checkout_destination_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "urlopen", lambda *a, **kw: pytest.fail("network before path check"))
    with pytest.raises(ValueError, match="outside the project checkout"):
        sources.fetch_sources(tmp_path / "cache", checkout=tmp_path)


def test_default_cache_is_revision_scoped(tmp_path, monkeypatch):
    monkeypatch.delenv(sources.SOURCE_DIRECTORY_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert sources.source_directory(OFFICIAL_REVISION) == tmp_path / "mandol-competition" / "aml" / OFFICIAL_REVISION
