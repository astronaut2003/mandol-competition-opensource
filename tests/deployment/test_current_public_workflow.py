"""The documented template selects the current methods without model requests."""
from pathlib import Path

from mandol_competition.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def test_current_template_selects_isolated_current_methods(monkeypatch):
    monkeypatch.delenv('MANDOL_CONFIG', raising=False)
    settings = load_config(ROOT / 'configs/deployment/current-cloud.example.toml')
    tracks = [settings.track(name) for name in ('textual', 'code', 'multimodal')]
    assert len({track.root for track in tracks}) == 3
    assert len({track.collection for track in tracks}) == 3
    assert {track.inference_contract for track in tracks} == {'aml-official-eligible-v1'}
    assert {track.reranker_backend for track in tracks} == {'dashscope-qwen37-v1'}
    assert settings.track('textual').evidence_profile == 'textual-t0-observation-history-v1'
    assert settings.track('code').retrieval_policy == 'code-native-sparse-rrf100-v1'
    assert settings.track('multimodal').retrieval_policy == 'mm-qwen37-same-pool-rank-only-v1'
    assert settings.track('multimodal').inference['vlm'] == 'https://vlm.example.invalid/v1'
    assert all(track.quality_candidate is None for track in tracks)
