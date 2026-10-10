from dataclasses import replace

import numpy as np
import pytest

from mandol_competition.textual.core.retrieval import fuse_legs
from mandol_substrate.representation import MemoryUnit
from mandol_substrate.lexical.sqlite_exact import LexicalHit
from mandol_substrate.state import StateUnavailable

pytestmark = pytest.mark.unit


def fixture_legs():
    return {
        "cosine": [LexicalHit("b", 0.9), LexicalHit("a", 0.8)],
        "bm25": [LexicalHit("a", 3), LexicalHit("c", 1)],
        "splade": [LexicalHit("c", 4), LexicalHit("b", 2)],
    }


def test_rrf_independent_rank_origin_missing_leg_and_dedup(profile):
    legs = fixture_legs()
    units = {u: MemoryUnit(u, {"text_content": u}) for u in "abc"}
    result = fuse_legs(legs, units, profile, top_k=3)
    expected = {
        u: sum(
            1 / (60 + i)
            for hits in legs.values()
            for i, h in enumerate(hits, 1)
            if h.logical_uid == u
        )
        for u in units
    }
    assert {u.uid: s for u, s in result} == pytest.approx(expected, abs=1e-12)
    legs["splade"] = []
    expected = {
        u: sum(
            1 / (60 + i)
            for hits in legs.values()
            for i, h in enumerate(hits, 1)
            if h.logical_uid == u
        )
        for u in units
    }
    assert {
        u.uid: s for u, s in fuse_legs(legs, units, profile, top_k=3)
    } == pytest.approx(expected, abs=1e-12)
    legs["bm25"].append(legs["bm25"][0])
    with pytest.raises(StateUnavailable):
        fuse_legs(legs, units, profile, top_k=3)


@pytest.mark.parametrize("normalization", ["min_max", "z_score", "none"])
def test_weighted_missing_zero_and_normalization(profile, normalization):
    weights = {"cosine": 2, "bm25": 3, "splade": 1}
    profile = replace(
        profile,
        fusion="weighted",
        options={"weights": weights, "normalization": normalization},
    )
    legs = fixture_legs()
    units = {u: MemoryUnit(u, {"text_content": u}) for u in "abc"}
    expected = np.zeros(3)
    for method, hits in legs.items():
        values = np.array(
            [next((h.score for h in hits if h.logical_uid == u), 0) for u in "abc"]
        )
        if normalization == "min_max":
            values = (values - values.min()) / (values.max() - values.min())
        elif normalization == "z_score":
            values = (values - values.mean()) / values.std()
            values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        expected += weights[method] * values / 6
    assert {
        u.uid: s for u, s in fuse_legs(legs, units, profile, top_k=3)
    } == pytest.approx(dict(zip("abc", expected, strict=True)), abs=1e-12)


def test_mmr_scalar_oracle_and_no_missing_vector_fallback(profile):
    profile = replace(profile, options={"mmr": True, "mmr_lambda": 0.6})
    legs = fixture_legs()
    units = {u: MemoryUnit(u, {"text_content": u}) for u in "abc"}
    vectors = {"a": [1, 0, 0], "b": [0.8, 0.6, 0], "c": [0, 1, 0]}
    query = [1, 0, 0]
    result = fuse_legs(
        legs, units, profile, top_k=3, query_vector=query, vectors=vectors
    )
    selected = []
    while len(selected) < 3:
        scores = {
            u: 0.6 * np.dot(query, v)
            - (1 - 0.6) * max([0] + [np.dot(v, vectors[s]) for s in selected])
            for u, v in vectors.items()
            if u not in selected
        }
        selected.append(max(scores, key=scores.get))
    assert [u.uid for u, s in result] == selected
    with pytest.raises(StateUnavailable):
        fuse_legs(legs, units, profile, top_k=3, query_vector=query, vectors={})
