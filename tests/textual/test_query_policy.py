import json
from dataclasses import asdict, replace

import pytest

from mandol_competition.textual.core.query_policy import QueryPolicy
from mandol_substrate.state.models import ProfileMismatch


def policy(profile, **kwargs):
    return QueryPolicy(
        profile.hash, ("cosine", "bm25", "splade"),
        (("cosine", 200), ("bm25", 20), ("splade", 100)), **kwargs,
    )


def test_query_policy_cannot_rewrite_physical_truth_or_share_mutable_options(profile):
    before, old_hash = asdict(profile), profile.hash
    chosen = policy(profile, fused_k=50, graph_hops=0, semantic_neighbors=0)
    effective = chosen.effective_profile(profile)
    assert effective.options["cosine_k"] == 200
    assert effective.options["bm25_k"] == 20
    assert effective.options["graph_hops"] == 0
    assert effective.embedding_model == profile.embedding_model
    assert effective.bm25_k1 == profile.bm25_k1
    effective.options["arbitrary"] = True
    assert asdict(profile) == before and profile.hash == old_hash
    assert chosen.hash != replace(chosen, fused_k=51).hash
    with pytest.raises(ProfileMismatch):
        chosen.effective_profile(replace(profile, embedding_revision="another"))


def test_roundtrip_and_unknown_model_change_rejected(profile, tmp_path):
    selected = policy(profile)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(asdict(selected)))
    assert QueryPolicy.load(path) == selected
    data = asdict(selected)
    data["embedding_model"] = "different"
    path.write_text(json.dumps(data))
    with pytest.raises(TypeError):
        QueryPolicy.load(path)


@pytest.mark.parametrize("change", [
    {"fused_k": 1025}, {"fused_k": True}, {"graph_hops": 3},
    {"semantic_neighbors": -1}, {"semantic_threshold": float("nan")},
    {"normalization": "none"}, {"rrf_k": 0}, {"rerank": "false"},
    {"weights": (("cosine", 1),)}, {"fusion": "raw_sum"},
])
def test_invalid_policy_fails_closed(profile, change):
    with pytest.raises(ValueError):
        policy(profile, **change)


def test_weights_and_requested_legs_are_explicit(profile):
    selected = policy(profile, fusion="weighted", weights=(
        ("cosine", 2), ("bm25", 1), ("splade", 1),
    ))
    assert selected.effective_profile(profile).options["weights"]["cosine"] == 2
    for depths in ((("cosine", 1),), (("cosine", 0), ("bm25", 1), ("splade", 1))):
        with pytest.raises(ValueError):
            replace(selected, leg_depths=depths)
    missing = replace(profile, methods=("cosine",))
    with pytest.raises(ProfileMismatch):
        replace(selected, storage_profile_hash=missing.hash).effective_profile(missing)


def test_actual_sqlite_lexical_query_policy_keeps_truth_and_exact_leg_cap(
    profile, store, scope,
):
    from types import SimpleNamespace

    from mandol_competition.textual.core.retrieval import CompetitionRetriever
    from mandol_substrate.state import CanonicalMemory, Mutation

    memories = tuple(CanonicalMemory(str(i), "alpha " * (i + 1)) for i in range(5))
    receipt = store.prepare(scope, "q-policy", "fixed", Mutation(memories=memories), {})
    store.publish(scope, receipt["mutation_seq"])
    before = store.scope_state(scope)
    stats = store.lexical.inspect_stats(scope)
    projection = SimpleNamespace(profile_hash=profile.hash)
    selected = QueryPolicy(profile.hash, ("bm25",), (("bm25", 2),))
    retriever = CompetitionRetriever(store, projection, profile, query_policy=selected)
    result, audit = retriever.retrieve(
        SimpleNamespace(scope=scope, published_seq=1), "alpha", None, None, top_k=4,
    )
    assert len(result) == 2  # No hidden max(top_k, leg_k) in an explicit policy.
    assert set(audit["legs"]) == {"bm25"}
    assert audit["query_policy_hash"] == selected.hash
    assert retriever.profile is profile and projection.profile_hash == profile.hash
    assert store.scope_state(scope) == before
    assert store.lexical.inspect_stats(scope) == stats
    assert all(unit.uid in {m.logical_uid for m in memories} for unit, _ in result)


def test_explicit_query_policy_does_not_relax_vector_version_fencing(profile, store, scope):
    from types import SimpleNamespace

    from mandol_competition.textual.core.retrieval import CompetitionRetriever
    from mandol_substrate.state import CanonicalMemory, Mutation, StateUnavailable

    receipt = store.prepare(
        scope, "q-policy", "fixed", Mutation(memories=(CanonicalMemory("a", "alpha"),)), {},
    )
    store.publish(scope, receipt["mutation_seq"])
    projection = SimpleNamespace(
        profile_hash=profile.hash,
        search_dense=lambda *a, **k: [SimpleNamespace(
            logical_uid="a", score=.9, metadata={"version": 0, "commit_seq": 1},
        )],
    )
    retriever = CompetitionRetriever(
        store, projection, profile,
        query_policy=QueryPolicy(profile.hash, ("cosine",), (("cosine", 1),)),
    )
    with pytest.raises(StateUnavailable, match="version differ"):
        retriever.retrieve(SimpleNamespace(scope=scope, published_seq=1), "q", [1, 0, 0], {})
