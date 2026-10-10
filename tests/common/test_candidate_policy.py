from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as N

import pytest

from mandol_substrate.retrieval import RetrievalBackend
from mandol_substrate.state import CanonicalMemory
from mandol_substrate.retrieval.candidates import CandidateRoute
from .fakes import test_profile as profile_fixture


def backend(policy=None):
    calls = {}
    def leg(name, ids):
        def run(*args, limit, **kwargs):
            calls[name] = limit
            return [N(logical_uid=i, score=1.) for i in ids[:limit]]
        return run
    state = N(lexical=N(search=leg("bm25", ["b", "c"])),
        get_memories=lambda scope, ids: {i: CanonicalMemory(i, i, commit_seq=1) for i in ids})
    projection = N(policy=N(physical_search_budget=1000), search_dense=leg("cosine", ["a", "b"]),
        search_splade=leg("splade", ["c", "b"]))
    profile = replace(profile_fixture(), options={"candidate_k": 100, **({"retrieval_policy": policy} if policy else {})})
    result = RetrievalBackend(state, projection, N(search_view=lambda _: nullcontext(N(published_seq=1))),
        N(encode_query=lambda _: ([1, 0, 0], {1: 1})), profile)
    result.verify_scope = lambda _: None  # This unit tests ranking; real verification has separate fault tests.
    return result, calls


def test_absent_policy_preserves_original_rrf_and_requested_count():
    value, calls = backend()
    result = value.search(None, "q", top_k=100)
    assert result[0][0].logical_uid == "b"
    assert result[0][1] == 1 / 62 + 1 / 62 + 1 / 61
    assert calls == {"cosine": 100, "splade": 100, "bm25": 100}


def test_explicit_depths_weights_and_trace_are_versioned():
    value, calls = backend({"version": "candidate-depth-weighted-rrf-v1",
        "depths": {"cosine": 1, "splade": 1, "bm25": 1}, "weights": {"cosine": 3}, "pre_rerank_k": 100})
    trace = {}
    value.observer = lambda name, values: trace.update({name: values})
    assert value.search(None, "q", top_k=2)[0][0].logical_uid == "a"
    assert calls == {"cosine": 1, "splade": 1, "bm25": 1}
    assert trace["fusion"]["union_count"] == 3
    assert "q" not in str(trace)


@pytest.mark.parametrize("policy", [
    {"version": "unversioned"},
    {"version": "candidate-depth-weighted-rrf-v1", "weights": {"cosine": float("nan")}},
    {"version": "candidate-depth-weighted-rrf-v1", "depths": {"bm25": 0}},
])
def test_invalid_policies_fail_before_search(policy):
    with pytest.raises(ValueError):
        backend(policy)


def test_navigation_keeps_raw_candidates_and_uses_one_encode_and_rerank():
    value, _ = backend()
    encodes, docs = [], []
    value.encoders.encode_query = lambda q: (encodes.append(q) or ([1,0,0], {1:1}))
    value.reranker = N(score=lambda q, texts: (docs.append(texts) or list(range(len(texts)))))
    result = value.search(None, 'q', top_k=100,
        candidate_route=lambda view: CandidateRoute(None, ('new-source',)))
    assert encodes == ['q'] and len(docs) == 1
    assert set(docs[0]) == {'a','b','c','new-source'}
    assert result[0][0].logical_uid == 'new-source'
    with pytest.raises(ValueError):
        CandidateRoute(('a',), ('foreign-source',))
    with pytest.raises(ValueError):
        CandidateRoute(None, tuple(str(i) for i in range(17)))


def test_navigation_documents_do_not_dilute_candidate_subset_bm25(tmp_path, profile, scope):
    from mandol_substrate.state import Mutation, SQLiteStateStore

    subset = replace(profile, options={"lexical_statistics": "candidate-subset-v1"})
    state = SQLiteStateStore(tmp_path / "subset.sqlite3", subset)
    try:
        raw = (CanonicalMemory("raw-a", "red station trip"),
               CanonicalMemory("raw-b", "blue station"))
        state.prepare(scope, "raw", "raw-body", Mutation(memories=raw), {})
        state.publish(scope, 1)
        ids = [memory.logical_uid for memory in raw]
        before = state.lexical.search(scope, "station trip", candidate_uids=ids)
        nav = CanonicalMemory("nav", "station trip " * 100, memory_type="visual_episode")
        state.prepare(scope, "nav", "nav-body", Mutation(memories=(nav,)), {})
        state.publish(scope, 2)
        after = state.lexical.search(scope, "station trip", candidate_uids=ids)
        assert before and after == before
    finally:
        state.close()
