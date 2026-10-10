from types import SimpleNamespace

from mandol_competition.textual.core.api_quality_trace import ApiQualityTrace
from mandol_substrate.state import Scope


def test_navigation_hydration_does_not_guess_ids_from_duplicate_text():
    scope = Scope("raw", "same_user")
    typed = Scope("typed", "same_user")
    memories = {"raw-a": SimpleNamespace(text_content="identical text"),
                "raw-b": SimpleNamespace(text_content="identical text")}
    state = SimpleNamespace(get_memories=lambda s, ids: {uid: memories[uid] for uid in ids if uid in memories})
    retriever = SimpleNamespace(retrieve=lambda view, *args, **kwargs: (
        [(SimpleNamespace(uid="raw-a"), 1)], {"legs": {"bm25": [("raw-a", 1)]}}))
    reranker = SimpleNamespace(rerank=lambda query, documents: [float(i) for i in range(len(documents))])

    def expand(scope, view, query, dense, sparse, raw_ids, canonical):
        state.get_memories(typed, [])
        # Eligibility often leaves the last raw hydration as a small support
        # subset. The observer must use explicit final navigation UID order.
        canonical.update(state.get_memories(scope, ["raw-b"]))
        return ["raw-b", "raw-a"], canonical, {"selected": {"procedure": ["derived"]}, "route": "RULE"}

    navigation = SimpleNamespace(expand=expand)
    service = SimpleNamespace(ready=True, scope=lambda user_id: Scope("raw", user_id), state=state,
                              retriever=retriever, reranker=reranker, navigation=navigation, query_policy=None)

    def search(user_id, query, top_k):
        view = SimpleNamespace(scope=scope, published_seq=1)
        fused, _ = retriever.retrieve(view, query, None, None)
        ids = [unit.uid for unit, _ in fused]
        canonical = state.get_memories(scope, ids)
        ids, canonical, _ = navigation.expand(scope, view, query, None, None, ids, canonical)
        scores = reranker.rerank(query, [canonical[uid].text_content for uid in ids])
        return [{"id": uid, "content": canonical[uid].text_content, "score": score}
                for uid, score in zip(ids, scores)]

    service.search = search
    with ApiQualityTrace(service) as trace:
        hits = service.search("same_user", "What rules?", 100)
        audit = trace.take()
        assert [hit["id"] for hit in hits] == audit["reranker"]["candidate_ids"] == ["raw-b", "raw-a"]
        assert audit["derived_hydration_calls"] == 1
        assert audit["navigation"]["route"] == "RULE"
    assert service.search is search
