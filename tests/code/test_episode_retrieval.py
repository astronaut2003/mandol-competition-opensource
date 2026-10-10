from mandol_substrate.representation import MemoryUnit
from mandol_competition.code.software.episodes import EngineeringEpisodeBuilder
from mandol_competition.code.software.experience_retrieval import (
    EpisodeRetriever, EpisodeRetrievalConfig, EvidencePacker, GraphAwareReranker,
)
from mandol_competition.code.software.lexical import CodeBM25
from mandol_competition.code.software.query import CodingQueryRouter, EvidenceNeedDecomposer


def _history(session, text, user="u"):
    raw = MemoryUnit(uid=f"raw:{user}:{session}", raw_data={"text_content": text},
                     metadata={"user_id": user, "session_id": session, "role": "user"})
    episode = EngineeringEpisodeBuilder().build([raw], user_id=user, session_id=session, repository="owner/repo")
    return raw, episode


def test_episode_union_is_scoped_and_diverse():
    a, ea = _history("a", "JWTInvalidError in src/auth.py")
    b, eb = _history("b", "Implement retry contract in src/network.py")
    other, eo = _history("other", "JWTInvalidError JWTInvalidError", user="other")
    sources = {raw.uid: raw for raw in (a, b, other)}
    retriever = EpisodeRetriever([ea, eb, eo], sources, user_id="u", config=EpisodeRetrievalConfig(mode="hybrid", code_aware=True))
    hits = retriever.search("JWTInvalidError src/auth.py", top_k=20)
    assert hits[0].episode.episode_id == ea.episode_id
    assert all(hit.episode.user_id == "u" for hit in hits)
    assert len({hit.episode.episode_id for hit in hits}) == len(hits)
    assert retriever.search("", top_k=8) == []


def test_packer_respects_total_budget_and_preserves_source_excerpt():
    raw, episode = _history("a", "prefix " * 500 + "TypeError: expected int, got None\npytest test_config -q")
    sources = {raw.uid: raw}
    hits = EpisodeRetriever([episode], sources, user_id="u", config=EpisodeRetrievalConfig(code_aware=True)).search("TypeError: expected int, got None")
    packed = EvidencePacker(max_chars=2000, per_source_chars=500).pack(hits, sources, query="TypeError: expected int, got None", top_k=1)
    assert len(packed) == 1
    assert sum(len(item["content"]) for item in packed) <= 2000
    assert "TypeError: expected int, got None" in packed[0]["content"]
    assert episode.source_hashes[raw.uid] in packed[0]["content"]
    assert raw.raw_data["text_content"].startswith("prefix " * 500)


def test_router_and_decomposition_use_query_only():
    router = CodingQueryRouter()
    assert router.route("What failed in the previous attempt?").asks_for_failure
    assert router.route("TypeError traceback in login").intent == "DEBUG"
    assert router.route("Add asynchronous HTTP retry support").intent == "DEVELOPMENT"
    needs = EvidenceNeedDecomposer().decompose("Add asynchronous retry support while preserving the cancellation contract")
    assert len(needs) == 2
    assert all(need in "Add asynchronous retry support while preserving the cancellation contract" for need in needs)


def test_lexical_statistics_do_not_cross_candidate_scope():
    index = CodeBM25({"a": "getUserProfile retry", "b": "network retry", "other": "retry " * 1000})
    scoped = index.search("get user profile retry", candidate_uids=["a", "b"], top_k=1)
    fresh = CodeBM25({"a": "getUserProfile retry", "b": "network retry"}).search("get user profile retry", top_k=1)
    assert scoped == fresh
    assert scoped[0][0] == "a"


def test_graph_features_require_typed_edges_and_repository_scope():
    class Graph:
        typed = False

        def get_all_units(self):
            return [MemoryUnit(uid=name, raw_data={"qualified_name": name}, metadata={"repo_uid": "repo:r"})
                    for name in ("pkg.caller", "pkg.callee")]

        def get_successors(self, uid):
            return ["pkg.callee"] if uid == "pkg.caller" else []

        def get_predecessors(self, uid):
            return []

        def get_relationship(self, src, dst, kind):
            return {} if self.typed and (src, dst, kind) == ("pkg.caller", "pkg.callee", "CALLS") else None

    graph = Graph()
    _, episode = _history("a", "pkg.callee behavior")
    reranker = GraphAwareReranker(graph, repository="owner/repo", repository_uid="repo:r")
    assert reranker.features("pkg.caller", episode)["structural_proximity"] == 0
    graph.typed = True
    assert reranker.features("pkg.caller", episode)["structural_proximity"] == .5
    episode.repository = "another/repo"
    assert reranker.features("pkg.caller", episode)["structural_proximity"] == 0


def test_model_channel_failure_and_cross_user_results_fall_back_to_scoped_lexical():
    raw, episode = _history("a", "JWTInvalidError refresh_token")

    class InvalidChannel:
        def search(self, query, *, top_k):
            return [("another-users-private-unit", 1.0)]

    retriever = EpisodeRetriever([episode], {raw.uid: raw}, user_id="u", raw_channel=InvalidChannel())
    assert retriever.search("JWTInvalidError")[0].episode.episode_id == episode.episode_id
    assert retriever.fallback_count == 1


def test_long_episode_cannot_exhaust_distinct_episode_candidate_budget():
    repeated = [MemoryUnit(uid=f"long:{i}", raw_data={"text_content": "retry"},
                           metadata={"user_id": "u", "session_id": "long", "role": "user"}) for i in range(20)]
    long = EngineeringEpisodeBuilder().build(repeated, user_id="u", session_id="long")
    raw, short = _history("short", "retry cancellation contract behavior")
    sources = {unit.uid: unit for unit in repeated + [raw]}
    retriever = EpisodeRetriever([long, short], sources, user_id="u", config=EpisodeRetrievalConfig(candidate_k=2))
    hits = retriever.search("retry", top_k=2)
    assert len(hits) == 2
    assert {hit.episode.episode_id for hit in hits} == {long.episode_id, short.episode_id}


def test_optional_card_cap_keeps_later_experience_with_exact_bounded_raw():
    from mandol_competition.code.software.experience_retrieval import EpisodeHit
    units = [MemoryUnit(uid=f"raw:{session}:{i}", raw_data={"text_content": "Task contract.\n" + "evidence " * 500},
                        metadata={"user_id": "u", "session_id": session, "role": "user" if i == 0 else "tool"})
             for session in ("first", "second") for i in range(3)]
    sources = {unit.uid: unit for unit in units}
    episodes = [EngineeringEpisodeBuilder().build([u for u in units if u.metadata["session_id"] == session],
                user_id="u", session_id=session) for session in ("first", "second")]
    hits = [EpisodeHit(ep, 1. / (i + 1), ep.source_uids[:1]) for i, ep in enumerate(episodes)]
    plain = EvidencePacker(max_cards=2, max_chars=3000).pack(hits, sources, query="contract", top_k=2)
    compact = EvidencePacker(max_cards=2, max_chars=3000, per_card_chars=1500).pack(hits, sources, query="contract", top_k=2)
    assert len(plain) == 1 and len(compact) == 2
    assert [card["id"] for card in compact] == [ep.episode_id for ep in episodes]
    assert all(len(card["content"]) <= 1500 and "[raw excerpt" in card["content"] for card in compact)
    assert all(ep.source_hashes[ep.source_uids[0]] in card["content"] for ep, card in zip(episodes, compact))
    assert all(unit.raw_data["text_content"].endswith("evidence " * 500) for unit in sources.values())
