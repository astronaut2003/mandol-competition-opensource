import pytest

from mandol_substrate.representation import MemoryUnit
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.persistence import CanonicalStore
from mandol_competition.multimodal.store import SPACE_NAME, UserGraphRegistry


class FakeGraph:
    def __init__(self, fail_after=None):
        self.units = {}
        self.calls = []
        self.closed = False
        self.fail_after = fail_after

    def add_unit(self, unit, **kwargs):
        if self.fail_after is not None and len(self.units) >= self.fail_after:
            raise RuntimeError("index failed")
        assert kwargs["content_type_for_embedding"] == "text"
        assert kwargs["explicit_content_for_embedding"] == unit.raw_data["text_content"]
        assert kwargs["space_names"] == [SPACE_NAME]
        assert isinstance(kwargs["generate_sparse_embedding"], bool)
        self.calls.append((unit.uid, kwargs))
        self.units[unit.uid] = unit

    def rebuild_all_indexes(self):
        self.calls.append("rebuild")

    def get_multi_retriever(self):
        return self

    def smart_search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        matches = [unit for unit in self.units.values() if query in unit.text_cached]
        return [(unit, 0.9) for unit in matches[: kwargs["top_k"]]]

    def close(self):
        self.closed = True


def unit(uid, user, text):
    return MemoryUnit(uid, {"text_content": text}, {"user_id": user})


def setup(tmp_path, *, max_graphs=64, fail_after=None):
    config = MultimodalConfig(data_dir=tmp_path, max_graphs=max_graphs)
    store = CanonicalStore(tmp_path / "canonical.db")
    built = []

    def factory(user_id):
        graph = FakeGraph(fail_after if not built else None)
        built.append((user_id, graph))
        return graph

    registry = UserGraphRegistry(config, store, graph_factory=factory)
    return registry, store, built


def test_uses_smart_search_and_prevents_cross_user_units(tmp_path):
    registry, store, built = setup(tmp_path)
    with registry.acquire("alice") as session:
        session.add_units([unit("a", "alice", "red bag")])
        assert [hit.uid for hit, _ in session.search("red", top_k=100)] == ["a"]
        assert built[0][1].calls[-1][1]["fusion_method"] == "rrf"
        with pytest.raises(ValueError):
            session.add_units([unit("b", "bob", "wrong")])
    with registry.acquire("bob") as session:
        assert session.search("red", top_k=100) == []
    registry.close()
    store.close()


def test_rebuilds_only_committed_units_after_evict(tmp_path):
    registry, store, built = setup(tmp_path)
    with store.hold("r1", "h1", "alice"):
        store.stage("r1", "alice", "h1", [unit("committed", "alice", "red")])
        store.mark_succeeded("r1", "h1", {"success": True})
    with store.hold("r2", "h2", "alice"):
        store.stage("r2", "alice", "h2", [unit("prepared", "alice", "blue")])
    with registry.acquire("alice") as session:
        assert [hit.uid for hit, _ in session.search("red", top_k=10)] == ["committed"]
        assert session.search("blue", top_k=10) == []
    registry.evict("alice")
    assert built[0][1].closed
    with registry.acquire("alice") as session:
        assert [hit.uid for hit, _ in session.search("red", top_k=10)] == ["committed"]
    registry.close()
    store.close()


def test_transaction_failure_invalidates_partial_index(tmp_path):
    registry, store, built = setup(tmp_path, fail_after=1)
    with registry.acquire("alice") as session:
        with pytest.raises(RuntimeError, match="index failed"):
            with session.transaction():
                session.add_units([unit("a", "alice", "red"), unit("b", "alice", "blue")])
    assert built[0][1].closed
    with registry.acquire("alice") as session:
        assert session.search("red", top_k=10) == []
    registry.close()
    store.close()


def test_pinned_graph_survives_capacity_pressure(tmp_path):
    registry, store, built = setup(tmp_path, max_graphs=1)
    with registry.acquire("alice") as alice:
        with registry.acquire("bob") as bob:
            assert registry.loaded_count() == 2
            bob.search("missing", top_k=10)
        assert not built[0][1].closed
        assert alice.search("missing", top_k=10) == []
    assert registry.loaded_count() <= 1
    registry.close()
    store.close()


def test_disabled_splade_skips_sparse_generation_on_add_and_restore(tmp_path):
    config = MultimodalConfig(data_dir=tmp_path, retrieval_methods=("bm25", "cosine_similarity"))
    store = CanonicalStore(tmp_path / "canonical.db")
    graphs = []

    def factory(user_id):
        graph = FakeGraph()
        graphs.append(graph)
        return graph

    registry = UserGraphRegistry(config, store, graph_factory=factory)
    memory = unit("u1", "alice", "blue backpack")
    with store.hold("r1", "h1", "alice"):
        store.stage("r1", "alice", "h1", [memory])
        with registry.acquire("alice") as session:
            session.add_units([memory])
        store.mark_succeeded("r1", "h1", {"success": True})
    assert registry.evict("alice")
    with registry.acquire("alice"):
        pass
    sparse_flags = [call[1]["generate_sparse_embedding"] for graph in graphs
                    for call in graph.calls if isinstance(call, tuple)
                    and "generate_sparse_embedding" in call[1]]
    assert sparse_flags == [False, False]
    registry.close()
    store.close()
