import json
from types import SimpleNamespace

import pytest

from mandol_competition.textual.core.graph import GraphViewManager
from mandol_substrate.state import (
    CanonicalMemory,
    Edge,
    Mutation,
    Scope,
    SQLiteStateStore,
    StateUnavailable,
)

pytestmark = pytest.mark.integration_sqlite


def populate(store, scope):
    docs = tuple(CanonicalMemory(u, u) for u in "abcd")
    edges = (
        Edge("e1", "a", "b", "X", {"weight": 0.7}),
        Edge("e2", "a", "b", "Y", {"nested": {"p": 3}}),
        Edge("e3", "b", "c", "X"),
        Edge("e4", "d", "a", "Y"),
        Edge("self", "a", "a", "X"),
    )
    row = store.prepare(scope, "r", "r", Mutation(memories=docs, edges=edges), {})
    store.publish(scope, row["mutation_seq"])
    return docs, edges


def test_structural_rebuild_parallel_edges_paths_and_legacy(store, scope):
    from mandol_substrate.representation import MemoryUnit
    from tests.archived_oracles import graph_oracles
    SemanticGraph, GraphContextExpander = graph_oracles()

    docs, edges = populate(store, scope)
    manager = GraphViewManager(store)
    view = manager.get(scope, 1)
    units = {
        m.logical_uid: MemoryUnit(m.logical_uid, {"text_content": m.text_content})
        for m in docs
    }
    legacy = SemanticGraph(SimpleNamespace(get_unit=units.get))
    for uid in units:
        idx = legacy.rx_graph.add_node({"uid": uid})
        legacy._uid_to_index[uid] = idx
        legacy._index_to_uid[idx] = uid
    for e in edges:
        legacy.rx_graph.add_edge(
            legacy._uid_to_index[e.source_uid],
            legacy._uid_to_index[e.target_uid],
            {"type": e.relation_type, **e.properties},
        )
    for direction in ("successors", "predecessors", "all"):
        for relation in (None, "X", "Y"):
            old = {
                u.uid
                for u in legacy.traverse_explicit_nodes(
                    "a", relationship_type=relation, direction=direction
                )
            } - {"a"}
            assert {
                e[0]
                for e in view.neighbors(
                    "a", direction=direction, relations=[relation] if relation else None
                )
            } == old
    assert len(view.neighbors("a", direction="successors")) == 2
    assert view.shortest_path("a", "c") == ["a", "b", "c"]
    assert view.shortest_path("c", "a") == []
    assert view.shortest_path("a", "c", relations=["Y"]) == []
    old = GraphContextExpander(legacy).expand_context_by_hops(
        ["a"], n_hops=2, max_neighbors_per_hop=20, semantic_threshold=2
    )
    assert set(view.traverse(["a"], hops=2, per_node=20)) == {
        u.uid for u in old.expanded_units
    }
    manager.evict()
    assert manager.get(scope, 1).shortest_path("a", "c") == ["a", "b", "c"]
    assert all(isinstance(n, str) for n in view.graph.nodes())


@pytest.mark.parametrize(
    "damage",
    [
        "none",
        "missing",
        "truncated",
        "json_corrupt",
        "checksum",
        "uuid",
        "seq",
        "profile",
    ],
)
def test_optional_graph_cache_damage_rebuild(store, scope, tmp_path, damage):
    populate(store, scope)
    manager = GraphViewManager(store)
    path = tmp_path / "graph.cache"
    if damage != "missing":
        assert manager.save_graph(scope, path)
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:15])
    elif damage == "json_corrupt":
        path.write_bytes(b"\x00bad")
    elif damage in {"checksum", "uuid", "seq", "profile"}:
        data = json.loads(path.read_text())
        data[
            {
                "checksum": "sha256",
                "uuid": "db_uuid",
                "seq": "published_seq",
                "profile": "profile_hash",
            }[damage]
        ] = "wrong"
        path.write_text(json.dumps(data))
    manager.evict()
    assert manager.load_graph(scope, path).shortest_path("a", "c") == ["a", "b", "c"]


def test_eviction_isolation_restart_without_snapshot(store, scope, profile):
    populate(store, scope)
    other = Scope(scope.namespace, "other")
    populate(store, other)
    manager = GraphViewManager(store, max_users=1)
    manager.get(scope, 1)
    manager.get(other, 1)
    assert manager.stats()["loaded_users"] == 1
    row = store.prepare(scope, "delete", "delete", Mutation(delete_uids=("b",)), {})
    store.publish(scope, row["mutation_seq"])
    assert manager.get(scope, 2).shortest_path("a", "c") == []
    assert manager.get(other, 1).shortest_path("a", "c") == ["a", "b", "c"]
    path = store.path
    store.close()
    reopened = SQLiteStateStore(path, profile)
    try:
        assert GraphViewManager(reopened).get(other, 1).shortest_path("a", "c") == [
            "a",
            "b",
            "c",
        ]
    finally:
        reopened.close()


@pytest.mark.parametrize("envelope", ["[]", "null", "42", '"bad cache"'])
def test_valid_json_wrong_cache_shape_rebuilds(store, scope, tmp_path, envelope):
    from mandol_substrate.state import CanonicalMemory, Mutation

    mutation = store.prepare(
        scope, "r", "r", Mutation(memories=(CanonicalMemory("a", "alpha"),)), {}
    )
    store.publish(scope, mutation["mutation_seq"])
    manager = GraphViewManager(store)
    path = tmp_path / "graph.cache"
    path.write_text(envelope)
    assert "a" in manager.load_graph(scope, path).uid_to_index


def test_cache_capacity_fails_closed_without_deleting_graph(store, scope):
    populate(store, scope)
    with pytest.raises(StateUnavailable):
        GraphViewManager(store, max_nodes=2).get(scope, 1)
    assert GraphViewManager(store).get(scope, 1).graph.num_edges() == 5
