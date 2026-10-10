import json
import sqlite3
from dataclasses import asdict, replace

import pytest

from mandol_substrate.state import (
    BuildJob,
    Conflict,
    Edge,
    Mutation,
    ProfileMismatch,
    Scope,
    SQLiteStateStore,
    StateUnavailable,
)
from mandol_substrate.state.models import digest


def commit(store, scope, request_id, mutation):
    row = store.prepare(
        scope,
        request_id,
        digest(asdict(mutation)),
        mutation,
        {"count": len(mutation.memories)},
    )
    store.publish(scope, row["mutation_seq"])
    return row


def test_pragmas_migration_owner_and_no_vector_schema(store, profile):
    with store.transaction() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        from mandol_substrate.state.schema import SCHEMA_VERSION

        assert db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        schema = " ".join(
            r[0]
            for r in db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
        )
        assert "BLOB" not in schema
        assert "dense_vector" not in schema and "splade_sparse" not in schema
    with pytest.raises(BlockingIOError):
        SQLiteStateStore(store.path, profile)
    old_uuid = store.db_uuid
    store.close()
    reopened = SQLiteStateStore(store.path, profile)
    assert reopened.db_uuid == old_uuid
    reopened.close()


def test_profile_mismatch_fails_closed(store, profile, scope):
    store.ensure_scope(scope)
    store.close()
    with pytest.raises(ProfileMismatch):
        SQLiteStateStore(
            store.path, replace(profile, embedding_revision="other-revision")
        )


def test_unknown_schema_rejected(tmp_path, profile):
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA user_version=99")
    with pytest.raises(StateUnavailable):
        SQLiteStateStore(path, profile)


def test_deterministic_identity_uses_full_namespace_user_session(scope):
    expected = scope.memory_uid("s", "r", 0)
    assert expected == Scope(scope.namespace, scope.user_id).memory_uid("s", "r", 0)
    assert (
        len(
            {
                expected,
                scope.memory_uid("s2", "r", 0),
                scope.memory_uid("s", "r", 1),
                Scope("other-run", scope.user_id).memory_uid("s", "r", 0),
                Scope(scope.namespace, "other-user").memory_uid("s", "r", 0),
            }
        )
        == 5
    )
    assert Scope("ab", "c").key != Scope("a", "bc").key


def test_t1_t2_idempotency_and_conflict(store, scope, memory):
    mutation = Mutation(memories=(memory,))
    first = store.prepare(scope, "r", "payload-hash", mutation, {"count": 1})
    assert first["status"] == "pending"
    assert store.scope_state(scope)["canonical_seq"] == 1
    assert store.scope_state(scope)["published_seq"] == 0
    assert store.pending_count() == 1
    assert store.prepare(scope, "r", "payload-hash", mutation, {"count": 99}) == first
    with pytest.raises(Conflict):
        store.prepare(scope, "r", "DIFFERENT", mutation, {})
    with pytest.raises(StateUnavailable):
        store.prepare(scope, "r2", "x", Mutation(), {})
    store.publish(scope, 1)
    assert store.request(scope, "r")["result"] == {"count": 1}
    assert store.request(scope, "r")["status"] == "success"
    assert store.pending_count() == 0
    store.publish(scope, 1)  # idempotent T2
    with pytest.raises(Conflict):
        store.publish(scope, 3)


def test_scope_sequences_are_independent(store, scope, memory):
    other = Scope(scope.namespace, "user-B")
    commit(store, scope, "r", Mutation(memories=(memory,)))
    commit(store, scope, "r2", Mutation())
    commit(store, other, "r", Mutation(memories=(memory,)))
    assert store.scope_state(scope)["published_seq"] == 2
    assert store.scope_state(other)["published_seq"] == 1
    assert len(list(store.iter_scopes(batch_size=1))) == 2


def test_atomic_rollback_on_bad_membership(store, scope, memory):
    with pytest.raises(sqlite3.IntegrityError):
        store.prepare(
            scope,
            "r",
            "digest",
            Mutation(memories=(replace(memory, spaces=("missing",)),)),
            {},
        )
    assert store.get_memories(scope, [memory.logical_uid]) == {}
    assert store.scope_state(scope)["canonical_seq"] == 0
    assert store.request(scope, "r") is None


def test_hierarchy_recursive_and_cycle_rollback(store, scope, memory):
    commit(
        store,
        scope,
        "r",
        Mutation(
            memories=(replace(memory, spaces=("leaf",)),),
            spaces=("root", "mid", "leaf"),
            space_edges=(("root", "mid"), ("mid", "leaf")),
        ),
    )
    assert store.descendant_spaces(scope, ["root"]) == ["leaf", "mid", "root"]
    assert store.get_memories(scope, [memory.logical_uid])[
        memory.logical_uid
    ].spaces == ("leaf",)
    with pytest.raises(Conflict):
        store.prepare(
            scope, "cycle", "d", Mutation(space_edges=(("leaf", "root"),)), {}
        )
    assert store.scope_state(scope)["canonical_seq"] == 1
    commit(store, scope, "unlink", Mutation(remove_space_edges=(("mid", "leaf"),)))
    assert store.descendant_spaces(scope, ["root"]) == ["mid", "root"]


def test_parallel_edges_properties_and_space_nodes(store, scope, memory, another):
    edges = (
        Edge(
            "e1",
            memory.logical_uid,
            another.logical_uid,
            "WORKS_AT",
            {"weight": 0.5, "list": [1, 2]},
        ),
        Edge(
            "e2", memory.logical_uid, another.logical_uid, "WORKS_AT", {"weight": 0.8}
        ),
        Edge("e3", "ms:hierarchical_memory", memory.logical_uid, "CONTAINS"),
    )
    commit(store, scope, "r", Mutation(memories=(memory, another), edges=edges))
    rows = [
        r for batch in store.graph_rows(scope, "edges", batch_size=1) for r in batch
    ]
    assert len(rows) == 3
    assert json.loads(rows[0]["properties"])["list"] == [1, 2]
    assert rows[2]["source_kind"] == "space"
    commit(store, scope, "del-edge", Mutation(delete_edge_ids=("e1",)))
    assert len([r for b in store.graph_rows(scope, "edges") for r in b]) == 2


def test_cross_scope_edge_rejected_atomically(store, scope, memory):
    other = Scope(scope.namespace, "B")
    commit(store, other, "r", Mutation(memories=(memory,)))
    with pytest.raises(Conflict):
        store.prepare(
            scope,
            "edge",
            "d",
            Mutation(edges=(Edge("e", memory.logical_uid, memory.logical_uid, "X"),)),
            {},
        )
    assert store.request(scope, "edge") is None


def test_lifecycle_supersession_version_tombstone(store, scope, memory, another):
    commit(store, scope, "r", Mutation(memories=(memory,)))
    inactive = replace(
        memory, version=2, active=False, lifecycle="superseded", valid_to_ms=9000
    )
    newer = replace(another, supersedes_uid=memory.logical_uid, valid_from_ms=9000)
    commit(store, scope, "update", Mutation(memories=(inactive, newer)))
    actual = store.get_memories(scope, [memory.logical_uid, newer.logical_uid])
    assert actual[memory.logical_uid].version == 2
    assert not actual[memory.logical_uid].active
    assert actual[newer.logical_uid].supersedes_uid == memory.logical_uid
    with pytest.raises(Conflict):
        store.prepare(scope, "stale", "d", Mutation(memories=(memory,)), {})
    commit(store, scope, "delete", Mutation(delete_uids=(memory.logical_uid,)))
    deleted = store.get_memories(scope, [memory.logical_uid])[memory.logical_uid]
    assert deleted.lifecycle == "deleted" and deleted.version == 3
    assert [m.logical_uid for b in store.iter_memories(scope) for m in b] == [
        newer.logical_uid
    ]
    with pytest.raises(Conflict):
        store.prepare(
            scope,
            "resurrect",
            "d",
            Mutation(memories=(replace(memory, version=4),)),
            {},
        )


def test_hlm_job_and_generated_content_commit_together(store, scope, memory, another):
    job = BuildJob.planned(scope, {memory.logical_uid: 1}, "recipe-revision-1")
    commit(store, scope, "input", Mutation(memories=(memory,), jobs=(job,)))
    completed = replace(
        job,
        status="completed",
        output_uids=(another.logical_uid,),
        progress={"stage": "done"},
    )
    commit(store, scope, "output", Mutation(memories=(another,), jobs=(completed,)))
    with pytest.raises(Conflict):
        store.prepare(scope, "regenerate", "d", Mutation(jobs=(job,)), {})
    assert (
        store.get_memories(scope, [another.logical_uid])[
            another.logical_uid
        ].text_content
        == "gamma"
    )


def test_backup_contains_pending_and_full_state(
    store, scope, memory, profile, tmp_path
):
    store.prepare(scope, "r", "d", Mutation(memories=(memory,)), {"count": 1})
    path = tmp_path / "backup.sqlite"
    store.backup(path)
    restored = SQLiteStateStore(path, profile)
    assert restored.pending_count() == 1
    assert (
        restored.get_memories(scope, [memory.logical_uid])[
            memory.logical_uid
        ].text_content
        == memory.text_content
    )
    restored.close()
