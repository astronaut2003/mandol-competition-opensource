import sqlite3
from dataclasses import replace

import pytest

from mandol_substrate.state import (
    Mutation,
    SQLiteStateStore,
    StateUnavailable,
)

pytestmark = [pytest.mark.integration_sqlite, pytest.mark.fault]


def commit(store, scope, request, memories):
    row = store.prepare(scope, request, request, Mutation(memories=memories), {})
    store.publish(scope, row["mutation_seq"])


def test_sqlite_disk_full_rolls_back_canonical_and_lexical(store, scope, memory):
    commit(store, scope, "r", (memory,))
    pages = store._connection.execute("PRAGMA page_count").fetchone()[0]
    store._connection.execute(f"PRAGMA max_page_count={pages}")
    with pytest.raises(sqlite3.OperationalError, match="full"):
        store.prepare(
            scope,
            "full",
            "full",
            Mutation(
                memories=(
                    replace(memory, version=2, text_content="different " * 20000),
                )
            ),
            {},
        )
    assert store.scope_state(scope)["canonical_seq"] == 1
    assert (
        store.get_memories(scope, [memory.logical_uid])[memory.logical_uid].version == 1
    )
    assert store.lexical.verify_scope(scope)
    assert store.request(scope, "full") is None


def test_busy_external_writer_no_half_state(store, scope, memory):
    commit(store, scope, "r", (memory,))
    store._connection.execute("PRAGMA busy_timeout=20")
    external = sqlite3.connect(store.path, isolation_level=None)
    try:
        external.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store.prepare(
                scope,
                "busy",
                "busy",
                Mutation(memories=(replace(memory, version=2),)),
                {},
            )
    finally:
        external.close()
    assert store.lexical.verify_scope(scope)
    assert store.scope_state(scope)["canonical_seq"] == 1
    commit(store, scope, "busy", (replace(memory, version=2),))


def test_backup_restore_includes_lexical_pending_and_atomic_repair(
    store, scope, memory, tmp_path, profile
):
    store.prepare(scope, "r", "r", Mutation(memories=(memory,)), {})
    backup = tmp_path / "backup.sqlite"
    store.backup(backup)
    restored = SQLiteStateStore(backup, profile)
    try:
        assert restored.lexical.verify_scope(scope)
        with pytest.raises(StateUnavailable):
            restored.lexical.search(scope, "alpha")
        restored.publish(scope, 1)
        assert len(restored.lexical.search(scope, "alpha")) == 1
        with restored.transaction() as db:
            db.execute("DELETE FROM lexical_postings WHERE scope=?", (scope.key,))
        with pytest.raises(StateUnavailable):
            restored.lexical.verify_scope(scope)
        restored.lexical.rebuild_scope(scope)
        assert restored.lexical.verify_scope(scope)
    finally:
        restored.close()


def test_schema_v1_migration_requires_explicit_projection_repair(
    tmp_path, profile, scope
):
    from dataclasses import asdict

    from mandol_substrate.state.models import canonical_json
    from mandol_substrate.state.schema import SCHEMA

    path = tmp_path / "v1.sqlite"
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO manifest VALUES ('db_uuid','migration-fixture')")
        db.execute(
            "INSERT INTO scopes VALUES (?,?,?,0,0,?,?,1,0,0)",
            (
                scope.key,
                scope.namespace,
                scope.user_id,
                profile.hash,
                canonical_json(asdict(profile)),
            ),
        )
        db.execute("PRAGMA user_version=1")
    store = SQLiteStateStore(path, profile)
    try:
        assert store.db_uuid == "migration-fixture"
        with pytest.raises(StateUnavailable):
            store.lexical.verify_scope(scope)
        assert store.lexical.rebuild_scope(scope)
    finally:
        store.close()
