import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from tests.common.fakes import ProjectionCodec
from mandol_substrate.gates import ScopeGates
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.state import Conflict, Mutation, Scope, StateUnavailable

from .fakes import Encoders, FileProjection, GraphViews, Processor


def coordinator(store, profile, tmp_path, **kwargs):
    projection = FileProjection(tmp_path / "projection")
    c = PublicationCoordinator(
        store,
        projection,
        ProjectionCodec(profile, Processor()),
        Encoders(),
        GraphViews(store),
        **kwargs,
    )
    return c, projection


@pytest.mark.parametrize("failure", [None, "before", "after"])
def test_timeout_reconciliation_does_not_duplicate_or_rebuild(
    store, profile, scope, memory, tmp_path, failure
):
    c, projection = coordinator(store, profile, tmp_path)
    projection.fail = failure
    result = c.apply(scope, "r", "d", Mutation(memories=(memory,)), {"count": 1})
    assert result == {"count": 1}
    assert projection.writes == 1
    assert store.scope_state(scope)["published_seq"] == 1
    assert (
        c.apply(scope, "r", "d", Mutation(memories=(memory,)), {"count": 99}) == result
    )
    assert projection.writes == 1
    assert len(list(projection.root.glob("*.json"))) == 1
    with pytest.raises(Conflict):
        c.apply(scope, "r", "other", Mutation(), {})


def test_unresolved_timeout_blocks_search_and_rolls_forward(
    store, profile, scope, memory, tmp_path
):
    c, projection = coordinator(store, profile, tmp_path, retries=2)
    projection.fail = "always"
    with pytest.raises(StateUnavailable):
        c.apply(scope, "r", "d", Mutation(memories=(memory,)), {"count": 1})
    assert store.scope_state(scope)["canonical_seq"] == 1
    assert store.scope_state(scope)["published_seq"] == 0
    with pytest.raises(StateUnavailable), c.search_view(scope):
        pytest.fail("unpublished scope was exposed")
    projection.fail = None
    c.recover()
    with c.search_view(scope) as view:
        assert view.published_seq == 1
    assert store.request(scope, "r")["status"] == "success"


def test_update_delete_and_conflicting_projection_hash(
    store, profile, scope, memory, tmp_path
):
    c, projection = coordinator(store, profile, tmp_path)
    c.apply(scope, "r", "d", Mutation(memories=(memory,)), {})
    updated = replace(memory, version=2, text_content="new text", active=False)
    c.apply(scope, "update", "u", Mutation(memories=(updated,)), {})
    row = projection.get_projection_metadata(scope, [memory.logical_uid])[
        memory.logical_uid
    ]
    assert row["version"] == 2 and not row["active"]
    c.apply(scope, "delete", "x", Mutation(delete_uids=(memory.logical_uid,)), {})
    assert projection.get_projection_metadata(scope, [memory.logical_uid]) == {}
    assert (
        store.get_memories(scope, [memory.logical_uid])[memory.logical_uid].lifecycle
        == "deleted"
    )


def test_mismatched_hash_never_publishes(store, profile, scope, memory, tmp_path):
    c, projection = coordinator(store, profile, tmp_path)
    store.prepare(scope, "r", "d", Mutation(memories=(memory,)), {})
    projection.ensure_schema(scope, create=True)
    canonical = store.get_memories(scope, [memory.logical_uid])[memory.logical_uid]
    row = c.codec.metadata(scope, canonical)
    row["projection_hash"] = "wrong"
    projection.upsert_projection(scope, [row])
    with pytest.raises(StateUnavailable):
        c.recover()
    assert store.scope_state(scope)["published_seq"] == 0


def test_projection_hash_binds_all_canonical_state_not_vector_bytes(
    profile, scope, memory
):
    codec = ProjectionCodec(profile, Processor())
    base = codec.metadata(scope, memory)
    assert (
        codec.metadata(scope, replace(memory, metadata={"debug": "changed"}))[
            "projection_hash"
        ]
        != base["projection_hash"]
    )
    assert (
        codec.metadata(scope, replace(memory, version=2))["projection_hash"]
        != base["projection_hash"]
    )
    assert (
        codec.metadata(Scope(scope.namespace, "B"), memory)["projection_hash"]
        != base["projection_hash"]
    )
    assert (
        codec.encode(base, [1, 0, 0], {1: 2})["projection_hash"]
        == base["projection_hash"]
    )
    with pytest.raises(ValueError):
        codec.encode(base, [1, 0, 0], {1: -2})


def test_same_user_read_lease_blocks_publication_but_other_user_progresses(
    store, profile, scope, memory, tmp_path
):
    c, _ = coordinator(store, profile, tmp_path)
    c.apply(scope, "r", "d", Mutation(memories=(memory,)), {})
    entered = threading.Event()
    release = threading.Event()

    def reader():
        with c.search_view(scope) as view:
            entered.set()
            assert release.wait(5)
            assert store.scope_state(scope)["published_seq"] == view.published_seq == 1

    with ThreadPoolExecutor(max_workers=3) as pool:
        read = pool.submit(reader)
        assert entered.wait(2)
        write = pool.submit(c.apply, scope, "r2", "d2", Mutation(), {})
        other = pool.submit(
            c.apply,
            Scope(scope.namespace, "B"),
            "r",
            "d",
            Mutation(memories=(memory,)),
            {},
        )
        other.result(timeout=2)
        assert not write.done()
        release.set()
        read.result(timeout=2)
        write.result(timeout=2)
    assert c.gates.active_scopes == 0


def test_lease_timeout_and_registry_cleanup(scope):
    gates = ScopeGates(timeout=0.02)
    with gates.lease(scope, write=True), ThreadPoolExecutor() as pool:

        def read():
            with gates.lease(scope):
                pass

        with pytest.raises(StateUnavailable):
            pool.submit(read).result(timeout=1)
    assert gates.active_scopes == 0


@pytest.mark.parametrize(
    "point",
    [
        "before_T1",
        "after_T1",
        "before_projection_write",
        "after_projection_write",
        "after_projection_verify",
        "after_graph_prepare",
        "before_T2",
        "after_T2_before_http_return",
    ],
)
def test_real_process_exit_then_recovery_without_graph_snapshot(tmp_path, point):
    worker = Path(__file__).with_name("fault_worker.py")
    src = Path(__file__).resolve().parents[2] / "src"
    env = dict(os.environ, PYTHONPATH=str(src), MANDOL_FAULT_POINT=point)
    killed = subprocess.run(
        [sys.executable, str(worker), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert killed.returncode == 86, killed.stderr
    env.pop("MANDOL_FAULT_POINT")
    resumed = subprocess.run(
        [sys.executable, str(worker), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == {"count": 1}
    replay = subprocess.run(
        [sys.executable, str(worker), str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert replay.returncode == 0, replay.stderr
    assert json.loads(replay.stdout) == {"count": 1}


@pytest.mark.parametrize('point', ['before_T1', 'after_T1', 'before_projection_write'])
def test_completed_model_work_survives_retry_and_is_cleared_at_publication(store, profile, scope, memory, tmp_path, point):
    c, projection = coordinator(store, profile, tmp_path, retries=1)
    calls = []
    class Counting(Encoders):
        def encode_documents(self, texts):
            calls.extend(texts)
            return super().encode_documents(texts)
    c.encoders = Counting()
    def fault(where):
        if where == point:
            raise RuntimeError('simulated interruption')
    c.fault = fault
    with pytest.raises((RuntimeError, StateUnavailable)):
        c.apply(scope, 'r', 'digest', Mutation(memories=(memory,)), {'count': 1})
    c.fault = lambda point: None
    assert c.apply(scope, 'r', 'digest', Mutation(memories=(memory,)), {'count': 1}) == {'count': 1}
    assert calls == [memory.text_content]
    assert store.ledger.reserve(scope, 'r', 'digest') == {}
