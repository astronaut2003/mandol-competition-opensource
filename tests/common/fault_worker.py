"""Subprocess boundary test. os._exit deliberately bypasses close/finally."""

import json
import os
import sys
from pathlib import Path

from fakes import (
    Encoders,
    FileProjection,
    GraphViews,
    Processor,
    ProjectionCodec,
    test_profile,
)

from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.state import (
    CanonicalMemory,
    Edge,
    Mutation,
    Scope,
    SQLiteStateStore,
)

root = Path(sys.argv[1])
scope = Scope("fault-test", "user")
state = SQLiteStateStore(root / "state.sqlite", test_profile())
projection = FileProjection(root / "projection")


def fault(point):
    if os.environ.get("MANDOL_FAULT_POINT") == point:
        os._exit(86)


coordinator = PublicationCoordinator(
    state,
    projection,
    ProjectionCodec(test_profile(), Processor()),
    Encoders(),
    GraphViews(state),
    fault=fault,
)
coordinator.recover()
uid = scope.memory_uid("s", "r", 0)
mutation = Mutation(
    memories=(CanonicalMemory(uid, "alpha"),),
    edges=(Edge("edge", "ms:hierarchical_memory", uid, "CONTAINS"),),
)
result = coordinator.apply(scope, "r", "request-digest", mutation, {"count": 1})
with coordinator.search_view(scope) as view:
    assert view.published_seq == 1
    assert len(view.graph["edges"]) == 1
    assert len(projection.get_projection_metadata(scope, [uid])) == 1
    assert state.get_memories(scope, [uid])[uid].version == 1
assert state.pending_count() == 0
print(json.dumps(result), flush=True)
state.close()
