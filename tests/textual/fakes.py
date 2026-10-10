"""Test-only durable projection double. Never imported by deployed backends."""

import json
import os
import threading
from pathlib import Path

from mandol_substrate.text import MandolTextProcessor
from mandol_substrate.state import RetrievalProfile
from mandol_substrate.state.models import StateUnavailable


def test_profile():
    return RetrievalProfile(
        "test/dense",
        "dense-revision-1",
        3,
        "tokenizer-1",
        "test/splade",
        "sparse-revision-1",
        MandolTextProcessor().fingerprint,
    )


class Processor(MandolTextProcessor):
    pass


class Encoders:
    def encode_documents(self, texts):
        return [([1.0, float(len(t) % 5), 0.5], {7: 1.0}) for t in texts]


class FileProjection:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.fail = None
        self.writes = 0

    def path(self, scope):
        return self.root / (scope.key + ".json")

    def _read(self, scope):
        return json.loads(self.path(scope).read_text())

    def _write(self, scope, rows):
        path = self.path(scope)
        temp = path.with_suffix(".tmp")
        with temp.open("w") as handle:
            json.dump(rows, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)

    def ensure_schema(self, scope, *, create):
        with self.lock:
            if not self.path(scope).exists():
                if not create:
                    raise StateUnavailable("missing published collection")
                self._write(scope, {})

    def upsert_projection(self, scope, rows):
        with self.lock:
            if self.fail in {"before", "always"}:
                mode, self.fail = self.fail, None if self.fail == "before" else "always"
                raise TimeoutError(mode)
            current = self._read(scope)
            current.update({r["logical_uid"]: r for r in rows})
            self._write(scope, current)
            self.writes += 1
            if self.fail == "after":
                self.fail = None
                raise TimeoutError("server committed, client timed out")

    def delete_projection(self, scope, uids):
        with self.lock:
            rows = self._read(scope)
            for uid in uids:
                rows.pop(uid, None)
            self._write(scope, rows)

    def get_projection_metadata(self, scope, uids):
        with self.lock:
            rows = self._read(scope)
            return {
                uid: {
                    k: v
                    for k, v in rows[uid].items()
                    if k not in {"dense_vector", "splade_sparse"}
                }
                for uid in uids
                if uid in rows
            }

    def health(self):
        return True


class GraphViews:
    """Protocol test view: no cache persisted; materializes a tiny synthetic graph."""

    def __init__(self, state):
        self.state = state

    def prepare(self, scope, seq):
        assert self.state.scope_state(scope)["canonical_seq"] == seq
        nodes = [r for b in self.state.graph_rows(scope, "nodes") for r in b]
        edges = [r for b in self.state.graph_rows(scope, "edges") for r in b]
        assert all(e["source_uid"] in {n["node_uid"] for n in nodes} for e in edges)
        return {"seq": seq, "nodes": nodes, "edges": edges}

    get = prepare


def ProjectionCodec(profile, processor):
    from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
    from mandol_substrate.vector.codec import MinimalProjectionCodec

    assert processor.fingerprint == profile.preprocessing_fingerprint
    return MinimalProjectionCodec(profile, QdrantProjectionPolicy().hash)
