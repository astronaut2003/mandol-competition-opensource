"""Code ingestion policy over the common canonical store; no independent schema."""

import os
import sqlite3
from pathlib import Path
import time
import uuid

from mandol_substrate.contracts import ResourceIdentity, SpacePolicy
from mandol_substrate.gates import ScopeGates
from mandol_substrate.publication import PublicationCoordinator
from mandol_substrate.qdrant.policy import QdrantProjectionPolicy
from mandol_substrate.state import CanonicalMemory, Conflict, Mutation, Scope, SQLiteStateStore, StateUnavailable
from mandol_substrate.state.models import canonical_json, digest
from mandol_substrate.vector.codec import MinimalProjectionCodec

from ..policies import CodeProcessor, PUBLICATION, retrieval_profile

RequestConflict = Conflict
StorageBusy = StateUnavailable
canonical = canonical_json


class CapacityExceeded(ValueError):
    pass


class PreparationBudgetExceeded(ValueError):
    """A known service preparation limit, not an unexpected server failure."""


class CodingEvidenceStore:
    """Compatibility facade for the source-backed episode adapter.

    Schema, WAL ownership, lexical writes, request ledger and publication all
    belong to mandol_substrate. This class only maps Code's payload and retention
    semantics to canonical mutations and snapshot rows consumed by its ranker.
    """

    def __init__(self, path, *, ttl_days=30, busy_timeout_ms=2000, max_units_per_user=None,
                 clock=time.time, code_aware=False, namespace="code", collection="code_v1",
                 inference_contract="vllm-http-v1"):
        if not 0 < ttl_days <= 30 or not 1 <= busy_timeout_ms <= 60000:
            raise ValueError("invalid retention or writer timeout")
        self.path = Path(path).resolve()
        self.ttl_seconds, self.clock = ttl_days * 86400, clock
        self.busy_timeout_ms, self.max_units_per_user = busy_timeout_ms, max_units_per_user
        self.namespace, self.collection = namespace, collection
        self.processor = CodeProcessor(code_aware)
        self.profile = retrieval_profile(self.processor,inference_contract)
        self.state = self.coordinator = None
        self.experience_builder = None
        self.operations = ScopeGates(timeout=1800)

    def initialize(self):
        if self.state is not None:
            return
        self.state = SQLiteStateStore(self.path, self.profile, busy_timeout_ms=self.busy_timeout_ms,
            processor=self.processor, space_policy=SpacePolicy(("code_raw",)),
            identity=ResourceIdentity("code", self.namespace, self.collection))
        with self.state.transaction() as db:
            db.execute("PRAGMA secure_delete=ON")
        os.chmod(self.path, 0o600)
        codec = MinimalProjectionCodec(self.profile, QdrantProjectionPolicy().hash)
        self.coordinator = PublicationCoordinator(self.state, None, codec, None, policy=PUBLICATION)
        self.coordinator.recover()

    def scope(self, user_id):
        return Scope(self.namespace, user_id)

    def _erase(self, scope):
        self.state.begin_purge(scope, {"purged": True})
        # Optional projection is now unusable. Keep its cleanup obligation
        # durable even though canonical erasure can acknowledge independently.
        self.state.finish_purge(scope)
        self.state.mark_projection_degraded(scope, "dense", "PurgePending")

    def add(self, user_id, session_id, request_id, messages):
        try:
            return self._add(user_id, session_id, request_id, messages)
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                raise StorageBusy("Coding memory storage is busy; retry this request") from None
            raise

    def _add(self, user_id, session_id, request_id, messages):
        from .schemas import AddRequest
        request = AddRequest(user_id=user_id, session_id=session_id, request_id=request_id, messages=messages)
        payload = [message.model_dump() for message in request.messages]
        payload_digest = digest({"session_id": session_id, "messages": payload})
        scope, now = self.scope(user_id), self.clock()
        with self.operations.lease(scope, write=True):
            self.state.ensure_scope(scope)
            attributes = self.state.scope_attributes(scope)
            if attributes and attributes["expires_at"] <= now:
                self._erase(scope)
                attributes = {}
            prior = self.state.request(scope, request_id)
            if prior:
                return self.coordinator.apply(scope, request_id, payload_digest, Mutation(), {})["count"]
            if len(payload) > self.coordinator.max_mutation_memories:
                raise PreparationBudgetExceeded("Service request memory budget exceeded")
            # Capacity and session order need metadata, not every historical
            # message body. Loading the entire corpus on each small Add caused
            # growing hydration/JSON overhead for long engineering histories.
            if self.max_units_per_user is not None:
                count = self.state._rows("SELECT count(*) AS n FROM memories WHERE scope=? AND memory_type='engineering_raw' AND lifecycle<>'deleted'", (scope.key,))[0]["n"]
                if count + len(payload) > self.max_units_per_user:
                    raise CapacityExceeded("User memory capacity exceeded; no messages were added")
            attributes = attributes or {"generation": uuid.uuid4().hex, "created_at": now,
                                        "expires_at": now + self.ttl_seconds, "revision": 0}
            attributes = dict(attributes, revision=attributes["revision"] + 1)
            last = self.state._rows("SELECT max(json_extract(metadata,'$.sequence')) AS n FROM memories WHERE scope=? AND session_id=? AND memory_type='engineering_raw' AND lifecycle<>'deleted'", (scope.key, session_id))[0]["n"]
            sequence = (last if last is not None else -1) + 1
            memories = tuple(CanonicalMemory(
                "engineering_raw:" + digest([user_id, request_id, index]), message["content"],
                raw_data={"message": message, "text_content": message["content"]},
                metadata={"sequence": sequence + index, "stored_at": now, "source_kind": "raw"},
                session_id=session_id, request_id=request_id, turn_index=index, role=message["role"],
                business_timestamp_ms=message.get("timestamp"), memory_tower="engineering",
                memory_level="raw", memory_type="engineering_raw", spaces=("code_raw",))
                for index, message in enumerate(payload))
            derived = Mutation()
            if self.experience_builder:
                try:
                    # Experimental source-backed derivation still receives its
                    # complete input. The frozen C0 path has no such builder.
                    rows = [m for batch in self.state.iter_memories(scope) for m in batch]
                    derived = self.experience_builder.prepare(scope,session_id,request_id,payload_digest,(*rows,*memories))
                except Exception as exc:
                    self.state.record_event(scope,'derive','experience',type(exc).__name__)
            result = self.coordinator.apply(scope, request_id, payload_digest,
                Mutation(memories=(*memories,*derived.memories),delete_uids=derived.delete_uids,
                         edges=derived.edges,jobs=derived.jobs),
                {"count": len(payload)}, scope_attributes=attributes)
            return result["count"]

    def snapshot(self, user_id, *, include_frontier=False):
        scope = self.scope(user_id)
        with self.operations.lease(scope):
            attrs = self.state.scope_attributes(scope)
            if not attrs or attrs["expires_at"] <= self.clock():
                return None, []
            with self.coordinator.search_view(scope) as view:
                rows = [{"uid": m.logical_uid, "user_id": user_id, "session_id": m.session_id,
                         "request_id": m.request_id, "message_index": m.turn_index,
                         "sequence": m.metadata["sequence"], "payload": canonical_json(m.raw_data["message"]),
                         "stored_at": m.metadata["stored_at"]}
                        for batch in self.state.iter_memories(scope) for m in batch if m.memory_type == "engineering_raw"]
                identity = (attrs["generation"], attrs["revision"])
                if include_frontier:
                    identity += (view.published_seq,)
                return identity, sorted(rows, key=lambda r: (r["session_id"], r["sequence"]))

    def purge_expired(self):
        expired = []
        for scope in self.state.iter_scopes():
            with self.operations.lease(scope, write=True):
                attrs = self.state.scope_attributes(scope)
                if attrs and attrs["expires_at"] <= self.clock():
                    self._erase(scope)
                    expired.append(scope.user_id)
        with self.state._lock:
            checkpoint = self.state._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint[0]:
                raise StorageBusy("Retention deletion committed; WAL checkpoint needs retry")
        return expired

    def stats(self):
        return {"users": sum(bool(a := self.state.scope_attributes(s)) and a["expires_at"] > self.clock()
                             for s in self.state.iter_scopes()),
                "stored_units": self.state._rows("SELECT count(*) AS n FROM memories")[0]["n"],
                "database_bytes": sum(p.stat().st_size for p in (self.path, Path(str(self.path) + "-wal")) if p.exists())}

    def close(self):
        if self.state is not None:
            self.state.close()
            self.state = None
