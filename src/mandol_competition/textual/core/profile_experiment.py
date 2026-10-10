"""Explicit PUBLIC-DATA research service; never selected by the formal CLI.

This is an HLM ablation overlay on the frozen base. It reuses AML lifecycle and
publication. Its augmented corpus may change candidate ranks. This experiment
does not claim fixed first-stage candidate equivalence between raw/HLM arms.
"""
import asyncio
import contextvars
import json
import threading
from dataclasses import replace
from pathlib import Path

from mandol_substrate.state.models import BuildJob, Mutation, StateUnavailable, digest
from mandol_competition.textual.api.competition import CompetitionMemoryService

from .gates import ScopeGates
from .paired_quality import append_call
from .run_storage import atomic_json
from .user_profile import (
    ProfileProvider,
    ProfileRecipe,
    ProfileWorker,
    current_profile_ids,
)

QUERY = contextvars.ContextVar("hlm_query", default="")
STRATEGY = contextvars.ContextVar("hlm_strategy", default="retrieved")


class ModelBridge:
    """One owned async provider loop; concurrent audit uses task-local context."""
    def __init__(self, path, *, provider_factory=ProfileProvider):
        self.path = Path(path)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="hlm-gpt", daemon=True)
        self.thread.start()
        async def create():
            return provider_factory()
        self.provider = asyncio.run_coroutine_threadsafe(create(), self.loop).result(timeout=30)

    def __call__(self, messages):
        result = asyncio.run_coroutine_threadsafe(
            self.provider.complete(messages, max_tokens=6000, json_mode=True), self.loop
        ).result(timeout=180)
        append_call(self.path, {"messages": messages, "response": result})
        return result

    def close(self):
        asyncio.run_coroutine_threadsafe(self.provider.close(), self.loop).result(timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise StateUnavailable("Owned HLM provider loop failed to stop")
        self.loop.close()


def profile_live(state, scope, memory, *, strict_current=False, now_ms=None):
    if memory.metadata.get("user_id") != scope.user_id:
        raise StateUnavailable("Foreign profile hydration")
    item = memory.metadata.get("profile_item", {})
    if item.get("status") != "current":
        return False
    if strict_current and item.get("lifetime") == "temporary":
        # Unresolved relative time is not invented from ingestion time. Keep the
        # source retrievable, but do not expose this as a currently valid profile.
        import time
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        if memory.valid_to_ms is None or memory.valid_to_ms <= now_ms:
            return False
        if memory.valid_from_ms is not None and memory.valid_from_ms > now_ms:
            return False
    versions = memory.metadata.get("source_versions", {})
    if not versions or len(versions) > 8:
        raise StateUnavailable("Profile provenance missing or unbounded")
    sources = state.get_memories(scope, list(versions))
    if any(u not in sources or sources[u].version != version or not sources[u].active
           or sources[u].lifecycle == "deleted" for u, version in versions.items()):
        return False
    superseded = state._rows("""SELECT 1 FROM graph_edges e JOIN memories m
        ON e.scope=m.scope AND e.source_uid=m.logical_uid
        WHERE e.scope=? AND e.target_uid=? AND e.active=1 AND m.active=1
        AND m.lifecycle!='deleted' AND e.relation_type IN ('PROFILE_SUPERSEDES','PROFILE_RETRACTS') LIMIT 1""",
        (scope.key, memory.logical_uid))
    return not superseded


class ProfileContext:
    def __init__(self, state, raw_policy, *, strict_current=False):
        self.state, self.raw_policy = state, raw_policy
        self.strict_current = strict_current

    def render(self, scope, hits, canonical):
        raw, profiles = [], []
        for hit in hits:
            memory = canonical[hit["id"]]
            if memory.metadata.get("hlm_profile"):
                if profile_live(self.state, scope, memory, strict_current=self.strict_current):
                    profiles.append(hit)
            else:
                raw.append(hit)
        strategy = STRATEGY.get()
        if strategy == "always":
            ids = current_profile_ids(self.state, scope, limit=100)
            more = self.state.get_memories(scope, ids)
            canonical = {**canonical, **more}
            profiles = [{"id": u, "content": more[u].text_content, "score": None}
                        for u in ids if profile_live(self.state, scope, more[u], strict_current=self.strict_current)]
            selected = profiles + raw
        elif strategy == "category":
            # Query-only explainable diagnostic routing; no benchmark labels.
            import re
            relevant = bool(re.search(r"prefer|recommend|suggest|like|dislike|habit|routine|goal|for me|should I|喜欢|推荐|习惯", QUERY.get(), re.IGNORECASE))
            selected = hits if relevant else raw
        elif strategy == "reserve":
            chosen = profiles[:2]
            provenance = {u for h in chosen for u in canonical[h["id"]].metadata["source_uids"]}
            # Reserve at most two occupied slots, not two empty slots. Otherwise
            # the OFF/ON comparison silently drops raw evidence on lexical miss.
            selected = [h for h in raw if h["id"] not in provenance][:10 - len(chosen)] + chosen
        elif strategy == "retrieved":
            allowed = {h["id"] for h in raw + profiles}
            selected = [h for h in hits if h["id"] in allowed]
        elif strategy == "raw":
            selected = raw
        else:
            raise ValueError("Unknown explicit HLM strategy")
        output, seen_sources = [], set()
        for hit in selected:
            memory = canonical[hit["id"]]
            if not memory.metadata.get("hlm_profile"):
                if memory.logical_uid in seen_sources:
                    continue
                rendered = self.raw_policy.render(scope, [hit], {hit["id"]: memory})[0]
                seen_sources.add(memory.logical_uid)
            else:
                uids = memory.metadata["source_uids"]
                if any(u in seen_sources for u in uids):
                    continue
                sources = self.state.get_memories(scope, uids)
                raw_hits = [{"id": u, "content": sources[u].text_content, "score": hit.get("score")} for u in uids]
                provenance = self.raw_policy.render(scope, raw_hits, sources)
                # Full original source and source UID/date/role remain visible.
                item = memory.metadata["profile_item"]
                header = f"[Derived User Profile; {item['category']}; {item['lifetime']}; source-backed interpretation]"
                content = header + "\n" + "\n".join(f"Source {h['id']}: {h['content']}" for h in provenance)
                rendered = {**hit, "content": content}
                seen_sources.update(uids)
            output.append(rendered)
        return output[:len(hits)]


class ProfileExperimentService(CompetitionMemoryService):
    def __init__(self, settings, *, gate_path, governance_gate_path=None):
        super().__init__(settings)
        gate = json.loads(Path(gate_path).read_text())
        if (gate.get("status") != "PASS_FOR_SMALL_PUBLIC_PANEL_ONLY"
                or gate.get("recipe_hash") != ProfileRecipe("0" * 64).fingerprint):
            raise StateUnavailable("Experimental semantic gate does not bind this recipe")
        self.profile_gates = ScopeGates(timeout=300, max_inflight_scopes=16)
        self.bridge = None
        self.enable_governance = governance_gate_path is not None
        if self.enable_governance:
            from .profile_governance import fingerprint
            governance_gate = json.loads(Path(governance_gate_path).read_text())
            if governance_gate.get("status") != "PASS_FOR_SMALL_PUBLIC_PANEL_ONLY" or governance_gate.get("fingerprint") != fingerprint():
                raise StateUnavailable("Governance semantic gate mismatch")

    def start(self):
        super().start()
        self._ready = False
        self.context_policy = ProfileContext(self.state, self.context_policy, strict_current=self.enable_governance)
        root = Path(self.settings.competition_run_root)
        policy = {"schema": "hlm-public-experiment-v1", "base": "source_b1_public_v1",
            "recipe_hash": ProfileRecipe(self.state.profile_hash).fingerprint,
            "source_roles": ["user"], "context_roles": ["assistant"],
            "source_order": "canonical_commit_seq_then_turn_index", "max_source_chars": 1200, "batch_sources": 12,
            "context_policy": "profile-context-v2-reserve-backfills-vacant-slots",
            "existing_profile_limit": 8, "publication": "raw plus complete derived/rejection under outer scope gate",
            "formal": False}
        if self.enable_governance:
            from .profile_governance import fingerprint
            policy.update(governance=fingerprint(), temporary_visibility="requires_validated_canonical_interval")
        path = root / "hlm-experiment.json"
        if path.exists() and json.loads(path.read_text()) != policy:
            raise StateUnavailable("Experimental profile changed in an existing run")
        atomic_json(path, policy)
        self.bridge = ModelBridge(root / "artifacts" / "hlm-calls.jsonl")
        self._restore_hlm_frontiers()
        self._ready = True

    def add_messages(self, user_id, session_id, request_id, messages):
        scope = self.scope(user_id)
        with self.profile_gates.lease(scope, write=True):
            count = super().add_messages(user_id, session_id, request_id, messages)
            uids = [scope.memory_uid(session_id, request_id, i) for i in range(len(messages))]
            self._finish_hlm(scope, request_id, uids)
            return count

    def _finish_hlm(self, scope, request_id, uids):
        if not 0 < len(uids) <= 1024:
            raise StateUnavailable("Invalid bounded raw Add manifest")
        frontier = self.state.scope_state(scope)
        if frontier["canonical_seq"] != frontier["published_seq"]:
            pending = self.state._rows("SELECT request_id,payload_digest FROM requests WHERE scope=? AND mutation_seq=?",
                                       (scope.key, frontier["canonical_seq"]))
            if len(pending) != 1:
                raise StateUnavailable("Incomplete profile publication lacks durable request")
            self.coordinator.apply(scope, pending[0]["request_id"], pending[0]["payload_digest"], Mutation(), {})
        sources = self.state.get_memories(scope, uids)
        if set(sources) != set(uids) or any(m.metadata.get("request_id") != request_id for m in sources.values()):
            raise StateUnavailable("Raw Add provenance mismatch during HLM recovery")
        worker = ProfileWorker(self.coordinator, ProfileRecipe(self.state.profile_hash), self.bridge)
        master_policy = [worker.recipe.fingerprint, "hlm-add-master-v1", request_id]
        if getattr(self, "enable_governance", False):
            from .profile_governance import fingerprint
            master_policy.append(fingerprint())
        master = BuildJob.planned(scope, {u: m.version for u, m in sources.items()}, digest(master_policy))
        existing = worker._job(scope, master.job_id)
        if existing and existing.status == "completed":
            return existing.output_uids
        raw, rejected = [], []
        for uid in uids:
            source = sources[uid]
            if source.role == "assistant":
                raw.append(uid)
                continue
            if source.role != "user":
                continue
            if len(source.text_content) > 1200:
                rejected.append({"uid": uid, "reason": "EXPLICIT_RECIPE_SOURCE_LENGTH_LIMIT_RAW_RETAINED"})
            else:
                raw.append(uid)
        if existing is None:
            master = replace(master, progress={"original_request_id": request_id,
                "stage": "hlm_add", "raw_uids": raw, "rejected": rejected})
            worker._apply(scope, "intent", master)
        else:
            master = existing
            if master.progress["raw_uids"] != raw or master.progress["rejected"] != rejected:
                raise StateUnavailable("HLM input policy changed during retry")
        outputs = []
        for offset in range(0, len(raw), 12):
            if getattr(self, "enable_governance", False):
                from .profile_governance import GovernanceWorker
                user_sources = [u for u in raw[offset:offset + 12] if sources[u].role == "user"]
                if user_sources:
                    GovernanceWorker(self.coordinator, worker.recipe, self.bridge).run(scope, user_sources)
            outputs.append(worker.run_incremental(scope, raw[offset:offset + 12]))
        completed = replace(master, status="completed",
            output_uids=tuple(u for result in outputs for u in result["output_uids"]),
            progress={**master.progress, "children": outputs})
        worker._apply(scope, "completed", completed)
        append_call(Path(self.settings.competition_run_root) / "artifacts" / "hlm-adds.jsonl",
            {"request_id": request_id, "user_id": scope.user_id, "outputs": outputs,
             "rejected": rejected, "master_job": master.job_id})
        return completed.output_uids

    def _restore_hlm_frontiers(self):
        # Streaming raw request manifests, not full histories in Python. A crash
        # after raw publication but before any HLM intent is detected here too.
        for scope in self.state.iter_scopes():
            last = ""
            while True:
                requests = self.state._rows("""SELECT DISTINCT request_id FROM memories
                    WHERE scope=? AND request_id>? AND memory_level='L0_Observation'
                    AND json_extract(metadata,'$.aml')=1 ORDER BY request_id LIMIT 16""", (scope.key, last))
                if not requests:
                    break
                for row in requests:
                    request_id = row["request_id"]
                    memories = self.state._rows("""SELECT logical_uid FROM memories WHERE scope=? AND request_id=?
                        AND memory_level='L0_Observation' AND json_extract(metadata,'$.aml')=1
                        ORDER BY turn_index,logical_uid LIMIT 1025""", (scope.key, request_id))
                    self._finish_hlm(scope, request_id, [m["logical_uid"] for m in memories])
                    last = request_id

    def _live_candidates(self, scope, uids, spaces=None):
        live = super()._live_candidates(scope, uids, spaces)
        memories = self.state.get_memories(scope, live)
        return [u for u in live if not memories[u].metadata.get("hlm_profile")
                or profile_live(self.state, scope, memories[u], strict_current=self.enable_governance)]

    def health(self):
        result = super().health()
        if self.state._rows("SELECT 1 FROM build_jobs WHERE status!='completed' AND json_extract(progress,'$.stage')='hlm_add' LIMIT 1"):
            raise StateUnavailable("Incomplete HLM Add frontier")
        return result

    def search(self, user_id, query, top_k):
        with self.profile_gates.lease(self.scope(user_id), write=False):
            if self.state._rows("SELECT 1 FROM build_jobs WHERE scope=? AND status!='completed' AND json_extract(progress,'$.stage')='hlm_add' LIMIT 1", (self.scope(user_id).key,)):
                raise StateUnavailable("Incomplete HLM Add frontier")
            token = QUERY.set(query)
            try:
                return super().search(user_id, query, top_k)
            finally:
                QUERY.reset(token)

    def shutdown(self):
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        super().shutdown()
