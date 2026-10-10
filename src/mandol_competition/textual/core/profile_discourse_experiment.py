"""Public-data-only Profile overlay. Frozen formal service remains unmodified."""

import json
import logging
import re
import time
import traceback
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC
from pathlib import Path

from mandol_substrate.state import StateUnavailable
from mandol_competition.textual.api.competition import CompetitionMemoryService

from .gates import ScopeGates
from .paired_quality import append_call
from .profile_delta_worker import current_delta_ids
from .profile_discourse import DiscourseProvider, DiscourseRecipe
from .profile_discourse_worker import ProfileDiscourseRuntime
from .profile_experiment import QUERY, STRATEGY, ModelBridge, ProfileExperimentService
from .reranker_quality import fingerprint
from .run_storage import atomic_json, checksum
from .user_profile import checked_response


class DiscourseReplayBridge(ModelBridge):
    """Explicit bounded research receipt reuse, never a formal runtime cache.

    Match the complete provider request (including JSON schema and token budget).
    First successful receipt wins in declared file order. A miss calls the real
    provider; every hit is labeled replay, never counted as a new GPT response.
    """

    @staticmethod
    def request_key(messages):
        return fingerprint(
            {
                "model": "gpt-4o-mini",
                "messages": messages,
                "temperature": 0,
                "max_tokens": 6000,
                "response_format": DiscourseProvider.response_format(
                    None, messages, True
                ),
            }
        )

    @classmethod
    def load_receipts(cls, paths, limit=256):
        cache, bindings = {}, []
        for name in paths:
            path = Path(name)
            before = checksum(path)
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    response = row["response"]
                    if response["status"] != "PASS":
                        continue
                    key = cls.request_key(row["messages"])
                    if key != response["request_sha256"]:
                        raise StateUnavailable("Replay provider request mismatch")
                    checked_response(response)
                    cache.setdefault(key, (response, before))
                    if len(cache) > limit:
                        raise StateUnavailable("Bounded research replay cache exceeded")
            if checksum(path) != before:
                raise StateUnavailable("Replay source changed during load")
            bindings.append({"file": str(path), "sha256": before})
        return cache, bindings

    def __init__(self, path, replay_paths):
        self.cache, bindings = self.load_receipts(replay_paths)
        manifest = Path(path).with_suffix(".replay-sources.json")
        if manifest.exists() and json.loads(manifest.read_text()) != bindings:
            raise StateUnavailable("Research replay binding changed")
        atomic_json(manifest, bindings)
        super().__init__(path, provider_factory=DiscourseProvider)

    def __call__(self, messages):
        key = self.request_key(messages)
        if key not in self.cache:
            return super().__call__(messages)
        response, source_sha = self.cache[key]
        result = deepcopy(response)
        append_call(
            self.path,
            {
                "messages": messages,
                "response": result,
                "execution": "replayed",
                "replay_source_sha256": source_sha,
            },
        )
        return result


@contextmanager
def failure_audit():
    """Record code locations/types only; no payload, exception message or secret."""
    try:
        yield
    except Exception as exc:
        chain = []
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(
                {
                    "type": type(current).__name__,
                    "frames": [
                        (f.filename, f.name, f.lineno)
                        for f in traceback.extract_tb(current.__traceback__)
                    ],
                }
            )
            current = current.__cause__ or current.__context__
        # StrictQualityProvider suppresses legacy exception-body logging globally.
        # Explicitly deliver ONLY this sanitized record; do not re-enable those
        # loggers or attach exc_info, exception messages, input text or locals.
        logger = logging.getLogger(__name__)
        logger.handle(
            logger.makeRecord(
                __name__,
                logging.ERROR,
                __file__,
                0,
                "Discourse runtime failure trace: %s",
                (chain,),
                None,
            )
        )
        raise


def missing_discourse_requests(state, scope, recipe, limit=16):
    """Detect even a crash after raw T2 but BEFORE any discourse root intent."""
    return state._rows(
        """SELECT m.request_id,MIN(m.commit_seq) AS seq FROM memories m
        WHERE m.scope=? AND m.memory_level='L0_Observation'
        AND json_extract(m.metadata,'$.aml')=1
        AND NOT EXISTS(SELECT 1 FROM build_jobs j, json_each(j.progress,'$.raw_request_ids') r
          WHERE j.scope=m.scope AND j.recipe_fingerprint=? AND j.status='completed'
          AND json_extract(j.progress,'$.discourse_root')=1 AND r.value=m.request_id)
        GROUP BY m.request_id ORDER BY seq,m.request_id LIMIT ?""",
        (scope.key, recipe.fingerprint, limit),
    )


def live_profile(state, scope, memory, recipe):
    meta = memory.metadata
    item = meta.get("profile_delta_item")
    if meta.get("user_id") != scope.user_id:
        raise StateUnavailable("Foreign profile")
    if (
        not item
        or meta.get("recipe_hash") != recipe.fingerprint
        or item["status"] != "current"
    ):
        return False
    if item["lifetime"] == "temporary":
        from datetime import datetime

        today = datetime.now(UTC).date().isoformat()
        if not item["valid_to"] or item["valid_to"] < today:
            return False
        if item["valid_from"] and item["valid_from"] > today:
            return False
    versions = {**meta.get("context_versions", {}), **meta.get("source_versions", {})}
    if not meta.get("source_versions") or len(versions) > 16:
        raise StateUnavailable("Missing/unbounded profile evidence")
    sources = state.get_memories(scope, list(versions))
    if any(
        u not in sources
        or sources[u].version != v
        or not sources[u].active
        or sources[u].lifecycle == "deleted"
        for u, v in versions.items()
    ):
        return False
    return not state._rows(
        """SELECT 1 FROM graph_edges WHERE scope=? AND target_uid=? AND active=1
        AND relation_type IN ('PROFILE_REPLACES','PROFILE_SUPERSEDES','PROFILE_RETRACTS') LIMIT 1""",
        (scope.key, memory.logical_uid),
    )


class DiscourseContext:
    def __init__(self, state, raw_policy, recipe):
        self.state, self.raw_policy, self.recipe = state, raw_policy, recipe

    def render(self, scope, hits, canonical):
        raw = [h for h in hits if not canonical[h["id"]].metadata.get("hlm_profile")]
        profiles = [
            h
            for h in hits
            if canonical[h["id"]].metadata.get("hlm_profile")
            and live_profile(self.state, scope, canonical[h["id"]], self.recipe)
        ]
        strategy = STRATEGY.get()
        query = QUERY.get()
        if strategy == "always":
            ids = current_delta_ids(self.state, scope, limit=100)
            more = self.state.get_memories(scope, ids)
            canonical = {**canonical, **more}
            profiles = [
                {"id": u, "score": None}
                for u in ids
                if live_profile(self.state, scope, more[u], self.recipe)
            ]
            selected = profiles + raw
        elif strategy == "raw":
            selected = raw
        elif strategy == "category":
            relevant = bool(
                re.search(
                    r"prefer|recommend|suggest|for me|should I|my |personal|喜欢|推荐|习惯|适合我",
                    query,
                    re.IGNORECASE,
                )
            )
            allowed = {h["id"] for h in raw + (profiles if relevant else [])}
            selected = [h for h in hits if h["id"] in allowed]
        elif strategy in ("reserve", "reserve1", "dynamic"):
            limit = 1 if strategy == "reserve1" else 2
            if strategy == "dynamic" and not re.search(
                r"prefer|recommend|suggest|for me|should I|my |喜欢|推荐",
                query,
                re.IGNORECASE,
            ):
                limit = 0
            chosen = profiles[:limit]
            selected = raw[: 10 - len(chosen)] + chosen
        elif strategy == "retrieved":
            allowed = {h["id"] for h in raw + profiles}
            selected = [h for h in hits if h["id"] in allowed]
        else:
            raise ValueError("Unknown experimental Profile strategy")
        result = []
        for hit in selected[:10]:
            m = canonical[hit["id"]]
            if not m.metadata.get("hlm_profile"):
                result.extend(self.raw_policy.render(scope, [hit], {m.logical_uid: m}))
                continue
            item = m.metadata["profile_delta_item"]
            sources = self.state.get_memories(scope, m.metadata["source_uids"])
            evidence = []
            for span in m.metadata["source_spans"]:
                source = sources[span["source_uid"]]
                if span["quote"] not in source.text_content:
                    raise StateUnavailable("Profile quote changed")
                rendered = self.raw_policy.render(
                    scope,
                    [
                        {
                            "id": source.logical_uid,
                            "content": source.text_content,
                            "score": hit.get("score"),
                        }
                    ],
                    {source.logical_uid: source},
                )[0]
                evidence.append(
                    "Source " + source.logical_uid + ": " + rendered["content"]
                )
            content = (
                f"[Derived User Profile; {item['kind']}; {item['lifetime']}; evidence-backed interpretation]\n{item['statement']}\n"
                + "\n".join(evidence)
            )
            result.append({**hit, "content": content})
        return result


class DiscourseExperimentService(ProfileExperimentService):
    def __init__(self, settings, *, gate_path, candidate_version=None, replay_paths=()):
        CompetitionMemoryService.__init__(self, settings)
        self._recipe_options = (
            {"version": candidate_version} if candidate_version else {}
        )
        self._replay_paths = tuple(replay_paths)
        gate = json.loads(Path(gate_path).read_text())
        if (
            gate.get("status") != "PASS_FOR_SMALL_PUBLIC_PANEL_ONLY"
            or gate.get("recipe_hash")
            != DiscourseRecipe("0" * 64, **self._recipe_options).fingerprint
            or gate.get("admission_precision", 0) < 0.8
            or gate.get("positive_coverage", 0) < 0.75
            or gate.get("systematic_critical_contamination", True)
        ):
            raise StateUnavailable(
                "Discourse semantic qualification missing or mismatched"
            )
        self.profile_gates = ScopeGates(timeout=600, max_inflight_scopes=16)
        self.bridge = None
        self.enable_governance = False

    @failure_audit()
    def start(self):
        CompetitionMemoryService.start(self)
        self._ready = False
        self.discourse_recipe = DiscourseRecipe(
            self.state.profile_hash, **self._recipe_options
        )
        root = Path(self.settings.competition_run_root)
        policy = {
            "schema": "profile-discourse-overlay-v1",
            "base": "source_b1_public_v1",
            "recipe_hash": self.discourse_recipe.fingerprint,
            "formal": False,
            "context": "profile-interpretation-plus-raw-source-v1",
            "budget": {"k": 10, "chars": 24000},
            "temporary_visibility": "requires_explicit_nonexpired_interval",
            "publication": "raw plus complete frame/Profile under outer user lease",
        }
        path = root / "discourse-experiment.json"
        if path.exists() and json.loads(path.read_text()) != policy:
            raise StateUnavailable("Experimental run policy changed")
        atomic_json(path, policy)
        self.context_policy = DiscourseContext(
            self.state, self.context_policy, self.discourse_recipe
        )
        self.bridge = (
            DiscourseReplayBridge(
                root / "artifacts" / "hlm-calls.jsonl", self._replay_paths
            )
            if self._replay_paths
            else ModelBridge(
                root / "artifacts" / "hlm-calls.jsonl",
                provider_factory=DiscourseProvider,
            )
        )
        self.discourse_runtime = ProfileDiscourseRuntime(
            self.coordinator, self.bridge, recipe=self.discourse_recipe
        )
        self._restore_hlm_frontiers()
        self._ready = True

    def _finish_hlm(self, scope, request_id, uids):
        before = time.monotonic()
        result = self.discourse_runtime.run(scope, uids)
        append_call(
            Path(self.settings.competition_run_root) / "artifacts" / "hlm-adds.jsonl",
            {
                "request_id": request_id,
                "user_id": scope.user_id,
                "result": result,
                "hlm_seconds": time.monotonic() - before,
            },
        )
        return result["output_uids"]

    def _restore_scope_hlm(self, scope):
        while rows := missing_discourse_requests(
            self.state, scope, self.discourse_recipe
        ):
            for row in rows:
                ids = self.state._rows(
                    """SELECT logical_uid FROM memories WHERE scope=? AND request_id=?
                    AND memory_level='L0_Observation' ORDER BY turn_index,logical_uid LIMIT 1025""",
                    (scope.key, row["request_id"]),
                )
                if len(ids) > 1024:
                    raise StateUnavailable("Unbounded raw Add")
                self._finish_hlm(
                    scope, row["request_id"], [m["logical_uid"] for m in ids]
                )

    def _restore_hlm_frontiers(self):
        for scope in self.state.iter_scopes():
            self._restore_scope_hlm(scope)

    def add_messages(self, user_id, session_id, request_id, messages):
        scope = self.scope(user_id)
        with failure_audit(), self.profile_gates.lease(scope, write=True):
            self._restore_scope_hlm(scope)
            count = CompetitionMemoryService.add_messages(
                self, user_id, session_id, request_id, messages
            )
            self._finish_hlm(
                scope,
                request_id,
                [
                    scope.memory_uid(session_id, request_id, i)
                    for i in range(len(messages))
                ],
            )
            return count

    def _live_candidates(self, scope, uids, spaces=None):
        live = CompetitionMemoryService._live_candidates(self, scope, uids, spaces)
        memories = self.state.get_memories(scope, live)
        return [
            u
            for u in live
            if not memories[u].metadata.get("hlm_profile")
            or live_profile(self.state, scope, memories[u], self.discourse_recipe)
        ]

    def health(self):
        result = CompetitionMemoryService.health(self)
        if self.state._rows(
            "SELECT 1 FROM build_jobs WHERE status!='completed' AND json_extract(progress,'$.discourse_root')=1 LIMIT 1"
        ):
            raise StateUnavailable("Incomplete discourse frontier")
        if any(
            missing_discourse_requests(self.state, s, self.discourse_recipe, 1)
            for s in self.state.iter_scopes()
        ):
            raise StateUnavailable("Raw Add lacks complete discourse result")
        return result

    def search(self, user_id, query, top_k):
        with self.profile_gates.lease(self.scope(user_id), write=False):
            if self.state._rows(
                "SELECT 1 FROM build_jobs WHERE scope=? AND status!='completed' AND json_extract(progress,'$.discourse_root')=1 LIMIT 1",
                (self.scope(user_id).key,),
            ):
                raise StateUnavailable("Incomplete discourse frontier")
            if missing_discourse_requests(
                self.state, self.scope(user_id), self.discourse_recipe, 1
            ):
                raise StateUnavailable("Raw Add lacks complete discourse result")
            token = QUERY.set(query)
            try:
                return CompetitionMemoryService.search(self, user_id, query, top_k)
            finally:
                QUERY.reset(token)
