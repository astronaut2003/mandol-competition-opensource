"""Request-local observation of the unchanged Competition API for public evals.

Install only on an owned evaluation service, after startup and before requests.
No class/global methods are modified. Every wrapper calls the original exactly
once and records outputs without rescoring or adding context. The caller drains
one completed trace after each Search; there is no historical query cache.
"""

import threading
import time
from contextvars import ContextVar

from mandol_substrate.state.models import digest


class ApiQualityTrace:
    def __init__(self, service, *, max_candidates=30000, correlation_id=None):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 30000:
            raise ValueError("Invalid bounded trace candidate limit")
        self.service = service
        self.max_candidates = max_candidates
        self.correlation_id = correlation_id
        self._active = ContextVar("public_api_quality_trace", default=None)
        self._completed = None
        self._inflight = False
        self._lock = threading.Lock()
        self._originals = []

    def _replace(self, owner, name, wrapper):
        original = getattr(owner, name)
        # Preserve whether an attribute was originally defined on the instance,
        # rather than leaving a bound method that shadows its class after exit.
        previous = owner.__dict__.get(name)
        existed = name in owner.__dict__
        self._originals.append((owner, name, original, previous, existed, wrapper))
        setattr(owner, name, wrapper)
        return original

    def __enter__(self):
        if self._originals or not self.service.ready:
            raise ValueError("Trace requires a ready owned service and fresh observer")
        service = self.service
        original_search = service.search
        original_retrieve = service.retriever.retrieve
        original_hydrate = service.state.get_memories
        original_rerank = service.reranker.rerank

        def hydrate(scope, uids):
            memories = original_hydrate(scope, uids)
            trace = self._active.get()
            if trace is not None:
                if len(uids) > self.max_candidates or scope.user_id != trace["user_id"]:
                    raise ValueError("Trace hydration scope/candidate bound differs")
                if scope.key == trace["scope"]:
                    trace["last_hydration"] = (
                        list(uids),
                        {uid: digest(m.text_content) for uid, m in memories.items()},
                    )
                else:
                    # A typed route hydrates another same-user namespace. It
                    # does not replace the raw reranker input identity/order.
                    trace["derived_hydration_calls"] += 1
            return memories

        def retrieve(view, query, dense, sparse, **kwargs):
            started = time.perf_counter()
            fused, audit = original_retrieve(view, query, dense, sparse, **kwargs)
            trace = self._active.get()
            if trace is not None:
                if view.scope.key != trace["scope"] or trace["retrieval"] is not None:
                    raise ValueError("Trace requires exactly one same-scope retrieval")
                legs = {method: list(hits) for method, hits in audit["legs"].items()}
                if sum(len(hits) for hits in legs.values()) > self.max_candidates:
                    raise ValueError("Trace first-stage candidate budget exceeded")
                trace["retrieval"] = {
                    "published_seq": view.published_seq,
                    "query_sha256": digest(query),
                    "legs": legs,
                    "fused": [(unit.uid, score) for unit, score in fused],
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "query_policy_hash": audit.get("query_policy_hash"),
                }
            return fused, audit

        def rerank(query, documents):
            trace = self._active.get()
            identities = None
            if trace is not None:
                if (
                    trace["reranker"] is not None
                    or len(documents) > self.max_candidates
                ):
                    raise ValueError("Trace requires one bounded reranker call")
                navigation = getattr(service, "navigation", None)
                declared = getattr(navigation, "rerank_inputs", None)
                declared = declared.get() if declared is not None else None
                hydration = ((declared["ids"], declared["hashes"]) if declared is not None
                             else trace.get("last_hydration"))
                if hydration is None:
                    raise ValueError("Reranker trace lacks canonical hydration")
                identities, hashes = hydration
                # Current service hydrates its final ordered candidate UIDs
                # immediately before rerank. Assert alignment; never match
                # duplicate texts heuristically or invent an ID for a score.
                if len(identities) != len(documents) or any(
                    hashes.get(uid) != digest(text)
                    for uid, text in zip(identities, documents, strict=True)
                ):
                    raise ValueError(
                        "Canonical UID/text order differs from rerank input"
                    )
                if declared is not None:
                    trace["source_families"] = declared["source_families"]
            started = time.perf_counter()
            scores = original_rerank(query, documents)
            if trace is not None:
                if len(scores) != len(identities):
                    raise ValueError("Trace observed missing reranker scores")
                trace["reranker"] = {
                    "query_sha256": digest(query),
                    "candidate_ids": identities,
                    "content_hashes": [digest(text) for text in documents],
                    "scores": list(scores),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            return scores

        def search(user_id, query, top_k, *, choices=None):
            correlation = self.correlation_id() if self.correlation_id is not None else None
            if correlation is not None:
                import re
                if not isinstance(correlation, str) or re.fullmatch(r"[0-9a-f]{64}", correlation) is None:
                    raise ValueError("Invalid observation-only correlation ID")
            with self._lock:
                if self._completed is not None or self._inflight:
                    raise ValueError(
                        "Public trace requires one drained sequential Search"
                    )
                self._inflight = True
            trace = {
                "user_id": user_id,
                "scope": service.scope(user_id).key,
                "query_sha256": digest(query),
                "top_k": top_k,
                "retrieval": None,
                "reranker": None,
                "derived_hydration_calls": 0,
                "navigation": None,
                "status": "INCOMPLETE",
                "rerank_enabled": not (
                    getattr(service, "query_policy", None) is not None
                    and not service.query_policy.rerank
                ),
            }
            if correlation is not None:
                trace["evaluation_call_id"] = correlation
            token = self._active.set(trace)
            started = time.perf_counter()
            try:
                result = (original_search(user_id, query, top_k, choices=choices)
                          if choices is not None else original_search(user_id, query, top_k))
                if trace["retrieval"] is None or (
                    trace["rerank_enabled"] and trace["reranker"] is None
                ):
                    raise ValueError("Public trace missing a required retrieval stage")
                if not trace["rerank_enabled"] and trace["reranker"] is not None:
                    raise ValueError("OFF policy unexpectedly performed reranking")
                candidate_ids = (
                    trace["reranker"]["candidate_ids"] if trace["rerank_enabled"]
                    else trace["last_hydration"][0]
                )
                trace["candidate_ids_before_rerank"] = candidate_ids
                if any(hit["id"] not in candidate_ids for hit in result):
                    raise ValueError("Search returned an untraced candidate")
                trace["returned_ids"] = [hit["id"] for hit in result]
                seeds = {uid for uid, _ in trace["retrieval"]["fused"]}
                trace["expansion_candidate_ids"] = [
                    uid for uid in candidate_ids if uid not in seeds
                ]
                trace["expansion_kind"] = (
                    "combined structural/semantic; not individually inferred"
                )
                trace["status"] = "PASS"
                return result
            except BaseException as exc:
                trace["error_type"] = type(exc).__name__
                raise
            finally:
                trace.pop("last_hydration", None)
                trace["service_latency_ms_with_observation"] = (
                    time.perf_counter() - started
                ) * 1000
                self._active.reset(token)
                with self._lock:
                    self._inflight = False
                    if self._completed is not None:
                        raise ValueError(
                            "Observer supports sequential evaluation Search only"
                        )
                    self._completed = trace

        self._replace(service.state, "get_memories", hydrate)
        self._replace(service.retriever, "retrieve", retrieve)
        self._replace(service.reranker, "rerank", rerank)
        navigation = getattr(service, "navigation", None)
        if navigation is not None:
            original_expand = navigation.expand

            def expand(scope, view, query, dense, sparse, raw_ids, canonical):
                result = original_expand(scope, view, query, dense, sparse, raw_ids, canonical)
                trace = self._active.get()
                if trace is not None:
                    ids, expanded_canonical, audit = result
                    if scope.key != trace["scope"] or len(ids) > self.max_candidates:
                        raise ValueError("Trace navigation scope/bound differs")
                    trace["last_hydration"] = (list(ids), {
                        uid: digest(expanded_canonical[uid].text_content) for uid in ids})
                    trace["navigation"] = {k: audit[k] for k in (
                        "selected", "failures", "timings", "total_seconds", "route", "reason_codes",
                        "raw_ids", "rerank_ids", "graph_depth", "graph_reached") if k in audit}
                return result

            self._replace(navigation, "expand", expand)
        self._replace(service, "search", search)
        return self

    def take(self):
        with self._lock:
            trace, self._completed = self._completed, None
        if trace is None:
            raise ValueError("No completed public Search trace")
        return trace

    def __exit__(self, *_):
        for owner, name, _, previous, existed, wrapper in reversed(self._originals):
            if getattr(owner, name) is not wrapper:
                raise ValueError("Owned trace wrapper changed during evaluation")
            if existed:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._originals.clear()
