"""Optional episode ranking and bounded source-backed evidence packaging."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import logging
import math

from mandol_substrate.representation import MemoryUnit
from .anchors import CodeAnchorExtractor
from .episodes import EngineeringEpisodeMemory, ExperienceOutcome, VerificationLevel
from .lexical import CodeBM25
from .query import CodingQueryRouter, EvidenceNeedDecomposer


@dataclass(frozen=True)
class EpisodeRetrievalConfig:
    mode: str = "raw"  # raw, episode, hybrid
    code_aware: bool = False
    router: bool = False
    evidence_needs: bool = False
    anchor_weight: float = 0.0
    outcome_weight: float = 0.0
    graph_weight: float = 0.0
    route_weight: float = 0.0
    rrf_k: int = 60
    candidate_k: int = 100
    distinctive_anchors: bool = False

    def __post_init__(self):
        if self.mode not in {"raw", "episode", "hybrid"}:
            raise ValueError("Unknown episode retrieval mode")
        if any(not 0 <= weight <= .25 for weight in (self.anchor_weight, self.outcome_weight, self.graph_weight, self.route_weight)):
            raise ValueError("Experimental feature weights must be bounded to [0, .25]")
        if self.rrf_k < 1 or self.candidate_k < 1:
            raise ValueError("Retrieval limits must be positive")
        if type(self.distinctive_anchors) is not bool:
            raise ValueError('Distinctive anchor policy must be explicit boolean')


@dataclass
class EpisodeHit:
    episode: EngineeringEpisodeMemory
    score: float
    evidence_uids: list[str]
    features: dict[str, float] = field(default_factory=dict)


class GraphAwareReranker:
    """Use the caller's authorized repository graph as a bounded feature source.

    Typed structural distance is supportive evidence, not a calibrated relevance
    probability. No graph candidates are inserted into first-stage retrieval.
    """

    relations = ("CALLS", "IMPORTS", "INHERITS_FROM", "IMPLEMENTS", "TESTED_BY",
                 "AFFECTS_BEHAVIOR_OF", "MODIFIES", "HAS_EVOLUTION", "CONTAINS", "HAS_METHOD")

    def __init__(self, graph, *, repository: str, repository_uid: str, max_hops=2, max_neighbors=32):
        if not 1 <= max_hops <= 3 or not 1 <= max_neighbors <= 128:
            raise ValueError("Graph expansion must be bounded")
        self.graph, self.repository = graph, repository
        self.max_hops, self.max_neighbors = max_hops, max_neighbors
        self.units = {unit.uid: unit for unit in graph.get_all_units() if unit.metadata.get("repo_uid") == repository_uid}
        self.index = defaultdict(set)
        short_names = defaultdict(set)
        for uid, unit in self.units.items():
            for name in ("qualified_name", "path", "file_path", "module_name"):
                value = unit.raw_data.get(name)
                if isinstance(value, str) and value:
                    self.index[CodeAnchorExtractor.normalize(value)].add(uid)
            name = unit.raw_data.get("name")
            if isinstance(name, str) and name:
                short_names[CodeAnchorExtractor.normalize(name)].add(uid)
        for name, ids in short_names.items():
            if len(ids) == 1:
                self.index[name].update(ids)

    def features(self, query: str, episode: EngineeringEpisodeMemory) -> dict[str, float]:
        empty = {"same_symbol": 0.0, "same_file": 0.0, "same_module": 0.0, "structural_proximity": 0.0}
        if episode.repository != self.repository:
            return empty
        query_tokens = {anchor.normalized for anchor in CodeAnchorExtractor().extract(query)}
        evidence_tokens = {anchor["normalized"] for anchor in episode.code_anchors}
        seeds = set().union(*(self.index[token] for token in query_tokens)) if query_tokens else set()
        targets = set().union(*(self.index[token] for token in evidence_tokens)) if evidence_tokens else set()
        if seeds & targets:
            kinds = {self.units[uid].metadata.get("software_type", self.units[uid].raw_data.get("software_type", "")) for uid in seeds & targets}
            return {"same_symbol": float(bool(kinds & {"function", "class", "symbol_memory"})),
                    "same_file": float(bool(kinds & {"file", "file_memory"})),
                    "same_module": float(bool(kinds & {"directory", "module_memory"})),
                    "structural_proximity": 1.0}
        queue = deque((uid, 0) for uid in sorted(seeds))
        visited = set(seeds)
        while queue:
            uid, distance = queue.popleft()
            if distance >= self.max_hops:
                continue
            neighbors = sorted(set(self.graph.get_successors(uid)) | set(self.graph.get_predecessors(uid)))
            for neighbor in neighbors[:self.max_neighbors]:
                if neighbor in visited or neighbor not in self.units:
                    continue
                typed = any(self.graph.get_relationship(uid, neighbor, relation) is not None
                            or self.graph.get_relationship(neighbor, uid, relation) is not None for relation in self.relations)
                if not typed:
                    continue
                if neighbor in targets:
                    return {**empty, "structural_proximity": 1 / (distance + 2)}
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
        return empty


class EpisodeRetriever:
    def __init__(self, episodes: list[EngineeringEpisodeMemory], sources: dict[str, MemoryUnit], *,
                 user_id: str, config: EpisodeRetrievalConfig | None = None, graph_reranker: GraphAwareReranker | None = None,
                 raw_channel=None, episode_channel=None, raw_index=None):
        self.config = config or EpisodeRetrievalConfig()
        self.episodes = {episode.episode_id: episode for episode in episodes if episode.user_id == user_id}
        self.sources, self.raw_to_episode = {}, {}
        for episode in self.episodes.values():
            episode.validate_provenance(sources)
            for uid in episode.source_uids:
                if uid in self.raw_to_episode:
                    raise ValueError("Raw evidence belongs to multiple episode identities")
                self.sources[uid] = sources[uid]
                self.raw_to_episode[uid] = episode.episode_id
        self.raw_index = raw_index if raw_index is not None else CodeBM25({uid: str(source.raw_data.get("text_content", "")) for uid, source in self.sources.items()}, code_aware=self.config.code_aware)
        self.episode_index = CodeBM25({uid: episode.retrieval_text() for uid, episode in self.episodes.items()}, code_aware=self.config.code_aware)
        self.graph_reranker = graph_reranker
        self.raw_channel = raw_channel
        self.episode_channel = episode_channel
        self.fallback_count = 0
        self.anchor_df = Counter(anchor for episode in self.episodes.values()
                                 for anchor in {a['normalized'] for a in episode.code_anchors})

    def _raw_search(self, query):
        if self.raw_channel is not None:
            try:
                matches = self.raw_channel.search(query, top_k=len(self.sources))
                ids = [uid for uid, _ in matches]
                if len(ids) != len(set(ids)) or not set(ids) <= self.sources.keys() or any(not math.isfinite(score) for _, score in matches):
                    raise ValueError("Raw model channel returned invalid or out-of-scope candidates")
                return matches
            except Exception as exc:
                from mandol_competition.inference.errors import InferenceRateLimited, InferenceDeadlineExceeded
                if isinstance(exc, (InferenceRateLimited, InferenceDeadlineExceeded)):
                    raise
                from mandol_substrate.state import ProfileMismatch
                if isinstance(exc, ProfileMismatch):
                    raise
                self.fallback_count += 1
                logging.getLogger(__name__).warning("Raw model retrieval failed (%s); using local lexical fallback", type(exc).__name__)
        return self.raw_index.search(query, top_k=len(self.sources))

    def search(self, query: str, top_k: int = 8) -> list[EpisodeHit]:
        if not query.strip() or top_k <= 0:
            return []
        queries = (query,) + (EvidenceNeedDecomposer().decompose(query) if self.config.evidence_needs else ())
        scores, evidence = defaultdict(float), defaultdict(list)
        for need in queries:
            if self.config.mode in {"raw", "hybrid"}:
                seen_episodes = set()
                matches = self._raw_search(need)
                for uid, _ in matches:
                    eid = self.raw_to_episode[uid]
                    if uid not in evidence[eid] and len(evidence[eid]) < 4:
                        evidence[eid].append(uid)
                    # Best raw rank per episode prevents long sessions winning
                    # solely by emitting many near-duplicate snippets.
                    if eid not in seen_episodes:
                        seen_episodes.add(eid)
                        scores[eid] += 1 / (self.config.rrf_k + len(seen_episodes))
                        if len(seen_episodes) >= self.config.candidate_k:
                            break
            if self.config.mode in {"episode", "hybrid"}:
                for rank, (eid, _) in enumerate(self._episode_search(need), 1):
                    scores[eid] += 1 / (self.config.rrf_k + rank)
        plan = CodingQueryRouter().route(query)
        anchors = {a.normalized for a in CodeAnchorExtractor().extract(query)}
        hits = []
        for eid, relevance in scores.items():
            episode = self.episodes[eid]
            overlap = len(anchors & {a["normalized"] for a in episode.code_anchors}) / max(1, len(anchors))
            if self.config.distinctive_anchors:
                # Common error names/helpers provide weak evidence. Normalize
                # information content against the whole authorized episode
                # corpus, not just matched terms (which would cancel IDF when
                # a query has a single ubiquitous anchor). No cross-user stats.
                n = len(self.episodes)
                matched = anchors & {a['normalized'] for a in episode.code_anchors}
                overlap = sum(math.log((n+1)/(self.anchor_df[a]+1)) for a in matched) / max(
                    1e-12, len(anchors)*math.log(n+1))
            verification = {VerificationLevel.INFERRED: 0., VerificationLevel.OBSERVED: .5, VerificationLevel.VERIFIED: 1.}[episode.verification_level]
            failure = episode.outcome == ExperienceOutcome.FAILURE or any(attempt.outcome == ExperienceOutcome.FAILURE for attempt in episode.attempts)
            outcome_signal = (1. if failure and plan.asks_for_failure else
                              {ExperienceOutcome.SUCCESS: 1., ExperienceOutcome.PARTIAL: .5}.get(episode.outcome, 0.))
            outcome = (outcome_signal + verification) / 2
            route = float((plan.intent == "DEBUG" and bool(episode.symptoms or episode.attempts))
                          or (plan.intent == "DEVELOPMENT" and bool(episode.engineering_knowledge)))
            graph = self.graph_reranker.features(query, episode) if self.graph_reranker and self.config.graph_weight else {}
            multiplier = (1 + self.config.anchor_weight * overlap + self.config.outcome_weight * outcome
                          + self.config.graph_weight * graph.get("structural_proximity", 0.)
                          + (self.config.route_weight * route if self.config.router else 0.))
            features = {"anchor_overlap": overlap, "outcome": outcome, "verification": verification, "route": route, **graph}
            hits.append(EpisodeHit(episode, relevance * multiplier, evidence[eid], features))
        return sorted(hits, key=lambda hit: (-hit.score, hit.episode.episode_id))[:top_k]

    def _episode_search(self, query):
        if self.episode_channel is not None:
            try:
                matches = self.episode_channel.search(query, top_k=self.config.candidate_k)
                ids = [uid for uid, _ in matches]
                if (len(ids) != len(set(ids)) or not set(ids) <= self.episodes.keys()
                        or any(not math.isfinite(score) for _, score in matches)):
                    raise ValueError("Episode model channel returned invalid or out-of-scope candidates")
                return matches[:self.config.candidate_k]
            except Exception as error:
                from mandol_competition.inference.errors import InferenceRateLimited, InferenceDeadlineExceeded
                if isinstance(error, (InferenceRateLimited, InferenceDeadlineExceeded)):
                    raise
                self.fallback_count += 1
                logging.getLogger(__name__).warning("Episode model retrieval failed (%s); using local lexical fallback", type(error).__name__)
        return self.episode_index.search(query, top_k=self.config.candidate_k)


class EvidencePacker:
    def __init__(self, *, max_cards=8, max_chars=16000, per_source_chars=1200, include_derived=False,
                 per_card_chars=None):
        if max_cards < 1 or max_chars < 256 or per_source_chars < 64:
            raise ValueError("Evidence budgets are too small")
        self.max_cards, self.max_chars, self.per_source_chars = max_cards, max_chars, per_source_chars
        self.include_derived = include_derived
        if per_card_chars is not None and (type(per_card_chars) is not int or per_card_chars < 256):
            raise ValueError("Per-card budget must be an integer of at least 256 characters")
        self.per_card_chars = per_card_chars

    def pack(self, hits: list[EpisodeHit], sources: dict[str, MemoryUnit], *, query: str, top_k: int):
        remaining, result = self.max_chars, []
        for hit in hits[:min(max(0, top_k), self.max_cards)]:
            card_budget = min(remaining, self.per_card_chars) if self.per_card_chars is not None else remaining
            episode = hit.episode
            episode.validate_provenance(sources)
            summary_label = (f"Task [{episode.task_summary_claim.verification.value}]" if episode.task_summary_claim else "Task")
            header = "[Historical Engineering Experience]\n" + f"{summary_label}: {episode.task_summary[:400]}\n"
            if episode.repository:
                header += f"Repository: {episode.repository[:160]}\n"
            if episode.revision:
                header += f"Historical revision: {episode.revision[:80]}\n"
            header += f"Outcome: {episode.outcome.value}; Verification: {episode.verification_level.value}\n"
            parts = [header]
            used = len(header)
            if self.include_derived:
                claims = ([("Diagnosis", claim) for claim in episode.diagnosis]
                          + [("Historical change", claim) for claim in episode.resolution]
                          + [(knowledge.kind.value, knowledge.claim) for knowledge in episode.engineering_knowledge])
                derived_budget = min(1600, max(0, min(card_budget, self.max_chars // self.max_cards) // 2 - used))
                for label, claim in claims:
                    line = f"\n{label} [{claim.verification.value}]: {claim.text[:320]}\nSources: {', '.join(claim.source_uids)}\n"
                    if len(line) <= derived_budget:
                        parts.append(line)
                        used += len(line)
                        derived_budget -= len(line)
            raw_added = 0
            # Query hits first, then validation/failure and the originating task.
            uids = list(dict.fromkeys(hit.evidence_uids[:1] + episode.outcome_evidence + [v.source_uid for v in episode.validation]
                                     + [uid for a in episode.attempts for uid in a.negative_evidence] + episode.source_uids))
            for uid in uids[:4]:
                text = str(sources[uid].raw_data.get("text_content", ""))
                positions = [text.casefold().find(a.normalized) for a in CodeAnchorExtractor().extract(query)]
                positions = [position for position in positions if position >= 0]
                start = max(0, min(positions) - 100) if positions else 0
                prefix = f"\nEvidence {uid} sha256:{episode.source_hashes[uid]}\n"
                if sources[uid].raw_data.get("artifact_scope"):
                    prefix += "Evidence scope: " + str(sources[uid].raw_data.get("scope", sources[uid].raw_data["artifact_scope"]))[:240] + "\n"
                available = min(self.per_source_chars, card_budget - used - len(prefix) - 100)
                if available < 64:
                    break
                end = min(len(text), start + available)
                piece = prefix + f"[raw excerpt {start}:{end}/{len(text)}]\n" + text[start:end] + "\n"
                parts.append(piece)
                used += len(piece)
                raw_added += 1
            if not raw_added:
                continue
            content = "".join(parts)
            remaining -= len(content)
            result.append({"id": episode.episode_id, "content": content, "score": hit.score, "created_at": None})
        return result
