"""Optional HTTP reranking of existing, source-validated Code episodes.

No memory is written or generated here. Storage and vector identities remain
unchanged; an unavailable or malformed scorer preserves the original order.
"""
from dataclasses import asdict, dataclass, replace
import math
import threading

from mandol_competition.inference.dashscope_rerank import BACKEND, recipe
from mandol_competition.inference.errors import InferenceSchemaError, InferenceRateLimited, InferenceDeadlineExceeded
from mandol_substrate.state import ProfileMismatch
from mandol_substrate.state.models import digest

POLICY = "code-qwen37-episode-rank-only-v1"
CANDIDATE_K = 32


@dataclass(frozen=True)
class RerankResult:
    hits: list
    fallback: bool = False
    error_class: str | None = None


def descriptor(profile, raw_policy):
    from .native_sparse import POLICY as NATIVE, BASELINE, descriptor as native_descriptor
    if raw_policy not in {NATIVE, BASELINE}:
        raise ValueError("Unknown Code raw policy for reranking")
    return dict(schema="code-search-reranker-extension-v1", profile_id=POLICY,
        storage_profile_hash=profile.hash, raw_policy=raw_policy,
        raw_retrieval_profile_hash=digest(native_descriptor(profile)) if raw_policy == NATIVE else profile.hash,
        candidate_k=CANDIDATE_K, candidate_text="unchanged-c0-single-card-evidence-packer",
        packer=dict(max_cards=8, max_chars=16000, per_source_chars=1200),
        reranker=asdict(recipe()), reranker_recipe_hash=recipe().hash,
        score_semantics="request-relative-qwen-relevance; original-c0-score-on-fallback",
        required=False, fallback="unchanged-c0-episode-order", generation=False)


class CodeEpisodeReranker:
    def __init__(self, client):
        self.client = client
        self._lock = threading.Lock()
        self._counts = dict(requests=0, successes=0, fallbacks=0)
        self._last_error = None

    def rank(self, query, hits, sources, *, user_id):
        from .software.experience_retrieval import EvidencePacker
        if not hits:
            return RerankResult([])
        if len(hits) > CANDIDATE_K:
            raise ValueError("Code reranker candidate budget exceeded")
        ids = [h.episode.episode_id for h in hits]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate Code episode identity")
        documents = []
        # Source/tenant failures must fail closed before any external request.
        for hit in hits:
            episode = hit.episode
            if episode.user_id != user_id or any(
                    sources[uid].metadata.get("user_id") != user_id for uid in episode.source_uids):
                raise ProfileMismatch("Code reranker source scope mismatch")
            episode.validate_provenance(sources)
            cards = EvidencePacker(max_cards=1, max_chars=16000).pack(
                [hit], sources, query=query, top_k=1)
            if len(cards) != 1:
                raise ValueError("Code reranker requires a source-backed card")
            documents.append(cards[0]["content"])
        with self._lock:
            self._counts["requests"] += 1
        try:
            scores = self.client.score(query, documents)
            if (len(scores) != len(hits) or any(type(s) not in (int, float)
                    or not math.isfinite(s) or not 0 <= s <= 1 for s in scores)):
                raise InferenceSchemaError("invalid_code_reranker_scores")
            ranked = sorted(zip(hits, scores, strict=True),
                            key=lambda pair: (-pair[1], pair[0].episode.episode_id))
            result = [replace(hit, score=float(score)) for hit, score in ranked]
        except (InferenceRateLimited, InferenceDeadlineExceeded):
            raise  # Preserve the caller's retry/deadline signal at the HTTP boundary.
        except Exception as exc:
            with self._lock:
                self._counts["fallbacks"] += 1
                self._last_error = type(exc).__name__
            # No retries at this layer, score mixing, pruning or model switch.
            return RerankResult(list(hits), fallback=True, error_class=type(exc).__name__)
        with self._lock:
            self._counts["successes"] += 1
            self._last_error = None
        return RerankResult(result)

    def stats(self):
        with self._lock:
            return dict(backend=BACKEND, policy=POLICY, required=False,
                recipe_hash=recipe().hash, candidate_k=CANDIDATE_K,
                **self._counts, last_error=self._last_error)

    def close(self):
        self.client.close()
