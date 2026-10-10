"""Explicit selected reranker transport; each model keeps its own score semantics."""

from dataclasses import dataclass

from mandol_substrate.state.models import ProfileMismatch, StateUnavailable, digest
from .models import ServingProfile, VllmModelClient
from .native_reranker_http import NativeRerankerHttpClient, validate_profile


@dataclass(frozen=True)
class NativeServingIdentity:
    model: str
    revision: str
    tokenizer_revision: str
    native_profile_hash: str
    request_timeout: float
    task: str = "reranker"
    backend: str = "native_reference_http"


class NativeReferenceRerankerClient:
    """No score conversion, template rewriting, inference fallback or score cache.

    The reference sidecar verifies pinned weight/tokenizer files while loading;
    health binds its entire official inference profile, not merely a model name.
    """

    def __init__(self, url, profile, *, request_timeout=120, transport=None):
        validate_profile(profile)
        if type(request_timeout) not in (int, float) or not 0 < request_timeout <= 300:
            raise ProfileMismatch("Invalid explicit native request timeout")
        self.profile = NativeServingIdentity(
            profile["model_name"], profile["model_revision"],
            profile["tokenizer_revision"], digest(profile), request_timeout,
        )
        self.client = NativeRerankerHttpClient(
            url, profile, timeout_seconds=request_timeout, transport=transport,
        )

    def validate_profile(self, tokenizer=None):
        # Preparation executes only in the pinned official reference worker.
        # No local tokenizer is substituted into that path.
        return self.health()

    def health(self):
        return self.client.health()

    def rerank(self, query, documents):
        if not isinstance(query, str) or not isinstance(documents, (tuple, list)):
            raise TypeError("Reranker requires original query and document sequence")
        if not documents:
            return []
        try:
            return self.client.score(query, documents)
        except StateUnavailable:
            raise
        except Exception as exc:
            raise StateUnavailable("Native reranker failed: " + type(exc).__name__) from exc

    def close(self):
        self.client.close()


def create_reranker_client(url, options):
    backend = options.get("reranker_backend", "vllm")
    if backend == "vllm":
        return VllmModelClient(url, ServingProfile(**options["reranker_serving"]))
    if backend == "native_reference_http":
        return NativeReferenceRerankerClient(
            url, options["reranker_serving"],
            request_timeout=options["native_reranker_request_timeout"],
        )
    raise ProfileMismatch("Unknown explicit reranker backend; no fallback")
