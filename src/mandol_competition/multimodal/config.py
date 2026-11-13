"""Runtime configuration for multimodal Add and Search."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


class ConfigurationError(RuntimeError):
    """The runtime configuration cannot be used safely."""


DEFAULT_RETRIEVAL_METHODS = ("bm25", "cosine_similarity", "splade")


@dataclass(frozen=True)
class MultimodalConfig:
    env: str = "dev"
    data_dir: Path = field(default_factory=lambda: Path("data/multimodal"))
    auth_token: str | None = None

    vlm_backend: str = "fake"
    vlm_model: str = "gpt-4o-mini"
    # A provider may expose a different wire name for a deployment alias.
    # Explicit only: never guess a replacement after a failed model request.
    vlm_request_model: str | None = None
    vlm_base_url: str | None = None
    vlm_api_key_env: str = "OPENAI_API_KEY"
    prompt_version: str = "multimodal_sgm_v1"
    schema_version: str = "1"
    vlm_max_attempts: int = 3
    vlm_timeout_seconds: float = 180.0
    vlm_output_tokens: int = 1024
    vlm_max_output_tokens: int = 1024

    data_ttl_days: int = 30
    retention_interval_seconds: int = 3600
    max_request_bytes: int = 64 * 1024 * 1024
    add_timeout_seconds: float = 1800.
    search_timeout_seconds: float = 1800.

    max_image_bytes: int = 10 * 1024 * 1024
    max_request_image_bytes: int = 30 * 1024 * 1024
    max_pixels: int = 40_000_000
    image_byte_budget: int = 28 * 1024 * 1024
    max_images: int = 10

    retrieval_methods: tuple[str, ...] = DEFAULT_RETRIEVAL_METHODS
    rerank_method: str | None = "baai"
    search_policy: str = "m0-original-v1"
    # Search-only overlay: never enters the immutable encoding/storage profile.
    evidence_policy: str = "baseline"
    full_database_audit: bool = True
    vision_failure_policy: str = "degrade"
    # Legacy/development construction keeps its previous policy. The new strict
    # serving configuration enables source preservation unless explicitly disabled.
    vision_source_fallback: bool = False
    embedding_model_name: str | None = None
    embedding_dim: int | None = None
    max_graphs: int = 64
    snapshot_every: int = 100

    # Formal serving uses the independent canonical/projection substrate.
    namespace: str = "multimodal"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "multimodal_v1"
    qdrant_timeout: float = 10.0
    retrieval_profile_path: Path | None = None
    inference_contract: str = "vllm-http-v1"
    qdrant_api_key: str | None = None
    embedding_endpoint: str = "http://127.0.0.1:18201"
    sparse_endpoint: str = "http://127.0.0.1:18202"
    reranker_endpoint: str = "http://127.0.0.1:18203"
    inference_timeout: float = 60.0
    reranker_pool: dict = field(default_factory=dict, hash=False)
    reranker_credential_env: str = 'DASHSCOPE_API_KEY'
    model_device: str = "cpu"
    publication_retries: int = 3
    publication_timeout: float = 45.0
    profiling_path: Path | None = None

    def __post_init__(self) -> None:
        if type(self.full_database_audit) is not bool:
            raise ConfigurationError('full_database_audit must be a bool')
        if type(self.vision_source_fallback) is not bool:
            raise ConfigurationError('vision_source_fallback must be a bool')
        if self.vision_source_fallback and self.vision_failure_policy != 'retry-required-v1':
            raise ConfigurationError('source fallback requires bounded strict visual recovery')
        if self.vlm_backend == 'openai_compat':
            from mandol_competition.inference.generation import generation_route
            route = generation_route(self.vlm_base_url, self.vlm_request_model or self.vlm_model,
                                     credential_env=self.vlm_api_key_env)
            if route.routed:
                object.__setattr__(self, 'vlm_base_url', route.endpoint)
                object.__setattr__(self, 'vlm_request_model', route.model)
                object.__setattr__(self, 'vlm_api_key_env', route.credential_env)
        from .vlm.output_guard import validate_output_budgets
        try:
            validate_output_budgets(self.vlm_output_tokens, self.vlm_max_output_tokens)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from None
        if (type(self.vlm_timeout_seconds) not in (int, float)
                or not math.isfinite(self.vlm_timeout_seconds)
                or not 0 < self.vlm_timeout_seconds <= 600):
            raise ConfigurationError("VLM attempt timeout must be positive and at most 600 seconds")
        if self.evidence_policy not in {"baseline", "source-grounded-v1", "source-grounded-atomic-v2",
                                        "source-grounded-delivery-v3", "source-grounded-recall-v4"}:
            raise ConfigurationError("Unknown multimodal evidence policy")
        if self.vision_failure_policy not in {"degrade", "retry-required-v1"}:
            raise ConfigurationError("Unknown multimodal vision failure policy")
        if self.evidence_policy != "baseline" and self.vlm_backend != "openai_compat":
            raise ConfigurationError("Grounded evidence selection requires the real mini endpoint")
        if self.search_policy not in {"m0-original-v1", "mm-querit-same-pool-rank-only-v1", "mm-qwen37-same-pool-rank-only-v1"}:
            raise ConfigurationError("Unknown multimodal search policy")
        if self.search_policy == "mm-querit-same-pool-rank-only-v1" and (
                self.inference_contract != "aml-official-eligible-v1" or self.rerank_method != "baai"):
            raise ConfigurationError("MM Querit requires the named-v4 M0 index and enabled reranking")
        if self.search_policy == 'mm-qwen37-same-pool-rank-only-v1' and (
                self.inference_contract != 'aml-official-eligible-v1' or self.rerank_method != 'baai'):
            raise ConfigurationError('MM Qwen requires the named-v4 M0 index and enabled reranking')
        if self.inference_contract not in {"vllm-http-v1","aml-official-eligible-v1"}:
            raise ConfigurationError("Unknown multimodal inference contract")
        if self.vlm_backend not in {"fake", "openai_compat"}:
            raise ConfigurationError(f"Unsupported vlm_backend: {self.vlm_backend!r}")
        if self.env == "production" and self.vlm_backend != "openai_compat":
            raise ConfigurationError("Production requires a real VLM backend, not 'fake'")
        if not 0 < self.max_image_bytes <= 10 * 1024 * 1024:
            raise ConfigurationError("max_image_bytes must be between 1 and 10 MiB")
        if not 0 < self.max_request_image_bytes <= 30 * 1024 * 1024:
            raise ConfigurationError("max_request_image_bytes must be between 1 and 30 MiB")
        if not 0 < self.image_byte_budget <= self.max_request_image_bytes:
            raise ConfigurationError("image_byte_budget must fit within max_request_image_bytes")
        if not self.retrieval_methods:
            raise ConfigurationError("at least one retrieval method is required")
        if self.embedding_model_name and (self.embedding_dim is None or self.embedding_dim <= 0):
            raise ConfigurationError("custom embedding model requires a positive embedding_dim")
        if not self.namespace or not self.qdrant_collection or self.qdrant_timeout <= 0:
            raise ConfigurationError("nonempty namespace/collection and positive Qdrant timeout required")

    def user_key_for(self, user_id: str) -> str:
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()

    def user_root(self, user_id: str) -> Path:
        namespace_key = hashlib.sha256(self.namespace.encode("utf-8")).hexdigest()
        return Path(self.data_dir) / "media" / namespace_key / "users" / self.user_key_for(user_id)


@lru_cache(maxsize=1)
def get_config() -> MultimodalConfig:
    """Read the runtime configuration once from environment variables."""
    audit = os.getenv('MANDOL_MM_FULL_DATABASE_AUDIT', '1')
    if audit not in {'0', '1'}:
        raise ConfigurationError('MANDOL_MM_FULL_DATABASE_AUDIT must be 0 or 1')
    failure_policy = os.getenv('MANDOL_MM_VISION_FAILURE_POLICY', 'degrade')
    source_fallback = os.getenv('MANDOL_MM_VISION_SOURCE_FALLBACK',
        '1' if failure_policy == 'retry-required-v1' else '0')
    if source_fallback not in {'0', '1'}:
        raise ConfigurationError('MANDOL_MM_VISION_SOURCE_FALLBACK must be 0 or 1')
    return MultimodalConfig(
        full_database_audit=audit == '1',
        vision_source_fallback=source_fallback == '1',
        add_timeout_seconds=float(os.getenv("MANDOL_MM_ADD_TIMEOUT_SECONDS", "1800")),
        search_timeout_seconds=float(os.getenv("MANDOL_MM_SEARCH_TIMEOUT_SECONDS", "1800")),
        env=os.getenv("MANDOL_MM_ENV", "dev"),
        data_dir=Path(os.getenv("MANDOL_MM_DATA_DIR", "data/multimodal")),
        auth_token=os.getenv("MANDOL_MM_AUTH_TOKEN") or None,
        vlm_backend=os.getenv("MANDOL_MM_VLM_BACKEND", "fake"),
        vlm_model=os.getenv("MANDOL_MM_VLM_MODEL", "gpt-4o-mini"),
        vlm_request_model=os.getenv("MANDOL_MM_VLM_REQUEST_MODEL") or None,
        vlm_base_url=os.getenv("MANDOL_MM_VLM_BASE_URL") or None,
        vlm_api_key_env=os.getenv("MANDOL_MM_VLM_API_KEY_ENV", "OPENAI_API_KEY"),
        prompt_version=os.getenv("MANDOL_MM_PROMPT_VERSION", "multimodal_sgm_v1"),
        schema_version=os.getenv("MANDOL_MM_SCHEMA_VERSION", "1"),
        vlm_max_attempts=int(os.getenv("MANDOL_MM_VLM_MAX_ATTEMPTS", "3")),
        vlm_timeout_seconds=float(os.getenv("MANDOL_MM_VLM_TIMEOUT_SECONDS", "180")),
        vlm_output_tokens=int(os.getenv('MANDOL_MM_VLM_OUTPUT_TOKENS', '1024')),
        vlm_max_output_tokens=int(os.getenv('MANDOL_MM_VLM_MAX_OUTPUT_TOKENS', '1024')),
        max_images=int(os.getenv("MANDOL_MM_MAX_IMAGES", "10")),
        retrieval_methods=tuple(method.strip() for method in os.getenv(
            "MANDOL_MM_RETRIEVAL_METHODS", ",".join(DEFAULT_RETRIEVAL_METHODS)
        ).split(",") if method.strip()),
        rerank_method=os.getenv("MANDOL_MM_RERANK_METHOD", "baai").strip() or None,
        search_policy=os.getenv("MANDOL_MULTIMODAL_RETRIEVAL_POLICY", "m0-original-v1"),
        evidence_policy=os.getenv("MANDOL_MM_EVIDENCE_POLICY", "baseline"),
        vision_failure_policy=os.getenv("MANDOL_MM_VISION_FAILURE_POLICY", "degrade"),
        embedding_model_name=os.getenv("MANDOL_MM_EMBEDDING_MODEL") or None,
        embedding_dim=int(os.getenv("MANDOL_MM_EMBEDDING_DIM", "0")) or None,
        max_graphs=int(os.getenv("MANDOL_MM_MAX_GRAPHS", "64")),
        snapshot_every=int(os.getenv("MANDOL_MM_SNAPSHOT_EVERY", "100")),
        namespace=os.getenv("MANDOL_MM_NAMESPACE", "multimodal"),
        qdrant_url=os.getenv("MANDOL_MM_QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_collection=os.getenv("MANDOL_MM_QDRANT_COLLECTION", "multimodal_v1"),
        qdrant_timeout=float(os.getenv("MANDOL_MM_QDRANT_TIMEOUT", "10")),
        retrieval_profile_path=Path(os.environ["MANDOL_MM_RETRIEVAL_PROFILE"])
        if os.getenv("MANDOL_MM_RETRIEVAL_PROFILE") else None,
        inference_contract=os.getenv("MANDOL_MULTIMODAL_INFERENCE_CONTRACT","vllm-http-v1"),
        qdrant_api_key=os.getenv("MANDOL_QDRANT_API_KEY") or None,
        embedding_endpoint=os.getenv("MANDOL_MM_EMBEDDING_ENDPOINT", "http://127.0.0.1:18201"),
        sparse_endpoint=os.getenv("MANDOL_MM_SPARSE_ENDPOINT", "http://127.0.0.1:18202"),
        reranker_endpoint=os.getenv("MANDOL_MM_RERANKER_ENDPOINT", "http://127.0.0.1:18203"),
        inference_timeout=float(os.getenv("MANDOL_MM_INFERENCE_TIMEOUT", "60")),
        model_device=os.getenv("MANDOL_MM_MODEL_DEVICE", "cpu"),
        publication_retries=int(os.getenv("MANDOL_MM_PUBLICATION_RETRIES", "3")),
        publication_timeout=float(os.getenv("MANDOL_MM_PUBLICATION_TIMEOUT", "45")),
        profiling_path=Path(os.environ["MANDOL_MM_PROFILING_PATH"])
        if os.getenv("MANDOL_MM_PROFILING_PATH") else None,
    )
