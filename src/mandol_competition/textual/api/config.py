"""Qdrant-only runtime settings. Quality comes from the immutable run capsule."""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from mandol_competition.textual.core.run_quality import load_capsule, materialize_policies
import hashlib

def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
from mandol_substrate.state.models import RetrievalProfile, StateUnavailable

QUALITY_SHA256 = "d2d9570f2d704ba119e9c42b38012af71fc20d0df4875ed1e753f600830479c8"
AML_V4_QUALITY_SHA256 = "fdc4bb47fcab086346e2f91c9adf3c48f8e3618ae78d7e88e7ec181995802c66"
OFFICIAL_CONTRACT = "aml-official-eligible-v1"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    add_path: str = "/add"
    search_path: str = "/search"
    health_path: str = "/health"
    memory_system_key: str = ""
    allow_unauthenticated_dev: bool = False
    rate_limit_rpm: int = 0
    max_request_bytes: int = 10 * 1024 * 1024
    add_timeout_seconds: float = 60.0
    search_timeout_seconds: float = 30.0
    data_ttl_days: int = 30
    retention_interval_seconds: int = 3600
    log_level: str = "INFO"
    competition_run_root: str | None = None
    qdrant_binary: str | None = None
    qdrant_mode: str = "external"
    qdrant_endpoint: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "textual_v1"
    qdrant_api_key: str | None = None
    evaluation_namespace: str = ""
    embedding_url: str = "http://127.0.0.1:18101"
    reranker_url: str = "http://127.0.0.1:18106"
    inference_contract: str = "vllm-http-v1"
    splade_url: str | None = "http://127.0.0.1:18102"
    model_cache: str | None = None
    competition_profile: str | None = None
    competition_query_policy: str | None = None
    competition_context_policy: str | None = None
    competition_reranker_execution_policy: str | None = None
    competition_return_policy: str | None = None
    experimental_hlm_policy: str | None = None
    experimental_quality_profile: str | None = None
    evidence_profile: str = "default"
    selector_endpoint: str | None = None
    selector_api_key_env: str = "CLOSEAI_API_KEY"
    reranker_pool: dict = field(default_factory=dict)
    # 'profile' preserves historical capsules; unified named-v4 factories
    # explicitly select the current external default without editing storage.
    reranker_backend: str = 'profile'
    reranker_credential_env: str = 'DASHSCOPE_API_KEY'

    def validate(self):
        from mandol_competition.inference.dashscope_rerank import BACKEND
        if self.reranker_backend not in {'profile', BACKEND}:
            raise ValueError('unknown Textual reranker backend')
        if self.qdrant_mode not in {"external", "owned"}:
            raise ValueError("explicit external or owned Qdrant mode required")
        expected_collection = "textual_aml_v4_v1" if self.inference_contract == OFFICIAL_CONTRACT else "textual_v1"
        if self.qdrant_mode == "external" and self.qdrant_collection != expected_collection:
            raise ValueError("Textual production collection differs from admitted model family")
        if self.inference_contract not in {"reference-v1", "vpc-vllm-029-v1", "vllm-http-v1", OFFICIAL_CONTRACT}:
            raise ValueError("Unknown inference contract")
        if self.qdrant_mode == "external" and self.inference_contract == "reference-v1":
            raise ValueError("unified API requires external HTTP inference")
        if self.inference_contract in {"vllm-http-v1", OFFICIAL_CONTRACT}:
            from mandol_competition.inference.http import endpoint_origin
            for url in (self.embedding_url, self.splade_url, self.reranker_url):
                endpoint_origin(url)
            if self.inference_contract == OFFICIAL_CONTRACT and endpoint_origin(self.embedding_url) != endpoint_origin(self.splade_url):
                raise ValueError("Native paired v4 encoding requires one recipe endpoint")
        if self.inference_contract == "vpc-vllm-029-v1":
            from mandol_competition.textual.core.remote_inference import private_endpoint

            for url in (self.embedding_url, self.splade_url, self.reranker_url):
                private_endpoint(url)
        if not self.competition_run_root or not self.evaluation_namespace:
            raise RuntimeError(
                "Explicit dataset/run root and evaluation namespace required"
            )
        if not self.memory_system_key and not self.allow_unauthenticated_dev:
            raise RuntimeError(
                "AML_MEMORY_SYSTEM_KEY required; only explicit dev may disable authentication"
            )
        if not 1 <= self.port <= 65535 or not 1 <= self.data_ttl_days <= 30:
            raise ValueError("Invalid port or retention window")
        for value in (self.add_timeout_seconds, self.search_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Finite positive request deadline required")
        if self.max_request_bytes <= 0 or self.rate_limit_rpm < 0:
            raise ValueError("Invalid request capacity")

    def bind_run(self):
        """No original workspace paths, runtime module patching or model fallback."""
        self.validate()
        official = self.inference_contract == OFFICIAL_CONTRACT
        from ..core.evidence_profiles import (
            OBSERVATION_T0, observation_profile_path, resolve_evidence_profile,
        )
        selected_evidence = resolve_evidence_profile(self.evidence_profile, official=official,
            experimental=bool(self.experimental_quality_profile or self.experimental_hlm_policy))
        quality_sha = AML_V4_QUALITY_SHA256 if official else QUALITY_SHA256
        if self.qdrant_mode == "external":
            from ..runtime import seed_frozen_capsule
            seed_frozen_capsule(self.competition_run_root, "aml_official_v4_v1" if official else "source_b1_public_v1")
        root = Path(self.competition_run_root).resolve()
        if checksum(root / "quality-profile.json") != quality_sha:
            raise StateUnavailable("Only the explicitly admitted frozen Textual family is allowed")
        quality = json.loads((root / "quality-profile.json").read_text())
        profile = RetrievalProfile(**json.loads((root / "profile.json").read_text()))
        capsule = load_capsule(root, quality, profile)
        paths = materialize_policies(root, capsule)
        candidate = None
        candidate_path = self.experimental_quality_profile
        if selected_evidence == OBSERVATION_T0:
            candidate_path = observation_profile_path()
        if candidate_path:
            from ..core.quality_candidate import bind_candidate
            paths, candidate = bind_candidate(root, candidate_path,
                capsule=capsule, profile=profile, baseline_sha256=quality_sha)
            if candidate.get("observation_order_policy") and self.experimental_hlm_policy:
                raise StateUnavailable("Observation-order candidate requires the frozen raw-only pipeline")
            if selected_evidence == OBSERVATION_T0:
                candidate = dict(candidate, status="DEFAULT_USER_APPROVED_PUBLIC_PROXY")
        for field, policy in (
            ("competition_profile", "profile"),
            ("competition_query_policy", "query"),
            ("competition_context_policy", "context"),
            ("competition_reranker_execution_policy", "execution"),
            ("competition_return_policy", "returns"),
        ):
            existing = getattr(self, field)
            if existing and Path(existing).read_bytes() != paths[policy].read_bytes():
                raise StateUnavailable(
                    "Runtime policy override differs from frozen capsule"
                )
            setattr(self, field, str(paths[policy]))
        self.quality_candidate = candidate
        self.selected_evidence_profile = candidate["id"] if candidate else selected_evidence
        return self

    @classmethod
    def from_env(cls, *, bind_run=True):
        settings = cls(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=int(os.getenv("API_PORT", "8000")),
            memory_system_key=os.getenv("AML_MEMORY_SYSTEM_KEY", ""),
            allow_unauthenticated_dev=os.getenv("AML_ALLOW_UNAUTHENTICATED_DEV", "0")
            == "1",
            competition_run_root=os.getenv("MANDOL_TEXTUAL_DATA_DIR", "data/textual"),
            qdrant_binary=os.getenv("QDRANT_BINARY"),
            qdrant_mode=os.getenv("MANDOL_TEXTUAL_QDRANT_MODE", "external"),
            qdrant_endpoint=os.getenv("MANDOL_TEXTUAL_QDRANT_ENDPOINT", os.getenv("QDRANT_ENDPOINT", "http://127.0.0.1:6333")),
            qdrant_collection=os.getenv("MANDOL_TEXTUAL_QDRANT_COLLECTION", "textual_v1"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            experimental_hlm_policy=os.getenv("AML_EXPERIMENTAL_HLM_POLICY"),
            experimental_quality_profile=os.getenv("MANDOL_TEXTUAL_QUALITY_CANDIDATE"),
            evidence_profile=os.getenv("MANDOL_TEXTUAL_EVIDENCE_PROFILE", "default"),
            selector_endpoint=os.getenv("MANDOL_TEXTUAL_SELECTOR_ENDPOINT"),
            selector_api_key_env=os.getenv("MANDOL_TEXTUAL_SELECTOR_API_KEY_ENV", "CLOSEAI_API_KEY"),
            evaluation_namespace=os.getenv("AML_EVALUATION_NAMESPACE", "textual"),
            embedding_url=os.getenv("EMBEDDING_ENDPOINT", "http://127.0.0.1:18101"),
            reranker_url=os.getenv("RERANK_ENDPOINT", "http://127.0.0.1:18106"),
            inference_contract=os.getenv("AML_INFERENCE_CONTRACT", "vllm-http-v1"),
            splade_url=os.getenv("SPLADE_ENDPOINT", "http://127.0.0.1:18102"),
            model_cache=os.getenv("AML_MODEL_CACHE"),
            add_timeout_seconds=float(os.getenv("AML_ADD_TIMEOUT_SECONDS", "60")),
            search_timeout_seconds=float(os.getenv("AML_SEARCH_TIMEOUT_SECONDS", "30")),
            rate_limit_rpm=int(os.getenv("AML_RATE_LIMIT_RPM", "0")),
            max_request_bytes=int(
                os.getenv("AML_MAX_REQUEST_BYTES", str(10 * 1024 * 1024))
            ),
            data_ttl_days=int(os.getenv("AML_DATA_TTL_DAYS", "30")),
            log_level=os.getenv("AML_LOG_LEVEL", "INFO"),
        )
        return settings.bind_run() if bind_run else settings
