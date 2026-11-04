"""Runtime configuration for the AML adapter.

The unified factory supplies per-track storage and HTTP inference settings.
Legacy AML overrides remain available for explicit development configurations.
The optional HTTP reranker reads credentials from the environment or a secret
mount; API processes never load a model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_list(name: str, default: Optional[List[str]] = None) -> List[str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return list(default or [])
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Settings:
    # --- HTTP server -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    add_path: str = "/add"
    search_path: str = "/search"
    health_path: str = "/health"

    # --- Authentication -----------------------------------------------------
    # Memory System Key issued by the participant for the AML platform.
    # Empty => unauthenticated endpoints. AML allows "none" only for the public
    # smoke compatibility check, so set a strong key before a formal run.
    memory_system_key: str = ""

    # --- Mandol runtime -----------------------------------------------------
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: Optional[int] = None
    use_flash_attention: bool = False
    # Build SPLADE sparse vectors during Add. Improves hybrid retrieval quality
    # but slows writes and requires downloading the SPLADE model.
    generate_sparse_embedding: bool = False
    # Backend layout:
    #   "shared"   -> one SemanticMap/SemanticGraph for the process, each AML
    #                 user_id isolated by its own MemorySpace (memory-light,
    #                 dense retrieval only).
    #   "isolated" -> one SemanticMap/SemanticGraph per AML user_id (model
    #                 weights are still shared through Mandol's model manager),
    #                 which also allows the space-agnostic MultiRetriever
    #                 hybrid path without cross-user leakage.
    backend: str = "shared"
    # Retrieval path used by the "isolated" backend:
    #   "graph"  -> SemanticGraph.search_similarity_in_graph (dense, robust)
    #   "hybrid" -> MultiRetriever.smart_search over cosine + bm25 (+ splade)
    retrieval_mode: str = "graph"
    retrieval_methods: List[str] = field(default_factory=lambda: ["cosine", "bm25"])
    rerank_method: Optional[str] = None
    # Pre-load the embedding model at startup so the first Add is not slowed by
    # model download / warm-up. Health returns 503 until warm-up finishes.
    warmup_on_start: bool = True

    # --- Storage / capacity -------------------------------------------------
    l2_dir: Optional[str] = None  # optional RocksDB tiered paging directory
    max_units_per_user: Optional[int] = None

    # --- Retention (AML requires deleting evaluation data within 30 days) ----
    data_ttl_days: int = 30
    retention_interval_seconds: int = 3600

    # --- Rate limiting / capacity -------------------------------------------
    rate_limit_rpm: int = 0  # 0 = unlimited
    max_request_bytes: int = 10 * 1024 * 1024
    add_timeout_seconds: float = 60.0
    search_timeout_seconds: float = 30.0

    # --- Optional high-level memory (experimental, default off) --------------
    high_level_memory: bool = False
    high_level_debounce_seconds: float = 5.0
    high_level_wait_on_search_seconds: float = 0.0

    # --- Opt-in software profile; existing textual path keeps its defaults ---
    coding_memory: bool = True
    coding_db_path: str = "data/code/state.sqlite3"
    namespace: str = "code"
    qdrant_endpoint: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "code_v1"
    qdrant_api_key: str | None = None
    coding_busy_timeout_ms: int = 2000
    coding_cached_users: int = 16
    coding_retrieval_mode: str = "raw"
    coding_code_aware: bool = False
    coding_router: bool = False
    coding_evidence_needs: bool = False
    coding_anchor_weight: float = 0.0
    coding_outcome_weight: float = 0.0
    coding_route_weight: float = 0.0
    coding_max_cards: int = 8
    coding_evidence_chars: int = 16000
    coding_per_card_chars: Optional[int] = None
    coding_dense_model_path: Optional[str] = None
    coding_dense_base_url: Optional[str] = None
    coding_dense_model: str = "qwen3-embedding-0.6b"
    coding_dense_timeout_seconds: float = 10.0
    inference_contract: str = "vllm-http-v1"
    coding_raw_policy: str = "c0-original-v1"
    coding_reranker_backend: str = "profile"
    coding_reranker_endpoint: str = "https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank"
    coding_reranker_credential_env: str = "DASHSCOPE_API_KEY"
    coding_reranker_timeout_seconds: float = 15.0
    coding_device: str = "cpu"

    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        embedding_dim = _get_int("AML_EMBEDDING_DIM", 0)
        s = cls(
            host=os.getenv("AML_HOST", "0.0.0.0"),
            port=_get_int("AML_PORT", 8000),
            add_path=os.getenv("AML_ADD_PATH", "/add") or "/add",
            search_path=os.getenv("AML_SEARCH_PATH", "/search") or "/search",
            health_path=os.getenv("AML_HEALTH_PATH", "/health") or "/health",
            memory_system_key=os.getenv("AML_MEMORY_SYSTEM_KEY", "") or "",
            embedding_model=(
                os.getenv("AML_EMBEDDING_MODEL") or "Qwen/Qwen3-Embedding-0.6B"
            ),
            embedding_dim=embedding_dim or None,
            use_flash_attention=_get_bool("AML_USE_FLASH_ATTENTION", False),
            generate_sparse_embedding=_get_bool("AML_GENERATE_SPARSE_EMBEDDING", False),
            backend=(os.getenv("AML_BACKEND", "shared") or "shared").strip().lower(),
            retrieval_mode=(os.getenv("AML_RETRIEVAL_MODE", "graph") or "graph").strip().lower(),
            retrieval_methods=_get_list("AML_RETRIEVAL_METHODS", ["cosine", "bm25"]),
            rerank_method=os.getenv("AML_RERANK_METHOD") or None,
            warmup_on_start=_get_bool("AML_WARMUP_ON_START", True),
            l2_dir=os.getenv("AML_L2_DIR") or None,
            max_units_per_user=_get_int("AML_MAX_UNITS_PER_USER", 0) or None,
            data_ttl_days=_get_int("AML_DATA_TTL_DAYS", 30),
            retention_interval_seconds=_get_int("AML_RETENTION_INTERVAL_SECONDS", 3600),
            rate_limit_rpm=_get_int("AML_RATE_LIMIT_RPM", 0),
            max_request_bytes=_get_int("AML_MAX_REQUEST_BYTES", 10 * 1024 * 1024),
            add_timeout_seconds=_get_float("AML_ADD_TIMEOUT_SECONDS", 60.0),
            search_timeout_seconds=_get_float("AML_SEARCH_TIMEOUT_SECONDS", 30.0),
            high_level_memory=_get_bool("AML_HIGH_LEVEL_MEMORY", False),
            high_level_debounce_seconds=_get_float("AML_HIGH_LEVEL_DEBOUNCE_SECONDS", 5.0),
            high_level_wait_on_search_seconds=_get_float(
                "AML_HIGH_LEVEL_WAIT_ON_SEARCH_SECONDS", 0.0
            ),
            coding_memory=True,
            coding_db_path=os.getenv("MANDOL_CODE_SQLITE_PATH", "data/code/state.sqlite3"),
            namespace=os.getenv("MANDOL_CODE_NAMESPACE", "code"),
            qdrant_endpoint=os.getenv("MANDOL_CODE_QDRANT_ENDPOINT", os.getenv("QDRANT_ENDPOINT", "http://127.0.0.1:6333")),
            qdrant_collection=os.getenv("MANDOL_CODE_QDRANT_COLLECTION", "code_v1"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            coding_busy_timeout_ms=_get_int("AML_CODING_BUSY_TIMEOUT_MS", 2000),
            coding_cached_users=_get_int("AML_CODING_CACHED_USERS", 16),
            coding_retrieval_mode=os.getenv("AML_CODING_RETRIEVAL_MODE", "raw"),
            coding_code_aware=_get_bool("AML_CODING_CODE_AWARE", False),
            coding_router=_get_bool("AML_CODING_ROUTER", False),
            coding_evidence_needs=_get_bool("AML_CODING_EVIDENCE_NEEDS", False),
            coding_anchor_weight=_get_float("AML_CODING_ANCHOR_WEIGHT", 0.0),
            coding_outcome_weight=_get_float("AML_CODING_OUTCOME_WEIGHT", 0.0),
            coding_route_weight=_get_float("AML_CODING_ROUTE_WEIGHT", 0.0),
            coding_max_cards=_get_int("AML_CODING_MAX_CARDS", 8),
            coding_evidence_chars=_get_int("AML_CODING_EVIDENCE_CHARS", 16000),
            coding_per_card_chars=_get_int("AML_CODING_PER_CARD_CHARS", 0) or None,
            coding_dense_model_path=os.getenv("AML_CODING_DENSE_MODEL_PATH") or None,
            coding_dense_base_url=os.getenv("AML_CODING_DENSE_BASE_URL") or None,
            coding_dense_model=os.getenv("AML_CODING_DENSE_MODEL", "qwen3-embedding-0.6b"),
            coding_dense_timeout_seconds=float(os.getenv("AML_CODING_DENSE_TIMEOUT_SECONDS", "10")),
            inference_contract=os.getenv("MANDOL_CODE_INFERENCE_CONTRACT", "vllm-http-v1"),
            coding_raw_policy=os.getenv("MANDOL_CODE_RETRIEVAL_POLICY", "c0-original-v1"),
            coding_reranker_backend=os.getenv("MANDOL_CODE_RERANKER_BACKEND", "profile"),
            coding_reranker_endpoint=os.getenv("MANDOL_CODE_RERANKER_ENDPOINT", cls.coding_reranker_endpoint),
            coding_reranker_credential_env=os.getenv("MANDOL_CODE_RERANKER_CREDENTIAL_ENV", "DASHSCOPE_API_KEY"),
            coding_reranker_timeout_seconds=_get_float("AML_CODING_RERANKER_TIMEOUT_SECONDS", 15.0),
            coding_device=os.getenv("AML_CODING_DEVICE", "cpu"),
            log_level=(os.getenv("AML_LOG_LEVEL", "INFO") or "INFO").upper(),
        )
        if s.backend not in {"shared", "isolated"}:
            s.backend = "shared"
        if s.retrieval_mode not in {"graph", "hybrid"}:
            s.retrieval_mode = "graph"
        if s.backend == "shared":
            # MultiRetriever is not MemorySpace-scoped, so hybrid retrieval is
            # only safe with per-user isolation.
            s.retrieval_mode = "graph"
        return s
