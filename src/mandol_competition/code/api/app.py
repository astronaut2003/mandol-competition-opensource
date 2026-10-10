"""FastAPI application exposing the AML Add / Search / Health contract."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .coding_memory import CodingMemoryService
from .ratelimit import RateLimiter
from .retention import RetentionJanitor
from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .security import require_auth
from .version import __version__
from .body_limit import BodyLimitMiddleware
from .coding_store import CapacityExceeded, RequestConflict, StorageBusy

logger = logging.getLogger("mandol_competition.code.api.app")


def _not_ready() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"reason": "memory system is warming up or unavailable"},
    )


def _rate_limited() -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={"reason": "rate limit exceeded"},
        headers={"Retry-After": "60"},
    )


def create_app(settings: Optional[Settings] = None, *, memory_service=None) -> FastAPI:
    cfg = settings or Settings.from_env()
    if cfg.inference_contract not in {"vllm-http-v1","aml-official-eligible-v1"}:
        raise ValueError("Unknown Coding inference contract")
    expected="code_aml_v4_v1" if cfg.inference_contract=="aml-official-eligible-v1" else "code_v1"
    if cfg.qdrant_collection != expected:
        raise ValueError("Coding collection must be "+expected)
    memory = memory_service if memory_service is not None else CodingMemoryService(cfg)
    from mandol_competition.runtime.deadline import ApiDeadlineExecutor
    deadlines = ApiDeadlineExecutor()
    limiter = RateLimiter(cfg.rate_limit_rpm)
    janitor = RetentionJanitor(memory, cfg)
    ready_event_holder: Dict[str, Any] = {"ready": False, "started": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting Mandol-AML %s (backend=%s)", __version__, cfg.backend)

        # Mandol warm-up (embedding model download/load) can take minutes on the
        # first start. Run it in a background thread so the HTTP server starts
        # immediately and /health reports 503 until the memory system is ready -
        # this matches the AML health semantics and the Docker HEALTHCHECK.
        def _initialize() -> None:
            try:
                memory.start()
                ready_event_holder["ready"] = True
                logger.info("Mandol runtime is ready")
            except Exception:
                logger.exception("Mandol runtime failed to start; health will report not-ready")
                ready_event_holder["ready"] = False
            finally:
                ready_event_holder["started"] = True

        init_thread = threading.Thread(target=_initialize, name="aml-init", daemon=True)
        init_thread.start()
        janitor.start()
        try:
            yield
        finally:
            janitor.stop()
            # Timed-out durable work still owns a worker and its store. Drain
            # it and initialization before closing canonical/projection clients.
            deadlines.close()
            init_thread.join()
            if ready_event_holder["started"] and memory.ready:
                try:
                    memory.shutdown()
                except Exception:
                    logger.exception("error during shutdown")

    app = FastAPI(
        title="Mandol-AML",
        version=__version__,
        description="Agent Memory Leaderboard Add/Search adapter for the Mandol memory system.",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    app.add_middleware(BodyLimitMiddleware, max_bytes=cfg.max_request_bytes)
    app.state.deadline_executor = deadlines

    @app.exception_handler(RequestConflict)
    async def request_conflict_handler(request: Request, exc: RequestConflict):
        return JSONResponse(status_code=409, content={"detail": {"reason": str(exc)}})

    @app.exception_handler(StorageBusy)
    async def storage_busy_handler(request: Request, exc: StorageBusy):
        return JSONResponse(status_code=503, headers={"Retry-After": "1"},
                            content={"detail": {"reason": str(exc)}})

    @app.exception_handler(CapacityExceeded)
    async def capacity_handler(request: Request, exc: CapacityExceeded):
        return JSONResponse(status_code=507, content={"detail": {"reason": str(exc)}})

    @app.get(cfg.health_path, include_in_schema=False)
    def health() -> Dict[str, Any]:
        if not ready_event_holder["ready"] or not memory.ready:
            raise _not_ready()
        if hasattr(memory, "health"):
            memory.health()
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> Dict[str, Any]:
        return {
            "system": "Mandol-AML",
            "version": __version__,
            "upstream": "Mandol 0.1.0 (https://github.com/AgentCombo/Mandol)",
            "contract": "https://agentmemories.ai/docs",
            "endpoints": {
                "add": cfg.add_path,
                "search": cfg.search_path,
                "health": cfg.health_path,
            },
            "ready": ready_event_holder["ready"] and memory.ready,
        }

    @app.post(cfg.add_path, response_model=AddResponse)
    def add(payload: AddRequest, request: Request) -> AddResponse:
        require_auth(request, cfg.memory_system_key)
        if not limiter.allow():
            raise _rate_limited()
        if not ready_event_holder["ready"] or not memory.ready:
            raise _not_ready()
        deadlines.run(lambda: memory.add_messages(
            user_id=payload.user_id,
            session_id=payload.session_id,
            request_id=payload.request_id,
            messages=[m.model_dump() for m in payload.messages],
        ), seconds=cfg.add_timeout_seconds)
        # Echo the exact identifiers received (contract requirement).
        return AddResponse(
            success=True,
            request_id=payload.request_id,
            user_id=payload.user_id,
            session_id=payload.session_id,
        )

    @app.post(cfg.search_path, response_model=SearchResponse, response_model_exclude_none=True)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        require_auth(request, cfg.memory_system_key)
        if not limiter.allow():
            raise _rate_limited()
        if not ready_event_holder["ready"] or not memory.ready:
            raise _not_ready()
        hits = deadlines.run(lambda: memory.search(
            user_id=payload.user_id,
            query=payload.query,
            top_k=payload.top_k,
        ), seconds=cfg.search_timeout_seconds)
        data = [
            {
                "id": hit["id"],
                "content": hit["content"],
                "score": hit["score"],
                "created_at": hit["created_at"],
            }
            for hit in hits
        ]
        return SearchResponse(data=data)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled %s on %s", type(exc).__name__, request.method)
        return JSONResponse(
            status_code=500,
            content={"detail": {"reason": "internal server error"}},
        )

    app.state.memory_service = memory
    from mandol_competition.observability.runtime import install_observability
    install_observability(app, "code", ready=health, state=lambda: getattr(getattr(memory, "store", None), "state", None))
    return app
