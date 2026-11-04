"""FastAPI application exposing the AML Add / Search / Health contract."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from .competition import CompetitionMemoryService
from .config import Settings
from .deadline import ApiDeadlineExecutor
from .ratelimit import RateLimiter
from .retention import RetentionJanitor
from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .security import require_auth
from .version import __version__
from mandol_competition.runtime.http_boundary import install_http_boundary, ErrorRule
from ..core.rerank_query_views import RerankQueryLimitError
from mandol_competition.runtime.contract import SearchPlan,acknowledge_add,EXECUTION_TOP_K_LIMIT
from mandol_competition.runtime.deadline import add_work_key
from mandol_competition.observability.diagnostics import diagnostic_call

logger = logging.getLogger("mandol_competition.textual.api.app")


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


def create_app(settings: Settings | None = None, *, memory_service=None) -> FastAPI:
    cfg = settings or Settings.from_env()
    cfg.validate()
    memory = memory_service if memory_service is not None else CompetitionMemoryService(cfg)
    deadlines = ApiDeadlineExecutor()
    limiter = RateLimiter(cfg.rate_limit_rpm)
    janitor = RetentionJanitor(memory, cfg)
    ready_event_holder: dict[str, Any] = {"ready": False, "started": False}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting Mandol-AML %s (Qdrant)", __version__)

        # External dependency validation and recovery may take time. Run it in a background thread so the HTTP server starts
        # immediately and /health reports 503 until the memory system is ready -
        # this matches the AML health semantics and the Docker HEALTHCHECK.
        def _initialize() -> None:
            try:
                memory.start()
                ready_event_holder["ready"] = True
                logger.info("Mandol runtime is ready")
            except Exception:  # noqa: BLE001 -- failed startup stays unavailable; no fallback.
                logger.error(
                    "Mandol runtime failed to start; health will report not-ready"
                )
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
            deadlines.close()  # Drain unknown-outcome writes before storage shutdown.
            init_thread.join()  # Never close storage while initialization owns it.
            memory.shutdown()

    app = FastAPI(
        title="Mandol-AML",
        version=__version__,
        description="Agent Memory Leaderboard Add/Search adapter for the Mandol memory system.",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.memory_service = memory
    app.state.deadline_executor = deadlines

    from mandol_competition.runtime.body_limit import BodyLimitMiddleware
    app.add_middleware(BodyLimitMiddleware, max_bytes=cfg.max_request_bytes)

    @app.get(cfg.health_path, include_in_schema=False)
    def health() -> dict[str, Any]:
        if not ready_event_holder["ready"] or not memory.ready:
            raise _not_ready()
        memory.health()
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, Any]:
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

        def perform_add():
            return diagnostic_call('add', memory.add_messages,
                user_id=payload.user_id,
                session_id=payload.session_id,
                request_id=payload.request_id,
                messages=[m.model_dump() for m in payload.messages],
            )

        count=deadlines.run(perform_add,seconds=cfg.add_timeout_seconds,
                            key=add_work_key('textual', payload))
        return diagnostic_call('add_receipt',acknowledge_add,payload,count,AddResponse)

    @app.post(
        cfg.search_path, response_model=SearchResponse, response_model_exclude_none=True
    )
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        require_auth(request, cfg.memory_system_key)
        if not limiter.allow():
            raise _rate_limited()
        if not ready_event_holder["ready"] or not memory.ready:
            raise _not_ready()
        plan=SearchPlan(payload.top_k,EXECUTION_TOP_K_LIMIT)

        def perform_search():
            return diagnostic_call('search', memory.search,
                user_id=payload.user_id,query=payload.query,top_k=plan.execution_top_k,choices=payload.options,
            )

        hits=deadlines.run(perform_search,seconds=cfg.search_timeout_seconds)
        return diagnostic_call('response',plan.response,hits,SearchResponse)

    install_http_boundary(app,'textual',ready=health,state=lambda:getattr(memory,'state',None),
        rules=(ErrorRule(RerankQueryLimitError,413,'query exceeds bounded lossless reranking capacity'),))
    return app
