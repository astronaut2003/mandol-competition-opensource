"""FastAPI application exposing the AML Add / Search / Health contract."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request

from .config import Settings
from .coding_memory import CodingMemoryService
from .ratelimit import RateLimiter
from .retention import RetentionJanitor
from .schemas import AddRequest, AddResponse, SearchRequest, SearchResponse
from .security import require_auth
from .version import __version__
from .body_limit import BodyLimitMiddleware
from .coding_store import CapacityExceeded, RequestConflict, StorageBusy
from .coding_store import PreparationBudgetExceeded
from mandol_competition.runtime.contract import SearchPlan,acknowledge_add
from mandol_competition.runtime.http_boundary import install_http_boundary,ErrorRule
from mandol_competition.runtime.deadline import add_work_key
from mandol_competition.observability.diagnostics import diagnostic_call

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
            # Startup can open SQLite/HTTP before model admission fails. Those
            # resources still belong to this lifespan even when never ready.
            if ready_event_holder["started"]:
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
        count=deadlines.run(lambda: diagnostic_call('add',memory.add_messages,
            user_id=payload.user_id,
            session_id=payload.session_id,
            request_id=payload.request_id,
            messages=[m.memory_message() for m in payload.messages],
        ), seconds=cfg.add_timeout_seconds, key=add_work_key('code', payload))
        return diagnostic_call('add_receipt',acknowledge_add,payload,count,AddResponse)

    @app.post(cfg.search_path, response_model=SearchResponse, response_model_exclude_none=True)
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        require_auth(request, cfg.memory_system_key)
        if not limiter.allow():
            raise _rate_limited()
        if not ready_event_holder["ready"] or not memory.ready:
            raise _not_ready()
        plan=SearchPlan(payload.top_k)
        hits = deadlines.run(lambda: diagnostic_call('search',memory.search,
            user_id=payload.user_id,
            query=payload.query_text(),
            top_k=payload.top_k,
        ), seconds=cfg.search_timeout_seconds)
        return diagnostic_call('response',plan.response,hits,SearchResponse)

    app.state.memory_service = memory
    install_http_boundary(app,'code',ready=health,state=lambda:getattr(getattr(memory,'store',None),'state',None),
        rules=(ErrorRule(StorageBusy,503,'storage unavailable; retry the same request',{'Retry-After':'1'}),
               ErrorRule(PreparationBudgetExceeded,413,'service request memory budget exceeded'),
               ErrorRule(CapacityExceeded,507,'service storage capacity exceeded')))
    return app
