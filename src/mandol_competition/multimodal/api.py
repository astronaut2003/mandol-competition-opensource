"""HTTP boundary for the AML Add, Search and Health contract."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from mandol_substrate.state.models import Conflict, StateUnavailable
from qdrant_client.http.exceptions import ApiException

from .config import ConfigurationError, MultimodalConfig, get_config
from .protocol import AddRequest, AddResponse, SearchRequest, SearchResponse
from .profiling import HTTPProfileMiddleware, ProfiledJSONResponse
from .service import AddService, ConflictError, SearchService
from .validation import PayloadError
from .vlm.base import VisionExtractor
from .vlm.fake import FakeVisionExtractor
from .vlm.openai_compat import OpenAICompatExtractor, build_vision_client


def build_extractor(config: MultimodalConfig) -> VisionExtractor:
    if config.vlm_backend == "fake":
        if config.env == "production":
            raise ConfigurationError("production requires a real VLM backend")
        return FakeVisionExtractor({}, model=config.vlm_model,
                                   prompt_version=config.prompt_version,
                                   schema_version=config.schema_version)
    if config.vlm_backend == "openai_compat":
        from .vlm.base import ExtractionError

        try:
            client = build_vision_client(config)
        except ExtractionError as exc:
            raise ConfigurationError(str(exc)) from exc
        return OpenAICompatExtractor(model_name=config.vlm_model, config=config, client=client)
    raise ConfigurationError(f"Unsupported VLM backend: {config.vlm_backend}")


def create_app(
    config: MultimodalConfig | None = None,
    *,
    extractor: VisionExtractor | None = None,
    graph_factory: Callable[[str], Any] | None = None,
    backend=None,
) -> FastAPI:
    config = config or get_config()
    if config.env == "production" and not config.auth_token:
        raise ConfigurationError("production requires MANDOL_MM_AUTH_TOKEN")
    if extractor is None:
        extractor = build_extractor(config)

    if graph_factory is not None:
        if config.env == "production" or backend is not None:
            raise ConfigurationError("graph_factory is an experimental development adapter only")
        from .persistence import CanonicalStore
        from .store import UserGraphRegistry

        store = CanonicalStore(config.data_dir / "canonical.sqlite3")
        registry = UserGraphRegistry(config, store, graph_factory=graph_factory)
        add = AddService(config, extractor=extractor, registry=registry, store=store)
        search = SearchService(config, extractor=extractor, registry=registry, store=store)
    else:
        from .backend import MultimodalBackend

        backend = backend or MultimodalBackend(config)
        store, registry = backend.state, backend
        add = AddService(config, extractor=extractor, backend=backend)
        search = SearchService(config, extractor=extractor, backend=backend)

    from mandol_competition.runtime.deadline import ApiDeadlineExecutor
    deadlines = ApiDeadlineExecutor()
    from .profiling import RequestProfiler
    profiler = RequestProfiler(config.profiling_path, backend=backend, add=add, search=search, extractor=extractor)
    from mandol_competition.runtime.retention import RetentionJanitor
    janitor = RetentionJanitor(backend, config) if backend is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            if backend is not None:
                backend.start()
                janitor.start()
            yield
        finally:
            if janitor is not None:
                janitor.stop()
            deadlines.close()
            search.close()
            registry.close()
            if hasattr(extractor, "close"):
                extractor.close()
            if backend is None:
                store.close()

    app = FastAPI(lifespan=lifespan)
    if config.profiling_path is not None:
        app.add_middleware(HTTPProfileMiddleware, profiler=profiler)
    app.state.add_service = add
    app.state.search_service = search
    app.state.backend = backend
    app.state.deadline_executor = deadlines

    @app.exception_handler(StateUnavailable)
    async def unavailable(request: Request, exc: StateUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Conflict)
    async def conflict(request: Request, exc: Conflict):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ApiException)
    @app.exception_handler(TimeoutError)
    async def projection_unavailable(request: Request, exc: Exception):
        return JSONResponse(status_code=503, content={"detail": "Qdrant operation unavailable; retry or reconcile"})

    def authenticate(request: Request) -> None:
        from mandol_competition.runtime.auth import require_auth
        require_auth(request, config.auth_token or "")

    @app.get("/health")
    def health() -> dict[str, str]:
        return backend.health() if backend is not None else {"status": "ok"}

    @app.post("/admin/purge", dependencies=[Depends(authenticate)])
    def purge_memory(user_id: str) -> dict[str, Any]:
        if backend is not None:
            return backend.purge(user_id)
        from .purge import purge_user

        try:
            report = purge_user(user_id=user_id, config=config, store=store, registry=registry)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return vars(report)

    @app.post("/add", response_model=AddResponse, response_class=ProfiledJSONResponse, dependencies=[Depends(authenticate)])
    def add_memory(body: AddRequest, response: Response) -> dict[str, Any]:
        profile_id = profiler.response_id(response)
        try:
            return deadlines.run(lambda: profiler.call("add", add.handle, body, profile_id), seconds=config.add_timeout_seconds)
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except PayloadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc

    @app.post(
        "/search", response_model=SearchResponse, response_class=ProfiledJSONResponse,
        response_model_exclude_none=True, dependencies=[Depends(authenticate)],
    )
    def search_memory(body: SearchRequest, response: Response) -> dict[str, Any]:
        profile_id = profiler.response_id(response)
        try:
            return deadlines.run(lambda: profiler.call("search", search.handle, body, profile_id), seconds=config.search_timeout_seconds)
        except PayloadError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.reason) from exc

    from mandol_competition.runtime.body_limit import BodyLimitMiddleware
    app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_request_bytes)
    from mandol_competition.observability.runtime import install_observability
    install_observability(app, "multimodal", ready=health, state=lambda: backend.state if backend else None)
    return app
