"""HTTP boundary for the AML Add, Search and Health contract."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
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
from .response import validate_response_media
from mandol_competition.observability.diagnostics import diagnostic_call
from mandol_competition.runtime.contract import (
    SearchPlan,EXECUTION_TOP_K_LIMIT,validate_add_response,validate_search_response,
)
from mandol_competition.runtime.http_boundary import install_http_boundary,ErrorRule
from mandol_competition.runtime.deadline import add_work_key


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
            raise HTTPException(status_code=409,detail={'reason':'purge operation cannot complete'}) from None
        return vars(report)

    @app.post("/add", response_model=AddResponse, response_class=ProfiledJSONResponse, dependencies=[Depends(authenticate)])
    def add_memory(body: AddRequest, response: Response) -> dict[str, Any]:
        profile_id = profiler.response_id(response)
        value=deadlines.run(lambda:diagnostic_call('add',profiler.call,'add',add.handle,body,profile_id),
            seconds=config.add_timeout_seconds, key=add_work_key('multimodal', body))
        return diagnostic_call('add_receipt',validate_add_response,body,value,AddResponse)

    @app.post(
        "/search", response_model=SearchResponse, response_class=ProfiledJSONResponse,
        response_model_exclude_none=True, dependencies=[Depends(authenticate)],
    )
    def search_memory(body: SearchRequest, response: Response) -> dict[str, Any]:
        profile_id = profiler.response_id(response)
        plan=SearchPlan(body.top_k,EXECUTION_TOP_K_LIMIT)
        bounded=body.model_copy(update={'top_k':plan.execution_top_k})
        value=deadlines.run(lambda:diagnostic_call('search',profiler.call,'search',search.handle,bounded,profile_id),
            seconds=config.search_timeout_seconds)
        value=diagnostic_call('response',validate_search_response,value,plan,SearchResponse)
        return diagnostic_call('response',validate_response_media,value)

    from mandol_competition.runtime.body_limit import BodyLimitMiddleware
    app.add_middleware(BodyLimitMiddleware, max_bytes=config.max_request_bytes)
    # Media validation messages can reflect caller input or decoder messages.
    # Only documented typed client errors get 413/422, with static reasons.
    install_http_boundary(app,'multimodal',ready=health,state=lambda:backend.state if backend else None,
        rules=(ErrorRule(PayloadError,413,'media or publication budget exceeded',when=lambda exc:exc.status_code==413),
               ErrorRule(PayloadError,422,'invalid multimodal payload',when=lambda exc:exc.status_code==422),
               ErrorRule(ApiException,503,'Qdrant operation unavailable; retry or reconcile'),
               ErrorRule(TimeoutError,503,'dependency operation timed out; retry or reconcile')))
    return app
