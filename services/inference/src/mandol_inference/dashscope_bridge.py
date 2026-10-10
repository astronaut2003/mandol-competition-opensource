"""CPU-only provider bridge; credentials stay in the service environment.

This bridge exposes the provider's native query/document dense+sparse response.
It does not load models, assert a provider weight revision, or apply Qwen prompts.
"""
from contextlib import asynccontextmanager
from dataclasses import asdict
import math
import os
import threading
import time

from fastapi import FastAPI, HTTPException

from mandol_competition.inference.dashscope import (
    DashScopeEmbeddingClient, MODEL, official_embedding_recipes,
)
from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.inference.http import deadline_budget


class ProviderRecovery:
    """One failure-only recovery worker; healthy polling never spends tokens."""
    def __init__(self, client, *, delay=5.):
        if not math.isfinite(delay) or not 0 < delay <= 300:
            raise ValueError('bounded positive recovery delay required')
        self.client, self.delay = client, delay
        self._condition = threading.Condition()
        self._stopping = False
        self._available = True
        self._generation = self._failures = self._probes = self._recoveries = 0
        self._next_probe = None
        self._worker = threading.Thread(target=self._run, name='v4-recovery', daemon=True)
        self._worker.start()

    def snapshot(self):
        with self._condition:
            return {'available': self._available, 'failures': self._failures,
                    'recovery_probes': self._probes, 'recoveries': self._recoveries,
                    'next_probe_seconds': max(0., self._next_probe-time.monotonic())
                        if self._next_probe is not None else None}

    def failed(self, error):
        with self._condition:
            self._generation += 1
            self._available = False
            self._failures += 1
            wait = min(300., self.delay*2**min(self._failures-1, 10))
            header = getattr(error, 'retry_after', None)
            if isinstance(header, (int, float)) and math.isfinite(header) and header >= 0:
                wait = max(wait, header)
            self._next_probe = time.monotonic()+wait
            self._condition.notify_all()

    def succeeded(self):
        with self._condition:
            self._generation += 1
            self._available = True
            self._failures = 0
            self._next_probe = None
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                while not self._stopping:
                    if self._available or self._next_probe is None:
                        self._condition.wait()
                    else:
                        wait = self._next_probe-time.monotonic()
                        if wait <= 0:
                            break
                        # Even a very long valid Retry-After remains interruptible.
                        self._condition.wait(timeout=min(60., wait))
                if self._stopping:
                    return
                generation = self._generation
                self._probes += 1
            error = None
            try:
                # These are independent static canaries, never a redispatch of
                # an unknown user's paid operation. Both native modes validate.
                with deadline_budget(60.):
                    self.client.encode(['Memory evidence recovery.'], query=False)
                    self.client.encode(['Find memory evidence.'], query=True)
            except Exception as exc:
                error = exc
            with self._condition:
                if self._stopping:
                    return
                if generation != self._generation:
                    continue  # A newer real request already changed availability.
                if error is None:
                    self._recoveries += 1
                    self.succeeded()
                else:
                    self.failed(error)

    def close(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._worker.join(timeout=65.)
        if self._worker.is_alive():
            raise RuntimeError('provider recovery exceeded its shutdown deadline')


def create_app(client_factory=None, *, recovery_delay=5.):
    dense, sparse = official_embedding_recipes()

    @asynccontextmanager
    async def lifespan(app):
        client = (client_factory or (lambda: DashScopeEmbeddingClient(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            endpoint=os.environ.get("AML_V4_PROVIDER_ORIGIN", "https://dashscope.aliyuncs.com"),
        )))()
        app.state.client = client
        app.state.admitted = False
        recovery = None
        try:
            # Warmup verifies the live endpoint, including sparse output. It writes
            # no canonical state. Provider revisions remain explicitly unattested.
            client.encode(["Memory evidence admission."], query=False)
            client.encode(["Find memory evidence."], query=True)
            recovery = app.state.recovery = ProviderRecovery(client, delay=recovery_delay)
            app.state.admitted = True
            yield
        finally:
            app.state.admitted = False
            if recovery is not None:
                recovery.close()
            client.close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        if not app.state.admitted or not app.state.recovery.snapshot()['available']:
            raise HTTPException(503, "provider_not_admitted")
        return {"status": "ok", "model": MODEL,
                "attestation": "provider-api-name; weight-revision-unattested"}

    @app.get("/v1/recipe")
    def recipe():
        return {"recipes": [{"hash": r.hash, "recipe": asdict(r)} for r in (dense, sparse)]}

    def encode(body, channel):
        if not isinstance(body, dict) or body.get("model") != MODEL:
            raise HTTPException(409, "model_identity_mismatch")
        if channel == "pair":
            valid = (body.get("dense_recipe_hash") == dense.hash
                     and body.get("sparse_recipe_hash") == sparse.hash)
        else:
            expected = dense if channel == "embedding" else sparse
            # Compatibility callers still supply a model identity; recipe clients
            # must provide their hash. This path never guesses query mode.
            valid = body.get("recipe_hash", expected.hash) == expected.hash
        if not valid:
            raise HTTPException(409, "recipe_identity_mismatch")
        texts = body.get("input")
        mode = body.get("input_type", "document")
        if (mode not in {"document", "query"} or not isinstance(texts, list)
                or not 1 <= len(texts) <= 1024):
            raise HTTPException(422, "invalid_encoding_request")
        try:
            pairs = app.state.client.encode(texts, query=mode == "query")
        except ValueError:
            raise HTTPException(422, "invalid_encoding_request") from None
        except InferenceUnavailable as exc:
            app.state.recovery.failed(exc)
            raise HTTPException(503, "provider_inference_unavailable") from None
        app.state.recovery.succeeded()
        rows = []
        for index, pair in enumerate(pairs):
            value = {"embedding": pair.dense,
                     "sparse": {str(k): v for k, v in pair.sparse.items()}}
            rows.append({"index": index, channel: value if channel == "pair" else value[channel]})
        return {"model": MODEL, "data": rows}

    @app.post("/v1/encode")
    def paired(body: dict):
        return encode(body, "pair")

    @app.post("/v1/embeddings")
    def embeddings(body: dict):
        return encode(body, "embedding")

    @app.post("/v1/sparse")
    def sparse_encode(body: dict):
        return encode(body, "sparse")

    from .observability import install_bridge_observability
    install_bridge_observability(app, ready=health,
        transports=lambda: {'provider': getattr(app.state.client, 'http', None)},
        extra=lambda: {'provider_recovery': app.state.recovery.snapshot()})
    return app
