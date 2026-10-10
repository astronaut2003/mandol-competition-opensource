"""Strict, proxy-free OpenAI-compatible embeddings for the coding service only.

No generation API, provider credentials, local model, or fabricated vectors.
One connection is reused under a lock. A stale reused socket gets one fresh
connection attempt; other failures close it and fail clearly. No redirects.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import http.client
import json
import logging
import math
import threading
import time
from urllib.parse import urlsplit

import numpy as np

from mandol_competition.code.software.local_models import _DenseIndex

logger = logging.getLogger(__name__)
_prepared_query = ContextVar('code_request_dense_query', default=None)


@contextmanager
def prepared_dense_query(encoder, text):
    """Move one pure query HTTP call outside the cache/diagnostic lock.

    The exact array or exception is consumed by the existing Dense channel in
    this request only. No corpus, result, representation or cross-request cache
    is introduced. The original retrieval path still decides failure policy.
    """
    try:
        result, error = encoder.encode_query(text), None
    except Exception as exc:
        result, error = None, exc
    token = _prepared_query.set((encoder, text, result, error))
    try:
        yield
    finally:
        _prepared_query.reset(token)


class EmbeddingError(RuntimeError):
    """Remote transport, model, or response contract failure (safe to log)."""


class RecipeDenseEncoder:
    """Explicit named-v4 document/query recipe, with pooled HTTP only.

    Semantic recipe mismatch fails closed. Network/schema failures remain an
    unavailable optional channel; they can never publish malformed vectors.
    The research Qwen encoder below keeps its original normalization and wire.
    """
    dimension=1024

    def __init__(self,endpoint,*,timeout=60,client=None):
        from mandol_competition.inference.clients import HTTPEmbeddingClient
        from mandol_competition.inference.dashscope import official_embedding_recipes
        self.recipe=official_embedding_recipes()[0]
        self.client=client or HTTPEmbeddingClient(endpoint,self.recipe,timeout=timeout,
                                                  batch_size=10,max_attempts=1)

    def validate_profile(self):
        from mandol_competition.inference.errors import InferenceIdentityError
        from mandol_substrate.state import ProfileMismatch
        try:return self.client.validate_profile()
        except InferenceIdentityError:
            raise ProfileMismatch("Code Dense serving recipe mismatch") from None

    def _encode(self,texts,*,query):
        self.validate_profile()
        vectors=np.asarray(self.client.embed(texts,query=query),dtype=np.float32)
        if vectors.shape!=(len(texts),self.dimension) or not np.isfinite(vectors).all():
            raise EmbeddingError("Invalid named-v4 Dense shape or numeric values")
        norms=np.linalg.norm(vectors,axis=1)
        if not np.isfinite(norms).all() or np.any(np.abs(norms-1)>.01):
            raise EmbeddingError("Named-v4 Dense normalization contract mismatch")
        return vectors

    def encode(self,texts):
        return self._encode(texts,query=False)

    def encode_query(self,text):
        prepared = _prepared_query.get()
        if prepared is not None and prepared[0] is self and prepared[1] == text:
            if prepared[3] is not None:
                raise prepared[3]
            return prepared[2]
        return self._encode([text],query=True)[0]

    def close(self):self.client.close()


class RemoteDenseEncoder:
    dimension = 1024

    def __init__(self, base_url, *, model="qwen3-embedding-0.6b", timeout=10.0, batch_size=32):
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.password is not None
                or url.query or url.fragment or url.path not in {"", "/", "/v1", "/v1/"}):
            raise ValueError("Embedding URL must be an HTTP(S) origin or /v1 URL without credentials")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Embedding model must be nonempty")
        if not math.isfinite(timeout) or timeout <= 0 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding timeout and batch size must be positive and finite")
        from mandol_competition.inference.http import HTTPTransport
        origin = f"{url.scheme}://{url.netloc}"
        self.http = HTTPTransport(origin, timeout=timeout, max_attempts=1)
        self._lock = threading.RLock()
        self.model, self.batch_size = model, batch_size
        self.requests = self.successes = self.failures = self.vectors = 0
        self.reconnects = 0
        self.available = None

    def encode(self, texts):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Embedding input must be a nonempty string or batch")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding texts must be nonempty strings")
        with self._lock:
            batches = [self._batch(texts[start:start + self.batch_size])
                       for start in range(0, len(texts), self.batch_size)]
            result = np.concatenate(batches)
        return result[0] if single else result

    def _batch(self, texts):
        self.requests += 1
        try:
            data = self.http.request("POST", "/v1/embeddings", {
                "model": self.model, "input": texts, "encoding_format": "float"})
            result = self._validate(data, len(texts))
        except Exception as exc:
            self.failures += 1
            self.available = False
            raise EmbeddingError("Embedding failed (" + type(exc).__name__ + ")") from None
        self.successes += 1
        self.vectors += len(texts)
        self.available = True
        return result

    def _validate(self, payload, count):
        if not isinstance(payload, dict) or payload.get("model") != self.model:
            raise EmbeddingError("Embedding response model mismatch")
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != count:
            raise EmbeddingError("Embedding response count mismatch")
        ordered = [None] * count
        for item in data:
            if not isinstance(item, dict):
                raise EmbeddingError("Malformed embedding item")
            index, vector = item.get("index"), item.get("embedding")
            if type(index) is not int or not 0 <= index < count or ordered[index] is not None:
                raise EmbeddingError("Invalid or duplicate embedding index")
            if (not isinstance(vector, list) or len(vector) != self.dimension
                    or any(type(value) not in (int, float) for value in vector)):
                raise EmbeddingError("Embedding dimension or numeric type mismatch")
            array = np.asarray(vector, dtype=np.float64)
            norm = np.linalg.norm(array)
            if not np.isfinite(array).all() or not np.isfinite(norm) or norm <= 0:
                raise EmbeddingError("Embedding vector must be finite and nonzero")
            ordered[index] = (array / norm).astype(np.float32)
        return np.stack(ordered)

    def index(self, documents, *, previous=None):
        return _RemoteDenseIndex(self, documents, previous=previous)

    def stats(self):
        with self._lock:
            return {"requests": self.requests, "successes": self.successes,
                    "failures": self.failures, "vectors": self.vectors,
                    "reconnects": self.reconnects,
                    "available": self.available, "dimension": self.dimension}

    def close(self):
        with self._lock:
            self.http.close()


class _RemoteDenseIndex(_DenseIndex):
    """Existing cosine search with immutable, unchanged document vectors reused.

    The service supplies only its previous user/generation projection. No query
    vectors or search results are cached, and no previous-index chain is kept.
    """

    def __init__(self, encoder, documents, *, previous=None):
        self.encoder = encoder
        self.documents = dict(documents)
        self.ids = sorted(documents)
        self.vectors = None
        if not self.ids:
            return
        positions = ({uid: i for i, uid in enumerate(previous.ids)}
                     if isinstance(previous, _RemoteDenseIndex) and previous.encoder is encoder else {})
        missing, reused = [], []
        for position, uid in enumerate(self.ids):
            if uid in positions and previous.documents[uid] == documents[uid]:
                reused.append((position, positions[uid]))
            else:
                missing.append(position)
        # Publish only after all new embeddings pass the strict remote contract.
        fresh = encoder.encode([documents[self.ids[i]] for i in missing]) if missing else None
        vectors = np.empty((len(self.ids), encoder.dimension), dtype=np.float32)
        for position, old_position in reused:
            vectors[position] = previous.vectors[old_position]
        if missing:
            vectors[missing] = fresh
        self.vectors = vectors
        logger.info("coding_dense_index documents=%d embedded=%d reused=%d", len(self.ids), len(missing), len(reused))
