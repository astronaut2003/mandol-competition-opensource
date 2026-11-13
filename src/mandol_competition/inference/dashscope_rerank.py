"""Direct DashScope reranking; scores are restored to the input document order.

Local budget tokenization and disclosed rank fusion guard the native requests; no model weights are loaded.
The provider currently does not attest weights, precision or a fixed revision.
"""
from dataclasses import asdict
import math
import threading
import time
from urllib.parse import urlsplit

from .errors import InferenceAdmissionDeferred, InferenceDeadlineExceeded, InferenceIdentityError, InferenceSchemaError, InferenceUnavailable
from .http import HTTPTransport, endpoint_origin
from .models import InferenceRecipe

MODEL = "qwen3.7-text-rerank"
PATH = "/api/v1/services/rerank/text-rerank/text-rerank"
ENDPOINT = "https://cn-beijing.maas.qianwenaiapi.com" + PATH
BACKEND = "dashscope-qwen37-v1"


def recipe():
    return InferenceRecipe(
        "dashscope-qwen37-text-rerank-v1", "reranker", MODEL, MODEL,
        "provider-managed-unverified", "provider-managed-unverified",
        dtype="provider-undisclosed", pooling="provider-managed",
        max_length=30000, truncation="provider-reject-no-client-truncation",
        score_semantics="request-relative-relevance-0-to-1-higher-is-better",
        contract="dashscope-text-rerank-v1",
    )


def scores_in_input_order(data, count):
    if data.get("code"):
        raise InferenceSchemaError("provider_rerank_error")
    # The native endpoint normally omits model identity. If present, it must
    # agree; omission is disclosed rather than synthesized as an attestation.
    if data.get("model") not in (None, MODEL):
        raise InferenceIdentityError("provider reranker model mismatch")
    output = data.get("output")
    rows = output.get("results") if isinstance(output, dict) else None
    if not isinstance(rows, list) or len(rows) != count:
        raise InferenceSchemaError("partial_rerank_results")
    values = [None] * count
    for row in rows:
        if not isinstance(row, dict) or type(row.get("index")) is not int:
            raise InferenceSchemaError("invalid_rerank_index")
        index, score = row["index"], row.get("relevance_score")
        if not 0 <= index < count or values[index] is not None:
            raise InferenceSchemaError("duplicate_or_missing_rerank_index")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise InferenceSchemaError("invalid_rerank_score")
        values[index] = float(score)
    return values


class DashScopeRerankerClient:
    # Pools must not spend provider tokens while no business work exists.
    paid_health_checks = True

    def __init__(self, endpoint=ENDPOINT, *, api_key=None, transport=None,
                 timeout=120., health_ttl=60., clock=time.monotonic):
        endpoint_origin(endpoint)
        parsed = urlsplit(endpoint)
        if parsed.path.rstrip("/") != PATH or parsed.scheme != "https":
            raise ValueError("explicit HTTPS DashScope text rerank endpoint required")
        if transport is None and (not isinstance(api_key, str) or not api_key.strip()):
            raise ValueError("reranker credential required via environment or secret mount")
        if not math.isfinite(health_ttl) or not 0 < health_ttl <= 300:
            raise ValueError("bounded provider health cache required")
        self.recipe = recipe()
        self.endpoint = endpoint.rstrip("/")
        self.http = transport or HTTPTransport(
            parsed.scheme + "://" + parsed.netloc, api_key=api_key,
            timeout=timeout, max_attempts=2, max_connections=4,
        )
        self._clock, self._ttl = clock, health_ttl
        self._health_lock = threading.Lock()
        self._last_success = None
        self._admitted = False
        self.last_response_metadata = None

    def score(self, query, documents):
        from .rerank_query_views import rerank_query_views
        return rerank_query_views(self, query, documents)

    def _score_native(self, query, documents):
        if (not isinstance(query, str) or not query.strip()
                or not isinstance(documents, (list, tuple))
                or any(not isinstance(d, str) or not d.strip() for d in documents)):
            raise ValueError("nonempty original reranking texts required")
        if not documents:
            return []
        if len(documents) > 500:
            # Provider scores are request-relative: independent batches must
            # not be merged as if their numeric scales were comparable.
            raise ValueError("DashScope reranking supports at most 500 documents per request")
        started = self._clock()
        try:
            data = self.http.request("POST", PATH, {
                "model": MODEL,
                "input": {"query": query, "documents": list(documents)},
                "parameters": {"top_n": len(documents), "return_documents": False},
            }, retry_unknown=False)
            values = scores_in_input_order(data, len(documents))
        except (InferenceAdmissionDeferred, InferenceDeadlineExceeded):
            raise  # No dispatch occurred; keep the last real success evidence.
        except BaseException:
            self._last_success = None
            raise
        self.last_response_metadata = {
            "request_model": MODEL, "response_model": data.get("model"),
            "request_id": data.get("request_id") if isinstance(data.get("request_id"), str) else None,
            "usage": {k: v for k, v in (data.get("usage") or {}).items()
                      if k in {'prompt_tokens', 'total_tokens'} and type(v) is int and v >= 0}
                     if isinstance(data.get('usage'), dict) else {},
            "documents": len(documents), "seconds": self._clock() - started,
        }
        self._last_success = self._clock()
        return values

    rerank = score

    def validate_profile(self, tokenizer=None):
        with self._health_lock:
            if self._admitted and self._last_success is not None and self._clock() - self._last_success < self._ttl:
                return True
            values = self.score("Where was the authentication bug fixed?", [
                "The authentication bug was fixed in auth/middleware.py.",
                "The weather forecast predicts rain tomorrow.",
            ])
            if values[0] <= values[1]:
                self._last_success = None
                raise InferenceSchemaError("rerank_admission_direction")
            self._admitted = True
            return True

    def health(self):
        # This is last-success state, not a claim of live provider reachability.
        # Startup admission remains explicit; health endpoints never infer.
        if self._last_success is None:
            raise InferenceUnavailable('reranker_has_no_successful_observation')
        return True

    def attestation(self):
        from .rerank_query_views import descriptor
        return {"application_query_policy": descriptor(),
            "level": "provider-request-admission" if self._admitted else "not-checked",
            "endpoint": self.endpoint, "recipe": asdict(self.recipe),
            "recipe_hash": self.recipe.hash,
            "response_model": (self.last_response_metadata or {}).get("response_model"),
            "weights_revision_verified": False, "dtype_verified": False,
            "health_probe_cache_seconds": self._ttl,
            "periodic_health_probes": False, "health_is_passive": True,
        }

    def close(self):
        self._last_success = None
        self._admitted = False
        self.http.close()


def from_environment(endpoint=ENDPOINT, *, credential_env="DASHSCOPE_API_KEY", timeout=120.):
    from mandol_competition.config.unified import secret_from_environment
    return DashScopeRerankerClient(endpoint, api_key=secret_from_environment(credential_env), timeout=timeout)
