"""Explicit VPC vLLM 0.29 transport contract for the frozen source B1 policy.

Contract admission is not numerical equivalence or immutable weight attestation.
The reference clients and frozen capsule stay unchanged. No local models load.
"""

import hashlib
import ipaddress
import math
import threading
from urllib.parse import urlsplit

from mandol_competition.inference.http import HTTPTransport, endpoint_origin
import httpx

from mandol_substrate.state.models import ProfileMismatch, StateUnavailable
from .models import RetrievalEncoders, ServingProfile, VllmModelClient
from .querit_reference import PROMPT, QueritPreparation, expected_profile

CONTRACT = "vpc-vllm-029-v1"
NETWORKS = tuple(
    ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
LIMITATIONS = {
    "immutable_weight_revision": "NOT_ATTESTED",
    "tokenizer_revision": "NOT_ATTESTED",
    "precision": "NOT_ATTESTED",
    "numerical_reference_oracle": "NOT_COMPLETED",
}


def private_endpoint(url):
    try:
        parsed = urlsplit(url)
        address = ipaddress.ip_address(parsed.hostname or "")
        valid = (
            parsed.scheme == "http"
            and parsed.path in ("", "/")
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and any(address in network for network in NETWORKS)
            and 1 <= (parsed.port or 80) <= 65535
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ProfileMismatch(
            "VPC inference requires a credential-free RFC1918 HTTP IP endpoint"
        )
    return url.rstrip("/")


def finite_vector(values, size):
    if (
        not isinstance(values, list)
        or len(values) != size
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
    ):
        raise StateUnavailable("Invalid model vector dimension/value")
    return values


def sparse_vector(values):
    finite_vector(values, 30522)
    if any(v < 0 for v in values):
        raise StateUnavailable("SPLADE returned negative weights")
    sparse = {index: float(value) for index, value in enumerate(values) if value > 0}
    if not sparse:
        raise StateUnavailable("SPLADE returned an unexpected empty sparse vector")
    return sparse


def relevance(probs):
    finite_vector(probs, 2)
    if any(not 0 <= v <= 1 for v in probs):
        raise StateUnavailable("Invalid Querit probabilities")
    # Native BF16 softmax returns rounded BF16 values, promoted to JSON floats.
    # Two complementary probabilities can consequently sum to 1.001953125.
    # Admit only exactly BF16-representable values under their rounding bound;
    # other transports retain the strict FP32 tolerance. Never renormalize or
    # otherwise change the frozen p1-p0 scoring rule.
    bf16 = all(v == 0 or (math.frexp(v)[0] * 256).is_integer() for v in probs)
    fp16 = all(v == 0 or (math.frexp(v)[0] * 2048).is_integer() for v in probs)
    tolerance = 3 / 1024 if bf16 else 3 / 8192 if fp16 else 1e-5
    if abs(sum(probs) - 1) > tolerance:
        raise StateUnavailable("Invalid Querit probabilities")
    return float(probs[1] - probs[0])


def texts_batch(texts, limit):
    if (
        not isinstance(texts, (list, tuple))
        or len(texts) > limit
        or any(not isinstance(t, str) or not t.strip() for t in texts)
    ):
        raise ValueError("Expected bounded nonblank model input texts")


class RemoteClient:
    alias = root = path = task = ""
    max_tokens = 0

    def __init__(self, url, *, timeout=30, transport=None, endpoint_policy="vpc"):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("Finite bounded inference timeout required")
        self.url = private_endpoint(url) if endpoint_policy == "vpc" else endpoint_origin(url)
        self.timeout = timeout
        self.http = HTTPTransport(self.url, timeout=timeout, transport=transport, max_attempts=3)
        self.client = self.http.client
        self._admission = threading.BoundedSemaphore(4)
        self._metrics_lock = threading.Lock()
        self._calls = {}

    def _request(self, method, path, payload=None, *, empty=False):
        data = self.http.request(method, path, payload, empty=empty)
        with self._metrics_lock:
            self._calls[path] = self._calls.get(path, 0) + 1
        return data

    def _ordered(self, data, count, field):
        if data.get("model") != self.alias:
            raise ProfileMismatch("Remote response model identity mismatch")
        return VllmModelClient._ordered(data, count, field)

    def validate_identity(self):
        from mandol_competition.inference.deployment import bindings
        binding = bindings().get(self.alias, {})
        expected_version = binding.get('vllm_version', '0.29.0')
        expected_root = binding.get('root', self.root)
        self._request("GET", "/health", empty=True)
        if self._request("GET", "/version").get("version") != expected_version:
            raise ProfileMismatch("Remote serving version does not match deployment contract")
        rows = self._request("GET", "/v1/models").get("data")
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or rows[0].get("id") != self.alias
            or rows[0].get("root") != expected_root
            or type(rows[0].get("max_model_len")) is not int
            or rows[0]["max_model_len"] < self.max_tokens
        ):
            raise ProfileMismatch("Remote alias/root/context contract mismatch")
        paths = self._request("GET", "/openapi.json").get("paths", {})
        if not isinstance(paths, dict) or self.path not in paths:
            raise ProfileMismatch("Remote inference task endpoint missing")
        return True

    def attestation(self):
        with self._metrics_lock:
            calls = dict(self._calls)
        return {
            "contract": CONTRACT,
            "endpoint": self.url,
            "task": self.task,
            "alias": self.alias,
            "expected_root": self.root,
            "admission": "CONTRACT_ONLY",
            "limitations": dict(LIMITATIONS),
            "successful_http_calls": calls,
        }

    def close(self):
        self.client.close()


class RemoteDenseClient(RemoteClient):
    alias = "qwen3-embedding-0.6b"
    root = "Qwen/Qwen3-Embedding-0.6B"
    path, task, max_tokens = "/v1/embeddings", "embedding", 4096

    def __init__(self, url, profile, **kwargs):
        if (
            profile.model != self.root
            or profile.dimension != 1024
            or profile.task != self.task
            or profile.max_tokens != self.max_tokens
            or not profile.normalize
        ):
            raise ProfileMismatch("VPC Dense requires the frozen Qwen 1024 profile")
        self.profile = (
            profile  # desired quality identity, not observed server attestation
        )
        super().__init__(url, **kwargs)

    def embed(self, texts):
        texts_batch(texts, 1024)
        result = []
        for offset in range(0, len(texts), self.profile.batch_size):
            batch = texts[offset : offset + self.profile.batch_size]
            data = self._request(
                "POST",
                self.path,
                {
                    "model": self.alias,
                    "input": batch,
                    "encoding_format": "float",
                    "truncate_prompt_tokens": self.max_tokens,
                },
            )
            for vector in self._ordered(data, len(batch), "embedding"):
                finite_vector(vector, 1024)
                norm = math.hypot(*vector)
                if not math.isfinite(norm) or norm == 0:
                    raise StateUnavailable("Zero or invalid Dense norm")
                result.append([float(v / norm) for v in vector])
        return result

    def validate_profile(self, tokenizer=None):
        self.validate_identity()
        probes = ["My project is Orion.", "北京 café memory."]
        batch = self.embed(probes)
        single = [self.embed([text])[0] for text in probes]
        if (
            max(
                abs(a - b)
                for x, y in zip(batch, single, strict=True)
                for a, b in zip(x, y, strict=True)
            )
            > 0.01
        ):
            raise ProfileMismatch("Dense single/batch consistency failed")
        return True

    def health(self):
        self.validate_identity()
        self.embed(["memory health probe"])
        return True


class RemoteSpladeClient(RemoteClient):
    alias, root = "splade-v3", "naver/splade-v3"
    path, task, max_tokens = "/pooling", "splade", 512

    def count_tokens(self, text):
        texts_batch([text], 1)
        data = self._request("POST", "/tokenize", {
            "model": self.alias, "prompt": text, "add_special_tokens": True})
        ids = data.get("tokens")
        if (not isinstance(ids, list) or type(data.get("count")) is not int
                or data["count"] != len(ids) or not ids
                or any(type(i) is not int or not 0 <= i < 30522 for i in ids)):
            raise StateUnavailable("Malformed SPLADE tokenizer output")
        return data["count"]

    def encode_documents(self, texts):
        texts_batch(texts, 128)
        # Retain frozen document microbatch1; the public API still accepts batches.
        result = []
        for text in texts:
            data = self._request(
                "POST",
                self.path,
                {
                    "model": self.alias,
                    "input": [text],
                    "task": "embed",
                    "encoding_format": "float",
                    "truncate_prompt_tokens": 512,
                    "add_special_tokens": True,
                },
            )
            result.append(sparse_vector(self._ordered(data, 1, "data")[0]))
        return result

    def encode_query(self, text):
        return self.encode_documents([text])[0]

    def validate_profile(self):
        self.validate_identity()
        self.encode_query("memory health probe")
        return True

    health = validate_profile


class RemoteQueritTokenizer:
    """Tokenizer interface for the existing official preparation algorithm.

    These attributes specify its expected contract, not verified remote revisions.
    Special token identities are checked over /tokenize and /detokenize at admission.
    """

    pad_token_id = 151643
    model_max_length = 131072

    def __init__(self, client):
        self.client = client

    def convert_tokens_to_ids(self, text):
        ids = self.encode(text, add_special_tokens=False, truncation=False)
        if len(ids) != 1:
            raise ProfileMismatch("Remote Querit special token is not atomic")
        return ids[0]

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        if add_special_tokens or truncation:
            raise ProfileMismatch("Querit must preserve the complete official pair")
        data = self.client._request(
            "POST",
            "/tokenize",
            {
                "model": self.client.alias,
                "prompt": text,
                "add_special_tokens": False,
            },
        )
        ids = data.get("tokens")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or not 0 <= i < 151666 for i in ids)
            or type(data.get("count")) is not int
            or data["count"] != len(ids)
        ):
            raise StateUnavailable("Malformed Querit tokenizer output")
        return ids


class RemoteQueritClient(RemoteClient):
    alias, root = "querit-4b", "/opt/models/Querit-4B-vllm"
    path, task, max_tokens = "/classify", "querit", 40960

    def __init__(self, url, profile, **kwargs):
        if profile != expected_profile(batch_size=1):
            raise ProfileMismatch("Remote Querit requires the exact official B1 policy")
        self.expected_profile = profile
        super().__init__(url, **kwargs)
        self.preparation = None

    def validate_profile(self, tokenizer=None):
        self.validate_identity()
        remote = RemoteQueritTokenizer(self)
        for token, expected in (("[CLS]", 151665), ("<|endoftext|>", 151643)):
            if (
                remote.convert_tokens_to_ids(token) != expected
                or self._request(
                    "POST", "/detokenize", {"model": self.alias, "tokens": [expected]}
                ).get("prompt")
                != token
            ):
                raise ProfileMismatch("Remote Querit special token mismatch")
        self.preparation = QueritPreparation(remote, self.max_tokens)
        self.rerank("memory", ["memory health probe"])
        return True

    def classify_pair(self, pair):
        data = self._request(
            "POST",
            self.path,
            {
                "model": self.alias,
                "input": [list(pair.input_ids)],
                "add_special_tokens": False,
                "use_activation": True,
            },
        )
        probs = self._ordered(data, 1, "probs")[0]
        if data["data"][0].get("num_classes") != 2:
            raise StateUnavailable("Querit must return two classes")
        return probs, relevance(probs)

    def rerank(self, query, documents):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Querit requires a nonblank query")
        texts_batch(documents, 10000)
        if self.preparation is None:
            raise StateUnavailable("Querit has not passed contract admission")
        # Original-order B1: no prompt substitution, truncation or synthetic scores.
        return [
            self.classify_pair(self.preparation.prepare(query, doc))[1]
            for doc in documents
        ]

    def health(self):
        self.validate_identity()
        self.rerank("memory", ["memory health probe"])
        return True

    def attestation(self):
        return dict(
            super().attestation(),
            batch_size=1,
            final_cls=151665,
            prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
            score_semantics="probs[1]-probs[0]",
        )


def remote_encoders(profile, embedding_url, splade_url, *, endpoint_policy="vpc"):
    if (
        profile.splade_model != "naver/splade-v3"
        or profile.splade_revision != "fdfeceb91d7b9de7985b38addd3ba9f53a59a355"
        or profile.options["splade_microbatch"] != 1
    ):
        raise ProfileMismatch("Unexpected frozen SPLADE policy")
    dense = RemoteDenseClient(
        embedding_url, ServingProfile(**profile.options["embedding_serving"]), endpoint_policy=endpoint_policy
    )
    sparse = None
    try:
        sparse = RemoteSpladeClient(splade_url, endpoint_policy=endpoint_policy)
        dense.validate_profile()
        sparse.validate_profile()
        return RetrievalEncoders(dense, sparse)
    except BaseException:
        dense.close()
        if sparse:
            sparse.close()
        raise
