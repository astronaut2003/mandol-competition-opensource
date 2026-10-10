"""Recipe HTTP clients. Transport is shared; each track selects its own recipe."""

import math
from dataclasses import asdict

from .errors import InferenceIdentityError, InferenceSchemaError
from .http import HTTPTransport


def ordered(data, count, field, alias):
    if data.get("model") != alias:
        raise InferenceIdentityError("inference response model mismatch")
    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) != count:
        raise InferenceSchemaError("partial_batch")
    if any(not isinstance(row, dict) or type(row.get("index")) is not int for row in rows):
        raise InferenceSchemaError("invalid_index")
    if sorted(row["index"] for row in rows) != list(range(count)):
        raise InferenceSchemaError("duplicate_or_missing_index")
    if any(field not in row for row in rows):
        raise InferenceSchemaError("missing_result")
    return [row[field] for row in sorted(rows, key=lambda row: row["index"])]


def finite_vector(value, dimension):
    if (not isinstance(value, list) or len(value) != dimension
            or any(type(v) not in {float, int} or not math.isfinite(v) for v in value)):
        raise InferenceSchemaError("invalid_vector")
    return value


class RecipeClient:
    def __init__(self, endpoint, recipe, *, transport=None, **http_options):
        self.recipe = recipe
        self.http = transport or HTTPTransport(endpoint, **http_options)
        self._attestation = None

    def validate_profile(self):
        observed = self.http.request("GET", "/v1/recipe")
        recipes = observed.get("recipes", [])
        if not any(r.get("hash") == self.recipe.hash and r.get("recipe") == asdict(self.recipe)
                   for r in recipes if isinstance(r, dict)):
            raise InferenceIdentityError("serving recipe differs from frozen client recipe")
        # This is server-declared recipe attestation, not independent weight inspection.
        self._attestation = {"level": "server-declared-recipe", "recipe_hash": self.recipe.hash}
        return True

    def health(self):
        return self.http.health()

    def attestation(self):
        return self._attestation or {"level": "not-checked", "recipe_hash": self.recipe.hash}

    def close(self):
        self.http.close()


class HTTPEmbeddingClient(RecipeClient):
    def __init__(self, *args, batch_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size < 1 or self.recipe.dimension is None:
            raise ValueError("embedding dimension and positive batch size required")
        self.batch_size = batch_size

    def embed(self, texts, *, query=False):
        result = []
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("nonempty embedding texts required")
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            # A recipe server applies its pinned prompt/tokenizer policy. Plain
            # vLLM recipes use separate adapters so prompts cannot be applied twice.
            data = self.http.request("POST", "/v1/embeddings", {
                "model": self.recipe.served_alias, "input": batch,
                "input_type": "query" if query else "document",
                "encoding_format": "float", "recipe_hash": self.recipe.hash})
            for row in ordered(data, len(batch), "embedding", self.recipe.served_alias):
                values = finite_vector(row, self.recipe.dimension)
                if not any(values):
                    raise InferenceSchemaError("zero_vector")
                result.append(values)
        return result


class HTTPSparseEncoderClient(RecipeClient):
    def encode_documents(self, texts):
        return self._encode(texts)

    def _encode(self,texts,*,input_type=None):
        result = []
        for text in texts:
            body={"model": self.recipe.served_alias,"input": [text],"recipe_hash": self.recipe.hash}
            if input_type is not None:body["input_type"]=input_type
            data = self.http.request("POST", "/v1/sparse", body)
            value = ordered(data, 1, "sparse", self.recipe.served_alias)[0]
            if not isinstance(value, dict):
                raise InferenceSchemaError("invalid_sparse")
            sparse = {}
            for key, weight in value.items():
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    raise InferenceSchemaError("invalid_sparse_index") from None
                if str(index) != str(key) or not 0 <= index < (self.recipe.dimension or 2**32 - 1):
                    raise InferenceSchemaError("invalid_sparse_index")
                if type(weight) not in {int, float} or not math.isfinite(weight) or weight < 0:
                    raise InferenceSchemaError("invalid_sparse_weight")
                if weight > self.recipe.sparse_threshold:
                    sparse[index] = float(weight)
            result.append(sparse)
        return result

    def encode_query(self, text):
        return self._encode([text],input_type="query")[0]


class HTTPRerankerClient(RecipeClient):
    def score(self, query, documents):
        if not documents:
            return []
        data = self.http.request("POST", "/v1/score", {"model": self.recipe.served_alias,
            "query": query, "documents": documents, "recipe_hash": self.recipe.hash})
        scores = ordered(data, len(documents), "score", self.recipe.served_alias)
        finite_vector(scores, len(documents))
        return scores

    rerank = score


class HTTPEncoders:
    def __init__(self, dense, sparse):
        self.dense, self.sparse = dense, sparse

    def encode_documents(self, texts):
        return list(zip(self.dense.embed(texts), self.sparse.encode_documents(texts), strict=True))

    def encode_query(self, text):
        return self.dense.embed([text], query=True)[0], self.sparse.encode_query(text)

    def close(self):
        self.dense.close()
        self.sparse.close()


class HTTPJointEncoders(HTTPEncoders):
    """Versioned paired encoding for a provider that returns both channels once."""
    def _encode(self, texts, *, query):
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("nonempty embedding texts required")
        data = self.dense.http.request("POST", "/v1/encode", {
            "model": self.dense.recipe.served_alias, "input": list(texts),
            "input_type": "query" if query else "document",
            "dense_recipe_hash": self.dense.recipe.hash,
            "sparse_recipe_hash": self.sparse.recipe.hash,
        }, retry_unknown=False)
        rows = ordered(data, len(texts), "pair", self.dense.recipe.served_alias)
        pairs = []
        for row in rows:
            if not isinstance(row, dict):
                raise InferenceSchemaError("invalid_joint_pair")
            dense = finite_vector(row.get("embedding"), self.dense.recipe.dimension)
            sparse = row.get("sparse")
            if not any(dense) or not isinstance(sparse, dict) or not sparse:
                raise InferenceSchemaError("empty_joint_vector")
            values = {}
            for key, value in sparse.items():
                try:
                    index = int(key)
                except (TypeError, ValueError):
                    raise InferenceSchemaError("invalid_joint_sparse") from None
                if (str(index) != str(key) or not 0 <= index < self.sparse.recipe.dimension
                        or type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                    raise InferenceSchemaError("invalid_joint_sparse")
                if value > self.sparse.recipe.sparse_threshold:
                    values[index] = float(value)
            if not values:
                raise InferenceSchemaError("empty_joint_sparse")
            pairs.append((dense, values))
        return pairs

    def encode_documents(self, texts):
        return self._encode(texts, query=False)

    def encode_query(self, text):
        return self._encode([text], query=True)[0]
