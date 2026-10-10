"""Frozen FP32 implementation for an external model worker or legacy qualification."""
import re
import threading
from importlib.metadata import version
from mandol_substrate.splade import MandolSpladeEncoder
from mandol_substrate.state.models import ProfileMismatch

def _check_runtime(profile):
    if profile.options.get("inference_recipe") != "multimodal-local-fp32-v1":
        raise ProfileMismatch("unsupported frozen inference recipe")
    for package in ("torch", "transformers", "sentence-transformers"):
        expected = profile.options.get("runtime", {}).get(package)
        if not expected or version(package).split("+")[0] != expected:
            raise ProfileMismatch(f"frozen {package} runtime mismatch")
    for revision in (profile.embedding_revision, profile.splade_revision):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ProfileMismatch("model revisions must be immutable commit hashes")
    if profile.tokenizer_fingerprint != profile.embedding_revision:
        raise ProfileMismatch("dense tokenizer must use the frozen model revision")


def _snapshot(model, revision):
    from huggingface_hub import snapshot_download

    # Some tokenizer versions perform model_info() for a repository name even
    # with local_files_only. A resolved immutable snapshot avoids that RPC.
    return snapshot_download(repo_id=model, revision=revision, local_files_only=True)


class LocalEncoders:
    def __init__(self, profile, *, device="cpu"):
        _check_runtime(profile)
        self.profile, self.device = profile, device
        self._lock = threading.RLock()
        self.dense = self.sparse = None

    def _load(self):
        if self.dense is None:
            import torch
            from sentence_transformers import SentenceTransformer

            self.dense = SentenceTransformer(_snapshot(self.profile.embedding_model, self.profile.embedding_revision),
                revision=self.profile.embedding_revision, local_files_only=True,
                trust_remote_code=False, device=self.device,
                model_kwargs={"torch_dtype": torch.float32})
            self.dense.max_seq_length = self.profile.options.get("dense_max_length", 4096)
            if self.dense.get_sentence_embedding_dimension() != self.profile.embedding_dimension:
                raise ProfileMismatch("dense model dimension differs from frozen profile")
        if self.sparse is None:
            self.sparse = MandolSpladeEncoder(_snapshot(self.profile.splade_model, self.profile.splade_revision),
                                             self.profile.splade_revision, device=self.device)

    def encode_dense(self, texts, *, query=False):
        with self._lock:
            self._load()
            return self.dense.encode(texts, batch_size=1, normalize_embeddings=True,
                prompt=self.profile.options.get("dense_query_prompt", "") if query else "",
                convert_to_numpy=True, show_progress_bar=False).tolist()

    def encode_sparse(self, texts):
        with self._lock:
            self._load()
            return self.sparse.encode_documents(texts)

    def _encode(self, texts, *, query=False):
        with self._lock:
            return list(zip(self.encode_dense(texts, query=query), self.encode_sparse(texts), strict=True))

    def encode_documents(self, texts):
        return self._encode(texts)

    def encode_query(self, text):
        return self._encode([text], query=True)[0]

    def close(self):
        self.dense = self.sparse = None


class LocalReranker:
    def __init__(self, profile, *, device="cpu"):
        _check_runtime(profile)
        options = profile.options
        self.model_name = options["reranker_model"]
        self.revision = options["reranker_revision"]
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ProfileMismatch("reranker revision must be an immutable commit hash")
        self.max_length = options.get("reranker_max_length", 4096)
        self.device, self.model = device, None
        self._lock = threading.Lock()

    def score(self, query, documents):
        with self._lock:
            if self.model is None:
                import torch
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(_snapshot(self.model_name, self.revision), revision=self.revision,
                    local_files_only=True, trust_remote_code=False, device=self.device,
                    max_length=self.max_length, model_kwargs={"torch_dtype": torch.float32})
            return self.model.predict([(query, doc) for doc in documents],
                                      batch_size=1, show_progress_bar=False).tolist()

    def close(self):
        self.model = None
