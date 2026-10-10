"""Deterministic inference for real SQLite/Qdrant fault tests (never serving)."""

import hashlib
from pathlib import Path

from mandol_substrate.state import RetrievalProfile
from mandol_substrate.text import MandolTextProcessor

from mandol_competition.multimodal.backend import MultimodalBackend
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.protocol import AddRequest, SearchRequest
from mandol_competition.multimodal.service import AddService, SearchService
from mandol_competition.multimodal.vlm.fake import FakeVisionExtractor


OWNED_COLLECTIONS = set()


class Encoders:
    def __init__(self, *, forbid_documents=False):
        self.documents = self.queries = 0
        self.forbid_documents = forbid_documents

    def _vector(self, text):
        tokens = text.lower().split()
        return ([1., len(tokens) / 100., .25],
                {int(hashlib.sha256(t.encode()).hexdigest()[:6], 16): 1. for t in tokens})

    def encode_documents(self, texts):
        assert not self.forbid_documents, "restart re-embedded a published document"
        self.documents += len(texts)
        return [self._vector(text) for text in texts]

    def encode_query(self, text):
        self.queries += 1
        return self._vector(text)


def profile():
    return RetrievalProfile("fixture", "frozen", 3, "frozen", "fixture", "frozen",
                            MandolTextProcessor().fingerprint,
                            methods=("bm25", "cosine", "splade"), options={"rerank_method": None})


def runtime(root, uri, collection, *, encoders=None, fault=None, namespace="multimodal", reranker=None):
    config = MultimodalConfig(data_dir=Path(root), qdrant_url=uri, qdrant_collection=collection,
                              rerank_method=None, namespace=namespace)
    backend = MultimodalBackend(config, profile=profile(), encoders=encoders or Encoders(),
                                 fault=fault, reranker=reranker)
    OWNED_COLLECTIONS.add((uri, collection))
    return backend


def services(backend, extractor=None):
    extractor = extractor or FakeVisionExtractor({}, model="fixture", prompt_version="v1", schema_version="1")
    return (AddService(backend.config, extractor=extractor, backend=backend),
            SearchService(backend.config, extractor=extractor, backend=backend))


def request(user="alice", request_id="r1", content="red backpack"):
    return AddRequest(request_id=request_id, user_id=user, session_id="s1",
                      messages=[{"role": "user", "content": content}])


def query(user="alice", text="backpack"):
    return SearchRequest(user_id=user, query=[{"type": "text", "text": text}], top_k=10)
