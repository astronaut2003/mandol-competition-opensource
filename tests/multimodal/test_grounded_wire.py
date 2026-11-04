"""Real SQLite/Qdrant and HTTP; artificial media and offline inference only."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json

from fastapi.testclient import TestClient
import pytest

from mandol_competition.inference.errors import InferenceRateLimited
from mandol_competition.multimodal.api import create_app
from mandol_competition.multimodal.service import AddService
from mandol_competition.multimodal.vlm.base import Extraction
from mandol_competition.multimodal.vlm.openai_compat import VisionExtractionUnavailable
from .backend_support import request
from .test_grounded_evidence import Mini
from .test_sqlite_qdrant import image_part


class Vision:
    model_name = "artificial"
    calls = 0
    failure = None

    def extract(self, **_):
        self.calls += 1
        if self.failure is not None and self.calls == 2:
            raise self.failure
        return Extraction(description="An artificial colored square.", model=self.model_name,
                          prompt_version="v1", schema_version="1")


def grounded_app(backend, monkeypatch, mini, vision=None, **extra):
    from mandol_competition.inference.evidence_selector import EvidenceSelector
    config = replace(backend.config, vlm_backend="openai_compat", vlm_base_url="http://unused.invalid",
                     evidence_policy="source-grounded-v1", vision_failure_policy="retry-required-v1", **extra)
    monkeypatch.setattr("mandol_competition.multimodal.evidence.create_selector",
                        lambda _: EvidenceSelector(None, client=mini, strict=True))
    return create_app(config, backend=backend, extractor=vision or Vision())


def test_http_failure_resume_and_replay_only_retries_failed_image(backend, monkeypatch):
    vision = Vision()
    vision.failure = VisionExtractionUnavailable(retry_safe=True, status_code=503)
    body = request(content=[image_part(), image_part("blue")]).model_dump()
    with TestClient(grounded_app(backend, monkeypatch, Mini(indices=[0]), vision), raise_server_exceptions=False) as client:
        first = client.post("/add", json=body)
        assert first.status_code == 503
        assert vision.calls == 2
        assert list(backend.state.iter_memories(backend.scope("alice"))) == []
        assert backend.state.scope_state(backend.scope("alice"))["published_seq"] == 0
        vision.failure = None
        assert client.post("/add", json=body).status_code == 200 and vision.calls == 3
        assert client.post("/add", json=body).status_code == 200 and vision.calls == 3
        memories = list(backend.state.iter_memories(backend.scope("alice")))
        assert len(memories) == 1 and not memories[0][0].metadata.get("vlm_degraded")
        assert client.post("/add", json={**body, "session_id": "different"}).status_code == 409


def test_unknown_paid_attempt_recovers_with_a_durable_cap_without_degraded_publication(backend):
    class Interrupted(Vision):
        def extract(self, **_):
            self.calls += 1
            raise SystemExit("artificial interruption after provider dispatch")
    vision = Interrupted()
    config = replace(backend.config, vision_failure_policy="retry-required-v1")
    add = AddService(config, extractor=vision, backend=backend)
    body = request(content=[image_part()])
    with pytest.raises(SystemExit):
        add.handle(body)
    for attempt in range(1, 4):
        # Each retry can recover once. Another process loss must consume that
        # durable allowance instead of either deadlocking or resetting it.
        with pytest.raises(SystemExit):
            AddService(config, extractor=vision, backend=backend).handle(body)
        assert vision.calls == attempt + 1
        assert list(backend.state.iter_memories(backend.scope("alice"))) == []
    with pytest.raises(InferenceRateLimited):
        AddService(config, extractor=vision, backend=backend).handle(body)
    assert vision.calls == 4
    assert list(backend.state.iter_memories(backend.scope("alice"))) == []


def test_wire_provenance_original_media_unique_budget_and_user_isolation(backend, monkeypatch):
    mini = Mini(indices=[0, 1, 2])
    with TestClient(grounded_app(backend, monkeypatch, mini, max_images=2)) as client:
        for index, color in enumerate(("red", "red", "blue")):
            body = request(request_id=f"r{index}", content=[image_part(color)]).model_dump()
            body["session_id"] = f"session-{index}"
            assert client.post("/add", json=body).status_code == 200
        result = client.post("/search", json={"user_id": "alice", "query": "colored squares", "top_k": 100})
        assert result.status_code == 200
        data = result.json()["data"]
        assert len(data) == 3 and len({r["id"] for r in data}) == 3
        assert all("Source metadata" in r["content"][0]["text"] and "session-" in r["content"][0]["text"] for r in data)
        images = [p["image_url"]["url"] for r in data for p in r["content"] if p["type"] == "image_url"]
        assert len(images) == 2 and set(images) == {image_part("red")["image_url"]["url"], image_part("blue")["image_url"]["url"]}
        assert "Shared original image" in json.dumps(data)
        json.dumps(data, allow_nan=False)
        with ThreadPoolExecutor(max_workers=16) as workers:
            concurrent = list(workers.map(lambda _: client.post("/search", json={"user_id": "alice",
                "query": "colored squares", "options": [], "top_k": 100}), range(16)))
        assert len(concurrent) == 16 and all(r.status_code == 200 and len(r.json()["data"]) == 3 for r in concurrent)
        calls = len(mini.calls)
        assert client.post("/search", json={"user_id": "other-user", "query": "squares", "top_k": 100}).json() == {"data": []}
        assert len(mini.calls) == calls
        mini.indices = []
        assert client.post("/search", json={"user_id": "alice", "query": "unobserved birthplace", "top_k": 100}).json() == {"data": []}
        mini.error = InferenceRateLimited(retry_after=71)
        failed = client.post("/search", json={"user_id": "alice", "query": "squares", "top_k": 100})
        assert failed.status_code == 429 and failed.headers["Retry-After"] == "71"
        assert sum(len(page) for page in backend.state.iter_memories(backend.scope("alice"))) == 3


def test_image_query_failure_is_not_replaced_by_text_only_success(backend, monkeypatch):
    class EmptyVision(Vision):
        def extract(self, **_):
            self.calls += 1
            return Extraction(model=self.model_name, prompt_version="v1", schema_version="1")
    mini, vision = Mini(), EmptyVision()
    with TestClient(grounded_app(backend, monkeypatch, mini, vision), raise_server_exceptions=False) as client:
        response = client.post("/search", json={"user_id": "alice", "query": [image_part()], "top_k": 100})
        assert response.status_code == 503 and vision.calls == 1
        assert backend.encoders.queries == 0 and not mini.calls
