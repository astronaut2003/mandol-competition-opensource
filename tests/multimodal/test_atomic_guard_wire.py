"""Baseline/candidate wire comparison with real scratch SQLite and Qdrant."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

from fastapi.testclient import TestClient

from mandol_competition.inference.errors import InferenceRateLimited
from mandol_competition.multimodal.api import create_app
from .backend_support import request
from .test_atomic_recall_guard import Guard
from .test_grounded_wire import Vision
from .test_sqlite_qdrant import image_part


def app_for(backend, monkeypatch, guard):
    config = replace(backend.config, vlm_backend="openai_compat", vlm_base_url="http://unused.invalid",
        evidence_policy="source-grounded-atomic-v2", vision_failure_policy="retry-required-v1")
    guard.close = lambda: None
    monkeypatch.setattr("mandol_competition.multimodal.evidence.create_selector", lambda _: guard)
    return create_app(config, backend=backend, extractor=Vision())


def test_nonempty_wire_retains_baseline_ids_scores_order_and_original_image_under_16_concurrency(backend, monkeypatch):
    guard = Guard(primary=(0,))
    app = app_for(backend, monkeypatch, guard)
    body = request().model_dump()
    body["messages"] = [{"role": "user", "content": f"Artificial note {i}."} for i in range(24)]
    body["messages"].append({"role": "user", "content": [
        {"type": "text", "text": "The label reads ORBIT-47."}, image_part("blue")]})
    query = {"user_id": "alice", "query": "Which label is on the blue folder?", "top_k": 100}
    with TestClient(app) as client:
        assert client.post("/add", json=body).status_code == 200
        service = app.state.search_service
        candidate_config = service.config
        service.config = replace(candidate_config, evidence_policy="baseline")
        baseline = client.post("/search", json=query).json()["data"]
        service.config = candidate_config
        assert len(baseline) == 25
        expected = [(r["id"], r["score"]) for r in baseline]
        result = client.post("/search", json=query)
        assert result.status_code == 200
        assert [(r["id"], r["score"]) for r in result.json()["data"]] == expected
        image_urls = [p["image_url"]["url"] for r in result.json()["data"] for p in r["content"] if p["type"] == "image_url"]
        assert image_part("blue")["image_url"]["url"] in image_urls
        assert "ORBIT-47" in json.dumps(result.json())
        assert all("Source metadata" in r["content"][0]["text"] for r in result.json()["data"])
        with ThreadPoolExecutor(max_workers=16) as workers:
            replies = list(workers.map(lambda _: client.post("/search", json={**query, "options": []}), range(16)))
        assert len(replies) == 16 and all(r.status_code == 200 for r in replies)
        assert all([(item["id"], item["score"]) for item in r.json()["data"]] == expected for r in replies)
        assert not any(kind == "verification" for kind, _ in guard.calls)
        assert client.post("/search", json={**query, "user_id": "other"}).json() == {"data": []}


def test_false_negative_rescue_absence_and_failed_verification_are_distinct_http_outcomes(backend, monkeypatch):
    guard = Guard(primary=(), verification=(0,))
    with TestClient(app_for(backend, monkeypatch, guard), raise_server_exceptions=False) as client:
        body = request(content="The folder label is ORBIT-47.").model_dump()
        assert client.post("/add", json=body).status_code == 200
        query = {"user_id": "alice", "query": "Which folder label?", "top_k": 100}
        rescued = client.post("/search", json=query)
        assert rescued.status_code == 200 and len(rescued.json()["data"]) == 1
        assert "ORBIT-47" in json.dumps(rescued.json())
        guard.verification = ()
        rejected = client.post("/search", json={**query, "query": "Unobserved birthplace?"})
        assert rejected.status_code == 200 and rejected.json() == {"data": []}
        guard.error = InferenceRateLimited(retry_after=83)
        unavailable = client.post("/search", json=query)
        assert unavailable.status_code == 429 and unavailable.headers["Retry-After"] == "83"
        assert backend.state._rows("SELECT count(*) AS n FROM memories")[0]["n"] == 1
        assert client.post("/add", json=body).status_code == 200
