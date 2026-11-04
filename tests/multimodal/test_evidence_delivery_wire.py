"""Offline HTTP/SQLite/Qdrant checks against the complete v2 wire floor."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json

from fastapi.testclient import TestClient
import pytest

from mandol_competition.inference.errors import InferenceRateLimited
from mandol_competition.multimodal.api import create_app
from mandol_competition.multimodal.evidence_delivery import HEADER, POLICY
from mandol_competition.multimodal.vlm.base import ExtractionError
from .backend_support import request
from .test_atomic_recall_guard import Guard
from .test_grounded_wire import Vision
from .test_sqlite_qdrant import image_part


def app_for(backend, monkeypatch, guard=None, vision=None, **extra):
    guard = guard or Guard(primary=(0,))
    guard.close = lambda: None
    monkeypatch.setattr("mandol_competition.multimodal.evidence.create_selector", lambda _: guard)
    config = replace(backend.config, vlm_backend="openai_compat", vlm_base_url="http://unused.invalid",
                     evidence_policy=POLICY, vision_failure_policy="retry-required-v1", **extra)
    return create_app(config, backend=backend, extractor=vision or Vision())


def notes_and_evidence(data):
    clean, notes = deepcopy(data), []
    for row in clean:
        kept = []
        for part in row["content"]:
            if part["type"] == "text" and part["text"].startswith(HEADER):
                notes.append(json.loads(part["text"][len(HEADER):]))
            else:
                kept.append(part)
        row["content"] = kept
    return notes, clean


def previous_response(client, query):
    service = client.app.state.search_service
    config = service.config
    try:
        service.config = replace(config, evidence_policy="source-grounded-atomic-v2")
        response = client.post("/search", json=query)
        assert response.status_code == 200
        return response.json()["data"]
    finally:
        service.config = config


def test_complete_atomic_and_cross_session_sources_are_byte_for_byte_v2(backend, monkeypatch):
    with TestClient(app_for(backend, monkeypatch)) as client:
        bodies = [request(request_id="choice", content=[
            {"type": "text", "text": "I chose the blue folder marked ORBIT-47."}, image_part("blue")]).model_dump(),
            request(request_id="purchase", content="I bought the folder selected in the previous session.").model_dump()]
        for index, body in enumerate(bodies):
            body["session_id"] = "session-" + str(index)
            body["messages"][0]["timestamp"] = 1767312000000 + index * 86400000
            assert client.post("/add", json=body).status_code == 200
        query = {"user_id": "alice", "query": "Which folder did I buy?", "top_k": 100}
        before = previous_response(client, query)
        assert len(before) == 2 and "ORBIT-47" in json.dumps(before)
        for options in ({}, {"options": []}, {"options": ["blue", "green"]}):
            result = client.post("/search", json={**query, **options})
            assert result.status_code == 200
            assert result.json()["data"] == previous_response(client, {**query, **options})
            assert notes_and_evidence(result.json()["data"])[0] == []
        assert all("session-" in row["content"][0]["text"] for row in before)
        assert image_part("blue")["image_url"]["url"] in json.dumps(before)


@pytest.mark.parametrize("limit,reason", [({"max_images": 1}, "image_count_budget"),
    ({"max_image_bytes": 1}, "single_image_byte_budget"),
    ({"image_byte_budget": 1}, "response_image_byte_budget"), ({}, "media_unavailable")])
def test_each_real_omission_keeps_every_original_text_id_score_and_image_admission(backend, monkeypatch, limit, reason):
    guard = Guard(primary=(0,))
    with TestClient(app_for(backend, monkeypatch, guard)) as client:
        for index, color in enumerate(("red", "blue")):
            body = request(request_id="r" + str(index), content=[
                {"type": "text", "text": f"Artificial folder {index} label ORBIT-{index}."}, image_part(color)]).model_dump()
            assert client.post("/add", json=body).status_code == 200
        service = client.app.state.search_service
        service.config = replace(service.config, **limit)
        if reason == "media_unavailable":
            monkeypatch.setattr("mandol_competition.multimodal.media_store.UserMediaStore.get", lambda *_: None)
            guard.primary = ()  # Missing media makes a negative review incomplete.
        query = {"user_id": "alice", "query": "Which folder label?", "top_k": 100}
        baseline = previous_response(client, query)
        before_seq = backend.state.scope_state(backend.scope("alice"))["published_seq"]
        response = client.post("/search", json=query)
        assert response.status_code == 200
        notes, actual = notes_and_evidence(response.json()["data"])
        assert actual == baseline and len(actual) == 2 and len(notes) == 1
        assert set(row["memory_id"] for row in notes[0]["sources"]) <= {row["id"] for row in actual}
        assert notes[0]["omitted_image_references"] == (1 if reason == "image_count_budget" else 2)
        assert all(row["not_attached"] == {reason: 1} for row in notes[0]["sources"])
        assert sum(len(page) for page in backend.state.iter_memories(backend.scope("alice"))) == 2
        assert backend.state.scope_state(backend.scope("alice"))["published_seq"] == before_seq
        assert len(guard.calls[-1][1]["candidates"]) == 2
        assert not any(kind == "verification" for kind, _ in guard.calls)
        assert client.post("/search", json={**query, "user_id": "other"}).json() == {"data": []}


def test_historical_failed_summary_keeps_the_original_image_and_replay_does_not_rewrite(backend, monkeypatch):
    class OldFailure(Vision):
        def extract(self, **_):
            raise ExtractionError("artificial historical extraction failure")
    with TestClient(app_for(backend, monkeypatch, vision=OldFailure())) as client:
        add = client.app.state.add_service
        original = add.config
        # Reproduce data written by the previous permissive policy in scratch
        # storage; the new strict Add policy itself is never weakened.
        add.config = replace(original, vision_failure_policy="degrade")
        body = request(content=[image_part("blue"), image_part("blue")]).model_dump()
        assert client.post("/add", json=body).status_code == 200
        add.config = original
        query = {"user_id": "alice", "query": "Which color is visible?", "top_k": 100}
        baseline = previous_response(client, query)
        result = client.post("/search", json=query)
        assert result.status_code == 200
        notes, actual = notes_and_evidence(result.json()["data"])
        assert actual == baseline and len(notes) == 1
        report = notes[0]
        assert report["recorded_degraded_sources"] == 1 and report["omitted_image_references"] == 0
        assert report["sources"][0]["attached_here"] == 1
        assert report["sources"][0]["shared_in_response"] == 0
        assert image_part("blue")["image_url"]["url"] in json.dumps(actual)
        assert client.post("/add", json=body).status_code == 200
        assert sum(len(page) for page in backend.state.iter_memories(backend.scope("alice"))) == 1


def test_16_searches_have_independent_notes_and_preserve_the_complete_low_level_response(backend, monkeypatch):
    guard = Guard(primary=(0,))
    with TestClient(app_for(backend, monkeypatch, guard, max_images=1)) as client:
        for index, color in enumerate(("red", "blue")):
            assert client.post("/add", json=request(request_id=str(index), content=[image_part(color)]).model_dump()).status_code == 200
        query = {"user_id": "alice", "query": "Visible colors", "options": [], "top_k": 100}
        baseline = previous_response(client, query)
        with ThreadPoolExecutor(max_workers=16) as workers:
            replies = list(workers.map(lambda _: client.post("/search", json=query), range(16)))
        assert all(reply.status_code == 200 for reply in replies)
        for reply in replies:
            notes, evidence = notes_and_evidence(reply.json()["data"])
            assert evidence == baseline and len(notes) == 1
            assert notes[0]["affected_sources"] == notes[0]["omitted_image_references"] == 1
            json.dumps(reply.json(), allow_nan=False)
        assert not any(kind == "verification" for kind, _ in guard.calls)


def test_v3_keeps_partial_evidence_and_failure_semantics_of_the_v2_gate(backend, monkeypatch):
    guard = Guard(primary=(), verification=(0,))
    with TestClient(app_for(backend, monkeypatch, guard), raise_server_exceptions=False) as client:
        assert client.post("/add", json=request(content="Yesterday's chosen folder was blue.").model_dump()).status_code == 200
        query = {"user_id": "alice", "query": "Which folder did I buy today?", "top_k": 100}
        rescued = client.post("/search", json=query)
        assert rescued.status_code == 200 and len(rescued.json()["data"]) == 1
        guard.verification = ()
        assert client.post("/search", json=query).json() == {"data": []}
        guard.error = InferenceRateLimited(retry_after=97)
        failed = client.post("/search", json=query)
        assert failed.status_code == 429 and failed.headers["Retry-After"] == "97"
