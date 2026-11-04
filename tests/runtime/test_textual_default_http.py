"""Formal three-API lifecycle; real SQLite/Qdrant, synthetic HTTP models."""

from contextlib import ExitStack
import json
import os
import uuid

import httpx
import pytest

from mandol_competition.runtime.__main__ import profile_path
from tests.support.api_process import running
from tests.support.frozen_http import frozen_http
from tests.support.scratch_qdrant import free_port


pytestmark = pytest.mark.skipif(not os.getenv("MANDOL_TEST_QDRANT"),
                                reason="owned disposable Qdrant required")


def test_default_output_restart_and_explicit_original_rollback(tmp_path, isolated_formal_qdrant):
    key = "synthetic-default-lifecycle-key"
    user = "default-lifecycle-user"
    with ExitStack() as stack:
        provider, control = stack.enter_context(frozen_http("provider"))
        querit, _ = stack.enter_context(frozen_http("querit"))
        bge, _ = stack.enter_context(frozen_http("bge"))
        vlm, _ = stack.enter_context(frozen_http("vlm"))
        ports = []
        while len(ports) < 3:
            port = free_port()
            if port not in ports:
                ports.append(port)
        text = profile_path("local").read_text()
        for old, new in {
            'runtime_root = "data/local"': 'runtime_root = ' + json.dumps(str(tmp_path / "data")),
            "http://127.0.0.1:19733": isolated_formal_qdrant,
            "http://127.0.0.1:18087": provider, "http://127.0.0.1:18083": querit,
            "http://127.0.0.1:18084": bge, "http://127.0.0.1:18085/v1": vlm + "/v1",
        }.items():
            text = text.replace(old, new)
        for track, old, new in zip(("textual", "code", "multimodal"), (18080, 18081, 18082), ports):
            text = text.replace(f"{track}_port = {old}", f"{track}_port = {new}")
        # Namespace is diagnostic; the dedicated Qdrant fixture, rather than
        # a namespace change alone, isolates the fixed collection ownership.
        namespace = "default-lifecycle-" + uuid.uuid4().hex
        text = text.replace('namespace = "mandol-local"', 'namespace = ' + json.dumps(namespace))
        assert "quality_candidate" not in text and "evidence_profile" not in text
        config = tmp_path / "local.toml"
        config.write_text(text)
        fixture_env = {"MANDOL_LOCAL_API_KEY": key, "MANDOL_LOCAL_VLM_KEY": "fixture-only"}
        bodies = [dict(user_id=user, session_id="coffee", request_id="first", messages=[
            dict(role="user", content="I used to prefer sweet coffee."),
            dict(role="assistant", content="You previously preferred sweet coffee.")]),
            dict(user_id=user, session_id="coffee-update", request_id="second", messages=[
            dict(role="user", content="Now I prefer unsweetened coffee."),
            dict(role="assistant", content="Your current preference is unsweetened coffee.")])]
        query = dict(user_id=user, query="How did my coffee preference change over time?", top_k=100,
                     options=["sweet coffee", "unsweetened coffee"])
        original = upgraded = None
        known_documents = set()
        for phase in ("default", "original", "default-restored"):
            phase_root = tmp_path / phase
            phase_root.mkdir()
            env = dict(fixture_env)
            if phase == "original":
                env["MANDOL_LOCAL_TEXTUAL_EVIDENCE_PROFILE"] = "t0-original-v1"
            before = len(control.calls)
            with running(phase_root, config, ports, env) as (urls, _):
                if phase != "default":
                    assert not any(x["input_digest"] in known_documents for x in control.calls[before:])
                with httpx.Client(timeout=30, trust_env=False, headers={"X-Api-Key": key}) as client:
                    if phase == "default":
                        for body in bodies:
                            start = len(control.calls)
                            added = client.post(urls["textual"] + "/add", json=body)
                            assert added.status_code == 200
                            known_documents.update(x["input_digest"] for x in control.calls[start:]
                                                   if x["path"] == "/v1/encode" and x["count"] == 2)
                        assert len(known_documents) == 2  # Restart accounting is not vacuous.
                    start = len(control.calls)
                    for body in bodies:
                        assert client.post(urls["textual"] + "/add", json=body).status_code == 200
                    assert len(control.calls) == start  # Replay does not invoke inference.
                    result = client.post(urls["textual"] + "/search", json=query)
                    assert result.status_code == 200
                    data = result.json()["data"]
                    assert len(data) == 4
                    if phase == "original":
                        original = data
                        assert not any("Source observation order:" in x["content"] for x in data)
                    else:
                        assert all("Source observation order:" in x["content"] for x in data)
                        assert all("NOT event time or truth priority" in x["content"] for x in data)
                        if upgraded is None:
                            upgraded = data
                        else:
                            assert data == upgraded
                    for track in ("code", "multimodal"):
                        response = client.post(urls[track] + "/search", json=query)
                        assert response.status_code == 200 and response.json() == {"data": []}
        assert [x["id"] for x in upgraded] == [x["id"] for x in original]
        assert all(x["content"].endswith(y["content"]) and x["score"] == y["score"]
                   for x, y in zip(upgraded, original, strict=True))
