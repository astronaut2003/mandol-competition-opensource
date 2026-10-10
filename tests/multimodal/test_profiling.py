import json
from types import SimpleNamespace

import pytest

from mandol_competition.multimodal.profiling import RequestProfiler, timed


def profiler(path):
    return RequestProfiler(path, backend=None, add=SimpleNamespace(), search=SimpleNamespace(), extractor=SimpleNamespace())


def test_spans_contain_no_payload_and_profiling_does_not_change_result(tmp_path):
    measured = timed(lambda body: {"success": True}, "embedding_remote")
    p = profiler(tmp_path / "trace.jsonl")
    assert p.call("add", measured, {"user_id": "PRIVATE", "content": "RAW SECRET"}, "id") == {"success": True}
    text = p.path.read_text()
    assert "PRIVATE" not in text and "RAW SECRET" not in text
    record = json.loads(text)
    assert record["spans"][0]["stage"] == "embedding_remote"
    assert record["spans"][0]["seconds"] >= 0
    assert p.call("add", measured, None, "id2") == {"success": True}
    assert len(p.path.read_text().splitlines()) == 2


def test_failed_profile_write_cannot_turn_durable_success_into_failure(tmp_path, monkeypatch):
    p = profiler(tmp_path / "trace.jsonl")
    monkeypatch.setattr(type(tmp_path), "open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    assert p.call("add", lambda _: {"success": True}, None, "id") == {"success": True}


def test_error_details_are_redacted_and_context_is_reset(tmp_path):
    p = profiler(tmp_path / "trace.jsonl")
    def fail(_):
        raise ValueError("SECRET RESPONSE BODY")
    with pytest.raises(ValueError):
        p.call("add", timed(fail, "model"), None, "failed")
    record = json.loads(p.path.read_text())
    assert record["error_class"] == "ValueError" and "SECRET" not in p.path.read_text()
    assert record["spans"][0]["status"] == "error"


def test_sdk_error_cause_classes_preserve_diagnostics_without_response_body(tmp_path):
    from qdrant_client.http.exceptions import ResponseHandlingException
    import httpx
    p = profiler(tmp_path / "trace.jsonl")
    def fail(_):
        raise ResponseHandlingException(httpx.ReadError("SECRET REQUEST/RESPONSE"))
    with pytest.raises(ResponseHandlingException):
        p.call("search", fail, None, "failed")
    record = json.loads(p.path.read_text())
    assert record["error_classes"] == ["ResponseHandlingException", "ReadError"]
    assert "SECRET" not in p.path.read_text()


def test_asgi_timings_include_serialization_and_cross_worker_dispatch_without_content(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import FastAPI,Response
    from fastapi.testclient import TestClient
    from mandol_competition.multimodal.profiling import HTTPProfileMiddleware,ProfiledJSONResponse
    p=profiler(tmp_path/'profile.jsonl');app=FastAPI()
    app.add_middleware(HTTPProfileMiddleware,profiler=p)
    @app.post('/search',response_class=ProfiledJSONResponse)
    def search(body:dict,response:Response):
        identifier=p.response_id(response)
        # The service runs in a different deadline executor thread.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(p.call,'search',lambda _: {'data':[{'id':'i','content':'PRIVATE RESULT'}]},body,identifier).result()
    with TestClient(app) as client:
        response=client.post('/search',json={'query':'PRIVATE QUERY'})
        assert response.status_code==200
        value=json.loads((tmp_path/'profile-http.jsonl').read_text())
        assert value['profile_id']==response.headers['X-Mandol-Profile']
        assert value['json_serialization']>=0 and value['validation_auth_dispatch']>=0
        assert value['response_validation_dispatch']>=0
        assert 'PRIVATE' not in (tmp_path/'profile-http.jsonl').read_text()
        assert not p._http_records
