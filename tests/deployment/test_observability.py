import logging
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel
from mandol_competition.observability.runtime import SafeJSONFormatter, install_observability


def test_logs_and_validation_do_not_echo_request_material():
    record = logging.LogRecord('application', logging.ERROR, __file__, 1, 'failed content=%s', ('do-not-log-secret',), None)
    assert 'do-not-log-secret' not in SafeJSONFormatter().format(record)
    app = FastAPI()
    class Body(BaseModel): value: int
    @app.post('/add')
    def add(body: Body): return {}
    install_observability(app, 'code', ready=lambda: True)
    with TestClient(app) as client:
        response = client.post('/add', json={'value': 'do-not-log-secret'})
        assert response.status_code == 422 and 'do-not-log-secret' not in response.text
        assert client.get('/livez').status_code == client.get('/readyz').status_code == 200
        metrics = client.get('/metrics').text
        assert 'track="code"' in metrics and 'do-not-log-secret' not in metrics and 'user_id' not in metrics


def test_production_operational_routes_reject_nonlocal_and_spoofed_peers(monkeypatch):
    monkeypatch.setenv('MANDOL_INTERNAL_ENDPOINTS_LOOPBACK_ONLY','1')
    app=FastAPI()
    @app.get('/health')
    def health():return {'status':'ok'}
    @app.post('/admin/purge')
    def purge():raise AssertionError('remote administrative mutation must not run')
    install_observability(app,'multimodal',ready=lambda:True)
    with TestClient(app,client=('10.0.0.12',12345)) as remote:
        assert remote.get('/health').status_code==200
        for route in ('/metrics','/metrics/','/readyz','/livez','/admin/purge'):
            response=remote.request('POST' if route.startswith('/admin') else 'GET',route,
                headers={'X-Forwarded-For':'127.0.0.1','Forwarded':'for=127.0.0.1'})
            assert response.status_code==404
    with TestClient(app,client=('127.0.0.1',12345)) as local:
        assert local.get('/metrics').status_code==local.get('/readyz').status_code==200
