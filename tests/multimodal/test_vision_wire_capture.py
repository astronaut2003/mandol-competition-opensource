"""Capture the actual paid call without repeats or changes to its outcome."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import stat
import time

import httpx
import pytest

from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport
from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.multimodal.vlm.capture import VisionWireCapture
from mandol_competition.multimodal.vlm.openai_compat import OpenAICompatVisionClient
from mandol_competition.multimodal.vlm.output_guard import VisionOutputTruncated


def response(*, finish='stop', tokens=18, content='{"description":"Owned red square"}'):
    return {'model': 'gpt-4o-mini', 'choices': [{'finish_reason': finish,
        'message': {'role': 'assistant', 'content': content}}],
        'usage': {'prompt_tokens': 30, 'completion_tokens': tokens, 'total_tokens': 30 + tokens}}


def payload():
    return {'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'Owned JSON source'}],
        'max_completion_tokens': 16384, 'temperature': 0., 'response_format': {'type': 'json_object'}}


@pytest.fixture
def capture_path(tmp_path, monkeypatch):
    monkeypatch.delenv('MANDOL_GENERATION_PROVIDER', raising=False)
    path = tmp_path / 'private-capture'
    monkeypatch.setenv('MANDOL_MM_VISION_CAPTURE_DIR', str(path))
    monkeypatch.setenv('MANDOL_MM_VISION_CAPTURE_UNTIL_UNIX', str(time.time() + 600))
    return path


def test_actual_wire_body_and_truncated_output_are_captured_once_without_headers(capture_path):
    seen = []
    reply = response(finish='length', tokens=16384, content='{"description":"unfinished')
    def serve(request):
        seen.append(request)
        return httpx.Response(200, json=reply)
    client = OpenAICompatVisionClient(model='gpt-4o-mini', api_key='private-owned-secret',
        base_url='https://owned.test/v1', timeout=1)
    client._http.http.close()
    client._http.http = HTTPTransport('https://owned.test/v1', api_key='private-owned-secret',
        max_attempts=1, transport=httpx.MockTransport(serve))
    try:
        with pytest.raises(VisionOutputTruncated):
            client.complete(prompt='Owned JSON extraction.', image_data_uri='data:image/png;base64,QUJD')
    finally:
        client.close()
    assert len(seen) == 1
    saved = list(capture_path.glob('*.json'))
    assert len(saved) == 1
    artifact = json.loads(saved[0].read_text())
    assert artifact['request'] == json.loads(seen[0].content)
    assert artifact['response'] == reply and artifact['kind'] == 'output_failure'
    encoded = saved[0].read_text()
    assert 'private-owned-secret' not in encoded and 'Authorization' not in encoded
    assert 'Idempotency-Key' not in encoded and 'X-Request-ID' not in encoded
    assert stat.S_IMODE(saved[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(capture_path.stat().st_mode) == 0o700


def test_capture_limits_are_shared_across_threads_instances_and_restarts(capture_path):
    def observe(_):
        VisionWireCapture(capture_path, time.time() + 60).observe(payload(), response(finish='length', tokens=16384))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(observe, range(40)))
    assert len(list(capture_path.glob('output_failure-*.json'))) == 2
    for item in capture_path.glob('*.json'):
        assert json.loads(item.read_text())['response']['choices'][0]['finish_reason'] == 'length'
    capture = VisionWireCapture.from_environment()
    for _ in range(4):
        capture.observe(payload(), response())
        capture.observe(payload(), None, InferenceUnavailable('ReadTimeout', unknown_outcome=True))
    assert len(list(capture_path.glob('*.json'))) == 4
    transport = json.loads(next(capture_path.glob('transport_failure-*.json')).read_text())
    assert transport['response'] is None and transport['error_type'] == 'InferenceUnavailable'


@pytest.mark.parametrize('finish,tokens,content', [('length', 16384, '{}'),
    ('stop', 16384, '{}'), ('content_filter', 5, '{}'), ('stop', 5, '{'),
    ('stop', 5, '{"x":1,"x":2}'), ('stop', 5, '{"x":NaN}')])
def test_invalid_output_is_retained_as_failure_not_success(capture_path, finish, tokens, content):
    capture = VisionWireCapture.from_environment()
    capture.observe(payload(), response(finish=finish, tokens=tokens, content=content))
    assert len(list(capture_path.glob('output_failure-*.json'))) == 1
    assert not list(capture_path.glob('success-*.json'))


def test_expiry_and_missing_or_unsafe_configuration_never_capture(capture_path, monkeypatch):
    capture = VisionWireCapture(capture_path, time.time() - 1)
    capture.observe(payload(), response())
    assert not capture_path.exists()
    monkeypatch.delenv('MANDOL_MM_VISION_CAPTURE_UNTIL_UNIX')
    assert VisionWireCapture.from_environment() is None
    for expiry in ('nan', 'inf', '0', str(time.time() + 3600)):
        monkeypatch.setenv('MANDOL_MM_VISION_CAPTURE_UNTIL_UNIX', expiry)
        assert VisionWireCapture.from_environment() is None
    monkeypatch.setenv('MANDOL_MM_VISION_CAPTURE_UNTIL_UNIX', str(time.time() + 300))
    monkeypatch.setenv('MANDOL_MM_VISION_CAPTURE_DIR', 'relative-path')
    assert VisionWireCapture.from_environment() is None


def test_public_directory_symlink_and_write_failure_do_not_affect_result(capture_path, monkeypatch, tmp_path):
    capture_path.mkdir(mode=0o755)
    capture = VisionWireCapture.from_environment()
    capture.observe(payload(), response())
    assert not list(capture_path.iterdir())
    linked = tmp_path / 'linked'
    linked.symlink_to(capture_path, target_is_directory=True)
    VisionWireCapture(linked, time.time() + 60).observe(payload(), response())
    assert not list(capture_path.iterdir())
    capture_path.chmod(0o700)
    def failed_open(*args, **kwargs):
        raise OSError('owned diagnostic disk fault')
    monkeypatch.setattr(os, 'open', failed_open)
    capture.observe(payload(), response())
    assert not list(capture_path.iterdir())


def test_oversize_artifact_is_skipped_without_changing_inference(capture_path):
    capture = VisionWireCapture.from_environment()
    capture.FILE_BYTES = 100
    capture.observe(payload(), response())
    assert not list(capture_path.glob('*.json'))


def test_disabled_capture_never_reaches_http_options_or_files(tmp_path, monkeypatch):
    monkeypatch.delenv('MANDOL_MM_VISION_CAPTURE_DIR', raising=False)
    monkeypatch.delenv('MANDOL_MM_VISION_CAPTURE_UNTIL_UNIX', raising=False)
    seen = []
    client = HTTPVLMClient('https://owned.test/v1', 'gpt-4o-mini', transport=HTTPTransport(
        'https://owned.test/v1', max_attempts=1, transport=httpx.MockTransport(
            lambda request: seen.append(request) or httpx.Response(200, json=response()))))
    try:
        assert client.complete(payload()['messages'], request_id=None, max_completion_tokens=16384) == response()
        assert 'diagnostic' not in json.loads(seen[0].content)
        assert not list(tmp_path.iterdir())
    finally:
        client.close()


def test_diagnostic_fault_never_changes_success_or_original_error(capture_path):
    def broken(*args):
        raise RuntimeError('owned diagnostic fault')
    for fail in (False, True):
        seen = []
        def serve(request):
            seen.append(request)
            if fail:
                raise httpx.ReadTimeout('owned timeout')
            return httpx.Response(200, json=response())
        client = HTTPVLMClient('https://owned.test/v1', 'gpt-4o-mini', transport=HTTPTransport(
            'https://owned.test/v1', max_attempts=1, transport=httpx.MockTransport(serve)))
        try:
            if fail:
                with pytest.raises(InferenceUnavailable) as caught:
                    client.complete([], request_id=None, diagnostic=broken, max_completion_tokens=16384)
                assert caught.value.error_class == 'ReadTimeout' and caught.value.unknown_outcome
            else:
                assert client.complete([], request_id=None, diagnostic=broken, max_completion_tokens=16384) == response()
            assert len(seen) == 1
            assert 'diagnostic' not in json.loads(seen[0].content)
        finally:
            client.close()
