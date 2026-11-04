"""Known HTTP rejection headers survive incomplete error diagnostic bodies."""
import time
import httpx
import pytest

from mandol_competition.inference.errors import InferenceRateLimited,InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport


@pytest.mark.parametrize('status',[429,503])
def test_incomplete_error_body_keeps_status_and_retry_after(status):
    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"private":'
            raise httpx.ReadTimeout('PRIVATE DIAGNOSTIC TEXT')
    wire=HTTPTransport('https://provider.test',max_attempts=1,
        transport=httpx.MockTransport(lambda _:httpx.Response(status,headers={'Retry-After':'120'},stream=Broken())))
    try:
        with pytest.raises(InferenceUnavailable) as raised:wire.request('POST','/call',{},retry_unknown=False)
        assert raised.value.status_code==status and raised.value.retry_after==120
        assert 'PRIVATE' not in str(raised.value)
        assert wire.snapshot()['http_429' if status==429 else 'http_5xx']==1
    finally:wire.close()


def test_deadline_during_error_body_keeps_known_quota_rejection():
    class Slow(httpx.SyncByteStream):
        def __iter__(self):time.sleep(.03);yield b'{}'
    wire=HTTPTransport('https://provider.test',timeout=.01,max_attempts=1,
        transport=httpx.MockTransport(lambda _:httpx.Response(429,headers={'Retry-After':'120'},stream=Slow())))
    try:
        with pytest.raises(InferenceRateLimited) as raised:wire.request('POST','/call',{})
        assert raised.value.retry_after==120
    finally:wire.close()


def test_partial_success_body_is_still_unknown_and_not_automatically_repeated():
    seen=[]
    class Broken(httpx.SyncByteStream):
        def __iter__(self):yield b'{';raise httpx.ReadError('private')
    def reply(request):seen.append(1);return httpx.Response(200,stream=Broken())
    wire=HTTPTransport('https://provider.test',max_attempts=3,transport=httpx.MockTransport(reply))
    try:
        with pytest.raises(InferenceUnavailable) as raised:wire.request('POST','/call',{},retry_unknown=False)
        assert raised.value.unknown_outcome and len(seen)==1
    finally:wire.close()
