from datetime import datetime, timezone
from email.utils import format_datetime
import time

import httpx
import pytest

from mandol_competition.inference.errors import InferenceUnavailable
from mandol_competition.inference.http import HTTPTransport, deadline_budget, retry_after_seconds


@pytest.mark.parametrize('header,expected', [('2', 2.), ('0', 0.), ('1.5', 1.5),
    ('-1', None), ('nan', None), ('inf', None), ('bad', None), (None, None)])
def test_retry_after_numeric_is_finite_nonnegative(header, expected):
    assert retry_after_seconds(header) == expected


def test_retry_after_http_date_and_expired_date():
    date = format_datetime(datetime.fromtimestamp(110., timezone.utc), usegmt=True)
    assert retry_after_seconds(date, now=100.) == 10.
    assert retry_after_seconds(date, now=120.) == 0.


@pytest.mark.parametrize('status', [429, 503])
def test_retry_after_is_not_shortened_and_jitter_is_bounded(status):
    waits = []
    client = HTTPTransport('http://test', timeout=10, max_attempts=3,
        transport=httpx.MockTransport(lambda _: httpx.Response(status, headers={'Retry-After': '2'})),
        sleep=waits.append, jitter=lambda low, high: high / 2)
    try:
        with pytest.raises(InferenceUnavailable):
            client.request('GET', '/probe')
        assert waits == [2., 2.]
        assert client.snapshot()['retries'] == 2
    finally:
        client.close()


def test_retry_after_beyond_budget_fails_without_early_retry():
    seen = []
    def reply(request):
        seen.append(request)
        return httpx.Response(429, headers={'Retry-After': '120'})
    client = HTTPTransport('http://test', timeout=1, transport=httpx.MockTransport(reply),
                           sleep=lambda _: pytest.fail('must not sleep beyond budget'))
    try:
        with pytest.raises(InferenceUnavailable):
            client.request('POST', '/encode', {}, retry_unknown=False)
        assert len(seen) == 1
    finally:
        client.close()


def test_predispatch_connection_failure_can_retry_paid_call_but_read_reset_cannot():
    for failure, attempts, unknown in [(httpx.ConnectError, 3, False), (httpx.ReadError, 1, True)]:
        seen, waits = [], []
        def reply(request):
            seen.append(request)
            raise failure('sensitive upstream message', request=request)
        client = HTTPTransport('http://test', transport=httpx.MockTransport(reply),
            sleep=waits.append, jitter=lambda low, high: high / 2)
        try:
            with pytest.raises(InferenceUnavailable) as exc:
                client.request('POST', '/paid', {}, retry_unknown=False)
            assert len(seen) == attempts and exc.value.unknown_outcome is unknown
            assert 'sensitive' not in str(exc.value)
            if attempts == 3:
                assert waits == [.05, .1]
        finally:
            client.close()


def test_sequence_budget_prevents_a_new_dispatch_after_deadline():
    seen = []
    client = HTTPTransport('http://test', transport=httpx.MockTransport(
        lambda request: seen.append(request) or httpx.Response(200, json={'ok': True})))
    try:
        with deadline_budget(.02):
            assert client.request('GET', '/first') == {'ok': True}
            time.sleep(.025)
            with pytest.raises(InferenceUnavailable):
                client.request('GET', '/second')
        assert len(seen) == 1
        assert client.snapshot()['inflight'] == client.snapshot()['waiting'] == 0
    finally:
        client.close()
