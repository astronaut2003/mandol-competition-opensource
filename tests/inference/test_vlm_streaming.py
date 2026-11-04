"""Opt-in SSE never bypasses bounded transport, model identity or cost guards."""
import json
import time

import httpx
import pytest

from mandol_competition.inference.errors import (
    InferenceDeadlineExceeded, InferenceIdentityError, InferenceRateLimited,
    InferenceSchemaError, InferenceUnavailable,
)
from mandol_competition.inference.http import HTTPTransport, deadline_budget
from mandol_competition.inference.vlm import HTTPVLMClient
from mandol_competition.multimodal.vlm.output_guard import VisionOutputError


def event(content=None, *, finish=None, model='gpt-4o-mini', usage=None):
    value=dict(model=model,choices=[dict(index=0,delta={} if content is None else dict(content=content),finish_reason=finish)])
    if usage is not None:value['usage']=usage
    return ('data: '+json.dumps(value,ensure_ascii=False)+'\n\n').encode()


def successful(text='artificial', *, finish='stop'):
    return (b': keepalive\r\n\r\n'+event(text)+event(finish=finish)
        +b'data: {"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\n'
        +b'data: [DONE]\n\n')


class Bytes(httpx.SyncByteStream):
    def __init__(self,chunks):self.chunks=chunks;self.closed=False;self.yielded=0
    def __iter__(self):
        for chunk in self.chunks:
            self.yielded+=1
            if isinstance(chunk,Exception):raise chunk
            if callable(chunk):chunk=chunk()
            yield chunk
    def close(self):self.closed=True


@pytest.fixture
def factory(monkeypatch):
    import mandol_competition.inference.budget as budget
    monkeypatch.setattr(budget,'from_environment',lambda:None)
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER','configured')
    clients=[]
    def make(handler,*,endpoint='https://provider.test',**kwargs):
        wire=HTTPTransport(endpoint,transport=httpx.MockTransport(handler),
            sleep=lambda _:None,jitter=lambda low,high:0,**kwargs)
        client=HTTPVLMClient(endpoint,'gpt-4o-mini',transport=wire)
        clients.append(client);return client,wire
    yield make
    for client in clients:client.close()


def call(client,**kwargs):
    return client.complete_stream([dict(role='user',content='artificial')],request_id='artificial-request',
        max_completion_tokens=32,**kwargs)


def test_utf8_sse_chunk_boundaries_return_normal_reply_and_cumulative_content(factory):
    body=successful('人工测试');stream=Bytes([bytes([b]) for b in body]);seen=[];payloads=[]
    def handle(request):
        payloads.append(json.loads(request.content));return httpx.Response(200,stream=stream)
    client,wire=factory(handle)
    result=call(client,on_content=seen.append)
    assert result['choices'][0]['message']['content']=='人工测试'
    assert result['choices'][0]['finish_reason']=='stop'
    assert result['model']=='gpt-4o-mini' and result['usage']['completion_tokens']==3
    assert seen==['人工测试'] and stream.closed
    assert payloads[0]['stream'] is True and payloads[0]['stream_options']==dict(include_usage=True)
    assert wire.snapshot()['inflight']==wire.snapshot()['waiting']==0


def test_multiple_content_deltas_are_cumulative_and_refusal_is_preserved(factory):
    stream=Bytes([event('a'),event('b'),event(finish='content_filter'),b'data: [DONE]\n\n'])
    client,_=factory(lambda _:httpx.Response(200,stream=stream));seen=[]
    result=call(client,on_content=seen.append)
    assert seen==['a','ab'] and result['choices'][0]['finish_reason']=='content_filter'
    assert result['usage'] is None


def test_length_is_returned_truthfully_not_changed_to_stop(factory):
    client,_=factory(lambda _:httpx.Response(200,stream=Bytes([successful(finish='length')])))
    assert call(client)['choices'][0]['finish_reason']=='length'


@pytest.mark.parametrize('body,exception',[
    (event('a',model='gpt-4o')+event(finish='stop')+b'data: [DONE]\n\n',InferenceIdentityError),
    (event('a')+event(finish='stop'),InferenceSchemaError),
    (event('a')+b'data: [DONE]\n\n',InferenceSchemaError),
    (b'data: {"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n\xff',InferenceSchemaError),
    (event('a')+event(finish='stop')+event('late')+b'data: [DONE]\n\n',InferenceSchemaError),
    (b'data: {"model":"gpt-4o-mini","model":"gpt-4o","choices":[]}\n\n',InferenceSchemaError),
])
def test_accepted_bad_streams_fail_paid_closed_without_retry(factory,body,exception):
    calls=[];stream=Bytes([body])
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream),max_attempts=3)
    with pytest.raises(exception) as exc:call(client)
    assert len(calls)==1 and getattr(exc.value,'vision_paid_output_failure',False)
    assert stream.closed and wire.snapshot()['inflight']==wire.snapshot()['waiting']==0


def test_content_callback_abort_stops_reading_without_repeating_paid_stream(factory):
    calls=[];stream=Bytes([event('repeated artificial'),event('must not consume'),event(finish='stop'),b'data: [DONE]\n\n'])
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream),max_attempts=3)
    error=VisionOutputError('vision_output_ocr_repetition',retryable=False)
    def stop(_):raise error
    with pytest.raises(VisionOutputError) as exc:call(client,on_content=stop)
    assert exc.value is error and error.vision_paid_output_failure is True
    assert len(calls)==stream.yielded==1 and stream.closed
    assert wire.snapshot()['inflight']==wire.snapshot()['waiting']==0


@pytest.mark.parametrize('failure',[httpx.ReadError('artificial read failure'),httpx.ConnectError('after received 200')])
def test_after_200_transport_break_is_paid_unknown_and_not_repeated(factory,failure):
    calls=[];stream=Bytes([event('a'),failure]);client,wire=factory(
        lambda request:calls.append(request) or httpx.Response(200,stream=stream),max_attempts=3)
    with pytest.raises(InferenceUnavailable) as exc:call(client)
    assert len(calls)==1 and exc.value.unknown_outcome and exc.value.vision_paid_output_failure
    assert stream.closed and wire.snapshot()['inflight']==0


def test_known_http_429_preserves_existing_retry_after_and_retry_path(factory):
    calls=[]
    def handle(request):
        calls.append(request)
        return httpx.Response(429,headers={'Retry-After':'0'}) if len(calls)==1 else httpx.Response(200,stream=Bytes([successful()]))
    client,wire=factory(handle,max_attempts=2)
    assert call(client)['choices'][0]['finish_reason']=='stop'
    assert len(calls)==2 and wire.snapshot()['http_429']==1 and wire.snapshot()['retries']==1


def test_http_429_never_invokes_sse_decoder_or_callback(factory):
    client,wire=factory(lambda _:httpx.Response(429,headers={'Retry-After':'60'}),max_attempts=1)
    with pytest.raises(InferenceRateLimited) as exc:call(client,on_content=lambda _:pytest.fail('not accepted'))
    assert exc.value.retry_after==60 and not getattr(exc.value,'vision_paid_output_failure',False)
    assert wire.snapshot()['inflight']==0


def test_sse_429_after_http_acceptance_is_paid_unknown_no_repeat(factory):
    calls=[];client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,
        stream=Bytes([b'data: {"error":{"code":429,"message":"never expose"}}\n\n'])),max_attempts=3)
    with pytest.raises(InferenceRateLimited) as exc:call(client)
    assert len(calls)==1 and exc.value.unknown_outcome and exc.value.vision_paid_output_failure
    assert 'never expose' not in str(exc.value) and wire.snapshot()['inflight']==0


def test_stream_response_byte_bound_closes_and_releases_admission(factory):
    stream=Bytes([b'x'*33]);client,wire=factory(lambda _:httpx.Response(200,stream=stream),max_response_bytes=32)
    with pytest.raises(InferenceSchemaError,match='response_too_large'):call(client)
    assert stream.closed and wire.snapshot()['inflight']==0


def test_successful_done_drains_bounded_tail_but_abort_does_not(factory):
    stream=Bytes([successful(),b': transport tail\n\n'])
    client,_=factory(lambda _:httpx.Response(200,stream=stream))
    assert call(client)['choices'][0]['finish_reason']=='stop'
    assert stream.yielded==2 and stream.closed


def test_done_does_not_bypass_transport_total_byte_bound(factory):
    body=successful();calls=[];stream=Bytes([body,b'x'*50])
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream),
                        max_response_bytes=len(body)+10,max_attempts=3)
    with pytest.raises(InferenceSchemaError,match='response_too_large') as exc:call(client)
    assert len(calls)==1 and exc.value.vision_paid_output_failure and stream.closed
    assert wire.snapshot()['inflight']==0


def test_deadline_before_dispatch_does_not_send_and_midstream_deadline_is_unknown(factory):
    calls=[];client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=Bytes([successful()])))
    with deadline_budget(.001):
        time.sleep(.003)
        with pytest.raises(InferenceDeadlineExceeded):call(client)
    assert not calls
    stream=Bytes([event('a'),lambda:(time.sleep(.02) or event(finish='stop')),b'data: [DONE]\n\n'])
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream))
    with deadline_budget(.01),pytest.raises(InferenceUnavailable) as exc:call(client)
    assert exc.value.unknown_outcome and exc.value.vision_paid_output_failure and stream.closed
    assert len(calls)==1 and wire.snapshot()['inflight']==0


def test_stream_decoder_cannot_be_enabled_for_other_endpoints_or_repeating_requests(factory):
    client,wire=factory(lambda _:pytest.fail('no dispatch'))
    for path,retry,payload in [('/encode',False,{'stream':True}),('/chat/completions',True,{'stream':True}),
                              ('/chat/completions',False,{'stream':False})]:
        with pytest.raises(ValueError):wire.request('POST',path,payload,retry_unknown=retry,stream_decoder=lambda _: {})


def test_original_complete_path_stays_nonstreaming(factory):
    seen=[]
    def handle(request):
        seen.append(json.loads(request.content));return httpx.Response(200,json=dict(model='gpt-4o-mini',
            choices=[dict(message=dict(content='artificial'),finish_reason='stop')]))
    client,_=factory(handle)
    result=client.complete([],request_id='artificial',max_completion_tokens=32)
    assert result['choices'][0]['message']['content']=='artificial' and 'stream' not in seen[0]


def without_done(*,finish='stop'):
    return successful(finish=finish).removesuffix(b'data: [DONE]\n\n')


@pytest.mark.parametrize('finish',['stop','length'])
def test_canonical_closeai_clean_finish_then_usage_eof_preserves_actual_reason(factory,finish):
    stream=Bytes([without_done(finish=finish)]);calls=[]
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream),
        endpoint='https://api.openai-proxy.org/v1',max_attempts=3)
    assert client.route.provider=='openai-proxy'  # The probe deliberately has no provider override.
    result=call(client)
    assert result['choices'][0]['finish_reason']==finish and result['usage']['completion_tokens']==3
    assert result['choices'][0]['message']['content']=='artificial'
    assert len(calls)==1 and stream.closed and wire.snapshot()['inflight']==0


def test_explicit_closeai_route_can_use_same_clean_eof_terminal(factory,monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER','closeai')
    monkeypatch.setenv('MANDOL_CLOSEAI_BASE_URL','https://custom-closeai.test/v1')
    client,_=factory(lambda _:httpx.Response(200,stream=Bytes([without_done()])),endpoint='https://custom-closeai.test/v1')
    assert client.route.closeai and call(client)['choices'][0]['finish_reason']=='stop'


def test_general_proxy_still_requires_actual_done_even_after_finish_and_usage(factory):
    client,_=factory(lambda _:httpx.Response(200,stream=Bytes([without_done()])))
    with pytest.raises(InferenceSchemaError,match='incomplete_vlm_stream'):call(client)


@pytest.mark.parametrize('body',[
    event('a')+event(finish='stop'),  # No usage-only terminal.
    event('a')+b'data: {"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'+event(finish='stop'),
    event('a')+b'data: {"model":"gpt-4o-mini","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n',
    without_done().removesuffix(b'\n'),  # Final SSE usage event lacks its empty frame delimiter.
    without_done()+b'garbage',
    without_done()+b'data: {"unfinished":',
    without_done()+b'\xe4',  # Incomplete UTF-8, not a clean EOF.
    event('a')+event(finish='stop')+b'data: {"model":"gpt-4o-mini","choices":[],"usage":{"completion_tokens":1}}\n\n',
])
def test_closeai_eof_requires_complete_ordered_frames_and_full_usage(factory,body):
    calls=[];stream=Bytes([body])
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream),
        endpoint='https://api.openai-proxy.org/v1',max_attempts=3)
    with pytest.raises(InferenceSchemaError) as exc:call(client)
    assert len(calls)==1 and exc.value.vision_paid_output_failure and stream.closed
    assert wire.snapshot()['inflight']==0


def test_closeai_clean_eof_cannot_hide_false_model_or_a_provider_error(factory):
    for body,error in [
        (without_done().replace(b'gpt-4o-mini',b'gpt-4o'),InferenceIdentityError),
        (without_done()+b'data: {"error":{"code":429,"message":"not retained"}}\n\n',InferenceRateLimited),
    ]:
        calls=[];client,_=factory(lambda request:calls.append(request) or httpx.Response(200,stream=Bytes([body])),
            endpoint='https://api.openai-proxy.org/v1',max_attempts=3)
        with pytest.raises(error) as exc:call(client)
        assert len(calls)==1 and exc.value.vision_paid_output_failure


def test_closeai_eof_after_finish_usage_still_rejects_exceptional_http_termination(factory):
    calls=[];stream=Bytes([without_done(),httpx.ReadError('artificial abnormal termination')])
    client,wire=factory(lambda request:calls.append(request) or httpx.Response(200,stream=stream),
        endpoint='https://api.openai-proxy.org/v1',max_attempts=3)
    with pytest.raises(InferenceUnavailable) as exc:call(client)
    assert len(calls)==1 and exc.value.unknown_outcome and exc.value.vision_paid_output_failure
    assert stream.closed and wire.snapshot()['inflight']==0
