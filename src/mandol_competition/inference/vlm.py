import codecs
import json

from .http import HTTPTransport
from .errors import InferenceIdentityError, InferenceSchemaError, InferenceUnavailable, InferenceRateLimited
from .generation import generation_route, is_mini_response, MINI_REQUEST_MODELS, CLOSEAI_ENDPOINT


class HTTPVLMClient:
    def __init__(self, endpoint, model, *, api_key=None, timeout=60, transport=None,
                 credential_env='OPENAI_API_KEY'):
        self.route = generation_route(endpoint, model, credential_env=credential_env)
        self.model = self.route.model
        self._mini = model in MINI_REQUEST_MODELS
        # The adapter owns provider-specific retries. Unknown outcomes are not
        # automatically repeated by this transport on top of that policy.
        if transport is not None and self.route.routed and hasattr(transport, 'endpoint'):
            if transport.endpoint != self.route.endpoint:
                raise ValueError('generation transport endpoint differs from selected route')
        self.http = transport or HTTPTransport(self.route.endpoint,
            api_key=self.route.key(api_key) if self.route.routed else api_key,
            proxy=self.route.proxy, timeout=timeout, max_attempts=1)

    def prepare(self, messages, **options):
        return {'model': self.model, 'messages': messages, **self.route.options(options)}

    def complete(self, messages, *, request_id, diagnostic=None, **options):
        payload = self.prepare(messages, **options)
        try:
            result = self.http.request("POST", "/chat/completions", payload,
                request_id=request_id, retry_unknown=False)
        except Exception as exc:
            if diagnostic is not None:
                try:
                    diagnostic(payload, None, exc)
                except Exception:
                    pass
            raise
        if diagnostic is not None:
            try:
                diagnostic(payload, result, None)
            except Exception:
                pass
        if not isinstance(result, dict):
            raise InferenceSchemaError("invalid_vlm_response")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise InferenceSchemaError("invalid_vlm_choices")
        if self._mini and not is_mini_response(result.get('model')):
            raise InferenceIdentityError("mini wire response identity mismatch")
        return result

    def complete_stream(self, messages, *, request_id, on_content=None, diagnostic=None, **options):
        """Opt-in SSE using the same bounded transport and ordinary reply shape.

        The callback receives cumulative decoded content and may stop a paid
        response by raising. Neither that abort nor an incomplete accepted
        stream is automatically dispatched again.
        """
        if on_content is not None and not callable(on_content):
            raise ValueError('stream content callback must be callable')
        # Keep the original route's model/token/provider validation. Only this
        # explicitly named method enables streaming after those checks.
        if 'stream' in options or 'stream_options' in options:
            raise ValueError('stream framing is owned by complete_stream')
        payload = self.prepare(messages, **options)
        payload.update(stream=True, stream_options={'include_usage': True})
        try:
            result = self.http.request('POST', '/chat/completions', payload,
                request_id=request_id, retry_unknown=False,
                stream_decoder=lambda chunks: self._decode_stream(chunks, on_content,
                    allow_closeai_eof=self._mini and (self.route.closeai or self.route.endpoint == CLOSEAI_ENDPOINT)))
        except Exception as exc:
            if diagnostic is not None:
                try:
                    diagnostic(payload, None, exc)
                except Exception:
                    pass
            raise
        if diagnostic is not None:
            try:
                diagnostic(payload, result, None)
            except Exception:
                pass
        return result

    def _decode_stream(self, chunks, on_content, *, allow_closeai_eof=False):
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        buffered, data = '', []
        parts, refusal_parts = [], []
        model, finish, usage = None, None, None
        done = False
        last_event = None

        def invalid(code='invalid_vlm_stream'):
            raise InferenceSchemaError(code)

        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    invalid()
                result[key] = value
            return result

        def event(text):
            nonlocal model, finish, usage, done, last_event
            if text == '[DONE]':
                if finish is None or model is None:
                    invalid('incomplete_vlm_stream')
                done = True
                return
            try:
                value = json.loads(text, object_pairs_hook=object_pairs,
                    parse_constant=lambda _: invalid())
            except (ValueError, UnicodeError):
                invalid()
            if not isinstance(value, dict):
                invalid()
            if 'error' in value:
                nested = value['error']
                code = nested.get('code') if isinstance(nested, dict) else None
                if code in (429, '429'):
                    exc = InferenceRateLimited()
                    exc.unknown_outcome = True
                    raise exc
                raise InferenceUnavailable('generation_stream_error', status_code=200, unknown_outcome=True)
            identity = value.get('model')
            if identity is not None:
                if not isinstance(identity, str) or not identity or model is not None and identity != model:
                    invalid('inconsistent_vlm_stream_identity')
                if self._mini and not is_mini_response(identity):
                    raise InferenceIdentityError('mini wire response identity mismatch')
                model = identity
            other = value.get('usage')
            if other is not None:
                if (usage is not None or not isinstance(other, dict)
                        or any(type(other.get(k)) is not int or other[k] < 0
                               for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))):
                    invalid('invalid_vlm_stream_usage')
                usage = {k: other[k] for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
            choices = value.get('choices')
            if not isinstance(choices, list) or len(choices) > 1:
                invalid('invalid_vlm_choices')
            if not choices:
                if other is None:
                    invalid('invalid_vlm_choices')
                last_event = 'terminal_usage' if finish is not None else 'usage'
                return
            choice = choices[0]
            if (not isinstance(choice, dict) or type(choice.get('index')) is not int or choice['index'] != 0
                    or not isinstance(choice.get('delta'), dict)):
                invalid('invalid_vlm_choices')
            delta = choice['delta']
            if any(k not in {'role', 'content', 'refusal'} for k in delta) or delta.get('role', 'assistant') != 'assistant':
                invalid()
            content, refusal = delta.get('content'), delta.get('refusal')
            if content is not None and not isinstance(content, str) or refusal is not None and not isinstance(refusal, str):
                invalid()
            if finish is not None and (content or refusal or choice.get('finish_reason') is not None):
                invalid('content_after_vlm_finish')
            if content:
                if model is None:
                    invalid('missing_vlm_stream_identity')
                parts.append(content)
                if on_content is not None:
                    on_content(''.join(parts))
            if refusal:
                refusal_parts.append(refusal)
            reason = choice.get('finish_reason')
            if reason is not None:
                if not isinstance(reason, str) or not reason:
                    invalid('invalid_vlm_finish_reason')
                finish = reason
            last_event = 'finish' if reason is not None else 'choices'

        def reply():
            return dict(model=model, choices=[dict(index=0,
                message=dict(role='assistant', content=''.join(parts),
                             refusal=''.join(refusal_parts) or None), finish_reason=finish)], usage=usage)

        try:
            for chunk in chunks:
                buffered += decoder.decode(chunk)
                while '\n' in buffered:
                    line, buffered = buffered.split('\n', 1)
                    if line.endswith('\r'):
                        line = line[:-1]
                    if not line:
                        if data:
                            event('\n'.join(data)); data = []
                            if done:
                                return reply()
                    elif not line.startswith(':'):
                        field, separator, value = line.partition(':')
                        if not separator or field not in {'data', 'event', 'id', 'retry'}:
                            invalid()
                        if field == 'data':
                            data.append(value[1:] if value.startswith(' ') else value)
            buffered += decoder.decode(b'', final=True)
        except UnicodeError:
            invalid('invalid_vlm_stream_utf8')
        # CloseAI was observed to close a normally exhausted HTTP stream after
        # its complete finish + final usage-only event, without [DONE]. This is
        # an origin-specific terminal frame, never a general truncated-stream
        # fallback: no pending bytes/event and no iterator exception is allowed.
        if (allow_closeai_eof and self._mini and is_mini_response(model)
                and finish in {'stop', 'length', 'content_filter', 'tool_calls', 'function_call'}
                and last_event == 'terminal_usage' and usage is not None and not buffered and not data):
            return reply()
        invalid('incomplete_vlm_stream')

    def close(self):
        self.http.close()
