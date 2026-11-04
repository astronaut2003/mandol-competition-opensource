"""Shared HTTP error boundary; domain diagnostics contain no HTTP dependencies."""
from dataclasses import dataclass
from collections.abc import Callable
import math
import uuid

from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse

from mandol_substrate.state import Conflict, StateUnavailable
from mandol_competition.inference.errors import InferenceUnavailable, InferenceInputRejected
from mandol_competition.observability.diagnostics import (
    DiagnosticContext, bind_context, reset_context, configure_diagnostics,
    diagnostic_call, diagnostic_stage, record_failure, validation_details,
)


class SanitizedResponseFailure(RuntimeError):
    """Fixed message for the server when a response can no longer be replaced."""


@dataclass(frozen=True)
class ErrorRule:
    exception: type
    status: int
    reason: str
    headers: dict | None = None
    when: Callable | None = None


class ErrorPolicy:
    def __init__(self,rules=()):
        self.rules=tuple(rules)+(
            ErrorRule(StateUnavailable,503,'dependency unavailable; completion may be unknown'),
            ErrorRule(Conflict,409,'request identity or version conflict'),
        )

    def rule(self,exc):
        if isinstance(exc, InferenceInputRejected):
            return ErrorRule(InferenceInputRejected, exc.status_code, 'model input rejected')
        if isinstance(exc, InferenceUnavailable) and exc.status_code == 429:
            delay = exc.retry_after
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                delay = 60.
            return ErrorRule(InferenceUnavailable, 429, 'upstream rate limit exceeded',
                             {'Retry-After': str(math.ceil(delay))})
        return next((rule for rule in self.rules if isinstance(exc,rule.exception) and (rule.when is None or rule.when(exc))),
                    ErrorRule(Exception,500,'internal server error'))

    def status(self,exc):
        return self.rule(exc).status


def exception_status(exc):
    return ErrorPolicy().status(exc)


async def malformed_request(request, exc):
    details = validation_details(exc.errors())
    record_failure(exc,stage='request_validation',validation_errors=exc.errors())
    return JSONResponse(status_code=422,content={'detail':{'reason':'malformed request','errors':details}})


class DiagnosticContextMiddleware:
    def __init__(self,app,*,track='textual',policy=None):
        self.app=app
        assert track in ('textual','code','multimodal')
        self.track,self.policy=track,policy or ErrorPolicy()

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':
            return await self.app(scope,receive,send)
        context=DiagnosticContext(correlation_id=uuid.uuid4().hex,track=self.track,operation={
            '/add':'add','/search':'search','/health':'health','/readyz':'ready','/livez':'live',
        }.get(scope['path'],'other'))
        scope.setdefault('state',{})['diagnostic_context']=context
        token=bind_context(context)
        started=False

        async def correlated_send(message):
            nonlocal started
            if message['type']=='http.response.start':
                message=dict(message)
                message['headers']=[(k,v) for k,v in message['headers'] if k.lower()!=b'x-request-id']
                message['headers'].append((b'x-request-id',context.correlation_id.encode('ascii')))
                started=True
            await send(message)

        try:
            await self.app(scope,receive,correlated_send)
        except Exception as exc:
            validation = exc.errors() if isinstance(exc,(RequestValidationError,ResponseValidationError)) else None
            record_failure(exc,validation_errors=validation)
            if started:
                raise SanitizedResponseFailure('response failed after headers were sent') from None
            rule=self.policy.rule(exc)
            await JSONResponse(status_code=rule.status,content={'detail':{'reason':rule.reason}},
                headers=rule.headers)(scope,receive,correlated_send)
        finally:
            reset_context(token)


def install_http_boundary(app,track,*,ready,state=lambda:None,rules=()):
    from mandol_competition.observability.runtime import install_observability
    policy=ErrorPolicy(rules)
    install_observability(app,track,ready=ready,state=state,
        validation_error_handler=malformed_request,exception_status=policy.status)
    configure_diagnostics(track)
    app.add_middleware(DiagnosticContextMiddleware,track=track,policy=policy)
