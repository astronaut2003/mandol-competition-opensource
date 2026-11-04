"""Request and nested operation deadlines shared by storage and inference."""
from contextlib import contextmanager
from contextvars import ContextVar
import time

_deadline = ContextVar('inference_deadline', default=None)
_request_deadline = ContextVar('api_request_deadline', default=None)


@contextmanager
def deadline_budget(seconds):
    until = time.monotonic() + seconds
    current = _deadline.get()
    token = _deadline.set(min(current, until) if current is not None else until)
    try:
        yield
    finally:
        _deadline.reset(token)


@contextmanager
def request_budget(seconds):
    until = time.monotonic() + seconds
    for current in (_request_deadline.get(), _deadline.get()):
        if current is not None:
            until = min(current, until)
    token = _request_deadline.set(until)
    try:
        with deadline_budget(max(0., until-time.monotonic())):
            yield
    finally:
        _request_deadline.reset(token)


def remaining_budget(seconds):
    until = _deadline.get()
    return max(0., min(seconds, until-time.monotonic())) if until is not None else seconds


def scope_wait_budget(fallback=None):
    """HTTP work may use its remaining request time; maintenance keeps its limit."""
    until = _request_deadline.get()
    if until is None:
        return fallback
    operation = _deadline.get()
    return max(0., min(until, operation if operation is not None else until)-time.monotonic())
