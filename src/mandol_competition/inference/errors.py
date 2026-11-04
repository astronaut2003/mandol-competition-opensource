import math

from mandol_substrate.state import ProfileMismatch
from mandol_substrate.state.models import DependencyUnavailable


class InferenceUnavailable(DependencyUnavailable):
    def __init__(self, error_class, *, status_code=None, unknown_outcome=False, retry_after=None,
                 provider_error_code=None, provider_request_id=None):
        self.error_class = error_class
        self.status_code = status_code
        self.unknown_outcome = unknown_outcome
        self.retry_after = retry_after
        self.provider_error_code = provider_error_code
        self.provider_request_id = provider_request_id
        super().__init__(f"inference unavailable ({error_class})")


class InferenceRateLimited(InferenceUnavailable):
    propagate_retry_signal = True
    def __init__(self, error_class='provider_rate_limited', *, retry_after=None,
                 provider_error_code=None, provider_request_id=None):
        if (type(retry_after) not in (int, float) or not math.isfinite(retry_after)
                or retry_after < 0):
            retry_after = 60.
        super().__init__(error_class, status_code=429, retry_after=retry_after,
                         provider_error_code=provider_error_code,
                         provider_request_id=provider_request_id)


class InferenceInputRejected(InferenceUnavailable):
    """A known permanent input rejection, not provider unavailability."""
    propagate_retry_signal = True

    def __init__(self, error_class='embedding_input_rejected', *, status_code=400,
                 provider_error_code=None, provider_request_id=None):
        if status_code not in {400, 413, 422}:
            raise ValueError('permanent input status required')
        super().__init__(error_class, status_code=status_code,
            provider_error_code=provider_error_code, provider_request_id=provider_request_id)


class InferenceDeadlineExceeded(InferenceUnavailable):
    """The caller's budget expired before a new provider dispatch."""
    propagate_retry_signal = True
    def __init__(self):
        super().__init__('deadline_before_dispatch', unknown_outcome=False)


class InferenceAdmissionDeferred(InferenceRateLimited):
    """Local model budget could not admit this call; no provider failure."""


class InferenceSchemaError(InferenceUnavailable):
    def __init__(self, error_class="invalid_response"):
        super().__init__(error_class)


class InferenceIdentityError(ProfileMismatch):
    pass
