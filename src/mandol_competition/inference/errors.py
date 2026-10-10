from mandol_substrate.state import ProfileMismatch, StateUnavailable


class InferenceUnavailable(StateUnavailable):
    def __init__(self, error_class, *, status_code=None, unknown_outcome=False, retry_after=None):
        self.error_class = error_class
        self.status_code = status_code
        self.unknown_outcome = unknown_outcome
        self.retry_after = retry_after
        super().__init__(f"inference unavailable ({error_class})")


class InferenceSchemaError(InferenceUnavailable):
    def __init__(self, error_class="invalid_response"):
        super().__init__(error_class)


class InferenceIdentityError(ProfileMismatch):
    pass
