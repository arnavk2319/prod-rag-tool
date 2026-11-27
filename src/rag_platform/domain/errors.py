"""Typed application errors; each maps to one HTTP status and one stable code."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str = "") -> None:
        super().__init__(message or self.code)
        self.message = message or self.code


class ValidationFailed(AppError):
    status_code = 422
    code = "validation_failed"


class Unauthenticated(AppError):
    status_code = 401
    code = "unauthenticated"


class Forbidden(AppError):
    status_code = 403
    code = "forbidden"


class NotFound(AppError):
    status_code = 404
    code = "not_found"


class Conflict(AppError):
    status_code = 409
    code = "conflict"


class PayloadTooLarge(AppError):
    status_code = 413
    code = "payload_too_large"


class RateLimited(AppError):
    status_code = 429
    code = "rate_limited"


class UpstreamUnavailable(AppError):
    status_code = 503
    code = "upstream_unavailable"


class UpstreamTimeout(AppError):
    status_code = 504
    code = "upstream_timeout"


class Internal(AppError):
    status_code = 500
    code = "internal_error"
