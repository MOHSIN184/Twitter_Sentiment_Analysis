import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request metadata and attach an X-Request-ID response header."""

    def __init__(self, app, logger: logging.Logger, enabled: bool = True) -> None:
        super().__init__(app)
        self.logger = logger
        self.enabled = enabled

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000
        response.headers["X-Request-ID"] = request_id

        if self.enabled:
            self.logger.info(
                "request method=%s path=%s status_code=%s duration_ms=%.2f request_id=%s",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                request_id,
            )

        return response
