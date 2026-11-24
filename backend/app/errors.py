"""Consistent JSON error responses: {"error": {"code", "message", "details"?}}.

Stack traces, file paths and request bodies (which may contain user text) are never returned.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ml_models.inference import ModelUnavailableError, UnknownModelError

log = logging.getLogger("predictai.api")

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    422: "validation_error",
    429: "rate_limited",
}


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str, details=None, headers: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


def error_response(status: int, code: str, message: str, details=None, headers: dict | None = None) -> JSONResponse:
    body: dict = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return error_response(exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        message = exc.detail if isinstance(exc.detail, str) else "Request could not be processed"
        return error_response(exc.status_code, code, message, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        # NB: the offending input is deliberately NOT echoed back (it may be user text)
        details = [
            {"field": ".".join(str(p) for p in err["loc"][1:]) or str(err["loc"][0]), "message": err["msg"]}
            for err in exc.errors()
        ]
        return error_response(422, "validation_error", "The request is invalid.", details)

    @app.exception_handler(UnknownModelError)
    async def _unknown_model(_: Request, exc: UnknownModelError):
        return error_response(422, "unknown_model", str(exc))

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(_: Request, exc: ModelUnavailableError):
        return error_response(503, "model_unavailable", str(exc))

    @app.exception_handler(PyMongoError)
    async def _db_error(_: Request, exc: PyMongoError):
        log.error("database error: %s", type(exc).__name__)
        return error_response(
            503,
            "database_unavailable",
            "The database is currently unavailable. Check MONGODB_URI and the Atlas network access list.",
        )

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("unhandled error")
        return error_response(500, "internal_error", "An unexpected error occurred.")
