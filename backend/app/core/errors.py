"""Application errors with stable codes and user-safe messages."""


class AppError(Exception):
    code = "app_error"
    http_status = 400

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequestError(AppError):
    code, http_status = "bad_request", 400


class NotFoundError(AppError):
    code, http_status = "not_found", 404


class ConflictError(AppError):
    code, http_status = "conflict", 409


class PayloadTooLargeError(AppError):
    code, http_status = "file_too_large", 413


class UnsupportedTypeError(AppError):
    code, http_status = "unsupported_file_type", 415


class IngestionError(AppError):
    code, http_status = "ingestion_failed", 422


class EmbeddingError(AppError):
    code, http_status = "embedding_unavailable", 503
