"""Qortex exception hierarchy.

All public-facing exceptions inherit from QortexError so callers can catch
the entire library with a single except clause if desired.
"""

from __future__ import annotations

import logging
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any


class QortexError(Exception):
    """Base class for all Qortex exceptions.

    Qortex exceptions carry structured context in addition to the human message
    so CLI, API, notebook, and logging surfaces can present the same failure
    without scraping strings.
    """

    default_code = "qortex.error"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        context: dict[str, Any] | None = None,
        suggestion: str | None = None,
        retriable: bool | None = None,
    ) -> None:
        self.message = message or self.__class__.__name__
        self.code = code or self.default_code
        self.context = dict(context or {})
        self.suggestion = suggestion
        self.retriable = retriable
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the error."""
        data: dict[str, Any] = {
            "type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
        }
        if self.context:
            data["context"] = self.context
        if self.suggestion:
            data["suggestion"] = self.suggestion
        if self.retriable is not None:
            data["retriable"] = self.retriable
        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(code={self.code!r}, "
            f"message={self.message!r}, context={self.context!r})"
        )


class QortexWarning(UserWarning):
    """Base warning category for non-fatal Qortex conditions."""


@dataclass(frozen=True)
class WarningRecord:
    """Structured non-fatal warning emitted by Qortex internals."""

    code: str
    message: str
    severity: str = "warning"
    context: dict[str, Any] = field(default_factory=dict)
    suggestion: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }
        if self.context:
            data["context"] = self.context
        if self.suggestion:
            data["suggestion"] = self.suggestion
        return data


def emit_warning(
    code: str,
    message: str,
    *,
    category: type[Warning] = QortexWarning,
    logger: logging.Logger | None = None,
    severity: str = "warning",
    context: dict[str, Any] | None = None,
    suggestion: str | None = None,
    stacklevel: int = 2,
) -> WarningRecord:
    """Emit a Python warning and optional log record with structured context."""
    record = WarningRecord(
        code=code,
        message=message,
        severity=severity,
        context=dict(context or {}),
        suggestion=suggestion,
    )
    if logger is not None:
        log_method = getattr(logger, severity, logger.warning)
        log_method("%s: %s", code, message, extra={"qortex_warning": record.to_dict()})
    _warnings.warn(message, category, stacklevel=stacklevel)
    return record


# ── Network / API ─────────────────────────────────────────────────────────────

class APIError(QortexError):
    """Raised when the OpenNeuro GraphQL API returns an unexpected response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(
            message,
            code="api.error",
            context={"status_code": status_code} if status_code is not None else {},
            retriable=status_code in {408, 429, 500, 502, 503, 504, 522, 524}
            if status_code is not None else None,
        )


class AuthError(QortexError):
    """Raised when authentication fails or no credentials are configured."""

    default_code = "auth.error"


class RateLimitError(APIError):
    """Raised when the server returns HTTP 429 and retries are exhausted."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        msg = (
            f"Rate limited by OpenNeuro server. Retry after {retry_after}s."
            if retry_after else "Rate limited by OpenNeuro server."
        )
        super().__init__(msg, status_code=429)
        self.context["retry_after"] = retry_after
        self.retriable = True


class NetworkError(QortexError):
    """Raised on connection-level failures (timeout, DNS, etc.)."""

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        operation: str | None = None,
        retriable: bool = True,
    ) -> None:
        self.url = url
        self.operation = operation
        context = {k: v for k, v in {"url": url, "operation": operation}.items() if v is not None}
        super().__init__(message, code="network.error", context=context, retriable=retriable)


# ── Dataset / Manifest ────────────────────────────────────────────────────────

class DatasetNotFoundError(QortexError):
    """Raised when a dataset ID does not exist on OpenNeuro."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(
            f"Dataset not found: {dataset_id!r}",
            code="dataset.not_found",
            context={"dataset_id": dataset_id},
            suggestion="Check the dataset id and whether it is public on OpenNeuro.",
        )


class SnapshotNotFoundError(QortexError):
    """Raised when a specific snapshot tag does not exist."""

    def __init__(self, dataset_id: str, tag: str, available: list[str] | None = None) -> None:
        self.dataset_id = dataset_id
        self.tag = tag
        self.available = available or []
        msg = f"Snapshot {tag!r} not found for dataset {dataset_id!r}."
        if available:
            msg += f" Available tags: {', '.join(available)}"
        super().__init__(
            msg,
            code="snapshot.not_found",
            context={"dataset_id": dataset_id, "tag": tag, "available": self.available},
        )


class ManifestError(QortexError):
    """Raised when the manifest cannot be built or is malformed."""

    default_code = "manifest.error"


# ── Download ──────────────────────────────────────────────────────────────────

class DownloadError(QortexError):
    """Raised when a file download fails after all retries."""

    def __init__(self, path: str, url: str, reason: str) -> None:
        self.path = path
        self.url = url
        self.reason = reason
        super().__init__(
            f"Failed to download {path!r} from {url!r}: {reason}",
            code="download.failed",
            context={"path": path, "url": url, "reason": reason},
            retriable=True,
        )


class IntegrityError(QortexError):
    """Raised when a downloaded file fails hash or size verification."""

    def __init__(self, path: str, expected: str, got: str, check: str = "hash") -> None:
        self.path = path
        self.expected = expected
        self.got = got
        self.check = check
        super().__init__(
            f"{check.capitalize()} mismatch for {path!r}: expected {expected!r}, got {got!r}",
            code="integrity.mismatch",
            context={"path": path, "expected": expected, "got": got, "check": check},
            suggestion="Delete the local file and retry the download.",
            retriable=True,
        )


class VersionConflictError(QortexError):
    """Raised when the local dataset snapshot differs from the requested one."""

    def __init__(self, dataset_id: str, local_tag: str, requested_tag: str) -> None:
        self.dataset_id = dataset_id
        self.local_tag = local_tag
        self.requested_tag = requested_tag
        super().__init__(
            f"Dataset {dataset_id!r}: local snapshot is {local_tag!r} but you requested "
            f"{requested_tag!r}. Remove the local copy or specify the same tag.",
            code="snapshot.version_conflict",
            context={
                "dataset_id": dataset_id,
                "local_tag": local_tag,
                "requested_tag": requested_tag,
            },
        )


class StorageError(QortexError):
    """Raised when there is insufficient disk space or a filesystem error."""

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        required_bytes: int | None = None,
        available_bytes: int | None = None,
    ) -> None:
        context = {
            k: v for k, v in {
                "path": path,
                "required_bytes": required_bytes,
                "available_bytes": available_bytes,
            }.items() if v is not None
        }
        super().__init__(message, code="storage.error", context=context)


# ── Parse / Load ──────────────────────────────────────────────────────────────

class LoaderNotFoundError(QortexError):
    """Raised when no registered loader can handle a given file."""

    def __init__(self, modality: str, extension: str | None = None) -> None:
        self.modality = modality
        msg = f"No loader registered for modality {modality!r}"
        if extension:
            msg += f" (extension {extension!r})"
        msg += ". Install the required optional dependency or register a custom loader."
        context = {"modality": modality}
        if extension:
            context["extension"] = extension
        super().__init__(msg, code="loader.not_found", context=context)


class LoadError(QortexError):
    """Raised when a loader fails to open or parse a file."""

    default_code = "load.error"


# ── ETL / Convert ─────────────────────────────────────────────────────────────

class ConversionError(QortexError):
    """Raised when an ETL conversion pipeline fails."""

    default_code = "conversion.error"


class ValidationError(QortexError):
    """Raised when BIDS validation cannot be executed or parsed."""

    default_code = "validation.error"


class FormatNotSupportedError(QortexError):
    """Raised when a requested output format has no registered writer."""

    def __init__(self, fmt: str) -> None:
        self.fmt = fmt
        super().__init__(
            f"Output format {fmt!r} is not supported. "
            f"Use one of: parquet, zarr, hdf5, webdataset, huggingface, tfrecord.",
            code="format.not_supported",
            context={"format": fmt},
        )


# ── Selection / Planning ──────────────────────────────────────────────────────

class SelectionError(QortexError):
    """Raised when an include/exclude pattern or filter yields no files."""

    def __init__(self, pattern: str, suggestions: list[str] | None = None) -> None:
        self.pattern = pattern
        msg = f"Pattern {pattern!r} matched no files in the manifest."
        if suggestions:
            msg += " Did you mean:\n  " + "\n  ".join(suggestions)
        super().__init__(
            msg,
            code="selection.no_matches",
            context={"pattern": pattern, "suggestions": suggestions or []},
        )


# ── Cache / Lake ──────────────────────────────────────────────────────────────

class CacheError(QortexError):
    """Raised for cache registry or integrity issues."""

    default_code = "cache.error"


class ConfigurationError(QortexError):
    """Raised when Qortex configuration is invalid."""

    default_code = "config.error"


class DatasetNotDownloadedError(QortexError):
    """Raised when a local operation is attempted on a dataset not yet fetched."""

    def __init__(self, dataset_id: str, snapshot: str | None = None) -> None:
        self.dataset_id = dataset_id
        self.snapshot = snapshot
        snap_str = f" (snapshot {snapshot!r})" if snapshot else ""
        super().__init__(
            f"Dataset {dataset_id!r}{snap_str} is not downloaded. "
            f"Run `ds.download()` or `qortex download {dataset_id}` first.",
            code="dataset.not_downloaded",
            context={"dataset_id": dataset_id, "snapshot": snapshot},
        )


# ── NeuroAI Runtime ───────────────────────────────────────────────────────────

class CompatibilityError(QortexError):
    """Raised when a source cannot satisfy a model's input contract."""

    def __init__(self, source_id: str, model_id: str, blockers: list | None = None) -> None:
        self.source_id = source_id
        self.model_id = model_id
        self.blockers = blockers or []
        blocker_str = "\n  ".join(_stringify(v) for v in self.blockers) if self.blockers else "see CompatibilityReport"
        super().__init__(
            f"Source {source_id!r} is incompatible with model {model_id!r}.\n"
            f"Blockers:\n  {blocker_str}",
            code="compatibility.incompatible",
            context={"source_id": source_id, "model_id": model_id, "blockers": [_stringify(v) for v in self.blockers]},
        )


class SourceAdapterError(QortexError):
    """Raised when a source adapter cannot probe or stream data."""

    def __init__(self, message: str, source_type: str | None = None, path: str | None = None) -> None:
        self.source_type = source_type
        self.path = path
        ctx = (f" [type={source_type!r}]" if source_type else "") + (f" [path={path!r}]" if path else "")
        super().__init__(
            f"SourceAdapter error{ctx}: {message}",
            code="source_adapter.error",
            context={k: v for k, v in {"source_type": source_type, "path": path}.items() if v is not None},
        )


class ModelAdapterError(QortexError):
    """Raised when a model adapter cannot inspect, load, or run inference."""

    def __init__(self, message: str, model_id: str | None = None, provider: str | None = None) -> None:
        self.model_id = model_id
        self.provider = provider
        ctx = (f" [provider={provider!r}]" if provider else "") + (f" [model={model_id!r}]" if model_id else "")
        super().__init__(
            f"ModelAdapter error{ctx}: {message}",
            code="model_adapter.error",
            context={k: v for k, v in {"model_id": model_id, "provider": provider}.items() if v is not None},
        )


class PreprocessPlanningError(QortexError):
    """Raised when the preprocessing planner cannot build a valid transform chain."""

    default_code = "preprocess_planning.error"


class OutputAdapterError(QortexError):
    """Raised when an output adapter cannot open, write, or close its destination."""

    def __init__(self, message: str, output_type: str | None = None, path: str | None = None) -> None:
        self.output_type = output_type
        self.path = path
        ctx = (f" [type={output_type!r}]" if output_type else "") + (f" [path={path!r}]" if path else "")
        super().__init__(
            f"OutputAdapter error{ctx}: {message}",
            code="output_adapter.error",
            context={k: v for k, v in {"output_type": output_type, "path": path}.items() if v is not None},
        )


class RuntimeExecutionError(QortexError):
    """Raised when the NeuroAI runtime encounters a fatal execution error."""

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        self.stage = stage
        super().__init__(
            message,
            code="runtime.execution_error",
            context={"stage": stage} if stage is not None else {},
        )


class ContractValidationError(QortexError):
    """Raised when a contract fails validation."""

    def __init__(self, contract_type: str, violations: list) -> None:
        self.contract_type = contract_type
        self.violations = [_stringify(v) for v in violations]
        super().__init__(
            f"{contract_type} validation failed:\n  " + "\n  ".join(self.violations),
            code="contract.validation_failed",
            context={"contract_type": contract_type, "violations": self.violations},
        )


def _stringify(value: Any) -> str:
    if hasattr(value, "message"):
        return str(getattr(value, "message"))
    return str(value)
