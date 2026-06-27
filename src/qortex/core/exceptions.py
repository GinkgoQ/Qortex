"""Qortex exception hierarchy.

All public-facing exceptions inherit from QortexError so callers can catch
the entire library with a single except clause if desired.
"""

from __future__ import annotations


class QortexError(Exception):
    """Base class for all Qortex exceptions."""


# ── Network / API ─────────────────────────────────────────────────────────────

class APIError(QortexError):
    """Raised when the OpenNeuro GraphQL API returns an unexpected response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


class AuthError(QortexError):
    """Raised when authentication fails or no credentials are configured."""


class RateLimitError(APIError):
    """Raised when the server returns HTTP 429 and retries are exhausted."""

    def __init__(self, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        super().__init__(
            f"Rate limited by OpenNeuro server. "
            f"Retry after {retry_after}s." if retry_after else "Rate limited."
        )


class NetworkError(QortexError):
    """Raised on connection-level failures (timeout, DNS, etc.)."""


# ── Dataset / Manifest ────────────────────────────────────────────────────────

class DatasetNotFoundError(QortexError):
    """Raised when a dataset ID does not exist on OpenNeuro."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Dataset not found: {dataset_id!r}")


class SnapshotNotFoundError(QortexError):
    """Raised when a specific snapshot tag does not exist."""

    def __init__(self, dataset_id: str, tag: str, available: list[str] | None = None) -> None:
        self.dataset_id = dataset_id
        self.tag = tag
        self.available = available or []
        msg = f"Snapshot {tag!r} not found for dataset {dataset_id!r}."
        if available:
            msg += f" Available tags: {', '.join(available)}"
        super().__init__(msg)


class ManifestError(QortexError):
    """Raised when the manifest cannot be built or is malformed."""


# ── Download ──────────────────────────────────────────────────────────────────

class DownloadError(QortexError):
    """Raised when a file download fails after all retries."""

    def __init__(self, path: str, url: str, reason: str) -> None:
        self.path = path
        self.url = url
        self.reason = reason
        super().__init__(f"Failed to download {path!r} from {url!r}: {reason}")


class IntegrityError(QortexError):
    """Raised when a downloaded file fails hash or size verification."""

    def __init__(self, path: str, expected: str, got: str, check: str = "hash") -> None:
        self.path = path
        super().__init__(
            f"{check.capitalize()} mismatch for {path!r}: expected {expected!r}, got {got!r}"
        )


class VersionConflictError(QortexError):
    """Raised when the local dataset snapshot differs from the requested one."""

    def __init__(self, dataset_id: str, local_tag: str, requested_tag: str) -> None:
        super().__init__(
            f"Dataset {dataset_id!r}: local snapshot is {local_tag!r} but you requested "
            f"{requested_tag!r}. Remove the local copy or specify the same tag."
        )


class StorageError(QortexError):
    """Raised when there is insufficient disk space or a filesystem error."""


# ── Parse / Load ──────────────────────────────────────────────────────────────

class LoaderNotFoundError(QortexError):
    """Raised when no registered loader can handle a given file."""

    def __init__(self, modality: str, extension: str | None = None) -> None:
        self.modality = modality
        msg = f"No loader registered for modality {modality!r}"
        if extension:
            msg += f" (extension {extension!r})"
        msg += ". Install the required optional dependency or register a custom loader."
        super().__init__(msg)


class LoadError(QortexError):
    """Raised when a loader fails to open or parse a file."""


# ── ETL / Convert ─────────────────────────────────────────────────────────────

class ConversionError(QortexError):
    """Raised when an ETL conversion pipeline fails."""


class ValidationError(QortexError):
    """Raised when BIDS validation cannot be executed or parsed."""


class FormatNotSupportedError(QortexError):
    """Raised when a requested output format has no registered writer."""

    def __init__(self, fmt: str) -> None:
        super().__init__(
            f"Output format {fmt!r} is not supported. "
            f"Use one of: parquet, zarr, hdf5, webdataset, huggingface, tfrecord."
        )


# ── Selection / Planning ──────────────────────────────────────────────────────

class SelectionError(QortexError):
    """Raised when an include/exclude pattern or filter yields no files."""

    def __init__(self, pattern: str, suggestions: list[str] | None = None) -> None:
        self.pattern = pattern
        msg = f"Pattern {pattern!r} matched no files in the manifest."
        if suggestions:
            msg += " Did you mean:\n  " + "\n  ".join(suggestions)
        super().__init__(msg)


# ── Cache / Lake ──────────────────────────────────────────────────────────────

class CacheError(QortexError):
    """Raised for cache registry or integrity issues."""


class DatasetNotDownloadedError(QortexError):
    """Raised when a local operation is attempted on a dataset not yet fetched."""

    def __init__(self, dataset_id: str, snapshot: str | None = None) -> None:
        snap_str = f" (snapshot {snapshot!r})" if snapshot else ""
        super().__init__(
            f"Dataset {dataset_id!r}{snap_str} is not downloaded. "
            f"Run `ds.download()` or `qortex download {dataset_id}` first."
        )


# ── NeuroAI Runtime ───────────────────────────────────────────────────────────

class CompatibilityError(QortexError):
    """Raised when a source cannot satisfy a model's input contract."""

    def __init__(self, source_id: str, model_id: str, blockers: list | None = None) -> None:
        self.source_id = source_id
        self.model_id = model_id
        self.blockers = blockers or []
        blocker_str = "\n  ".join(self.blockers) if self.blockers else "see CompatibilityReport"
        super().__init__(
            f"Source {source_id!r} is incompatible with model {model_id!r}.\n"
            f"Blockers:\n  {blocker_str}"
        )


class SourceAdapterError(QortexError):
    """Raised when a source adapter cannot probe or stream data."""

    def __init__(self, message: str, source_type: str | None = None, path: str | None = None) -> None:
        self.source_type = source_type
        self.path = path
        ctx = (f" [type={source_type!r}]" if source_type else "") + (f" [path={path!r}]" if path else "")
        super().__init__(f"SourceAdapter error{ctx}: {message}")


class ModelAdapterError(QortexError):
    """Raised when a model adapter cannot inspect, load, or run inference."""

    def __init__(self, message: str, model_id: str | None = None, provider: str | None = None) -> None:
        self.model_id = model_id
        self.provider = provider
        ctx = (f" [provider={provider!r}]" if provider else "") + (f" [model={model_id!r}]" if model_id else "")
        super().__init__(f"ModelAdapter error{ctx}: {message}")


class PreprocessPlanningError(QortexError):
    """Raised when the preprocessing planner cannot build a valid transform chain."""


class OutputAdapterError(QortexError):
    """Raised when an output adapter cannot open, write, or close its destination."""

    def __init__(self, message: str, output_type: str | None = None, path: str | None = None) -> None:
        self.output_type = output_type
        self.path = path
        ctx = (f" [type={output_type!r}]" if output_type else "") + (f" [path={path!r}]" if path else "")
        super().__init__(f"OutputAdapter error{ctx}: {message}")


class RuntimeExecutionError(QortexError):
    """Raised when the NeuroAI runtime encounters a fatal execution error."""


class ContractValidationError(QortexError):
    """Raised when a contract fails validation."""

    def __init__(self, contract_type: str, violations: list) -> None:
        self.contract_type = contract_type
        self.violations = violations
        super().__init__(f"{contract_type} validation failed:\n  " + "\n  ".join(violations))
