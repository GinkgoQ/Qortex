"""Global configuration for Qortex.

All settings are read from environment variables (prefixed QORTEX_) and can
be overridden programmatically via `qortex.configure(...)`.

Environment variables:
    QORTEX_CACHE_DIR              Override default cache directory
    QORTEX_API_TOKEN              OpenNeuro API token
    QORTEX_MAX_CONCURRENT_DOWNLOADS
    QORTEX_MAX_CONCURRENT_HEADS
    QORTEX_MAX_RETRIES
    QORTEX_VERIFY_HASH            "true" / "false"
    QORTEX_VERIFY_SIZE            "true" / "false"
    QORTEX_GQL_ENDPOINT
    QORTEX_OPENNEURO_ENDPOINT
    QORTEX_TOTALSEGMENTATOR_LICENSE
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import platformdirs
from pydantic import Field, ValidationError as PydanticValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qortex.core.exceptions import ConfigurationError

log = logging.getLogger(__name__)


def _default_cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("qortex", appauthor=False))


class QortexConfig(BaseSettings):
    """Immutable configuration object.  Create via ``get_config()``."""

    model_config = SettingsConfigDict(
        env_prefix="QORTEX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )

    # ── Cache ────────────────────────────────────────────────────────────
    cache_dir: Annotated[Path, Field(default_factory=_default_cache_dir)]

    # ── Network ──────────────────────────────────────────────────────────
    openneuro_endpoint: str = "https://openneuro.org/"
    gql_endpoint: str = "https://openneuro.org/crn/graphql"

    # ── Concurrency ───────────────────────────────────────────────────────
    max_concurrent_downloads: Annotated[int, Field(ge=1, le=64)] = 8
    max_concurrent_heads: Annotated[int, Field(ge=1, le=256)] = 64

    # ── Retry policy ──────────────────────────────────────────────────────
    max_retries: Annotated[int, Field(ge=0, le=20)] = 5
    retry_backoff_base: Annotated[float, Field(ge=0.0)] = 0.5
    retry_backoff_max: Annotated[float, Field(ge=0.0)] = 60.0
    retry_codes: tuple[int, ...] = (408, 429, 500, 502, 503, 504, 522, 524)

    # ── Timeouts (seconds) ────────────────────────────────────────────────
    metadata_timeout: Annotated[float, Field(gt=0)] = 20.0
    download_timeout: Annotated[float, Field(gt=0)] = 120.0
    head_timeout: Annotated[float, Field(gt=0)] = 10.0

    # ── Integrity ─────────────────────────────────────────────────────────
    verify_hash: bool = True
    verify_size: bool = True

    # ── Defaults ──────────────────────────────────────────────────────────
    exclude_derivatives_default: bool = True

    # ── Auth ──────────────────────────────────────────────────────────────
    api_token: str | None = Field(default=None, alias="QORTEX_API_TOKEN")
    totalsegmentator_license: str | None = Field(
        default=None,
        alias="QORTEX_TOTALSEGMENTATOR_LICENSE",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def _expand_cache_dir(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    def redacted(self) -> dict[str, Any]:
        """Return config values safe for logs, reports, and CLI diagnostics."""
        data = self.model_dump(mode="python")
        if data.get("api_token"):
            data["api_token"] = "***"
        if data.get("totalsegmentator_license"):
            data["totalsegmentator_license"] = "***"
        return data

    def with_overrides(self, **kwargs) -> "QortexConfig":
        """Return a new config with specific fields replaced and validated."""
        unknown = sorted(set(kwargs) - set(self.model_fields))
        if unknown:
            raise ConfigurationError(
                f"Unknown Qortex configuration field(s): {', '.join(unknown)}",
                context={"unknown_fields": unknown},
                suggestion=f"Known fields: {', '.join(sorted(self.model_fields))}",
            )
        data = self.model_dump(mode="python")
        data.update(kwargs)
        try:
            return type(self).model_validate(data)
        except PydanticValidationError as exc:
            raise ConfigurationError(
                "Invalid Qortex configuration override",
                context={"errors": exc.errors(include_url=False), "overrides": _redact_mapping(kwargs)},
            ) from exc


# ── Module-level singleton ────────────────────────────────────────────────────

_config: QortexConfig | None = None


def get_config() -> QortexConfig:
    """Return the current global config, initialising from env/defaults if needed."""
    global _config
    if _config is None:
        try:
            _config = QortexConfig()
        except PydanticValidationError as exc:
            raise ConfigurationError(
                "Invalid Qortex configuration from environment or .env",
                context={"errors": exc.errors(include_url=False)},
            ) from exc
    return _config


def configure(**kwargs) -> None:
    """Override specific config fields globally.

    Example
    -------
    >>> import qortex
    >>> qortex.configure(max_concurrent_downloads=16, cache_dir="~/neurodata")
    """
    global _config
    _config = get_config().with_overrides(**kwargs)
    log.debug("Qortex configuration updated: %s", _config.redacted())


def reset_config() -> None:
    """Reset to defaults (mainly useful in tests)."""
    global _config
    _config = None


def _redact_mapping(values: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(values)
    for key in list(redacted):
        if (
            "token" in key.lower()
            or "password" in key.lower()
            or "secret" in key.lower()
            or "license" in key.lower()
        ):
            redacted[key] = "***"
    return redacted
