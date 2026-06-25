"""Shared HTTP transport layer for the OpenNeuro GraphQL client.

Provides:
  - A reusable synchronous httpx.Client with retry / backoff.
  - A reusable asynchronous httpx.AsyncClient for the download engine.
  - System trust-store SSL context (via truststore, with fallback).
  - User-agent header injection.
  - 429 Retry-After handling.
"""

from __future__ import annotations

import ssl
import time

import httpx

from qortex._version import __version__
from qortex.core.config import QortexConfig, get_config
from qortex.core.exceptions import NetworkError, RateLimitError

# ── SSL context ───────────────────────────────────────────────────────────────

def _build_ssl_context() -> ssl.SSLContext:
    """Return a system-trust-store SSL context with graceful fallback."""
    try:
        import truststore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return ctx
    except (ImportError, OSError, ssl.SSLError):
        return ssl.create_default_context()


SSL_CONTEXT: ssl.SSLContext = _build_ssl_context()

USER_AGENT = f"qortex/{__version__}"

# HTTP status codes that are safe to retry
RETRYABLE_CODES = (408, 429, 500, 502, 503, 504, 522, 524)

# Exception types that indicate a transient network issue
RETRYABLE_EXCEPTIONS = (
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


# ── Retry helper ──────────────────────────────────────────────────────────────

def _backoff(attempt: int, base: float, maximum: float) -> float:
    """Exponential backoff with a cap."""
    return min(base * (2 ** attempt), maximum)


def _retry_after(response: httpx.Response) -> float | None:
    """Parse Retry-After header (seconds or HTTP-date)."""
    header = response.headers.get("retry-after")
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        # Could be an HTTP-date; just return a safe default
        return 30.0


# ── Synchronous client (for metadata / GraphQL) ───────────────────────────────

class SyncTransport:
    """Thin wrapper around httpx.Client with retry logic for GQL requests."""

    def __init__(self, config: QortexConfig | None = None) -> None:
        self._cfg = config or get_config()
        self._client = httpx.Client(
            verify=SSL_CONTEXT,
            headers={"user-agent": USER_AGENT},
            timeout=self._cfg.metadata_timeout,
        )

    def post(
        self,
        url: str,
        *,
        json: dict,
        cookies: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """POST with automatic retry/backoff.  Returns the final response."""
        cfg = self._cfg
        last_exc: Exception | None = None

        for attempt in range(cfg.max_retries + 1):
            try:
                response = self._client.post(
                    url,
                    json=json,
                    cookies=cookies or {},
                    timeout=timeout or cfg.metadata_timeout,
                )
            except RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt < cfg.max_retries:
                    delay = _backoff(attempt, cfg.retry_backoff_base, cfg.retry_backoff_max)
                    time.sleep(delay)
                    continue
                raise NetworkError(str(exc)) from exc

            if response.status_code == 429:
                delay = _retry_after(response) or _backoff(
                    attempt, cfg.retry_backoff_base, cfg.retry_backoff_max
                )
                if attempt < cfg.max_retries:
                    time.sleep(delay)
                    continue
                raise RateLimitError(retry_after=delay)

            if response.status_code in RETRYABLE_CODES:
                if attempt < cfg.max_retries:
                    delay = _backoff(attempt, cfg.retry_backoff_base, cfg.retry_backoff_max)
                    time.sleep(delay)
                    continue

            return response

        raise NetworkError(f"Request failed after {cfg.max_retries} retries: {last_exc}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SyncTransport":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Asynchronous client factory (for download engine) ────────────────────────

def build_async_client(
    config: QortexConfig | None = None,
    timeout: float | None = None,
) -> httpx.AsyncClient:
    """Return a configured httpx.AsyncClient for the download engine.

    The caller is responsible for using it as an async context manager.
    """
    cfg = config or get_config()
    return httpx.AsyncClient(
        verify=SSL_CONTEXT,
        headers={"user-agent": USER_AGENT},
        timeout=timeout or cfg.download_timeout,
        # Keep a generous connection pool — the semaphores control actual
        # parallelism, not the pool size.
        limits=httpx.Limits(
            max_connections=cfg.max_concurrent_downloads + cfg.max_concurrent_heads + 10,
            max_keepalive_connections=cfg.max_concurrent_downloads,
        ),
    )
