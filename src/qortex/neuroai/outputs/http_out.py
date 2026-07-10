"""HTTP callback output adapter.

POSTs model outputs as JSON to an HTTP/HTTPS endpoint with retry logic
and optional Bearer or Basic authentication.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 0.5


class HTTPCallbackOutputAdapter(OutputAdapter):
    """Output adapter that POSTs predictions to an HTTP endpoint.

    Parameters
    ----------
    url:
        Full HTTP/HTTPS URL for the callback endpoint.
    pipeline_ref:
        Short pipeline reference.
    auth:
        Optional dict: ``{"type": "bearer", "token": "..."}`` or
        ``{"type": "basic", "username": "...", "password": "..."}``.
    """

    def __init__(
        self,
        url: str,
        *,
        pipeline_ref: str | None = None,
        auth: dict | None = None,
    ) -> None:
        if not url:
            raise ValueError("HTTPCallbackOutputAdapter requires a URL")
        self._url = url
        self._pipeline_ref = pipeline_ref
        self._auth = auth or {}
        self._session = None
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        try:
            import requests
        except ImportError:
            raise ImportError(
                "HTTP callback output requires requests. "
                "Install with: pip install requests"
            )
        self._session = requests.Session()

        auth_type = self._auth.get("type", "")
        if auth_type == "bearer":
            self._session.headers["Authorization"] = f"Bearer {self._auth.get('token', '')}"
        elif auth_type == "basic":
            from requests.auth import HTTPBasicAuth
            self._session.auth = HTTPBasicAuth(
                self._auth.get("username", ""), self._auth.get("password", "")
            )

        self._session.headers.setdefault("Content-Type", "application/json")
        log.info("HTTP callback output ready: %s", self._url)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if self._session is None:
            raise RuntimeError("HTTPCallbackOutputAdapter: call open() first")
        meta = metadata or {}

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_ref": self._pipeline_ref,
            "output_type": output.output_type,
            "class_name": output.class_name,
            "class_index": output.class_index,
            "probabilities": output.probabilities,
            "regression_value": output.regression_value,
            "source_id": meta.get("source_id"),
            "model_id": meta.get("model_id"),
            "window_index": meta.get("window_index"),
        }

        import time
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.post(self._url, json=payload, timeout=5.0)
                resp.raise_for_status()
                self._n_written += 1
                return
            except Exception as exc:
                if attempt < _MAX_RETRIES - 1:
                    log.warning("HTTP callback attempt %d failed: %s — retrying", attempt + 1, exc)
                    time.sleep(_RETRY_BACKOFF_S * (attempt + 1))
                else:
                    log.error("HTTP callback failed after %d retries: %s", _MAX_RETRIES, exc)

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        log.info("HTTP callback adapter closed (%d POSTs succeeded)", self._n_written)
