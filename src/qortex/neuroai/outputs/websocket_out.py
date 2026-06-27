"""WebSocket event output adapter.

Pushes model outputs as JSON messages to a WebSocket server.
Tries websockets (async) first, falls back to websocket-client (sync).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)


class WebSocketOutputAdapter(OutputAdapter):
    """Output adapter that sends prediction results over WebSocket.

    Parameters
    ----------
    url:
        WebSocket server URL (``ws://...`` or ``wss://...``).
    pipeline_ref:
        Short pipeline reference.
    """

    def __init__(
        self,
        url: str,
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        if not url:
            raise ValueError("WebSocketOutputAdapter requires a URL")
        self._url = url
        self._pipeline_ref = pipeline_ref
        self._ws = None
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        try:
            import websocket  # websocket-client (sync)
            self._ws = websocket.create_connection(self._url)
            self._backend = "websocket-client"
        except ImportError:
            raise ImportError(
                "WebSocket output requires websocket-client or websockets. "
                "Install with: pip install websocket-client"
            )
        except Exception as exc:
            raise ConnectionError(f"WebSocket connection to {self._url} failed: {exc}") from exc
        log.info("WebSocket connected: %s", self._url)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocketOutputAdapter: call open() first")
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

        try:
            self._ws.send(json.dumps(payload))
            self._n_written += 1
        except Exception as exc:
            log.warning("WebSocket send failed: %s", exc)

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
        log.info("WebSocket closed (%d messages sent)", self._n_written)
