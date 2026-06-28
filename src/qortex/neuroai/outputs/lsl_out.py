"""LSL marker output adapter.

Pushes model prediction results as string markers onto an LSL outlet.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)


class LSLMarkerOutputAdapter(OutputAdapter):
    """Output adapter that pushes string markers to an LSL outlet.

    Parameters
    ----------
    stream_name:
        Name of the LSL stream to create.
    pipeline_ref:
        Short pipeline reference string used as LSL source_id.
    """

    def __init__(
        self,
        stream_name: str = "qortex_predictions",
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        self._stream_name = stream_name
        self._pipeline_ref = pipeline_ref or "qortex"
        self._outlet = None
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        pylsl = _require_pylsl()
        info = pylsl.StreamInfo(
            name=self._stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_string,
            source_id=f"qortex_{self._pipeline_ref}",
        )
        self._outlet = pylsl.StreamOutlet(info)
        log.info("LSL outlet opened: stream_name=%r source_id=qortex_%s",
                 self._stream_name, self._pipeline_ref)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if self._outlet is None:
            raise RuntimeError("LSLMarkerOutputAdapter: call open() first")

        # Build marker string
        trigger_value = (metadata or {}).get("trigger_value")
        if trigger_value is not None:
            marker = str(trigger_value)
        elif output.class_name is not None:
            marker = output.class_name
            if output.class_index is not None:
                marker = f"{output.class_name}:{output.class_index}"
        else:
            marker = output.output_type

        self._outlet.push_sample([marker])
        self._n_written += 1
        log.debug("LSL marker pushed: %r", marker)

    def write_marker(self, marker: Any) -> None:
        """Push a structured EventMarkerOutput to the LSL outlet.

        The marker is serialised as a compact JSON string so BCI receivers
        that understand JSON can decode the full payload; simple BCI2000-style
        receivers get a human-readable string.
        """
        if self._outlet is None:
            return
        label = getattr(marker, "label", "")
        confidence = getattr(marker, "confidence", None)
        emit = getattr(marker, "emit_payload", {})
        if emit:
            payload = json.dumps({"label": label, "confidence": confidence, **emit},
                                 ensure_ascii=False)
        else:
            payload = f"{label}:{confidence:.3f}" if confidence is not None else label
        self._outlet.push_sample([payload])
        log.debug("LSL trigger marker pushed: %r", payload)

    def close(self) -> None:
        if self._outlet is not None:
            del self._outlet
            self._outlet = None
        log.info("LSL outlet closed (pushed %d markers)", self._n_written)


def _require_pylsl():
    try:
        import pylsl
        return pylsl
    except ImportError:
        raise ImportError(
            "LSL output requires pylsl. "
            "Install with: pip install 'qortex[lsl]' or pip install pylsl"
        )
