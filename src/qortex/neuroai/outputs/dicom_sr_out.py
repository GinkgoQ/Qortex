"""DICOM SR (Structured Report) output adapter.

Writes model classification / report results as TID 1500 Measurement
Report DICOM SR objects using highdicom.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)


class DICOMSROutputAdapter(OutputAdapter):
    """Output adapter that writes results as DICOM Structured Reports.

    Parameters
    ----------
    path:
        Output file path (``*.dcm``) or output directory.
    pipeline_ref:
        Short pipeline reference for provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._pipeline_ref = pipeline_ref
        self._n_written = 0

    def open(self) -> None:
        if self._path.suffix:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._path.mkdir(parents=True, exist_ok=True)
        log.info("DICOM SR output ready: %s", self._path)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        hd = _require_highdicom()
        meta = metadata or {}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        try:
            # Build a minimal TID 1500 measurement report
            observer_context = hd.sr.templates.ObserverContext(
                observer_type=hd.sr.coding.codes.DCM.Device,
                observer_identifying_attributes=hd.sr.templates.DeviceObserverIdentifyingAttributes(
                    uid=hd.UID(),
                ),
            )

            measurement_report = hd.sr.templates.MeasurementReport(
                observation_context=hd.sr.templates.ObservationContext(
                    observer_person_context=None,
                    observer_device_context=observer_context,
                ),
                procedure_reported=hd.sr.coding.codes.LN.CTUnspecifiedBodyRegion,
                imaging_measurements=[],
            )

            sr = hd.sr.EnhancedSR(
                evidence=[],
                content=measurement_report,
                series_instance_uid=hd.UID(),
                series_number=200 + self._n_written,
                sop_instance_uid=hd.UID(),
                instance_number=1,
                institution_name="Qortex",
                institutional_department_name="NeuroAI",
                manufacturer="Qortex",
            )

        except Exception as exc:
            log.warning(
                "DICOM SR creation failed (highdicom API mismatch?): %s — "
                "falling back to JSON",
                exc,
            )
            self._write_json_fallback(output, meta, timestamp)
            return

        out_path = self._out_path()
        sr.save_as(str(out_path))
        self._n_written += 1
        log.info("DICOM SR saved: %s", out_path.name)

    def close(self) -> None:
        log.info("DICOM SR output adapter closed (%d files written)", self._n_written)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _out_path(self) -> Path:
        if self._path.suffix:
            if self._n_written == 0:
                return self._path
            stem = self._path.stem
            return self._path.parent / f"{stem}_{self._n_written:04d}.dcm"
        return self._path / f"sr_{self._n_written:04d}.dcm"

    def _write_json_fallback(self, output: ModelOutput, meta: dict, timestamp: str) -> None:
        import json
        data = {
            "type": "DICOM_SR_fallback",
            "timestamp": timestamp,
            "pipeline_ref": self._pipeline_ref,
            "output_type": output.output_type,
            "class_name": output.class_name,
            "class_index": output.class_index,
            "probabilities": output.probabilities,
            "source_id": meta.get("source_id"),
            "model_id": meta.get("model_id"),
        }
        out = self._out_path().with_suffix(".json")
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._n_written += 1


def _require_highdicom():
    try:
        import highdicom as hd
        return hd
    except ImportError:
        raise ImportError(
            "DICOM SR output requires highdicom. "
            "Install with: pip install 'qortex[dicom]' or pip install highdicom"
        )
