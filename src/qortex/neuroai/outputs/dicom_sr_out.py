"""DICOM SR (Structured Report) output adapter.

Writes model classification / report results as TID 1500 Measurement
Report DICOM SR objects using highdicom.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter, OutputAdapterError

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

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        if self._path.suffix:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._path.mkdir(parents=True, exist_ok=True)
        log.info("DICOM SR output ready: %s", self._path)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        hd = _require_highdicom()

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
            raise OutputAdapterError(f"DICOM SR creation failed: {exc}") from exc

        out_path = self._out_path()
        sr.save_as(str(out_path))
        _validate_written_dicom(out_path, expected_modality="SR")
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

def _require_highdicom():
    try:
        import highdicom as hd
        return hd
    except ImportError:
        raise ImportError(
            "DICOM SR output requires highdicom. "
            "Install with: pip install 'qortex[dicom]' or pip install highdicom"
        )


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "DICOM SR output requires pydicom. "
            "Install with: pip install 'qortex[dicom]' or pip install pydicom"
        )


def _validate_written_dicom(path: Path, *, expected_modality: str) -> None:
    pydicom = _require_pydicom()
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception as exc:
        raise OutputAdapterError(f"Cannot reopen written DICOM output {path}: {exc}") from exc
    modality = getattr(dataset, "Modality", None)
    if modality != expected_modality:
        raise OutputAdapterError(
            f"Written DICOM output modality is {modality!r}; expected {expected_modality!r}."
        )
