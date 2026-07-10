"""DICOM SEG output adapter.

Writes model segmentation masks as DICOM Segmentation Storage objects
using highdicom, with geometry validation against the source DICOM series.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter, OutputAdapterError

log = logging.getLogger(__name__)


class DICOMSEGOutputAdapter(OutputAdapter):
    """Output adapter that writes segmentation masks as DICOM SEG files.

    Parameters
    ----------
    path:
        Output file path (``*.dcm``) or directory.
    source_series_dir:
        Path to the source DICOM series folder.  When provided, the SEG
        object references the source instances for geometry consistency.
    pipeline_ref:
        Short pipeline reference for provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_series_dir: str | Path | None = None,
        pipeline_ref: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._source_series_dir = Path(source_series_dir) if source_series_dir else None
        self._pipeline_ref = pipeline_ref
        self._n_written = 0
        self._source_datasets: list | None = None

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        if self._path.suffix:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._path.mkdir(parents=True, exist_ok=True)

        if self._source_series_dir and self._source_series_dir.exists():
            self._source_datasets = _load_source_series(self._source_series_dir)

        log.info("DICOM SEG output ready: %s", self._path)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        hd = _require_highdicom()
        _require_pydicom()
        meta = metadata or {}

        mask = output.mask
        if mask is None:
            raise OutputAdapterError("DICOM SEG output requires a segmentation mask.")

        mask_arr = np.array(mask)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[np.newaxis, :, :]  # [1, H, W]

        # Validate shape against source series
        if self._source_datasets:
            expected_slices = len(self._source_datasets)
            expected_h = int(self._source_datasets[0].Rows)
            expected_w = int(self._source_datasets[0].Columns)
            if mask_arr.shape != (expected_slices, expected_h, expected_w):
                raise OutputAdapterError(
                    "DICOM SEG geometry mismatch: "
                    f"mask shape {mask_arr.shape} does not match source series "
                    f"{(expected_slices, expected_h, expected_w)}."
                )

        class_name = output.class_name or meta.get("class_name", "Segmentation")
        segment_label = str(class_name)

        try:
            desc = hd.seg.SegmentDescription(
                segment_number=1,
                segment_label=segment_label,
                segmented_property_category=hd.sr.coding.codes.SCT.MorphologicallyAbnormalStructure,
                segmented_property_type=hd.sr.coding.codes.SCT.Nodule,
                algorithm_type=hd.seg.SegmentAlgorithmTypes.AUTOMATIC,
                algorithm_identification=hd.AlgorithmIdentificationSequence(
                    name="Qortex",
                    family=hd.sr.coding.codes.DCM.ArtificialIntelligence,
                    version="1.0",
                ),
            )

            source_images = self._source_datasets if self._source_datasets else []
            seg = hd.seg.Segmentation(
                source_images=source_images,
                pixel_array=mask_arr.astype(np.uint8),
                segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
                segment_descriptions=[desc],
                series_instance_uid=hd.UID(),
                series_number=100 + self._n_written,
                sop_instance_uid=hd.UID(),
                instance_number=1,
                manufacturer="Qortex",
                manufacturer_model_name="Qortex NeuroAI",
                software_versions="1.0",
                device_serial_number="0001",
                content_creator_name="Qortex",
            )

        except Exception as exc:
            raise OutputAdapterError(f"DICOM SEG creation failed: {exc}") from exc

        out_path = self._out_path()
        seg.save_as(str(out_path))
        _validate_written_dicom(out_path, expected_modality="SEG")
        self._n_written += 1
        log.info("DICOM SEG saved: %s", out_path.name)

    def close(self) -> None:
        log.info("DICOM SEG output adapter closed (%d files written)", self._n_written)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _out_path(self) -> Path:
        if self._path.suffix:
            if self._n_written == 0:
                return self._path
            stem = self._path.stem
            return self._path.parent / f"{stem}_{self._n_written:04d}.dcm"
        return self._path / f"seg_{self._n_written:04d}.dcm"

def _load_source_series(folder: Path) -> list:
    try:
        import pydicom
        files = sorted(folder.rglob("*.dcm"))
        datasets = []
        for f in files:
            try:
                datasets.append(pydicom.dcmread(str(f)))
            except Exception:
                pass
        def _sort_key(ds):
            return int(getattr(ds, "InstanceNumber", 0))
        return sorted(datasets, key=_sort_key)
    except Exception:
        return []


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


def _require_highdicom():
    try:
        import highdicom as hd
        return hd
    except ImportError:
        raise ImportError(
            "DICOM SEG output requires highdicom. "
            "Install with: pip install 'qortex[dicom]' or pip install highdicom"
        )


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError(
            "DICOM SEG output requires pydicom. "
            "Install with: pip install 'qortex[dicom]' or pip install pydicom"
        )
