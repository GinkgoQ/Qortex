"""DICOM folder source adapter.

Reads DICOM series from a local directory, assembles 3-D volumes with proper
Rescale Slope / Intercept applied, and builds a 4×4 affine from DICOM geometry
tags.

PHI handling
------------
PatientName, PatientID, PatientBirthDate, PatientSex, and PatientAge are
NEVER written to SourceProfile fields, logs, or any Qortex provenance record.
The source_id uses only the directory name (no patient-derived strings).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qortex.core.exceptions import SourceAdapterError
from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    Modality,
    QortexVolume,
    SourceProfile,
    WarningItem,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)

# DICOM tags that contain PHI — must never appear in logs or SourceProfile
_PHI_TAGS = frozenset([
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "PatientAge",
    "PatientWeight",
    "OtherPatientIDs",
    "PatientAddress",
    "ReferringPhysicianName",
    "RequestingPhysician",
    "PerformingPhysicianName",
    "InstitutionName",
    "InstitutionAddress",
])


class DICOMFolderAdapter(SourceAdapter):
    """Source adapter for a local DICOM folder.

    Scans for ``.dcm`` files, groups by SeriesInstanceUID, assembles volumes
    slice-by-slice with Rescale Slope / Intercept applied.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="dicom"`` and ``path=...`` pointing to a
        folder that contains ``.dcm`` files (flat or nested one level).
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        window_spec: WindowSpec | None = None,
        channel_names: list[str] | None = None,
    ) -> None:
        if not spec.path:
            raise SourceAdapterError(
                "DICOMFolderAdapter requires spec.path",
                source_type="dicom",
            )
        self._root = Path(spec.path).expanduser().resolve()
        if not self._root.exists():
            raise SourceAdapterError(
                f"DICOM folder not found: {self._root}",
                source_type="dicom",
                path=str(self._root),
            )
        self._spec = spec
        self._window_spec = window_spec

    # ── SourceAdapter interface ───────────────────────────────────────────────

    @property
    def source_id(self) -> str:
        return f"dicom:{self._root.name}"

    def probe(self) -> SourceProfile:
        pydicom = _require_pydicom()
        dcm_files = self._collect_dcm_files()
        if not dcm_files:
            raise SourceAdapterError(
                f"No .dcm files found under {self._root}",
                source_type="dicom",
                path=str(self._root),
            )

        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
        except Exception as exc:
            raise SourceAdapterError(
                f"Cannot read DICOM header from {dcm_files[0].name}: {exc}",
                source_type="dicom",
                path=str(self._root),
            ) from exc

        # Safe (non-PHI) tags only
        modality_tag = str(getattr(ds, "Modality", "unknown")).lower()
        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))
        n_slices = len(dcm_files)
        pixel_spacing = [float(v) for v in getattr(ds, "PixelSpacing", [1.0, 1.0])]
        slice_thickness = float(getattr(ds, "SliceThickness", 1.0))
        rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
        rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        series_desc = str(getattr(ds, "SeriesDescription", ""))  # not PHI

        # Log only non-PHI fields
        log.info(
            "DICOM probe: modality=%s rows=%d cols=%d slices=%d",
            modality_tag, rows, cols, n_slices,
        )

        warnings: list[WarningItem] = []
        if not hasattr(ds, "PixelSpacing"):
            warnings.append(WarningItem(
                code="MISSING_PIXEL_SPACING",
                message="PixelSpacing tag absent; defaulting to 1.0 mm.",
                severity="warning",
                evidence={},
                suggestion="Verify DICOM header or supply spacing manually.",
            ))

        return SourceProfile(
            source_id=self.source_id,
            source_type="dicom",
            modality=modality_tag,
            n_channels=1,
            sampling_rate_hz=None,
            spatial_shape=(n_slices, rows, cols),
            voxel_sizes_mm=(pixel_spacing[0], pixel_spacing[1], slice_thickness),
            dtype="float32",
            axis_convention=AxisConvention.spatial_zyx,
            path=str(self._root),
            warnings=warnings,
            evidence_status=EvidenceStatus.confirmed,
            extra={
                "pixel_spacing_mm": pixel_spacing,
                "slice_thickness_mm": slice_thickness,
                "rescale_slope": rescale_slope,
                "rescale_intercept": rescale_intercept,
                "series_description": series_desc,
                "series_count": len(self._group_by_series(dcm_files)),
                "phi_redacted": True,
            },
        )

    def read_batch(self) -> list[QortexData]:
        series_groups = self._group_by_series(self._collect_dcm_files())
        return [self._assemble_volume(files) for files in series_groups.values()]

    def stream(self) -> Iterator[QortexData]:
        series_groups = self._group_by_series(self._collect_dcm_files())
        for series_uid, files in series_groups.items():
            # Log only the truncated UID — no PHI
            log.info("Streaming DICOM series %s… (%d slices)", series_uid[:12], len(files))
            yield self._assemble_volume(files)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect_dcm_files(self) -> list[Path]:
        files = list(self._root.rglob("*.dcm"))
        if not files:
            files = list(self._root.rglob("*.DCM"))
        return sorted(files)

    def _group_by_series(self, dcm_files: list[Path]) -> dict[str, list[Path]]:
        pydicom = _require_pydicom()
        groups: dict[str, list[Path]] = {}
        for f in dcm_files:
            try:
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                uid = str(getattr(ds, "SeriesInstanceUID", "default"))
            except Exception:
                uid = "default"
            groups.setdefault(uid, []).append(f)
        return groups

    def _assemble_volume(self, files: list[Path]) -> QortexVolume:
        pydicom = _require_pydicom()
        slices = []
        affine = np.eye(4, dtype=np.float64)
        voxel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0)

        datasets = []
        for f in files:
            try:
                datasets.append(pydicom.dcmread(str(f)))
            except Exception as exc:
                log.warning("Could not read DICOM file %s: %s", f.name, exc)

        if not datasets:
            raise SourceAdapterError(
                "No readable DICOM slices found in series",
                source_type="dicom",
                path=str(self._root),
            )

        # Sort by InstanceNumber or z-position
        def _sort_key(ds):
            if hasattr(ds, "InstanceNumber"):
                return int(ds.InstanceNumber)
            if hasattr(ds, "ImagePositionPatient"):
                return float(ds.ImagePositionPatient[2])
            return 0

        datasets.sort(key=_sort_key)

        for ds in datasets:
            arr = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            slices.append(arr)

        volume = np.stack(slices, axis=0)  # [Z, Y, X]

        # Build affine from first slice (no PHI tags used here)
        ds0 = datasets[0]
        if hasattr(ds0, "ImageOrientationPatient") and hasattr(ds0, "ImagePositionPatient"):
            iop = [float(v) for v in ds0.ImageOrientationPatient]
            ipp = [float(v) for v in ds0.ImagePositionPatient]
            ps = [float(v) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
            st = float(getattr(ds0, "SliceThickness", 1.0))
            row_cos = np.array(iop[:3])
            col_cos = np.array(iop[3:])
            normal = np.cross(row_cos, col_cos)
            affine[:3, 0] = col_cos * ps[1]
            affine[:3, 1] = row_cos * ps[0]
            affine[:3, 2] = normal * st
            affine[:3, 3] = ipp
            voxel_sizes = (ps[0], ps[1], st)

        return QortexVolume(
            data=volume,
            shape=volume.shape,
            axes=["z", "y", "x"],
            dtype="float32",
            units="HU",
            affine=affine.tolist(),
            voxel_sizes_mm=voxel_sizes,
            coordinate_frame="patient_lps",
            source_provenance={
                "source_type": "dicom",
                "root": str(self._root),
                "n_slices": len(slices),
                "phi_redacted": True,
            },
        )


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise SourceAdapterError(
            "DICOM support requires pydicom. "
            "Install with: pip install 'qortex[dicom]'",
            source_type="dicom",
        )
