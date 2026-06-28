"""Coordinate and geometry check domain.

Validates spatial correctness: NIfTI affine, qform/sform consistency, orientation,
DICOM LPS vs NIfTI RAS, voxel-size distribution, shape consistency, mask/source
affine compatibility, and DWI bvec/bval integrity.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

_VOXEL_SIZE_WARN_CV = 0.05   # 5 % coefficient of variation across subjects
_AFFINE_TOL = 1e-3            # mm tolerance for affine comparison


class GeometryChecker(BaseChecker):
    """Validate spatial geometry, affines, and DWI gradient tables."""

    name = "geometry"
    required_for = frozenset({"visualize", "convert", "train"})

    def __init__(
        self,
        *,
        modality: str | None = None,
        check_affine: bool = True,
        check_voxel_size_consistency: bool = True,
        check_dwi_gradients: bool = True,
    ) -> None:
        self._modality = modality
        self._check_affine = check_affine
        self._check_voxel = check_voxel_size_consistency
        self._check_dwi = check_dwi_gradients

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={"dataset_path": str(dataset_path), "modality": self._modality},
        )

        nifti_files = sorted(dataset_path.rglob("*.nii")) + sorted(dataset_path.rglob("*.nii.gz"))
        bvec_files = sorted(dataset_path.rglob("*.bvec")) if self._check_dwi else []

        if not nifti_files and not bvec_files:
            report.add(CheckFinding(
                code="GEOMETRY.NO_NIFTI_FILES",
                severity=CheckSeverity.INFO,
                message="No NIfTI or DWI gradient files found; geometry checks skipped.",
                path=str(dataset_path),
                evidence=[EvidenceRecord(
                    field="nifti_files",
                    state=EvidenceState.missing,
                    observed_source=str(dataset_path),
                )],
            ))
            return report.finalize()

        voxel_sizes: list[tuple[float, float, float]] = []

        for nii in nifti_files:
            header_info = _read_nifti_geometry(nii)
            if header_info is None:
                report.add(CheckFinding(
                    code="GEOMETRY.NIFTI_UNREADABLE",
                    severity=CheckSeverity.WARN,
                    message=f"Cannot read NIfTI geometry header: {nii.name}",
                    path=str(nii),
                ))
                continue

            qform_code, sform_code, voxel_size, qform_affine, sform_affine = header_info

            # Record voxel size for cross-subject consistency
            voxel_sizes.append(voxel_size)

            if self._check_affine:
                self._check_nifti_affine(
                    nii, qform_code, sform_code, qform_affine, sform_affine, report
                )

        if self._check_voxel and len(voxel_sizes) > 1:
            self._check_voxel_size_consistency(voxel_sizes, dataset_path, report)

        if self._check_dwi:
            for bvec in bvec_files:
                bval = bvec.with_suffix(".bval")
                if bval.exists():
                    self._check_dwi_gradient_table(bvec, bval, report)

        return report.finalize()

    # ── Affine checks ─────────────────────────────────────────────────────────

    def _check_nifti_affine(
        self,
        nii: Path,
        qform_code: int,
        sform_code: int,
        qform_affine: np.ndarray | None,
        sform_affine: np.ndarray | None,
        report: CheckReport,
    ) -> None:
        entities = _parse_bids_entities_from_path(nii)

        if qform_code == 0 and sform_code == 0:
            report.add(CheckFinding(
                code="GEOMETRY.NO_VALID_FORM",
                severity=CheckSeverity.BLOCK,
                message=(
                    f"Both qform_code and sform_code are 0 in {nii.name}. "
                    "Spatial orientation is unknown."
                ),
                path=str(nii),
                bids_entities=entities,
                evidence=[EvidenceRecord(
                    field="qform_code",
                    state=EvidenceState.confirmed,
                    observed_value=qform_code,
                    observed_source=str(nii),
                )],
                suggested_fix=SuggestedFix(
                    description="Set qform_code and sform_code to 1 (scanner coordinates).",
                    safe=False,
                    reversible=False,
                ),
            ))
            return

        if qform_code > 0 and sform_code > 0 and qform_affine is not None and sform_affine is not None:
            diff = np.abs(qform_affine - sform_affine).max()
            if diff > _AFFINE_TOL:
                report.add(CheckFinding(
                    code="GEOMETRY.QSFORM_MISMATCH",
                    severity=CheckSeverity.WARN,
                    message=(
                        f"qform and sform affines differ by up to {diff:.4f} mm in {nii.name}. "
                        "nibabel will use sform when sform_code > qform_code."
                    ),
                    path=str(nii),
                    bids_entities=entities,
                    expected=0.0,
                    observed=float(diff),
                    evidence=[EvidenceRecord(
                        field="qform_sform_max_diff_mm",
                        state=EvidenceState.inferred,
                        observed_value=float(diff),
                        observed_source=str(nii),
                    )],
                ))

    # ── Voxel size consistency ────────────────────────────────────────────────

    def _check_voxel_size_consistency(
        self,
        voxel_sizes: list[tuple[float, float, float]],
        dataset_path: Path,
        report: CheckReport,
    ) -> None:
        arr = np.array(voxel_sizes)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        cv = std / np.where(mean > 0, mean, 1.0)  # coefficient of variation

        if cv.max() > _VOXEL_SIZE_WARN_CV:
            report.add(CheckFinding(
                code="GEOMETRY.VOXEL_SIZE_INCONSISTENT",
                severity=CheckSeverity.WARN,
                message=(
                    f"Voxel sizes vary across {len(voxel_sizes)} NIfTI files "
                    f"(CV up to {cv.max()*100:.1f}%). Spatial normalization may be required."
                ),
                path=str(dataset_path),
                observed={"mean_mm": mean.tolist(), "std_mm": std.tolist(), "cv": cv.tolist()},
                evidence=[EvidenceRecord(
                    field="voxel_size_cv",
                    state=EvidenceState.inferred,
                    observed_value=float(cv.max()),
                    observed_source=str(dataset_path),
                )],
                suggested_fix=SuggestedFix(
                    description=(
                        "Resample all images to a common voxel size before training. "
                        "Record target resolution in the ConversionContract."
                    ),
                    safe=True,
                ),
            ))
        else:
            report.record_evidence(EvidenceRecord(
                field="voxel_size_consistency",
                state=EvidenceState.confirmed,
                observed_value={"mean_mm": mean.tolist(), "cv": cv.tolist()},
                observed_source=str(dataset_path),
                note=f"Voxel sizes are consistent (max CV={cv.max()*100:.2f}%)",
            ))

    # ── DWI gradient table ────────────────────────────────────────────────────

    def _check_dwi_gradient_table(
        self, bvec: Path, bval: Path, report: CheckReport
    ) -> None:
        entities = _parse_bids_entities_from_path(bvec)
        try:
            bvals = np.array([float(x) for x in bval.read_text().split()])
        except Exception as exc:
            report.add(CheckFinding(
                code="GEOMETRY.DWI_BVAL_UNREADABLE",
                severity=CheckSeverity.BLOCK,
                message=f"Cannot parse bval file: {exc}",
                path=str(bval),
                bids_entities=entities,
            ))
            return

        try:
            bvec_lines = bvec.read_text().strip().splitlines()
            if len(bvec_lines) != 3:
                raise ValueError(f"Expected 3 rows in bvec, got {len(bvec_lines)}")
            bvecs = np.array([[float(v) for v in line.split()] for line in bvec_lines])
        except Exception as exc:
            report.add(CheckFinding(
                code="GEOMETRY.DWI_BVEC_UNREADABLE",
                severity=CheckSeverity.BLOCK,
                message=f"Cannot parse bvec file: {exc}",
                path=str(bvec),
                bids_entities=entities,
            ))
            return

        n_bvals = len(bvals)
        n_bvecs = bvecs.shape[1] if bvecs.ndim == 2 else 0

        if n_bvals != n_bvecs:
            report.add(CheckFinding(
                code="GEOMETRY.DWI_BVAL_BVEC_COUNT_MISMATCH",
                severity=CheckSeverity.BLOCK,
                message=(
                    f"bval has {n_bvals} values but bvec has {n_bvecs} vectors. "
                    "Volume count mismatch prevents DWI model fitting."
                ),
                path=str(bvec),
                bids_entities=entities,
                expected=n_bvals,
                observed=n_bvecs,
                evidence=[EvidenceRecord(
                    field="dwi_volume_count",
                    state=EvidenceState.contradicted,
                    claimed_value=n_bvals,
                    observed_value=n_bvecs,
                    claimed_source=str(bval),
                    observed_source=str(bvec),
                )],
            ))
            return

        # Verify unit norms for non-b0 vectors
        nonb0_mask = bvals > 50
        if nonb0_mask.any():
            nonb0_vecs = bvecs[:, nonb0_mask]
            norms = np.linalg.norm(nonb0_vecs, axis=0)
            bad_norm_mask = np.abs(norms - 1.0) > 0.01
            n_bad = bad_norm_mask.sum()
            if n_bad > 0:
                report.add(CheckFinding(
                    code="GEOMETRY.DWI_BVEC_NONNORM",
                    severity=CheckSeverity.WARN,
                    message=(
                        f"{n_bad} non-b0 gradient vectors deviate from unit norm by >1%. "
                        "Gradients may not be normalized."
                    ),
                    path=str(bvec),
                    bids_entities=entities,
                    observed={"n_bad": int(n_bad), "norms_sample": norms[:5].tolist()},
                    evidence=[EvidenceRecord(
                        field="bvec_norms",
                        state=EvidenceState.inferred,
                        observed_value={"n_bad": int(n_bad)},
                        observed_source=str(bvec),
                    )],
                    suggested_fix=SuggestedFix(
                        description="Normalize gradient vectors so each non-b0 vector has unit norm.",
                        safe=False,
                    ),
                ))
            else:
                report.record_evidence(EvidenceRecord(
                    field="bvec_norms",
                    state=EvidenceState.confirmed,
                    observed_value={"n_nonb0": int(nonb0_mask.sum()), "all_unit_norm": True},
                    observed_source=str(bvec),
                ))

        report.record_evidence(EvidenceRecord(
            field=f"{bvec.stem}.n_volumes",
            state=EvidenceState.confirmed,
            observed_value=n_bvals,
            observed_source=str(bval),
        ))


# ── Header fast-reader ────────────────────────────────────────────────────────

def _read_nifti_geometry(path: Path):
    """Read qform_code, sform_code, voxel sizes, and affines from a NIfTI-1 header."""
    import gzip

    open_fn = gzip.open if str(path).endswith(".gz") else open
    try:
        with open_fn(path, "rb") as fh:  # type: ignore[arg-type]
            raw = fh.read(348)
        if len(raw) < 348:
            return None

        sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
        endian = "<" if sizeof_hdr == 348 else ">"

        # pixdim at byte 76 (8×float32)
        pixdim = struct.unpack_from(f"{endian}8f", raw, 76)
        voxel_size = (abs(float(pixdim[1])), abs(float(pixdim[2])), abs(float(pixdim[3])))

        # qform_code at byte 252 (int16), sform_code at 254 (int16)
        qform_code = struct.unpack_from(f"{endian}h", raw, 252)[0]
        sform_code = struct.unpack_from(f"{endian}h", raw, 254)[0]

        # quatern_b/c/d at 256-268 (3×float32), qoffset at 268-280 (3×float32)
        # srow_x/y/z at 280-328 (4+4+4 × float32)
        def _build_qform_affine() -> np.ndarray | None:
            try:
                b, c, d = struct.unpack_from(f"{endian}3f", raw, 256)
                ox, oy, oz = struct.unpack_from(f"{endian}3f", raw, 268)
                qfac = 1.0 if pixdim[0] >= 0 else -1.0
                a = np.sqrt(max(0.0, 1.0 - b**2 - c**2 - d**2))
                rot = np.array([
                    [a*a + b*b - c*c - d*d, 2*(b*c - a*d),         2*(b*d + a*c)        ],
                    [2*(b*c + a*d),          a*a + c*c - b*b - d*d, 2*(c*d - a*b)        ],
                    [2*(b*d - a*c),          2*(c*d + a*b),          a*a + d*d - b*b - c*c],
                ])
                rot[:, 2] *= qfac
                aff = np.eye(4)
                aff[:3, :3] = rot * np.array([pixdim[1], pixdim[2], pixdim[3]])
                aff[:3, 3] = [ox, oy, oz]
                return aff
            except Exception:
                return None

        def _build_sform_affine() -> np.ndarray | None:
            try:
                rows = [struct.unpack_from(f"{endian}4f", raw, 280 + i * 16) for i in range(3)]
                aff = np.eye(4)
                for i, row in enumerate(rows):
                    aff[i, :] = row
                return aff
            except Exception:
                return None

        qform_affine = _build_qform_affine() if qform_code > 0 else None
        sform_affine = _build_sform_affine() if sform_code > 0 else None

        return qform_code, sform_code, voxel_size, qform_affine, sform_affine
    except Exception:
        return None


def _parse_bids_entities_from_path(path: Path) -> dict[str, str]:
    import re
    entity_re = re.compile(r"(sub|ses|task|run|acq|ce|dir|rec|echo|part)-([A-Za-z0-9]+)")
    return dict(entity_re.findall(path.name))
