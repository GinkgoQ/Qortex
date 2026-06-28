"""Image quality control for MRI, fMRI, DWI, and other volumetric modalities.

Computes no-reference QC metrics from NIfTI images.
Requires: numpy.  Optional: nibabel (for full-image QC).

All metrics are numerical evidence — no clinical interpretation.

Install extras:
    pip install 'qortex[mri]'
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from qortex.neuroclassic._base import (
    MethodConfidence,
    MetricResult,
    NeuroClassicReport,
    NeuroClassicResult,
    NeuroClassicSpec,
    _timer,
)

__version__ = "0.1.0"

_SPEC = NeuroClassicSpec(
    method_name="image_qc",
    modality="mri",
    target_workflow="visualize,convert,train",
    required_evidence=["image_array", "voxel_sizes_mm", "affine"],
    optional_evidence=["mask", "tr_s", "n_volumes"],
    assumptions=[
        "Image is in RAS orientation (or orientation is known).",
        "Input array is float32 or float64.",
        "Voxel sizes are in millimeters.",
    ],
    invalid_input_states=[
        "Zero-size array",
        "NaN or Inf in the entire array",
        "Constant image (std = 0)",
        "Non-finite voxel sizes",
    ],
)


@dataclass
class ImageQualityReport:
    """Per-image QC report for one NIfTI volume or fMRI series."""
    scope: str
    shape: tuple
    voxel_sizes_mm: tuple | None
    n_volumes: int | None
    tr_s: float | None

    # Global statistics
    image_min: float | None = None
    image_max: float | None = None
    image_mean: float | None = None
    image_std: float | None = None
    image_median: float | None = None

    # QC flags
    has_nan: bool = False
    has_inf: bool = False
    is_constant: bool = False
    foreground_fraction: float | None = None  # fraction of voxels with signal

    # fMRI-specific
    tsnr: float | None = None              # temporal SNR across volumes
    volume_outlier_indices: list[int] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_result(self) -> NeuroClassicResult:
        metrics = [
            MetricResult("shape", list(self.shape)),
            MetricResult("voxel_sizes_mm", list(self.voxel_sizes_mm) if self.voxel_sizes_mm else None, unit="mm"),
            MetricResult("image_min", self.image_min),
            MetricResult("image_max", self.image_max),
            MetricResult("image_mean", self.image_mean),
            MetricResult("image_std", self.image_std),
            MetricResult("has_nan", self.has_nan),
            MetricResult("has_inf", self.has_inf),
            MetricResult("is_constant", self.is_constant),
            MetricResult("foreground_fraction", self.foreground_fraction),
        ]
        if self.tsnr is not None:
            metrics.append(MetricResult("tsnr", self.tsnr,
                                        interpretation="Temporal SNR across volumes"))
        if self.volume_outlier_indices:
            metrics.append(MetricResult("volume_outlier_indices", self.volume_outlier_indices))

        return NeuroClassicResult(
            method_name="image_qc",
            method_version=__version__,
            modality="mri",
            scope=self.scope,
            inputs={"shape": list(self.shape)},
            parameters={},
            assumptions=_SPEC.assumptions,
            metrics=metrics,
            warnings=self.warnings,
            blockers=self.blockers,
            unknowns=self.unknowns,
            runtime_s=self.runtime_s,
            confidence=self.confidence,
            provenance={"method": "image_qc", "version": __version__},
        )

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "shape": list(self.shape),
            "voxel_sizes_mm": list(self.voxel_sizes_mm) if self.voxel_sizes_mm else None,
            "n_volumes": self.n_volumes,
            "tr_s": self.tr_s,
            "image_min": self.image_min,
            "image_max": self.image_max,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "image_median": self.image_median,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "is_constant": self.is_constant,
            "foreground_fraction": self.foreground_fraction,
            "tsnr": self.tsnr,
            "volume_outlier_indices": self.volume_outlier_indices,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "runtime_s": self.runtime_s,
            "confidence": self.confidence.value,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def compute_image_qc(
    data: np.ndarray,
    *,
    voxel_sizes_mm: tuple[float, ...] | None = None,
    affine: np.ndarray | None = None,
    scope: str = "unknown",
    background_threshold_pct: float = 10.0,
    tsnr_robust: bool = True,
) -> ImageQualityReport:
    """Compute QC metrics for a 3-D or 4-D NIfTI image array.

    Parameters
    ----------
    data:
        Image array, shape [x, y, z] or [x, y, z, t].
    voxel_sizes_mm:
        Voxel dimensions in mm.
    affine:
        4×4 affine matrix (optional; used for orientation checks).
    scope:
        Identifier for this image (file path).
    background_threshold_pct:
        Voxels below this percentile of the image max are treated as background.
    tsnr_robust:
        If True, use median/IQR-based tSNR estimate (more robust to outlier volumes).

    Returns
    -------
    ImageQualityReport
    """
    t0 = time.perf_counter()
    ndim = data.ndim

    if ndim not in (3, 4):
        raise ValueError(f"data must be 3-D or 4-D; got shape {data.shape}")

    shape = data.shape
    n_volumes = shape[3] if ndim == 4 else None
    is_fmri = ndim == 4

    report = ImageQualityReport(
        scope=scope,
        shape=shape,
        voxel_sizes_mm=voxel_sizes_mm,
        n_volumes=n_volumes,
        tr_s=None,
    )

    if data.size == 0:
        report.blockers.append("Image data array is empty.")
        report.confidence = MethodConfidence.UNKNOWN
        report.runtime_s = time.perf_counter() - t0
        return report

    # ── NaN / Inf detection ───────────────────────────────────────────────────
    finite_mask = np.isfinite(data)
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())
    report.has_nan = n_nan > 0
    report.has_inf = n_inf > 0

    if n_nan > 0:
        report.warnings.append(f"{n_nan} NaN voxels detected ({n_nan / data.size * 100:.2f}%).")
    if n_inf > 0:
        report.blockers.append(f"{n_inf} Inf voxels detected.")

    # ── Global statistics (on finite voxels) ─────────────────────────────────
    finite_data = data[finite_mask]
    if finite_data.size == 0:
        report.blockers.append("No finite voxels remain after NaN/Inf removal.")
        report.confidence = MethodConfidence.UNKNOWN
        report.runtime_s = time.perf_counter() - t0
        return report

    report.image_min = float(finite_data.min())
    report.image_max = float(finite_data.max())
    report.image_mean = float(finite_data.mean())
    report.image_std = float(finite_data.std())
    report.image_median = float(np.median(finite_data))

    if report.image_std == 0.0:
        report.is_constant = True
        report.blockers.append("Image is constant (std = 0); no signal variation detected.")
        report.confidence = MethodConfidence.UNKNOWN
        report.runtime_s = time.perf_counter() - t0
        return report

    # ── Foreground fraction ───────────────────────────────────────────────────
    if report.image_max > 0:
        threshold = np.percentile(finite_data, background_threshold_pct)
        foreground_mask = finite_data > threshold
        report.foreground_fraction = float(foreground_mask.sum() / finite_data.size)
        if report.foreground_fraction < 0.01:
            report.warnings.append(
                f"Foreground fraction is very low ({report.foreground_fraction:.4f}). "
                "Image may be mostly background or incorrectly masked."
            )

    # ── fMRI-specific: tSNR and volume outliers ───────────────────────────────
    if is_fmri and n_volumes and n_volumes >= 4:
        report.tsnr, report.volume_outlier_indices = _compute_tsnr_and_outliers(
            data, robust=tsnr_robust
        )
        if report.volume_outlier_indices:
            report.warnings.append(
                f"{len(report.volume_outlier_indices)} volume outliers detected "
                f"(indices: {report.volume_outlier_indices[:10]})."
            )
        if report.tsnr is not None and report.tsnr < 30:
            report.warnings.append(
                f"tSNR = {report.tsnr:.1f} is below 30; may indicate acquisition issues."
            )
    elif is_fmri:
        report.unknowns.append(
            f"fMRI has only {n_volumes} volumes; tSNR estimate requires ≥ 4."
        )
        report.confidence = MethodConfidence.LOW_CONFIDENCE

    report.runtime_s = time.perf_counter() - t0
    return report


def run_image_qc_on_dataset(
    dataset_path: Path,
    *,
    modality: str = "mri",
    max_files: int | None = None,
) -> NeuroClassicReport:
    """Run image QC across all NIfTI files in a BIDS dataset.

    Requires nibabel: pip install 'qortex[mri]'
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "Image QC requires nibabel. Install with: pip install 'qortex[mri]'"
        ) from None

    nc_report = NeuroClassicReport(
        method_name="image_qc",
        method_version=__version__,
        modality=modality,
        dataset_path=str(dataset_path),
        spec=_SPEC,
    )

    files = sorted(Path(dataset_path).rglob("*.nii")) + sorted(Path(dataset_path).rglob("*.nii.gz"))
    if max_files:
        files = files[:max_files]

    for f in files:
        t0 = time.perf_counter()
        try:
            img = nib.load(str(f))
            img = nib.as_closest_canonical(img)
            data = img.get_fdata(dtype=np.float32)
            zooms = img.header.get_zooms()
            vox = tuple(abs(float(v)) for v in zooms[:3])
            affine = img.affine
            qc = compute_image_qc(data, voxel_sizes_mm=vox, affine=affine, scope=str(f))
            qc.runtime_s += time.perf_counter() - t0
            nc_report.add_result(qc.to_result())
        except Exception as exc:
            result = NeuroClassicResult(
                method_name="image_qc",
                method_version=__version__,
                modality=modality,
                scope=str(f),
                inputs={},
                parameters={},
                assumptions=[],
                blockers=[f"Could not load image: {exc}"],
                confidence=MethodConfidence.UNKNOWN,
            )
            nc_report.add_result(result)

    return nc_report


# ── fMRI helpers ──────────────────────────────────────────────────────────────

def _compute_tsnr_and_outliers(
    data: np.ndarray,
    *,
    robust: bool = True,
    outlier_threshold_sd: float = 3.0,
) -> tuple[float | None, list[int]]:
    """Compute temporal SNR and detect outlier volumes.

    tSNR = mean(signal) / std(signal) across the time dimension,
    averaged over foreground voxels.
    """
    if data.ndim != 4:
        return None, []
    n_vol = data.shape[3]
    if n_vol < 4:
        return None, []

    # Global mean timeseries (mean over x, y, z)
    with np.errstate(invalid="ignore"):
        ts_mean = np.nanmean(data, axis=(0, 1, 2))   # [n_vol]
        ts_std = np.nanstd(ts_mean)
        ts_med = np.nanmedian(ts_mean)

    # tSNR on per-voxel basis (foreground only)
    signal_mean = np.nanmean(data, axis=3)
    signal_std = np.nanstd(data, axis=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        voxel_tsnr = np.where(signal_std > 0, signal_mean / signal_std, np.nan)
    fg_mask = signal_mean > (np.nanmax(signal_mean) * 0.1)
    tsnr_vals = voxel_tsnr[fg_mask]
    tsnr = float(np.nanmedian(tsnr_vals)) if tsnr_vals.size > 0 else None

    # Volume outliers
    outliers: list[int] = []
    if ts_std > 0:
        z_scores = (ts_mean - ts_med) / ts_std
        outliers = [i for i, z in enumerate(z_scores) if abs(z) > outlier_threshold_sd]

    return tsnr, outliers
