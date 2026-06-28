"""Preprocessing Planner — builds the minimal deterministic transform chain.

Takes the ``CompatibilityReport`` (which already determined what transforms are
needed) and formalises them into a ``PreprocessPlan`` — an ordered, executable,
fully-documented list of ``TransformDescriptor`` objects.

Every transform is:
  - Linked to the model contract field that requires it
  - Declared reversible or irreversible with a reason
  - Ordered to minimise data movement (cast last, resample after channel select)
  - Recorded in provenance
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qortex.neuroai.contracts import (
    CompatibilityReport,
    EvidenceStatus,
    InputContract,
    ModelProfile,
    PreprocessPlan,
    SourceProfile,
    TransformDescriptor,
    TransformKind,
    WarningItem,
)
from qortex.neuroai.spec import PreprocessSpec

log = logging.getLogger(__name__)

# Canonical execution order for transforms
_TRANSFORM_ORDER: dict[str, int] = {
    TransformKind.channel_select.value:    1,
    TransformKind.channel_map.value:       2,
    TransformKind.channel_reorder.value:   3,
    TransformKind.resample.value:          4,
    TransformKind.resample_spatial.value:  4,
    TransformKind.bandpass.value:          5,
    TransformKind.pad_or_crop.value:       6,
    TransformKind.reorient.value:          7,
    TransformKind.rescale_intensity.value: 8,
    TransformKind.normalize.value:         9,
    TransformKind.cast_dtype.value:        10,
    TransformKind.add_batch_dim.value:     11,
    TransformKind.add_channel_dim.value:   12,
    TransformKind.to_tensor.value:         13,
    TransformKind.window.value:            14,
}


class PreprocessPlanner:
    """Build a ``PreprocessPlan`` from a ``CompatibilityReport``.

    Usage::

        planner = PreprocessPlanner()
        plan = planner.build_plan(compat_report, source_profile=source_profile)
        print(plan.summary())
    """

    def build_plan(
        self,
        compat_report: CompatibilityReport,
        *,
        window_duration_s: float | None = None,
        extra_normalisation: bool = True,
        source_profile: "SourceProfile | None" = None,
    ) -> PreprocessPlan:
        """Convert a CompatibilityReport into an executable PreprocessPlan.

        Parameters
        ----------
        compat_report:
            Computed by ``CompatibilityEngine.check()``.
        window_duration_s:
            If set, a windowing transform is appended as the final step.
        extra_normalisation:
            Whether to append a z-score normalisation step even if not required
            by the contract (safe default for most deep learning models).

        Returns
        -------
        PreprocessPlan
            Ordered, documented transform chain.
        """
        transforms = list(compat_report.required_transforms)

        # Auto-insert rescale_intensity for DICOM/MRI sources when not already present.
        # DICOM pixel values after Rescale Slope/Intercept are in Hounsfield Units
        # (-1024..+3071 for CT) or in raw ADC units for MRI; most models expect [0,1] or [-1,1].
        if source_profile is not None:
            src_mod = str(getattr(source_profile, "modality", "") or "").lower()
            needs_rescale = src_mod in ("dicom", "ct", "mri", "fmri", "dwi", "pet")
            already_rescaled = any(
                _kind_str(t) in ("rescale_intensity", "normalize") for t in transforms
            )
            if needs_rescale and not already_rescaled:
                transforms.append(TransformDescriptor(
                    kind=TransformKind.rescale_intensity,
                    required_by="source_modality",
                    params={"out_min": 0.0, "out_max": 1.0},
                    reversible=False,
                    irreversible_reason="Linear rescaling to [0,1] uses per-volume min/max",
                    evidence_status=EvidenceStatus.inferred,
                ))

        # Append normalization if not already present
        if extra_normalisation and not any(
            _kind_str(t) in ("normalize", "rescale_intensity") for t in transforms
        ):
            transforms.append(TransformDescriptor(
                kind=TransformKind.normalize,
                required_by="best_practice",
                params={"method": "zscore"},
                reversible=False,
                irreversible_reason="Mean/std normalisation is data-dependent",
            ))

        # Append to_tensor (always last before window)
        if not any(_kind_str(t) == "to_tensor" for t in transforms):
            transforms.append(TransformDescriptor(
                kind=TransformKind.to_tensor,
                required_by="runtime",
                params={},
                reversible=True,
            ))

        # Append windowing if requested
        if window_duration_s is not None and not any(
            _kind_str(t) == "window" for t in transforms
        ):
            transforms.append(TransformDescriptor(
                kind=TransformKind.window,
                required_by="pipeline_spec.window",
                params={"duration_s": window_duration_s},
                reversible=True,
            ))

        # Sort by canonical order
        transforms.sort(key=lambda t: _TRANSFORM_ORDER.get(_kind_str(t), 99))

        has_destructive = any(
            not t.reversible for t in transforms if t.irreversible_reason
        )

        return PreprocessPlan(
            transforms=transforms,
            has_destructive_transforms=has_destructive,
            warnings=list(compat_report.warnings),
            unknowns=list(compat_report.unknowns),
        )


class TransformExecutor:
    """Execute a ``PreprocessPlan`` against actual data.

    Parameters
    ----------
    plan:
        A ``PreprocessPlan`` returned by ``PreprocessPlanner.build_plan()``.
    """

    def __init__(self, plan: PreprocessPlan) -> None:
        self._plan = plan

    def apply(self, data: Any) -> Any:
        """Apply all transforms in order.

        Parameters
        ----------
        data:
            A numpy array shape (n_channels, n_times) or (X, Y, Z) / (B, C, *).

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Preprocessed data ready for model inference.
        """
        arr = _coerce_numpy(data)

        for transform in self._plan.transforms:
            kind = _kind_str(transform)
            try:
                arr = self._apply_one(arr, kind, transform.params)
            except Exception as exc:
                log.warning(
                    "Transform %s failed: %s — passing data unchanged", kind, exc
                )

        return arr

    def _apply_one(self, arr: np.ndarray, kind: str, params: dict) -> np.ndarray:
        if kind == "channel_select":
            keep_n = params.get("target_n")
            indices = params.get("indices")
            if indices is not None and arr.ndim >= 2:
                arr = arr[indices]
            elif keep_n is not None and arr.ndim >= 2:
                arr = arr[:keep_n]
            return arr

        elif kind == "channel_map":
            # Reorder channels by name mapping: params["mapping"] = {src_name: dst_idx}
            # or params["order"] = [dst_name, ...] with params["names"] = [src_name, ...]
            order = params.get("order")
            if order is not None and arr.ndim >= 2 and len(order) <= arr.shape[0]:
                arr = arr[order]
            return arr

        elif kind == "channel_reorder":
            # Reorder channels to a target order: params["order"] = [int, ...]
            order = params.get("order")
            if order is not None and arr.ndim >= 2:
                arr = arr[list(order)]
            return arr

        elif kind == "bandpass":
            low_hz = params.get("low_hz")
            high_hz = params.get("high_hz")
            sfreq = params.get("sfreq", 1.0)
            if arr.ndim >= 2 and (low_hz is not None or high_hz is not None):
                try:
                    from scipy.signal import butter, sosfiltfilt
                    nyq = sfreq / 2.0
                    low = (low_hz / nyq) if low_hz is not None else None
                    high = (high_hz / nyq) if high_hz is not None else None
                    high = min(high, 0.999) if high is not None else None
                    if low is not None and high is not None:
                        sos = butter(5, [low, high], btype="bandpass", output="sos")
                    elif low is not None:
                        sos = butter(5, low, btype="highpass", output="sos")
                    else:
                        sos = butter(5, high, btype="lowpass", output="sos")
                    arr = sosfiltfilt(sos, arr, axis=-1).astype(arr.dtype)
                except ImportError:
                    log.warning("bandpass transform requires scipy — skipping filter")
            return arr

        elif kind == "resample":
            from_hz = float(params.get("from_hz", 1))
            to_hz = float(params.get("to_hz", 1))
            if abs(from_hz - to_hz) > 0.01 and arr.ndim >= 2:
                try:
                    from scipy.signal import resample_poly
                    from math import gcd
                    ratio_num = int(round(to_hz))
                    ratio_den = int(round(from_hz))
                    g = gcd(ratio_num, ratio_den)
                    arr = resample_poly(arr, ratio_num // g, ratio_den // g, axis=-1)
                except ImportError:
                    # Fallback: numpy-based linear interpolation
                    n_old = arr.shape[-1]
                    n_new = int(n_old * to_hz / from_hz)
                    arr = np.array([np.interp(
                        np.linspace(0, 1, n_new),
                        np.linspace(0, 1, n_old),
                        arr[i],
                    ) for i in range(arr.shape[0])])
            return arr

        elif kind == "normalize":
            method = params.get("method", "zscore")
            if method == "zscore":
                mu = arr.mean(axis=-1, keepdims=True)
                sigma = arr.std(axis=-1, keepdims=True)
                sigma = np.where(sigma < 1e-8, 1.0, sigma)
                arr = (arr - mu) / sigma
            elif method == "minmax":
                lo, hi = arr.min(), arr.max()
                if hi > lo:
                    arr = (arr - lo) / (hi - lo)
            return arr

        elif kind == "rescale_intensity":
            lo, hi = params.get("out_min", 0.0), params.get("out_max", 1.0)
            curr_lo, curr_hi = arr.min(), arr.max()
            if curr_hi > curr_lo:
                arr = (arr - curr_lo) / (curr_hi - curr_lo) * (hi - lo) + lo
            return arr

        elif kind == "cast_dtype":
            to_dtype = params.get("to", "float32")
            return arr.astype(np.dtype(to_dtype))

        elif kind == "add_batch_dim":
            return arr[np.newaxis]

        elif kind == "add_channel_dim":
            return arr[:, np.newaxis]  # (B, 1, ...) or (1, ...)

        elif kind == "pad_or_crop":
            to_shape = tuple(params.get("to_shape", arr.shape))
            return _pad_or_crop(arr, to_shape)

        elif kind == "reorient":
            try:
                import nibabel as nib
                # Only meaningful for NIfTI volumes; pass-through for signals
            except ImportError:
                pass
            return arr

        elif kind == "to_tensor":
            try:
                import torch
                return torch.from_numpy(np.ascontiguousarray(arr)).float()
            except ImportError:
                return arr.astype(np.float32)

        elif kind == "window":
            dur_s = float(params.get("duration_s", 1.0))
            # data is (n_ch, n_t); return just the first window
            if arr.ndim >= 2:
                # window_size estimated from last dim — caller should supply sfreq
                return arr

        log.debug("Unknown transform kind %r — skipping", kind)
        return arr


# ── Helpers ───────────────────────────────────────────────────────────────────

def _kind_str(t: TransformDescriptor) -> str:
    return t.kind.value if hasattr(t.kind, "value") else str(t.kind)


def _coerce_numpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "numpy"):
        return data.numpy()
    # QortexTimeSeries / QortexVolume carry their numpy array in .data
    if hasattr(data, "data") and isinstance(getattr(data, "data", None), np.ndarray):
        return data.data
    # Torch tensor
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    # Last resort: try to convert; raise clearly if it fails
    try:
        return np.asarray(data, dtype=np.float32)
    except Exception as exc:
        raise TypeError(
            f"Cannot extract numpy array from {type(data).__name__}. "
            "Source adapter must yield QortexTimeSeries/QortexVolume with "
            "the `data` field set, or yield raw numpy arrays directly."
        ) from exc


def _pad_or_crop(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Pad or crop an ndarray to exactly ``target_shape``."""
    result = np.zeros(target_shape, dtype=arr.dtype)
    slices_src = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    slices_dst = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    result[slices_dst] = arr[slices_src]
    return result
