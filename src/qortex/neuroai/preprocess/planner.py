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
    PreprocessPlan,
    SourceProfile,
    TransformDescriptor,
    TransformKind,
)

log = logging.getLogger(__name__)


class TransformError(RuntimeError):
    """Raised when a critical preprocessing transform fails.

    Unlike non-critical structural transforms (add_batch_dim, to_tensor),
    critical transforms — resample, reorient, normalize, rescale_intensity,
    cast_dtype, bandpass, channel_select, pad_or_crop — directly affect the
    numerical distribution of the data.  Silently passing data through when
    these fail would produce wrong inference results without any indication.

    Catch this at the pipeline level and record the window as dropped.
    """


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
        source_profile: "SourceProfile | None" = None,
        model_provider: str = "",
    ) -> PreprocessPlan:
        """Convert a CompatibilityReport into an executable PreprocessPlan.

        Only transforms that the ``CompatibilityEngine`` determined are required
        by the model's ``InputContract`` are included.  No "best practice"
        normalization or per-modality rescaling is inserted automatically —
        adding the wrong normalization destroys the distribution a model expects.

        Parameters
        ----------
        compat_report:
            Computed by ``CompatibilityEngine.check()``.
        window_duration_s:
            If set, a windowing transform is appended as the final step.
        source_profile:
            Used for informational logging only; no longer auto-inserts rescaling.
        model_provider:
            Controls whether ``to_tensor`` is emitted as numpy or torch.

        Returns
        -------
        PreprocessPlan
            Ordered, documented transform chain — contract-driven, not heuristic.
        """
        transforms = list(compat_report.required_transforms)

        # Append to_tensor (always last before window).
        # ``as_numpy=True`` when the model provider handles numpy natively (e.g.
        # ONNX Runtime, HuggingFace pipeline) and a torch import is not needed.
        if not any(_kind_str(t) == "to_tensor" for t in transforms):
            as_numpy = model_provider in (
                "huggingface", "hf", "transformers", "onnx", "onnxruntime"
            )
            transforms.append(TransformDescriptor(
                kind=TransformKind.to_tensor,
                required_by="runtime",
                params={"as_numpy": as_numpy},
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

    # Transforms that must succeed — silent skip would corrupt inference results.
    _CRITICAL_KINDS = frozenset({
        "resample", "resample_spatial", "reorient",
        "normalize", "rescale_intensity", "cast_dtype", "bandpass",
        "channel_select", "channel_map", "channel_reorder", "window",
        "pad_or_crop",
    })

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

        Raises
        ------
        TransformError
            When a critical transform (resample, reorient, normalize,
            rescale_intensity, cast_dtype, bandpass, channel_select,
            pad_or_crop) fails.  Non-critical structural transforms
            (add_batch_dim, add_channel_dim, to_tensor, window) log a
            warning and pass the data through unchanged.
        """
        arr = _coerce_numpy(data)

        for transform in self._plan.transforms:
            kind = _kind_str(transform)
            try:
                arr = self._apply_one(arr, kind, transform.params)
            except Exception as exc:
                if kind in self._CRITICAL_KINDS:
                    raise TransformError(
                        f"Critical transform '{kind}' failed: {exc}. "
                        "Pipeline aborted to prevent silent data corruption. "
                        "Check the transform parameters and input data shape."
                    ) from exc
                log.warning(
                    "Non-critical transform '%s' failed: %s — passing data unchanged.",
                    kind, exc,
                )

        return arr

    def _apply_one(self, arr: np.ndarray, kind: str, params: dict) -> np.ndarray:
        if kind == "channel_select":
            return _apply_channel_select(arr, params)

        elif kind == "channel_map":
            return _apply_channel_map(arr, params)

        elif kind == "channel_reorder":
            return _apply_channel_reorder(arr, params)

        elif kind == "bandpass":
            low_hz = params.get("low_hz")
            high_hz = params.get("high_hz")
            sfreq = params.get("sfreq", 1.0)
            if arr.ndim < 2:
                raise TransformError(f"bandpass expects at least 2D data, got shape {arr.shape}")
            if low_hz is None and high_hz is None:
                raise TransformError("bandpass requires low_hz, high_hz, or both")
            try:
                from scipy.signal import butter, sosfiltfilt
            except ImportError as exc:
                raise TransformError(
                    "bandpass requires scipy. Install scipy or remove the bandpass transform."
                ) from exc
            nyq = float(sfreq) / 2.0
            if nyq <= 0:
                raise TransformError(f"bandpass requires positive sfreq, got {sfreq!r}")
            low = (float(low_hz) / nyq) if low_hz is not None else None
            high = (float(high_hz) / nyq) if high_hz is not None else None
            if low is not None and not 0.0 < low < 1.0:
                raise TransformError(f"bandpass low_hz={low_hz!r} is outside valid range")
            if high is not None and not 0.0 < high < 1.0:
                raise TransformError(f"bandpass high_hz={high_hz!r} is outside valid range")
            if low is not None and high is not None and low >= high:
                raise TransformError("bandpass low_hz must be lower than high_hz")
            if low is not None and high is not None:
                sos = butter(5, [low, high], btype="bandpass", output="sos")
            elif low is not None:
                sos = butter(5, low, btype="highpass", output="sos")
            else:
                sos = butter(5, high, btype="lowpass", output="sos")
            arr = sosfiltfilt(sos, arr, axis=-1).astype(arr.dtype)
            return arr

        elif kind == "resample":
            from_hz = float(params.get("from_hz", 1))
            to_hz = float(params.get("to_hz", 1))
            if abs(from_hz - to_hz) > 0.01 and arr.ndim >= 2:
                try:
                    from scipy.signal import resample_poly
                    from math import gcd
                except ImportError as exc:
                    raise TransformError(
                        "resample requires scipy.signal.resample_poly. "
                        "Install scipy or provide source data at the model sampling rate."
                    ) from exc
                ratio_num = int(round(to_hz))
                ratio_den = int(round(from_hz))
                if ratio_num <= 0 or ratio_den <= 0:
                    raise TransformError(
                        f"resample requires positive rates, got from_hz={from_hz}, to_hz={to_hz}"
                    )
                g = gcd(ratio_num, ratio_den)
                arr = resample_poly(arr, ratio_num // g, ratio_den // g, axis=-1)
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
            from_frame = params.get("from", "LPS")
            to_frame = params.get("to", "RAS")
            return _reorient_volume(arr, from_frame=from_frame, to_frame=to_frame)

        elif kind == "to_tensor":
            # If data is already a torch tensor, leave it as-is.
            if hasattr(arr, "dtype") and type(arr).__module__.startswith("torch"):
                return arr
            as_numpy = params.get("as_numpy", False)
            if as_numpy:
                # Caller (e.g. HuggingFace pipeline) wants plain numpy, not a tensor.
                return np.ascontiguousarray(arr).astype(np.float32)
            try:
                import torch
                return torch.from_numpy(np.ascontiguousarray(arr)).float()
            except ImportError:
                # torch not installed — return float32 numpy; most adapters accept it.
                return np.ascontiguousarray(arr).astype(np.float32)

        elif kind == "window":
            raise TransformError(
                "The window transform is not executed in TransformExecutor. "
                "Windowing is performed by source adapters and recorded in runtime metadata."
            )

        raise TransformError(f"Unsupported transform kind: {kind!r}")


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


def _apply_channel_select(arr: np.ndarray, params: dict) -> np.ndarray:
    if arr.ndim < 2:
        raise TransformError(f"channel_select expects at least 2D data, got shape {arr.shape}")

    mode = params.get("mode")
    indices = params.get("indices")
    names = params.get("names") or params.get("keep")
    source_names = params.get("source_names")
    target_n = params.get("target_n")
    missing_policy = params.get("missing_policy", "error")

    if mode in (None, "indices") and indices is not None:
        idx = _validate_indices(indices, arr.shape[0], "channel_select.indices")
        return arr[idx]

    if mode in (None, "names") and names is not None:
        if not source_names:
            if indices is not None:
                idx = _validate_indices(indices, arr.shape[0], "channel_select.indices")
                return arr[idx]
            raise TransformError(
                "channel_select by names requires source_names or precomputed indices"
            )
        missing = [name for name in names if name not in source_names]
        if missing and missing_policy == "error":
            raise TransformError(f"channel_select missing required channel(s): {missing}")
        idx = [source_names.index(name) for name in names if name in source_names]
        if not idx:
            raise TransformError("channel_select produced an empty channel set")
        return arr[idx]

    if mode in (None, "first_n") and target_n is not None:
        n = int(target_n)
        if n <= 0:
            raise TransformError(f"channel_select target_n must be positive, got {target_n!r}")
        if n > arr.shape[0]:
            raise TransformError(
                f"channel_select target_n={n} exceeds available channels={arr.shape[0]}"
            )
        return arr[:n]

    raise TransformError(
        "channel_select requires one of: indices, names/source_names, or target_n"
    )


def _apply_channel_map(arr: np.ndarray, params: dict) -> np.ndarray:
    if arr.ndim < 2:
        raise TransformError(f"channel_map expects at least 2D data, got shape {arr.shape}")
    mapping = params.get("mapping")
    source_names = params.get("source_names") or params.get("names")
    target_names = params.get("target_names") or params.get("order")

    if mapping and source_names and target_names:
        resolved = []
        for target in target_names:
            source = mapping.get(target, target)
            if source not in source_names:
                raise TransformError(
                    f"channel_map target {target!r} maps to missing source {source!r}"
                )
            resolved.append(source_names.index(source))
        return arr[resolved]

    order = params.get("indices") or params.get("order")
    if order is not None and all(isinstance(v, int) for v in order):
        idx = _validate_indices(order, arr.shape[0], "channel_map.order")
        return arr[idx]

    missing = params.get("missing_channels")
    if missing:
        raise TransformError(
            "channel_map received missing_channels without an explicit mapping. "
            f"Missing: {missing}"
        )

    raise TransformError(
        "channel_map requires mapping + source_names + target_names, or integer order"
    )


def _apply_channel_reorder(arr: np.ndarray, params: dict) -> np.ndarray:
    if arr.ndim < 2:
        raise TransformError(f"channel_reorder expects at least 2D data, got shape {arr.shape}")
    order = params.get("indices") or params.get("order")
    if order is None:
        source_names = params.get("source_names")
        target_names = params.get("target_names")
        if source_names and target_names:
            missing = [name for name in target_names if name not in source_names]
            if missing:
                raise TransformError(f"channel_reorder missing channel(s): {missing}")
            order = [source_names.index(name) for name in target_names]
    if order is None:
        raise TransformError("channel_reorder requires order/indices or source_names+target_names")
    idx = _validate_indices(order, arr.shape[0], "channel_reorder.order")
    return arr[idx]


def _validate_indices(indices: Any, n_channels: int, label: str) -> list[int]:
    try:
        idx = [int(i) for i in indices]
    except Exception as exc:
        raise TransformError(f"{label} must be a list of integer indices") from exc
    if not idx:
        raise TransformError(f"{label} must not be empty")
    bad = [i for i in idx if i < 0 or i >= n_channels]
    if bad:
        raise TransformError(
            f"{label} contains out-of-range indices {bad}; available channels={n_channels}"
        )
    return idx


def _pad_or_crop(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Pad or crop an ndarray to exactly ``target_shape``."""
    result = np.zeros(target_shape, dtype=arr.dtype)
    slices_src = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    slices_dst = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    result[slices_dst] = arr[slices_src]
    return result


def _reorient_volume(arr: np.ndarray, from_frame: str, to_frame: str) -> np.ndarray:
    """Reorient a 3-D (or 4-D) volume between named coordinate frames.

    When nibabel is available, uses ``nibabel.orientations`` for any valid
    3-character orientation code pair (e.g. 'LPS'→'RAS', 'LAS'→'RPS', etc.).
    The array must be indexed so that axis 0, 1, 2 correspond to the first,
    second, and third characters of ``from_frame`` respectively.

    Without nibabel, only the LPS↔RAS pair is handled via a direct axis flip,
    and a warning is emitted because the flip assumes [i,j,k] axis order.

    Parameters
    ----------
    arr:
        Numpy array with ndim >= 3.  For 4-D arrays the extra axes (e.g. time)
        are left unchanged; reorientation is applied to the first 3 axes only.
    from_frame:
        3-character orientation code of the source (e.g. 'LPS', 'RAS').
    to_frame:
        3-character orientation code of the target.

    Raises
    ------
    TransformError
        When nibabel is not available and the frame pair is not LPS↔RAS.
    """
    if arr.ndim < 3:
        raise TransformError(
            f"reorient: expected a 3-D or 4-D volume but got shape {arr.shape}. "
            "Reorientation only applies to volumetric data."
        )

    from_up = from_frame.upper().strip()[:3]
    to_up = to_frame.upper().strip()[:3]

    if from_up == to_up:
        return arr

    try:
        from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
        from_ornt = axcodes2ornt(from_up)
        to_ornt = axcodes2ornt(to_up)
        transform = ornt_transform(from_ornt, to_ornt)
        return apply_orientation(arr, transform)
    except ImportError:
        if {from_up, to_up} == {"LPS", "RAS"}:
            log.warning(
                "reorient %r→%r: nibabel not installed. Using axis-flip approximation "
                "which assumes axis-0=L/R, axis-1=P/A, axis-2=S/I. Install nibabel "
                "for correct reorientation: pip install 'qortex[mri]'",
                from_frame, to_frame,
            )
            # LPS→RAS and RAS→LPS are both self-inverse: flip axes 0 and 1
            return arr[::-1, ::-1, ...].copy()
        raise TransformError(
            f"Cannot reorient {from_frame!r}→{to_frame!r}: nibabel is required for "
            "all frame pairs except LPS↔RAS. Install with: pip install 'qortex[mri]'"
        )
