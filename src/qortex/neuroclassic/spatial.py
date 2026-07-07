"""Spatial feature extraction for classical EEG workflows."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qortex.neuroclassic._base import (
    MethodConfidence,
    MetricResult,
    NeuroClassicResult,
    NeuroClassicSpec,
)

__version__ = "0.1.0"

_SPEC_CSP = NeuroClassicSpec(
    method_name="common_spatial_patterns",
    modality="eeg",
    target_workflow="convert,train",
    required_evidence=["epochs_array", "binary_labels"],
    optional_evidence=["channel_names"],
    assumptions=[
        "Epochs are [n_epochs, n_channels, n_times].",
        "CSP is fitted only on training data to avoid leakage.",
        "Returned features are log-variance features from projected epochs.",
    ],
    invalid_input_states=[
        "Non-binary labels",
        "Less than two epochs per class",
        "Constant or non-finite channel covariance",
    ],
)


@dataclass
class CSPReport:
    """Common Spatial Patterns filters and log-variance features.

    CSP finds spatial filters that maximise variance for one class while
    minimising it for the other.  It is a feature extractor for supervised
    binary EEG tasks, not a clinical interpretation.
    """
    scope: str
    classes: tuple[str, str]
    filters: np.ndarray
    patterns: np.ndarray
    features: np.ndarray
    selected_component_indices: list[int]
    channel_names: list[str]
    eigenvalues: list[float]
    regularization: float
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    confidence: MethodConfidence = MethodConfidence.HIGH

    @property
    def n_components(self) -> int:
        return int(self.filters.shape[0])

    @property
    def n_epochs(self) -> int:
        return int(self.features.shape[0])

    def transform(self, epochs: np.ndarray) -> np.ndarray:
        """Project new epochs with fitted filters and return log-variance features."""
        _validate_epochs(epochs, n_channels=len(self.channel_names))
        return _csp_features(epochs.astype(np.float64, copy=False), self.filters)

    def to_result(self) -> NeuroClassicResult:
        return NeuroClassicResult(
            method_name="common_spatial_patterns",
            method_version=__version__,
            modality="eeg",
            scope=self.scope,
            inputs={
                "n_epochs": self.n_epochs,
                "n_channels": len(self.channel_names),
                "classes": list(self.classes),
            },
            parameters={
                "n_components": self.n_components,
                "selected_component_indices": self.selected_component_indices,
                "regularization": self.regularization,
            },
            assumptions=_SPEC_CSP.assumptions,
            metrics=[
                MetricResult("n_components", self.n_components),
                MetricResult("n_epochs", self.n_epochs),
                MetricResult("eigenvalues", self.eigenvalues),
                MetricResult("feature_shape", list(self.features.shape)),
            ],
            warnings=self.warnings,
            runtime_s=self.runtime_s,
            confidence=self.confidence,
            provenance={"method": "common_spatial_patterns", "version": __version__},
        )

    def to_dict(self, include_arrays: bool = False) -> dict:
        data = {
            "scope": self.scope,
            "classes": list(self.classes),
            "n_components": self.n_components,
            "n_epochs": self.n_epochs,
            "selected_component_indices": self.selected_component_indices,
            "channel_names": self.channel_names,
            "eigenvalues": self.eigenvalues,
            "regularization": self.regularization,
            "warnings": self.warnings,
            "confidence": self.confidence.value,
        }
        if include_arrays:
            data["filters"] = self.filters.tolist()
            data["patterns"] = self.patterns.tolist()
            data["features"] = self.features.tolist()
        return data


def compute_common_spatial_patterns(
    epochs: np.ndarray,
    labels: list[str] | np.ndarray,
    *,
    channel_names: list[str] | None = None,
    n_components: int = 4,
    regularization: float = 1e-6,
    scope: str = "unknown",
) -> CSPReport:
    """Fit binary Common Spatial Patterns and return log-variance features.

    Parameters
    ----------
    epochs:
        Float array with shape ``[n_epochs, n_channels, n_times]``.
    labels:
        Binary class labels, one per epoch.  Fit on training epochs only.
    n_components:
        Number of spatial filters/features to return.  Components are selected
        symmetrically from both ends of the CSP eigenvalue spectrum.
    regularization:
        Diagonal covariance regularization added before whitening.
    """
    t0 = time.perf_counter()
    _validate_epochs(epochs)
    if regularization < 0:
        raise ValueError(f"regularization must be >= 0; got {regularization}")

    x = epochs.astype(np.float64, copy=False)
    n_epochs, n_channels, _n_times = x.shape
    y = np.asarray(labels)
    if y.shape[0] != n_epochs:
        raise ValueError(f"labels length {y.shape[0]} != n_epochs {n_epochs}")
    classes_arr = np.unique(y)
    if classes_arr.size != 2:
        raise ValueError(f"CSP requires exactly two classes; got {classes_arr.tolist()}")
    classes = (str(classes_arr[0]), str(classes_arr[1]))
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError(f"channel_names length {len(channel_names)} != n_channels {n_channels}")
    if not (1 <= n_components <= n_channels):
        raise ValueError(f"n_components must be in [1, {n_channels}]; got {n_components}")

    covs = []
    warnings: list[str] = []
    for class_value in classes_arr:
        class_epochs = x[y == class_value]
        if class_epochs.shape[0] < 2:
            raise ValueError(f"class {class_value!r} has fewer than two epochs")
        covs.append(_mean_normalized_covariance(class_epochs, regularization=regularization))

    cov_a, cov_b = covs
    composite = cov_a + cov_b
    evals, evecs = np.linalg.eigh(composite)
    keep = evals > np.finfo(np.float64).eps
    if keep.sum() < n_channels:
        warnings.append("Composite covariance was rank deficient; zero-variance dimensions were ignored.")
    if keep.sum() < 2:
        raise ValueError("Composite covariance is rank deficient; CSP cannot be fitted.")

    whitening = (evecs[:, keep] / np.sqrt(evals[keep])).T
    s_a = whitening @ cov_a @ whitening.T
    csp_evals, csp_vecs = np.linalg.eigh(s_a)
    order = np.argsort(csp_evals)[::-1]
    full_filters = csp_vecs[:, order].T @ whitening
    full_evals = csp_evals[order]

    selected = _select_csp_components(len(full_evals), n_components)
    filters = full_filters[selected]
    patterns = np.linalg.pinv(filters).T
    features = _csp_features(x, filters)

    confidence = MethodConfidence.HIGH
    if n_epochs < 10:
        confidence = MethodConfidence.LOW_CONFIDENCE
        warnings.append("Fewer than 10 epochs were used to fit CSP; filters are low-confidence.")

    return CSPReport(
        scope=scope,
        classes=classes,
        filters=filters,
        patterns=patterns,
        features=features,
        selected_component_indices=selected,
        channel_names=channel_names,
        eigenvalues=[float(full_evals[i]) for i in selected],
        regularization=regularization,
        runtime_s=time.perf_counter() - t0,
        warnings=warnings,
        confidence=confidence,
    )


def _validate_epochs(epochs: np.ndarray, *, n_channels: int | None = None) -> None:
    if epochs.ndim != 3:
        raise ValueError(f"epochs must be [n_epochs, n_channels, n_times]; got {epochs.shape}")
    if epochs.shape[0] == 0 or epochs.shape[1] == 0 or epochs.shape[2] < 2:
        raise ValueError(f"epochs must be non-empty with at least two time samples; got {epochs.shape}")
    if n_channels is not None and epochs.shape[1] != n_channels:
        raise ValueError(f"epochs n_channels {epochs.shape[1]} != fitted n_channels {n_channels}")
    if not np.isfinite(epochs).all():
        raise ValueError("epochs must contain only finite values")


def _mean_normalized_covariance(epochs: np.ndarray, *, regularization: float) -> np.ndarray:
    covs = []
    for epoch in epochs:
        centered = epoch - epoch.mean(axis=1, keepdims=True)
        cov = centered @ centered.T / max(centered.shape[1] - 1, 1)
        trace = float(np.trace(cov))
        if trace <= 0:
            continue
        covs.append(cov / trace)
    if not covs:
        raise ValueError("all epochs had zero covariance")
    mean_cov = np.mean(covs, axis=0)
    if regularization:
        mean_cov = mean_cov + np.eye(mean_cov.shape[0]) * regularization
    return mean_cov


def _select_csp_components(n_available: int, n_components: int) -> list[int]:
    selected: list[int] = []
    left = 0
    right = n_available - 1
    while len(selected) < n_components and left <= right:
        selected.append(left)
        if len(selected) < n_components and right != left:
            selected.append(right)
        left += 1
        right -= 1
    return selected


def _csp_features(epochs: np.ndarray, filters: np.ndarray) -> np.ndarray:
    projected = np.einsum("kc,ect->ekt", filters, epochs)
    variances = np.var(projected, axis=2)
    denom = np.sum(variances, axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return np.log(np.maximum(variances / denom, np.finfo(np.float64).tiny))
