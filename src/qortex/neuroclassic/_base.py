"""Base types for the qortex.neuroclassic classical methods layer.

Every classical method exposes a NeuroClassicSpec → NeuroClassicResult → NeuroClassicReport
pipeline.  All results carry method metadata, provenance, runtime cost, and structured
findings so they can feed into CheckReport, PreflightReport, and ArtifactWriter.
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MethodConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"   # small N, edge-case data
    UNKNOWN = "UNKNOWN"                  # evidence insufficient


@dataclass(frozen=True)
class MetricResult:
    """One named scalar or vector metric from a classical method."""
    name: str
    value: Any                    # scalar, list, dict, or None
    unit: str | None = None
    threshold: float | None = None
    threshold_source: str | None = None   # e.g. "ACNS 2017", "Qortex default"
    interpretation: str | None = None     # numerical description only — no clinical claims
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "threshold_source": self.threshold_source,
            "interpretation": self.interpretation,
            "confidence": self.confidence.value,
        }


@dataclass
class NeuroClassicSpec:
    """Configuration for running a classical neuroanalytic method.

    Every method must declare its target modality, required inputs, and parameters.
    Specs are immutable after construction and re-used across subjects/sessions.
    """
    method_name: str
    modality: str
    target_workflow: str                   # visualize, convert, train, neuroai-run
    required_evidence: list[str]
    optional_evidence: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    invalid_input_states: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "modality": self.modality,
            "target_workflow": self.target_workflow,
            "required_evidence": self.required_evidence,
            "optional_evidence": self.optional_evidence,
            "parameters": self.parameters,
            "assumptions": self.assumptions,
            "invalid_input_states": self.invalid_input_states,
        }


@dataclass
class NeuroClassicResult:
    """Raw output of one classical method run on one input unit (file/subject/session).

    Contains metrics, findings, and provenance — not rendered for display.
    """
    method_name: str
    method_version: str
    modality: str
    scope: str                         # e.g. "sub-01_task-rest_eeg.edf"
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    assumptions: list[str]
    metrics: list[MetricResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0
    confidence: MethodConfidence = MethodConfidence.HIGH

    def add_metric(self, m: MetricResult) -> None:
        self.metrics.append(m)

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "method_version": self.method_version,
            "modality": self.modality,
            "scope": self.scope,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "assumptions": self.assumptions,
            "metrics": [m.to_dict() for m in self.metrics],
            "warnings": self.warnings,
            "blockers": self.blockers,
            "unknowns": self.unknowns,
            "provenance": self.provenance,
            "runtime_s": self.runtime_s,
            "confidence": self.confidence.value,
        }


@dataclass
class NeuroClassicReport:
    """Aggregated report from a classical method across multiple input units.

    Equivalent to a CheckReport but specific to neuroclassic methods.
    """
    method_name: str
    method_version: str
    modality: str
    dataset_path: str
    spec: NeuroClassicSpec
    results: list[NeuroClassicResult] = field(default_factory=list)
    computed_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    total_runtime_s: float = 0.0

    def add_result(self, r: NeuroClassicResult) -> None:
        self.results.append(r)
        self.total_runtime_s += r.runtime_s

    @property
    def all_warnings(self) -> list[str]:
        return [w for r in self.results for w in r.warnings]

    @property
    def all_blockers(self) -> list[str]:
        return [b for r in self.results for b in r.blockers]

    @property
    def has_blockers(self) -> bool:
        return any(r.blockers for r in self.results)

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "method_version": self.method_version,
            "modality": self.modality,
            "dataset_path": self.dataset_path,
            "spec": self.spec.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "computed_at": self.computed_at.isoformat(),
            "total_runtime_s": self.total_runtime_s,
            "n_results": len(self.results),
            "n_with_warnings": sum(1 for r in self.results if r.warnings),
            "n_with_blockers": sum(1 for r in self.results if r.blockers),
        }


@dataclass
class CohortMetricReport:
    """Cohort-level summary across a set of NeuroClassicResults.

    Computes descriptive statistics over per-subject/session metrics.
    """
    method_name: str
    metric_name: str
    modality: str
    n_subjects: int
    values: list[float] = field(default_factory=list)
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    iqr: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    outlier_indices: list[int] = field(default_factory=list)
    outlier_threshold_sd: float = 3.0
    confidence: MethodConfidence = MethodConfidence.HIGH

    def compute(self) -> "CohortMetricReport":
        if not self.values:
            self.confidence = MethodConfidence.UNKNOWN
            return self

        import math
        n = len(self.values)
        s = sorted(self.values)
        self.mean = sum(self.values) / n
        self.std = math.sqrt(sum((v - self.mean) ** 2 for v in self.values) / max(n - 1, 1))
        mid = n // 2
        self.median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
        q1 = s[n // 4]
        q3 = s[3 * n // 4]
        self.iqr = q3 - q1
        self.min_val = s[0]
        self.max_val = s[-1]

        if n < 5:
            self.confidence = MethodConfidence.LOW_CONFIDENCE

        # Robust outlier detection via the median absolute deviation (MAD).
        # A single extreme value inflates the standard deviation enough to mask
        # itself, so a mean/SD rule is unreliable here.  The MAD-based modified
        # z-score (Iglewicz & Hoaglin) resists this masking effect.
        self.outlier_indices = self._robust_outliers()
        return self

    def _robust_outliers(self) -> list[int]:
        import math
        n = len(self.values)
        if n < 3:
            return []
        med = self.median if self.median is not None else (sum(self.values) / n)
        abs_dev = sorted(abs(v - med) for v in self.values)
        mid = n // 2
        mad = abs_dev[mid] if n % 2 else (abs_dev[mid - 1] + abs_dev[mid]) / 2

        if mad > 0:
            # 0.6745 scales MAD to be consistent with the SD of a normal dist.
            return [
                i for i, v in enumerate(self.values)
                if abs(0.6745 * (v - med) / mad) > self.outlier_threshold_sd
            ]
        # Degenerate MAD (≥50% identical values): fall back to mean/SD.
        if self.std and self.std > 0:
            return [
                i for i, v in enumerate(self.values)
                if abs(v - med) > self.outlier_threshold_sd * self.std
            ]
        return []

    def to_dict(self) -> dict:
        return {
            "method_name": self.method_name,
            "metric_name": self.metric_name,
            "modality": self.modality,
            "n_subjects": self.n_subjects,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "iqr": self.iqr,
            "min": self.min_val,
            "max": self.max_val,
            "outlier_indices": self.outlier_indices,
            "outlier_threshold_sd": self.outlier_threshold_sd,
            "confidence": self.confidence.value,
        }


def _timer():
    """Context manager that returns elapsed seconds."""
    class _Timer:
        def __enter__(self):
            self._start = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self._start

    return _Timer()
