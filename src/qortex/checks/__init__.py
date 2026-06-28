"""Qortex data integrity and check system.

Three modes:
  targeted   — one concern, one CheckReport
  preflight  — goal-aware, aggregated PreflightReport
  lazy       — automatic fast hints during normal operations

Evidence model: confirmed / inferred / claimed / missing / contradicted / unknown / blocked

Usage::

    from qortex.checks import run_preflight, lazy_check_dataset, CheckReport

    report = run_preflight("./dataset", goal="train", target="diagnosis")
    if report.status.value == "BLOCK":
        for b in report.blockers:
            print(b.message)

    hints = lazy_check_dataset(Path("./dataset"))
    hints.emit()
"""

from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    PreflightReport,
    SuggestedFix,
)
from qortex.checks._base import BaseChecker
from qortex.checks.preflight import PreflightChecker
from qortex.checks.lazy import lazy_check_dataset, get_lazy_mode, LazyHint, LazyCheckResult
from qortex.checks.converter import (
    ConversionContract,
    ConversionProvenanceRecord,
    FitScope,
    NormalizerFitRecord,
    NormalizerSpec,
    ReversibilityStatus,
    StandardizerReport,
)
from qortex.checks.domains import (
    ConversionReadinessChecker,
    EventsChecker,
    GeometryChecker,
    LeakageChecker,
    MetadataChecker,
    RuntimeCompatibilityChecker,
    StructureChecker,
    UnitsChecker,
)


def run_preflight(
    dataset_path,
    *,
    goal: str,
    modality: str | None = None,
    target: str | None = None,
    split_unit: str = "subject",
    source_profile=None,
    pipeline_yaml=None,
) -> PreflightReport:
    """Run a goal-aware preflight check on a local dataset.

    Parameters
    ----------
    dataset_path:
        Path to the BIDS dataset root.
    goal:
        One of ``visualize``, ``convert``, ``train``, ``neuroai-run``.
    modality:
        Optional modality filter.
    target:
        Label column name for leakage / label checks.
    split_unit:
        Grouping unit for leakage checks (``subject``, ``session``, ``run``).
    source_profile:
        Optional SourceProfile for runtime compatibility checks.
    pipeline_yaml:
        Optional path to a pipeline YAML for transform extraction.

    Returns
    -------
    PreflightReport
        Aggregated result with ``.status``, ``.blockers``, ``.warnings``, etc.
    """
    from pathlib import Path
    checker = PreflightChecker(
        goal=goal,
        modality=modality,
        target=target,
        split_unit=split_unit,
        source_profile=source_profile,
        pipeline_yaml=Path(pipeline_yaml) if pipeline_yaml else None,
    )
    return checker.run(Path(dataset_path))


__all__ = [
    # Report types
    "CheckFinding",
    "CheckReport",
    "CheckSeverity",
    "EvidenceRecord",
    "EvidenceState",
    "PreflightReport",
    "SuggestedFix",
    # Base
    "BaseChecker",
    # Preflight
    "PreflightChecker",
    "run_preflight",
    # Lazy
    "lazy_check_dataset",
    "get_lazy_mode",
    "LazyHint",
    "LazyCheckResult",
    # Converter / normalizer policy
    "ConversionContract",
    "ConversionProvenanceRecord",
    "FitScope",
    "NormalizerFitRecord",
    "NormalizerSpec",
    "ReversibilityStatus",
    "StandardizerReport",
    # Domain checkers
    "StructureChecker",
    "MetadataChecker",
    "EventsChecker",
    "GeometryChecker",
    "UnitsChecker",
    "LeakageChecker",
    "ConversionReadinessChecker",
    "RuntimeCompatibilityChecker",
]
