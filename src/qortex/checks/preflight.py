"""Preflight check orchestrator.

Runs multiple targeted checks and aggregates results into a PreflightReport
based on the downstream goal (visualize, convert, train, neuroai-run).
"""

from __future__ import annotations

from pathlib import Path

from qortex.checks._report import CheckSeverity, PreflightReport
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

# Which checkers are required per goal
_GOAL_CHECKERS: dict[str, list[type]] = {
    "visualize": [
        StructureChecker,
        MetadataChecker,
        GeometryChecker,
    ],
    "convert": [
        StructureChecker,
        MetadataChecker,
        GeometryChecker,
        UnitsChecker,
        EventsChecker,
        ConversionReadinessChecker,
    ],
    "train": [
        StructureChecker,
        MetadataChecker,
        EventsChecker,
        UnitsChecker,
        LeakageChecker,
    ],
    "neuroai-run": [
        StructureChecker,
        MetadataChecker,
        UnitsChecker,
        RuntimeCompatibilityChecker,
    ],
}

_VALID_GOALS = frozenset(_GOAL_CHECKERS.keys())


class PreflightChecker:
    """Orchestrate multiple targeted checks for a downstream workflow goal.

    Parameters
    ----------
    goal:
        One of ``visualize``, ``convert``, ``train``, ``neuroai-run``.
    modality:
        Optional modality filter (e.g. ``eeg``, ``mri``, ``dwi``).
    target:
        Target column for label checks (e.g. ``diagnosis``).
    split_unit:
        Grouping unit for leakage checks (``subject``, ``session``, ``run``).
    source_profile:
        Optional SourceProfile for runtime compatibility checks.
    pipeline_yaml:
        Path to a pipeline YAML for neuroai-run goal transform extraction.
    extra_checker_kwargs:
        Extra kwargs passed to individual checker constructors.
    """

    def __init__(
        self,
        goal: str,
        *,
        modality: str | None = None,
        target: str | None = None,
        split_unit: str = "subject",
        source_profile=None,
        pipeline_yaml: Path | None = None,
        extra_checker_kwargs: dict | None = None,
    ) -> None:
        if goal not in _VALID_GOALS:
            raise ValueError(
                f"Unknown preflight goal '{goal}'. Valid goals: {sorted(_VALID_GOALS)}"
            )
        self._goal = goal
        self._modality = modality
        self._target = target
        self._split_unit = split_unit
        self._source_profile = source_profile
        self._pipeline_yaml = pipeline_yaml
        self._extra_kwargs = extra_checker_kwargs or {}

    def run(self, dataset_path: Path) -> PreflightReport:
        dataset_path = Path(dataset_path)
        report = PreflightReport(
            goal=self._goal,
            dataset_path=str(dataset_path),
            modality=self._modality,
            target=self._target,
            split_unit=self._split_unit,
        )

        pipeline_transforms = self._load_pipeline_transforms()

        checker_classes = _GOAL_CHECKERS[self._goal]
        for cls in checker_classes:
            checker = self._build_checker(cls, pipeline_transforms)
            kwargs: dict = {}
            if cls is RuntimeCompatibilityChecker and self._source_profile is not None:
                kwargs["source_profile"] = self._source_profile
            check_report = checker.run(dataset_path, **kwargs)
            report.add_check(check_report)

        return report.finalize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_checker(self, cls: type, pipeline_transforms: list[str]) -> object:
        """Instantiate a checker with appropriate kwargs for this preflight context."""
        if cls is StructureChecker:
            return cls(
                modality=self._modality,
                **self._extra_kwargs.get("structure", {}),
            )
        if cls is MetadataChecker:
            return cls(
                modality=self._modality,
                **self._extra_kwargs.get("metadata", {}),
            )
        if cls is GeometryChecker:
            return cls(
                modality=self._modality,
                **self._extra_kwargs.get("geometry", {}),
            )
        if cls is UnitsChecker:
            return cls(
                modality=self._modality,
                **self._extra_kwargs.get("units", {}),
            )
        if cls is EventsChecker:
            return cls(
                modality=self._modality,
                require_trial_type=(self._target is not None),
                **self._extra_kwargs.get("events", {}),
            )
        if cls is LeakageChecker:
            return cls(
                target=self._target,
                split_unit=self._split_unit,
                **self._extra_kwargs.get("leakage", {}),
            )
        if cls is ConversionReadinessChecker:
            return cls(
                modality=self._modality,
                **self._extra_kwargs.get("conversion", {}),
            )
        if cls is RuntimeCompatibilityChecker:
            return cls(
                required_modality=self._modality,
                pipeline_transforms=pipeline_transforms,
                **self._extra_kwargs.get("runtime", {}),
            )
        return cls()

    def _load_pipeline_transforms(self) -> list[str]:
        if self._pipeline_yaml is None:
            return []
        try:
            import yaml  # type: ignore[import-untyped]
            with open(self._pipeline_yaml) as fh:
                cfg = yaml.safe_load(fh)
            return [str(t) for t in (cfg.get("transforms") or [])]
        except Exception:
            return []
