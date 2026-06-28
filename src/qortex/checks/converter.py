"""Converter, Normalizer, and Standardizer policy contracts.

Every transform must declare its input/output contracts, required evidence,
preserved fields, invalidated assumptions, reversibility, and provenance.
Normalizers declare their fit scope.  Standardizers report what they changed
and what remains heterogeneous.
"""

from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FitScope(str, Enum):
    """Scope over which a normalizer fits its parameters."""
    per_window = "per_window"
    per_file = "per_file"
    per_subject = "per_subject"
    per_session = "per_session"
    train_split_only = "train_split_only"
    whole_dataset = "whole_dataset"
    external_reference = "external_reference"


class ReversibilityStatus(str, Enum):
    reversible = "reversible"
    irreversible = "irreversible"
    reversible_with_params = "reversible_with_params"


@dataclass(frozen=True)
class ConversionContract:
    """Declares the full intent of a conversion step.

    A converter must never silently lose provenance, units, axes, labels,
    coordinate frames, or split group membership.
    """
    name: str
    version: str

    # Contract fields
    input_format: str
    output_format: str

    # Required evidence that must be present before conversion begins
    required_evidence: list[str]

    # Fields that this conversion changes
    changed_fields: list[str]

    # Fields that this conversion preserves (checked after conversion)
    preserved_fields: list[str]

    # Assumptions this conversion invalidates in downstream artifacts
    invalidated_assumptions: list[str]

    # Parameters exposed by this conversion
    parameters: dict[str, Any] = field(default_factory=dict)

    reversibility: ReversibilityStatus = ReversibilityStatus.reversible

    # Provenance record that will be written alongside the output
    provenance_fields: list[str] = field(default_factory=lambda: [
        "source_file",
        "source_checksum",
        "subject",
        "session",
        "run",
        "task",
        "modality",
        "axes",
        "units",
        "coordinate_frame",
        "sampling_frequency_hz",
        "event_timebase",
        "labels",
        "split_group",
        "transform_history",
        "output_schema",
    ])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "required_evidence": self.required_evidence,
            "changed_fields": self.changed_fields,
            "preserved_fields": self.preserved_fields,
            "invalidated_assumptions": self.invalidated_assumptions,
            "parameters": self.parameters,
            "reversibility": self.reversibility.value,
            "provenance_fields": self.provenance_fields,
        }


@dataclass
class NormalizerSpec:
    """Declares how a normalizer is fit and what it changes.

    Whole-dataset fitting is unsafe for ML unless explicitly requested and
    recorded as leakage-relevant in the provenance.
    """
    name: str
    fit_scope: FitScope
    input_field: str
    output_field: str
    parameters: dict[str, Any] = field(default_factory=dict)
    leakage_relevant: bool = False
    fitted_at: datetime.datetime | None = None
    fit_source: str | None = None  # e.g. "train_split" or "external_reference_file"

    def __post_init__(self) -> None:
        if self.fit_scope == FitScope.whole_dataset:
            self.leakage_relevant = True

    def record_fit(
        self,
        *,
        fitted_params: dict[str, Any],
        source_description: str,
        fitted_at: datetime.datetime | None = None,
    ) -> "NormalizerFitRecord":
        return NormalizerFitRecord(
            normalizer_name=self.name,
            fit_scope=self.fit_scope,
            fitted_params=fitted_params,
            source_description=source_description,
            leakage_relevant=self.leakage_relevant,
            fitted_at=fitted_at or datetime.datetime.utcnow(),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fit_scope": self.fit_scope.value,
            "input_field": self.input_field,
            "output_field": self.output_field,
            "parameters": self.parameters,
            "leakage_relevant": self.leakage_relevant,
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "fit_source": self.fit_source,
        }


@dataclass(frozen=True)
class NormalizerFitRecord:
    """Immutable record of a completed normalizer fit."""
    normalizer_name: str
    fit_scope: FitScope
    fitted_params: dict[str, Any]
    source_description: str
    leakage_relevant: bool
    fitted_at: datetime.datetime

    def to_dict(self) -> dict:
        return {
            "normalizer_name": self.normalizer_name,
            "fit_scope": self.fit_scope.value,
            "fitted_params": self.fitted_params,
            "source_description": self.source_description,
            "leakage_relevant": self.leakage_relevant,
            "fitted_at": self.fitted_at.isoformat(),
        }


@dataclass
class StandardizerReport:
    """Records what a standardizer changed and what remains heterogeneous.

    A standardizer must not erase heterogeneity without reporting it.
    """
    standardized_fields: dict[str, Any] = field(default_factory=dict)
    unstandardized_fields: dict[str, str] = field(default_factory=dict)  # field → reason
    applied_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)
    n_subjects: int = 0
    n_files: int = 0

    def record_standardized(self, field: str, value: Any) -> None:
        self.standardized_fields[field] = value

    def record_unstandardized(self, field: str, reason: str) -> None:
        self.unstandardized_fields[field] = reason

    def to_dict(self) -> dict:
        return {
            "standardized_fields": self.standardized_fields,
            "unstandardized_fields": self.unstandardized_fields,
            "applied_at": self.applied_at.isoformat(),
            "n_subjects": self.n_subjects,
            "n_files": self.n_files,
        }

    def format_summary(self) -> str:
        lines = []
        for f, v in self.standardized_fields.items():
            lines.append(f"  {f}: standardized to {v}")
        for f, reason in self.unstandardized_fields.items():
            lines.append(f"  {f}: NOT standardized — {reason}")
        return "\n".join(lines) if lines else "  (no fields processed)"


@dataclass(frozen=True)
class ConversionProvenanceRecord:
    """Written alongside every converted file."""
    conversion_name: str
    conversion_version: str
    source_file: str
    source_checksum: str
    subject: str | None
    session: str | None
    run: str | None
    task: str | None
    modality: str | None
    axes: list[str]
    units: str | None
    coordinate_frame: str | None
    sampling_frequency_hz: float | None
    event_timebase: str | None
    labels: list[str]
    split_group: str | None
    transform_history: list[str]
    output_schema: str
    converted_at: datetime.datetime

    @classmethod
    def from_source_file(
        cls,
        *,
        source_file: Path,
        conversion_name: str,
        conversion_version: str,
        **kwargs,
    ) -> "ConversionProvenanceRecord":
        try:
            checksum = _sha256_head(source_file)
        except OSError:
            checksum = "unavailable"
        return cls(
            conversion_name=conversion_name,
            conversion_version=conversion_version,
            source_file=str(source_file),
            source_checksum=checksum,
            converted_at=datetime.datetime.utcnow(),
            **kwargs,
        )

    def to_dict(self) -> dict:
        return {
            "conversion_name": self.conversion_name,
            "conversion_version": self.conversion_version,
            "source_file": self.source_file,
            "source_checksum": self.source_checksum,
            "subject": self.subject,
            "session": self.session,
            "run": self.run,
            "task": self.task,
            "modality": self.modality,
            "axes": self.axes,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "event_timebase": self.event_timebase,
            "labels": self.labels,
            "split_group": self.split_group,
            "transform_history": self.transform_history,
            "output_schema": self.output_schema,
            "converted_at": self.converted_at.isoformat(),
        }


def _sha256_head(path: Path, n_bytes: int = 65536) -> str:
    """SHA-256 of the first n_bytes of a file — fast fingerprint without full read."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n_bytes))
    return h.hexdigest()
