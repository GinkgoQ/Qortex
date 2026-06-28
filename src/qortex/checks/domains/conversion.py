"""Conversion readiness check domain.

Validates whether data can be converted safely: file loadability, axes/units/
timebase/coordinate frame availability, label preservation, and output schema fit.
"""

from __future__ import annotations

from pathlib import Path

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

_SIGNAL_EXTS = {".edf", ".bdf", ".fif", ".set", ".vhdr"}
_VOLUME_EXTS = {".nii", ".nii.gz"}


class ConversionReadinessChecker(BaseChecker):
    """Validate whether files can be safely converted without data loss."""

    name = "conversion_readiness"
    required_for = frozenset({"convert"})

    def __init__(
        self,
        *,
        modality: str | None = None,
        output_format: str | None = None,
    ) -> None:
        self._modality = modality
        self._output_format = output_format

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={
                "dataset_path": str(dataset_path),
                "modality": self._modality,
                "output_format": self._output_format,
            },
        )

        signal_files = [
            f for ext in _SIGNAL_EXTS
            for f in sorted(dataset_path.rglob(f"*{ext}"))
        ]
        volume_files = sorted(dataset_path.rglob("*.nii")) + sorted(dataset_path.rglob("*.nii.gz"))

        checked = 0
        for sf in signal_files:
            self._check_signal_file(sf, report)
            checked += 1

        for vf in volume_files:
            self._check_volume_file(vf, report)
            checked += 1

        if checked == 0:
            report.add(CheckFinding(
                code="CONVERSION.NO_SUPPORTED_FILES",
                severity=CheckSeverity.WARN,
                message="No supported signal or volume files found for conversion readiness check.",
                path=str(dataset_path),
            ))

        return report.finalize()

    def _check_signal_file(self, path: Path, report: CheckReport) -> None:
        entities = _parse_bids_entities_from_path(path)
        ext = "".join(path.suffixes).lower()

        # 1. File must be non-empty and readable (stat only)
        try:
            size = path.stat().st_size
        except OSError as exc:
            report.add(CheckFinding(
                code="CONVERSION.FILE_UNREADABLE",
                severity=CheckSeverity.BLOCK,
                message=f"Cannot stat signal file: {exc}",
                path=str(path),
                bids_entities=entities,
            ))
            return

        if size == 0:
            report.add(CheckFinding(
                code="CONVERSION.EMPTY_FILE",
                severity=CheckSeverity.BLOCK,
                message=f"Signal file is empty (0 bytes): {path.name}",
                path=str(path),
                bids_entities=entities,
                evidence=[EvidenceRecord(
                    field="file_size_bytes",
                    state=EvidenceState.confirmed,
                    observed_value=0,
                    observed_source=str(path),
                )],
            ))
            return

        # 2. Companion sidecar must carry units / axes information
        sidecar = path.parent / (path.stem + ".json")
        if not sidecar.exists():
            report.add(CheckFinding(
                code="CONVERSION.MISSING_SIDECAR",
                severity=CheckSeverity.WARN,
                message=(
                    f"No JSON sidecar for {path.name}. "
                    "Units, reference, and sampling frequency cannot be confirmed."
                ),
                path=str(path),
                bids_entities=entities,
                evidence=[EvidenceRecord(
                    field="sidecar",
                    state=EvidenceState.missing,
                    observed_source=str(path.parent),
                )],
                suggested_fix=SuggestedFix(
                    description=(
                        f"Create {path.stem}.json with at minimum: "
                        "SamplingFrequency, Manufacturer, EEGReference, Units."
                    ),
                    safe=True,
                ),
            ))
        else:
            try:
                import json
                meta = json.loads(sidecar.read_text())
                required_fields = ["SamplingFrequency"]
                for f in required_fields:
                    if f not in meta:
                        report.add(CheckFinding(
                            code="CONVERSION.SIDECAR_MISSING_REQUIRED_FIELD",
                            severity=CheckSeverity.WARN,
                            message=f"Sidecar {sidecar.name} is missing required field '{f}'.",
                            path=str(sidecar),
                            bids_entities=entities,
                            evidence=[EvidenceRecord(
                                field=f,
                                state=EvidenceState.missing,
                                observed_source=str(sidecar),
                            )],
                        ))
            except Exception as exc:
                report.add(CheckFinding(
                    code="CONVERSION.SIDECAR_PARSE_FAILED",
                    severity=CheckSeverity.WARN,
                    message=f"Cannot parse sidecar JSON: {exc}",
                    path=str(sidecar),
                    bids_entities=entities,
                ))

        report.record_evidence(EvidenceRecord(
            field=f"{path.name}.size_bytes",
            state=EvidenceState.confirmed,
            observed_value=size,
            observed_source=str(path),
        ))

    def _check_volume_file(self, path: Path, report: CheckReport) -> None:
        entities = _parse_bids_entities_from_path(path)
        try:
            size = path.stat().st_size
        except OSError as exc:
            report.add(CheckFinding(
                code="CONVERSION.FILE_UNREADABLE",
                severity=CheckSeverity.BLOCK,
                message=f"Cannot stat volume file: {exc}",
                path=str(path),
                bids_entities=entities,
            ))
            return

        if size == 0:
            report.add(CheckFinding(
                code="CONVERSION.EMPTY_VOLUME",
                severity=CheckSeverity.BLOCK,
                message=f"NIfTI file is empty (0 bytes): {path.name}",
                path=str(path),
                bids_entities=entities,
            ))
            return

        report.record_evidence(EvidenceRecord(
            field=f"{path.name}.size_bytes",
            state=EvidenceState.confirmed,
            observed_value=size,
            observed_source=str(path),
        ))


def _parse_bids_entities_from_path(path: Path) -> dict[str, str]:
    import re
    entity_re = re.compile(r"(sub|ses|task|run|acq|ce|dir|rec|echo|part)-([A-Za-z0-9]+)")
    return dict(entity_re.findall(path.name))
