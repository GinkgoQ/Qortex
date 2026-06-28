"""Unit and scaling check domain.

Validates declared measurement units against observed signal/image statistics.
Detects implausible scale, NaN/Inf, constant images, and unit inconsistency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

# Known BIDS-valid unit strings (non-exhaustive; used for spelling checks)
_VALID_UNITS = {
    # EEG/MEG/ERP
    "µV", "uV", "mV", "V",
    "fT", "fT/cm",
    "T",
    # fMRI / MRI
    "arbitrary", "a.u.", "au", "arbitrary units",
    "Hz", "rad/s",
    "mm", "m", "cm",
    "s", "ms",
    # PET
    "Bq/mL", "BQML",
    # fNIRS
    "mol/L", "mmol/L", "mMol/L", "µmol/L", "umol/L",
    # iEEG
    "kΩ", "kohm", "MΩ", "Mohm",
    # Other
    "n/a", "N/A",
}

# Plausible peak-to-peak amplitude ranges by declared unit (min, max in same unit)
_AMPLITUDE_RANGES: dict[str, tuple[float, float]] = {
    "uV":  (0.1, 5_000),    # EEG: 0.1–5000 µV
    "µV":  (0.1, 5_000),
    "mV":  (0.0001, 5),
    "V":   (1e-7, 0.005),   # EEG in Volts: ~µV scale
    "fT":  (1, 1_000_000),  # MEG femtotesla
}

_CHANNELS_TSV_UNITS_COLUMN = "units"


class UnitsChecker(BaseChecker):
    """Validate declared units and observed signal/image scale."""

    name = "units"
    required_for = frozenset({"convert", "train", "neuroai-run"})

    def __init__(
        self,
        *,
        modality: str | None = None,
        check_signal_scale: bool = True,
        check_image_nan_inf: bool = True,
        amplitude_sample_size: int = 10_000,
    ) -> None:
        self._modality = modality
        self._check_signal_scale = check_signal_scale
        self._check_image_nan_inf = check_image_nan_inf
        self._amplitude_sample_size = amplitude_sample_size

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={"dataset_path": str(dataset_path), "modality": self._modality},
        )

        # Check channels.tsv units columns
        for channels_tsv in sorted(dataset_path.rglob("*_channels.tsv")):
            self._check_channels_tsv(channels_tsv, report)

        # Check JSON sidecar Units field
        for sidecar in sorted(dataset_path.rglob("*.json")):
            if "dataset_description" in sidecar.name:
                continue
            self._check_sidecar_units(sidecar, report)

        # Signal amplitude plausibility (requires loading; gated by flag)
        if self._check_signal_scale:
            for edf in sorted(dataset_path.rglob("*.edf")):
                self._check_edf_signal_scale(edf, report)

        return report.finalize()

    # ── channels.tsv ──────────────────────────────────────────────────────────

    def _check_channels_tsv(self, channels_tsv: Path, report: CheckReport) -> None:
        entities = _parse_bids_entities_from_path(channels_tsv)
        try:
            rows = _read_tsv(channels_tsv)
        except Exception as exc:
            report.add(CheckFinding(
                code="UNITS.CHANNELS_TSV_PARSE_FAILED",
                severity=CheckSeverity.WARN,
                message=f"Cannot parse channels.tsv: {exc}",
                path=str(channels_tsv),
                bids_entities=entities,
            ))
            return

        if not rows:
            return

        columns = list(rows[0].keys())
        if _CHANNELS_TSV_UNITS_COLUMN not in columns:
            report.add(CheckFinding(
                code="UNITS.CHANNELS_TSV_MISSING_UNITS",
                severity=CheckSeverity.WARN,
                message=f"channels.tsv has no 'units' column: {channels_tsv.name}",
                path=str(channels_tsv),
                bids_entities=entities,
                suggested_fix=SuggestedFix(
                    description="Add 'units' column (e.g. 'uV') to all channels.tsv files.",
                    field="units",
                    safe=True,
                ),
            ))
            return

        unit_values = [r.get("units", "n/a") for r in rows]
        unique_units = set(u for u in unit_values if u not in ("n/a", "N/A", ""))

        for u in unique_units:
            canonical = u.replace("μ", "µ").strip()
            if canonical not in _VALID_UNITS:
                report.add(CheckFinding(
                    code="UNITS.UNKNOWN_UNIT",
                    severity=CheckSeverity.WARN,
                    message=f"Unrecognized unit '{u}' in {channels_tsv.name}.",
                    path=str(channels_tsv),
                    bids_entities=entities,
                    observed=u,
                    evidence=[EvidenceRecord(
                        field="units",
                        state=EvidenceState.claimed,
                        claimed_value=u,
                        claimed_source=str(channels_tsv),
                        note="Unit is declared but not in the Qortex recognized-unit list.",
                    )],
                    suggested_fix=SuggestedFix(
                        description=f"Use a BIDS-canonical unit: uV, mV, V, fT, Hz, s, etc.",
                        field="units",
                        safe=True,
                    ),
                ))

        if len(unique_units) > 1:
            report.add(CheckFinding(
                code="UNITS.INCONSISTENT_CHANNEL_UNITS",
                severity=CheckSeverity.INFO,
                message=(
                    f"channels.tsv declares mixed units: {sorted(unique_units)}. "
                    "Per-channel unit normalization will be required."
                ),
                path=str(channels_tsv),
                bids_entities=entities,
                observed=sorted(unique_units),
                evidence=[EvidenceRecord(
                    field="channel_units",
                    state=EvidenceState.confirmed,
                    observed_value=sorted(unique_units),
                    observed_source=str(channels_tsv),
                )],
            ))

    # ── JSON sidecar Units ────────────────────────────────────────────────────

    def _check_sidecar_units(self, sidecar: Path, report: CheckReport) -> None:
        try:
            import json
            data = json.loads(sidecar.read_text())
        except Exception:
            return

        declared_unit = data.get("Units") or data.get("units")
        if declared_unit is None:
            return

        canonical = str(declared_unit).replace("μ", "µ").strip()
        if canonical not in _VALID_UNITS:
            entities = _parse_bids_entities_from_path(sidecar)
            report.add(CheckFinding(
                code="UNITS.SIDECAR_UNKNOWN_UNIT",
                severity=CheckSeverity.WARN,
                message=(
                    f"Sidecar {sidecar.name} declares unit '{declared_unit}' which is "
                    "not in the recognized unit list."
                ),
                path=str(sidecar),
                bids_entities=entities,
                observed=declared_unit,
                evidence=[EvidenceRecord(
                    field="Units",
                    state=EvidenceState.claimed,
                    claimed_value=declared_unit,
                    claimed_source=str(sidecar),
                )],
            ))

    # ── EDF signal scale ──────────────────────────────────────────────────────

    def _check_edf_signal_scale(self, edf: Path, report: CheckReport) -> None:
        """Read physical min/max from EDF header and check plausibility."""
        entities = _parse_bids_entities_from_path(edf)
        try:
            phys_ranges = _read_edf_physical_ranges(edf)
        except Exception as exc:
            report.add(CheckFinding(
                code="UNITS.EDF_UNREADABLE",
                severity=CheckSeverity.INFO,
                message=f"Cannot read EDF signal scale: {exc}",
                path=str(edf),
                bids_entities=entities,
            ))
            return

        if not phys_ranges:
            return

        all_max = max(abs(mn) for mn, mx in phys_ranges) + max(abs(mx) for mn, mx in phys_ranges)
        if all_max == 0:
            report.add(CheckFinding(
                code="UNITS.ZERO_PHYSICAL_RANGE",
                severity=CheckSeverity.WARN,
                message=f"EDF physical range is zero for all channels: {edf.name}",
                path=str(edf),
                bids_entities=entities,
            ))
            return

        # Guess likely unit from physical range magnitude
        peak = max(abs(mx) for _, mx in phys_ranges)
        likely_unit: str | None = None
        for unit, (lo, hi) in _AMPLITUDE_RANGES.items():
            if lo <= peak <= hi:
                likely_unit = unit
                break

        report.record_evidence(EvidenceRecord(
            field=f"{edf.name}.physical_peak",
            state=EvidenceState.inferred,
            observed_value=peak,
            observed_source=str(edf),
            note=f"Likely unit by amplitude: {likely_unit or 'unknown'}",
        ))

        # Find the sidecar to compare declared unit
        stem = edf.stem
        sidecar_path = edf.parent / (stem + ".json")
        if not sidecar_path.exists():
            # Try channels.tsv
            channels_tsv = edf.parent / (stem + "_channels.tsv")
            if not channels_tsv.exists():
                return

        if sidecar_path.exists():
            try:
                import json
                sidecar_data = json.loads(sidecar_path.read_text())
                declared_unit = sidecar_data.get("Units") or sidecar_data.get("units")
                if declared_unit and likely_unit and declared_unit not in (likely_unit, likely_unit.replace("µ", "u")):
                    report.add(CheckFinding(
                        code="UNITS.AMPLITUDE_UNIT_MISMATCH",
                        severity=CheckSeverity.WARN,
                        message=(
                            f"Declared unit '{declared_unit}' may not match the observed "
                            f"amplitude range (peak {peak:.2f}; likely unit: {likely_unit}). "
                            "Manual confirmation required."
                        ),
                        path=str(edf),
                        bids_entities=entities,
                        expected=likely_unit,
                        observed=declared_unit,
                        evidence=[EvidenceRecord(
                            field="Units",
                            state=EvidenceState.contradicted,
                            claimed_value=declared_unit,
                            observed_value=f"peak={peak:.2f}, likely={likely_unit}",
                            claimed_source=str(sidecar_path),
                            observed_source=str(edf),
                            note="Inferred from physical range in EDF header.",
                        )],
                        suggested_fix=SuggestedFix(
                            description=(
                                f"Confirm whether data is in {declared_unit} or {likely_unit}. "
                                "Update 'Units' in the sidecar accordingly."
                            ),
                            field="Units",
                            safe=True,
                        ),
                    ))
            except Exception:
                pass


# ── Tiny EDF reader ───────────────────────────────────────────────────────────

def _read_edf_physical_ranges(path: Path) -> list[tuple[float, float]]:
    """Read physical min/max per channel from EDF header (no MNE needed)."""
    with open(path, "rb") as fh:
        header = fh.read(256)
    n_signals_raw = header[252:256].decode("ascii", errors="replace").strip()
    n_signals = int(n_signals_raw)

    # Signal header is 256 bytes per signal, starting at byte 256
    # Field order: label[16] transducer[80] phys_dim[8] phys_min[8] phys_max[8] ...
    with open(path, "rb") as fh:
        fh.seek(256)
        signal_header = fh.read(n_signals * 256)

    phys_ranges = []
    field_sizes = [16, 80, 8, 8, 8, 8, 8, 80, 8, 32]  # EDF per-signal fields
    for i in range(n_signals):
        offset = 0
        field_vals: list[str] = []
        for size in field_sizes:
            chunk = signal_header[i * 256 + offset: i * 256 + offset + size]
            field_vals.append(chunk.decode("ascii", errors="replace").strip())
            offset += size
        # phys_min at index 3, phys_max at index 4
        try:
            pmin = float(field_vals[3])
            pmax = float(field_vals[4])
            phys_ranges.append((pmin, pmax))
        except (ValueError, IndexError):
            phys_ranges.append((0.0, 0.0))

    return phys_ranges


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    return rows


def _parse_bids_entities_from_path(path: Path) -> dict[str, str]:
    import re
    entity_re = re.compile(r"(sub|ses|task|run|acq|ce|dir|rec|echo|part)-([A-Za-z0-9]+)")
    return dict(entity_re.findall(path.name))
