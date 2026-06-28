"""Lazy / fast monitoring check subsystem.

Lightweight checks that run automatically during common Qortex operations.
Inspects manifests, headers, sidecars, and cached profiles — never loads full data.
Controlled by the QORTEX_LAZY_CHECKS environment variable.

QORTEX_LAZY_CHECKS=off     disables all lazy checks
QORTEX_LAZY_CHECKS=warn    emits warnings (default)
QORTEX_LAZY_CHECKS=strict  turns warnings into errors
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


class LazyCheckMode(str):
    OFF = "off"
    WARN = "warn"
    STRICT = "strict"


@dataclass(frozen=True)
class LazyHint:
    """A lightweight hint from a fast lazy check."""
    code: str
    message: str
    recommendation: str | None = None
    command: str | None = None
    path: str | None = None


@dataclass
class LazyCheckResult:
    mode: str
    hints: list[LazyHint] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.hints)

    def emit(self) -> None:
        """Emit hints to the logger at the appropriate level."""
        if self.mode == LazyCheckMode.OFF:
            return
        level = logging.ERROR if self.mode == LazyCheckMode.STRICT else logging.WARNING
        for hint in self.hints:
            msg = f"[qortex lazy] {hint.code}: {hint.message}"
            if hint.recommendation:
                msg += f"\n  → {hint.recommendation}"
            if hint.command:
                msg += f"\n  Run: {hint.command}"
            log.log(level, msg)

    def raise_if_strict(self) -> None:
        if self.mode == LazyCheckMode.STRICT and self.hints:
            codes = [h.code for h in self.hints]
            raise RuntimeError(
                f"Qortex strict lazy checks failed: {codes}. "
                "Set QORTEX_LAZY_CHECKS=warn to continue with warnings."
            )


def get_lazy_mode() -> str:
    return os.environ.get("QORTEX_LAZY_CHECKS", "warn").lower().strip()


def lazy_check_dataset(dataset_path: Path) -> LazyCheckResult:
    """Run all fast lazy checks on a local dataset path.

    Safe to call before any load operation.  Never mutates data.
    """
    mode = get_lazy_mode()
    result = LazyCheckResult(mode=mode)

    if mode == LazyCheckMode.OFF:
        return result

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return result

    _check_missing_companion_files(dataset_path, result)
    _check_inconsistent_sampling_rates(dataset_path, result)
    _check_missing_events_tsv(dataset_path, result)
    _check_missing_channels_tsv(dataset_path, result)
    _check_partial_local_download(dataset_path, result)
    _check_derivative_raw_mixing(dataset_path, result)

    result.emit()
    return result


# ── Individual lazy checks ────────────────────────────────────────────────────

def _check_missing_companion_files(dataset_path: Path, result: LazyCheckResult) -> None:
    """Flag signal files without any JSON sidecar."""
    for ext in (".edf", ".bdf", ".fif"):
        for f in dataset_path.rglob(f"*{ext}"):
            sidecar = f.parent / (f.stem + ".json")
            if not sidecar.exists():
                result.hints.append(LazyHint(
                    code="LAZY.MISSING_SIDECAR",
                    message=f"Signal file {f.name} has no JSON sidecar.",
                    recommendation="Create a BIDS-compliant JSON sidecar.",
                    command=f"qortex check metadata {dataset_path}",
                    path=str(f),
                ))


def _check_inconsistent_sampling_rates(dataset_path: Path, result: LazyCheckResult) -> None:
    """Detect mixed sampling rates declared in JSON sidecars."""
    import json
    rates: dict[str, float] = {}
    for sidecar in dataset_path.rglob("*.json"):
        try:
            data = json.loads(sidecar.read_text())
        except Exception:
            continue
        sfreq = data.get("SamplingFrequency")
        if sfreq is None:
            continue
        try:
            rates[sidecar.name] = float(sfreq)
        except (TypeError, ValueError):
            continue

    if rates:
        unique_rates = set(rates.values())
        if len(unique_rates) > 1:
            result.hints.append(LazyHint(
                code="LAZY.MIXED_SAMPLING_RATES",
                message=f"Mixed SamplingFrequency values: {sorted(unique_rates)} Hz.",
                recommendation="Verify that files belong to the same protocol or resample before training.",
                command=f"qortex check metadata {dataset_path} --explain",
                path=str(dataset_path),
            ))


def _check_missing_events_tsv(dataset_path: Path, result: LazyCheckResult) -> None:
    """Flag task-based recordings that lack a matching events.tsv."""
    signal_exts = (".edf", ".bdf", ".fif")
    for ext in signal_exts:
        for f in dataset_path.rglob(f"*_task-*{ext}"):
            events_candidate = f.parent / (f.stem + "_events.tsv")
            alt_candidate = f.parent / (
                "_".join(p for p in f.stem.split("_") if not p.startswith("run-"))
                + "_events.tsv"
            )
            if not events_candidate.exists() and not alt_candidate.exists():
                result.hints.append(LazyHint(
                    code="LAZY.MISSING_EVENTS_TSV",
                    message=f"Task recording {f.name} has no matching events.tsv.",
                    recommendation="Add events.tsv to enable label extraction.",
                    command=f"qortex check events {dataset_path}",
                    path=str(f),
                ))


def _check_missing_channels_tsv(dataset_path: Path, result: LazyCheckResult) -> None:
    """Flag EEG/MEG files without a channels.tsv."""
    for ext in (".edf", ".bdf"):
        for f in dataset_path.rglob(f"*{ext}"):
            channels_tsv = f.parent / (f.stem + "_channels.tsv")
            if not channels_tsv.exists():
                result.hints.append(LazyHint(
                    code="LAZY.MISSING_CHANNELS_TSV",
                    message=f"{f.name} has no channels.tsv; channel metadata unavailable.",
                    recommendation="Add channels.tsv with name, type, and units columns.",
                    command=f"qortex check units {dataset_path}",
                    path=str(f),
                ))


def _check_partial_local_download(dataset_path: Path, result: LazyCheckResult) -> None:
    """Flag zero-byte or partially written files."""
    for f in dataset_path.rglob("*"):
        if f.is_file() and f.suffix in (".edf", ".bdf", ".nii", ".gz", ".fif"):
            try:
                if f.stat().st_size == 0:
                    result.hints.append(LazyHint(
                        code="LAZY.EMPTY_FILE",
                        message=f"File has 0 bytes: {f.name}. Download may be incomplete.",
                        recommendation="Re-download the file.",
                        path=str(f),
                    ))
            except OSError:
                pass


def _check_derivative_raw_mixing(dataset_path: Path, result: LazyCheckResult) -> None:
    """Warn if derivative files appear to be mixed into raw subject directories."""
    derivative_indicators = ("_desc-", "_space-", "_res-", "_den-", "_proc-")
    raw_dirs = [d for d in dataset_path.iterdir()
                if d.is_dir() and d.name.startswith("sub-")]

    for raw_dir in raw_dirs:
        for f in raw_dir.rglob("*"):
            if f.is_file() and any(tok in f.name for tok in derivative_indicators):
                if "derivatives" not in str(f.relative_to(dataset_path)):
                    result.hints.append(LazyHint(
                        code="LAZY.DERIVATIVE_IN_RAW",
                        message=(
                            f"Possible derivative file in raw directory: {f.name}. "
                            "Derived outputs belong under derivatives/."
                        ),
                        recommendation="Move to derivatives/<pipeline>/ to preserve raw integrity.",
                        command=f"qortex check structure {dataset_path}",
                        path=str(f),
                    ))
                    break  # one hint per subject is enough
