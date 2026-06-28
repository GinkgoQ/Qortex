"""Metadata / header cross-validation check domain.

Compares declared values in BIDS JSON sidecars against values extracted from
file headers (EDF, NIfTI, DICOM).  Returns contradicted evidence where they differ.
"""

from __future__ import annotations

import json
import struct
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

# Relative tolerance for numeric field comparison
_FREQ_REL_TOL = 0.01   # 1 % — sampling rate
_SHAPE_REL_TOL = 0.0   # exact match for voxel/sample counts

_SIDECAR_SIGNAL_FIELDS = {
    "SamplingFrequency": float,
    "PowerLineFrequency": float,
    "EEGReference": str,
    "ChannelCount": int,
    "RecordingDuration": float,
}

_SIDECAR_VOLUME_FIELDS = {
    "RepetitionTime": float,
    "NumberOfVolumesDiscardedByScanner": int,
    "NumberOfVolumesDiscardedByUser": int,
}


class MetadataChecker(BaseChecker):
    """Cross-check BIDS JSON sidecars against raw file headers."""

    name = "metadata"
    required_for = frozenset({"visualize", "convert", "train", "neuroai-run"})

    def __init__(self, *, modality: str | None = None) -> None:
        self._modality = modality

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={"dataset_path": str(dataset_path), "modality": self._modality},
        )

        if not dataset_path.exists():
            report.add(CheckFinding(
                code="METADATA.PATH_NOT_FOUND",
                severity=CheckSeverity.BLOCK,
                message=f"Dataset path does not exist: {dataset_path}",
            ))
            return report.finalize()

        for sidecar in sorted(dataset_path.rglob("*.json")):
            if "dataset_description" in sidecar.name:
                continue
            self._check_sidecar(sidecar, report)

        return report.finalize()

    # ── Per-sidecar dispatch ──────────────────────────────────────────────────

    def _check_sidecar(self, sidecar: Path, report: CheckReport) -> None:
        try:
            declared = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            report.add(CheckFinding(
                code="METADATA.SIDECAR_INVALID",
                severity=CheckSeverity.WARN,
                message=f"Cannot parse sidecar JSON: {exc}",
                path=str(sidecar),
            ))
            return

        stem = sidecar.stem  # e.g. sub-01_task-rest_eeg
        parent = sidecar.parent

        # Signal sidecars (EEG, MEG, iEEG, fNIRS)
        if any(tok in stem for tok in ("_eeg", "_meg", "_ieeg", "_fnirs")):
            for ext in (".edf", ".bdf", ".fif"):
                raw_file = parent / (stem + ext)
                if raw_file.exists():
                    self._cross_check_signal(raw_file, declared, sidecar, report)
                    break

        # Volume sidecars (BOLD, T1w, dwi)
        elif any(tok in stem for tok in ("_bold", "_T1w", "_T2w", "_dwi", "_T1", "_T2")):
            for ext in (".nii.gz", ".nii"):
                nii = parent / (stem + ext)
                if nii.exists():
                    self._cross_check_nifti(nii, declared, sidecar, report)
                    break

    # ── Signal checks ─────────────────────────────────────────────────────────

    def _cross_check_signal(
        self, raw_file: Path, declared: dict, sidecar: Path, report: CheckReport
    ) -> None:
        ext = "".join(raw_file.suffixes)
        observed_sfreq: float | None = None
        observed_n_ch: int | None = None

        if ext in (".edf", ".bdf"):
            observed_sfreq, observed_n_ch = _read_edf_header_fast(raw_file)

        if observed_sfreq is None:
            report.record_evidence(EvidenceRecord(
                field=f"{raw_file.name}.SamplingFrequency",
                state=EvidenceState.unknown,
                note="Could not read header without loading full file.",
                observed_source=str(raw_file),
            ))
            return

        claimed_sfreq = declared.get("SamplingFrequency")
        if claimed_sfreq is not None:
            try:
                claimed_f = float(claimed_sfreq)
                rel_diff = abs(claimed_f - observed_sfreq) / max(observed_sfreq, 1.0)
                if rel_diff > _FREQ_REL_TOL:
                    report.add(CheckFinding(
                        code="METADATA.SFREQ_MISMATCH",
                        severity=CheckSeverity.BLOCK,
                        message=(
                            f"SamplingFrequency mismatch: sidecar claims {claimed_f} Hz "
                            f"but raw header reports {observed_sfreq} Hz ({rel_diff*100:.2f}% diff)."
                        ),
                        path=str(raw_file),
                        expected=claimed_f,
                        observed=observed_sfreq,
                        evidence=[EvidenceRecord(
                            field="SamplingFrequency",
                            state=EvidenceState.contradicted,
                            claimed_value=claimed_f,
                            observed_value=observed_sfreq,
                            claimed_source=str(sidecar),
                            observed_source=str(raw_file),
                        )],
                        suggested_fix=SuggestedFix(
                            description=(
                                f"Update {sidecar.name} SamplingFrequency to {observed_sfreq} "
                                "if the raw header is authoritative."
                            ),
                            field="SamplingFrequency",
                            safe=True,
                        ),
                    ))
                else:
                    report.record_evidence(EvidenceRecord(
                        field="SamplingFrequency",
                        state=EvidenceState.confirmed,
                        claimed_value=claimed_f,
                        observed_value=observed_sfreq,
                        claimed_source=str(sidecar),
                        observed_source=str(raw_file),
                    ))
            except (TypeError, ValueError):
                pass
        else:
            report.record_evidence(EvidenceRecord(
                field="SamplingFrequency",
                state=EvidenceState.inferred,
                observed_value=observed_sfreq,
                observed_source=str(raw_file),
                note="Sidecar did not declare SamplingFrequency; using raw header value.",
            ))

        claimed_n_ch = declared.get("ChannelCount")
        if claimed_n_ch is not None and observed_n_ch is not None:
            if int(claimed_n_ch) != observed_n_ch:
                report.add(CheckFinding(
                    code="METADATA.CHANNEL_COUNT_MISMATCH",
                    severity=CheckSeverity.WARN,
                    message=(
                        f"ChannelCount mismatch: sidecar claims {claimed_n_ch} channels "
                        f"but raw header has {observed_n_ch}."
                    ),
                    path=str(raw_file),
                    expected=int(claimed_n_ch),
                    observed=observed_n_ch,
                    evidence=[EvidenceRecord(
                        field="ChannelCount",
                        state=EvidenceState.contradicted,
                        claimed_value=int(claimed_n_ch),
                        observed_value=observed_n_ch,
                        claimed_source=str(sidecar),
                        observed_source=str(raw_file),
                    )],
                ))

    # ── Volume checks ─────────────────────────────────────────────────────────

    def _cross_check_nifti(
        self, nii: Path, declared: dict, sidecar: Path, report: CheckReport
    ) -> None:
        try:
            tr_s, n_vols, pixdim = _read_nifti_header_fast(nii)
        except Exception as exc:
            report.add(CheckFinding(
                code="METADATA.NIFTI_UNREADABLE",
                severity=CheckSeverity.WARN,
                message=f"Cannot read NIfTI header: {exc}",
                path=str(nii),
            ))
            return

        claimed_tr = declared.get("RepetitionTime")
        if claimed_tr is not None and tr_s is not None:
            try:
                claimed_tr_f = float(claimed_tr)
                rel_diff = abs(claimed_tr_f - tr_s) / max(tr_s, 1e-6)
                if rel_diff > _FREQ_REL_TOL:
                    report.add(CheckFinding(
                        code="METADATA.TR_MISMATCH",
                        severity=CheckSeverity.BLOCK,
                        message=(
                            f"RepetitionTime mismatch: sidecar claims {claimed_tr_f} s "
                            f"but NIfTI header dim[4]/pixdim[4] = {tr_s:.4f} s."
                        ),
                        path=str(nii),
                        expected=claimed_tr_f,
                        observed=tr_s,
                        evidence=[EvidenceRecord(
                            field="RepetitionTime",
                            state=EvidenceState.contradicted,
                            claimed_value=claimed_tr_f,
                            observed_value=tr_s,
                            claimed_source=str(sidecar),
                            observed_source=str(nii),
                        )],
                        suggested_fix=SuggestedFix(
                            description=f"Update {sidecar.name} RepetitionTime to {tr_s:.4f}.",
                            field="RepetitionTime",
                            safe=True,
                        ),
                    ))
                else:
                    report.record_evidence(EvidenceRecord(
                        field="RepetitionTime",
                        state=EvidenceState.confirmed,
                        claimed_value=claimed_tr_f,
                        observed_value=tr_s,
                        claimed_source=str(sidecar),
                        observed_source=str(nii),
                    ))
            except (TypeError, ValueError):
                pass


# ── Header fast-readers (no heavy deps) ──────────────────────────────────────

def _read_edf_header_fast(path: Path) -> tuple[float | None, int | None]:
    """Read sampling frequency and channel count from an EDF/BDF header.

    EDF header layout (ASCII bytes):
      bytes 0-7:   version
      bytes 8-88:  patient info
      88-168:      recording info
      168-176:     startdate
      176-184:     starttime
      184-192:     n_header_bytes
      192-236:     reserved
      236-244:     n_records
      244-252:     record_duration_s
      252-256:     n_signals

    Then per signal: label[16], transducer[80], phys_dim[8], phys_min[8],
    phys_max[8], dig_min[8], dig_max[8], prefilter[80], n_samples[8], reserved[32]
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(256)
        if len(header) < 256:
            return None, None
        n_signals_raw = header[252:256].decode("ascii", errors="replace").strip()
        n_signals = int(n_signals_raw)

        with open(path, "rb") as fh:
            fh.seek(256 + n_signals * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80))
            n_samples_bytes = fh.read(n_signals * 8)

        record_dur_raw = header[244:252].decode("ascii", errors="replace").strip()
        record_dur = float(record_dur_raw) if record_dur_raw else 1.0

        n_samples_per_record = []
        for i in range(n_signals):
            chunk = n_samples_bytes[i * 8: (i + 1) * 8]
            n_samples_per_record.append(int(chunk.decode("ascii", errors="replace").strip()))

        if not n_samples_per_record:
            return None, n_signals

        # Sampling rate = n_samples / record_duration (use max of non-annotation signals)
        max_samples = max(n_samples_per_record)
        sfreq = max_samples / record_dur if record_dur > 0 else None
        return sfreq, n_signals
    except Exception:
        return None, None


def _read_nifti_header_fast(path: Path) -> tuple[float | None, int | None, tuple | None]:
    """Read TR, n_volumes, and pixdim from a NIfTI-1 header (no nibabel needed).

    NIfTI-1 binary layout (348-byte header):
      sizeof_hdr: 4 bytes (int32, must be 348)
      ...skipping to dim field...
      dim: at byte 40, 8×int16 — dim[0]=ndim, dim[1-7]=shape
      pixdim: at byte 76, 8×float32
    """
    import gzip

    open_fn = gzip.open if str(path).endswith(".gz") else open
    try:
        with open_fn(path, "rb") as fh:  # type: ignore[arg-type]
            raw = fh.read(348)
        if len(raw) < 348:
            return None, None, None

        # Endianness check via sizeof_hdr
        sizeof_hdr = struct.unpack_from("<i", raw, 0)[0]
        endian = "<" if sizeof_hdr == 348 else ">"

        dim = struct.unpack_from(f"{endian}8h", raw, 40)
        pixdim = struct.unpack_from(f"{endian}8f", raw, 76)

        ndim = dim[0]
        shape = tuple(dim[1: ndim + 1])
        n_vols = shape[3] if ndim >= 4 else None
        tr_s = float(pixdim[4]) if ndim >= 4 and pixdim[4] > 0 else None

        return tr_s, n_vols, pixdim[1:4]
    except Exception:
        return None, None, None
