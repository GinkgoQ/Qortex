"""ML-readiness quality checks — metadata-only, no data loading.

These checks analyse the manifest and local BIDS structure to produce
a QualityMetrics object.  They never load signal or image data.
"""

from __future__ import annotations

from pathlib import Path

from qortex.core.entities import Manifest, QualityMetrics


def compute_quality_metrics(
    manifest: Manifest,
    local_path: Path | None = None,
) -> QualityMetrics:
    """Compute QC and ML-readiness metrics from the manifest."""

    issues: list[str] = []
    risks: list[str] = []

    # ── BIDS essentials ───────────────────────────────────────────────────
    filenames = {f.filename for f in manifest.files}

    if "dataset_description.json" not in filenames:
        issues.append("Missing dataset_description.json")

    if "participants.tsv" not in filenames:
        issues.append("Missing participants.tsv")
        risks.append("No participant metadata table found.")

    # ── Events coverage ───────────────────────────────────────────────────
    signal_files = [
        f for f in manifest.files
        if f.modality in {"eeg", "meg", "ieeg", "fmri"} and not f.is_dir
    ]
    events_files = [
        f for f in manifest.files
        if f.suffix == "events" and f.extension == ".tsv"
    ]

    missing_events_pct = 0.0
    if signal_files:
        # Simple heuristic: compare task-level signal files to events files
        signal_tasks = {(f.entities.subject, f.entities.task)
                        for f in signal_files if f.entities.task}
        events_tasks = {(f.entities.subject, f.entities.task)
                        for f in events_files if f.entities.task}
        if signal_tasks:
            covered = len(signal_tasks & events_tasks)
            missing_events_pct = 1.0 - covered / len(signal_tasks)
            if missing_events_pct > 0.1:
                risks.append(
                    f"{missing_events_pct * 100:.0f}% of task signal files "
                    f"have no matching events file — label extraction will fail."
                )

    # ── Sampling frequency consistency ────────────────────────────────────
    # (metadata only — we read .json sidecars if local_path is given)
    if local_path is not None:
        sfreqs = _extract_sfreqs_from_sidecars(manifest, local_path)
        unique_sfreqs = set(sfreqs)
        if len(unique_sfreqs) > 1:
            risks.append(
                f"Inconsistent sampling frequencies detected: "
                f"{sorted(unique_sfreqs)} Hz. Resampling required before batching."
            )

    # ── Missing sidecar coverage ──────────────────────────────────────────
    data_files = [f for f in manifest.files
                  if f.datatype and f.extension not in {".json", ".tsv"} and not f.is_dir]
    json_stems = {
        f.filename.replace(f.extension, "")
        for f in manifest.files if f.extension == ".json"
    }
    missing_sidecar_pct = 0.0
    if data_files:
        without_sidecar = [
            f for f in data_files
            if f.filename.replace(f.extension, "") not in json_stems
        ]
        missing_sidecar_pct = len(without_sidecar) / len(data_files)
        if missing_sidecar_pct > 0.2:
            issues.append(
                f"{missing_sidecar_pct * 100:.0f}% of data files have no .json sidecar."
            )

    # ── Derivatives ───────────────────────────────────────────────────────
    has_raw = any(not f.path.startswith("derivatives/")
                  for f in manifest.files if not f.is_dir and f.datatype)
    if not has_raw and manifest.summary.has_derivatives:
        risks.append(
            "Dataset contains only derivative files — raw data not available."
        )

    # ── Class imbalance check ─────────────────────────────────────────────
    # (based on events suffix — simple heuristic without loading data)
    if missing_events_pct > 0.5:
        risks.append(
            "Most signal files lack events files — class-balanced splitting "
            "will not be possible."
        )

    # ── Scoring ───────────────────────────────────────────────────────────
    bids_score = max(0.0, 100.0 - len(issues) * 10.0)
    ml_penalty = len(risks) * 8.0 + missing_events_pct * 30.0 + missing_sidecar_pct * 20.0
    ml_readiness_score = max(0.0, 100.0 - ml_penalty)
    loadability_score = max(0.0, 100.0 - missing_sidecar_pct * 50.0 - len(issues) * 5.0)

    return QualityMetrics(
        bids_score=round(bids_score, 1),
        ml_readiness_score=round(ml_readiness_score, 1),
        loadability_score=round(loadability_score, 1),
        missing_events_pct=round(missing_events_pct, 3),
        missing_sidecar_pct=round(missing_sidecar_pct, 3),
        issues=issues,
        risks=risks,
    )


def _extract_sfreqs_from_sidecars(manifest: Manifest, local_path: Path) -> list[float]:
    """Read SamplingFrequency from JSON sidecars for electrophysiology data."""
    import json
    sfreqs: list[float] = []
    for f in manifest.files:
        if f.extension == ".json" and f.datatype in {"eeg", "meg", "ieeg"}:
            candidate = local_path / f.path
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    sf = data.get("SamplingFrequency")
                    if isinstance(sf, (int, float)):
                        sfreqs.append(float(sf))
                except Exception:
                    pass
    return sfreqs
