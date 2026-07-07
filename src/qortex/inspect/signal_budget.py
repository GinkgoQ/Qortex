"""Signal Budget Estimator — remote acquisition parameter inspection.

This module fetches JSON sidecar files concurrently from the CDN and extracts
acquisition parameters (sampling frequency, recording duration, channel counts)
to estimate:

  * Total signal content (hours of recorded data per modality)
  * Windowed sample counts (given a window duration and overlap)
  * Per-class sample estimates (combined with LabelLandscape)
  * Minimum viable subset for a given training budget
  * Data adequacy for common EEG/MEG/fMRI ML benchmarks

For NIfTI (fMRI/MRI/DWI) files, parameters are extracted from the file header
via Range requests (352 bytes) rather than JSON sidecars.

Key insight: Most signal file parameters that matter for ML planning are in the
JSON sidecar, not the signal file itself. A sidecar is typically 1–50 KB, while
the signal file is 10 MB–4 GB. Fetching sidecars gives you all the information
you need to decide whether to download the signal data at all.

BIDS sidecar fields used
------------------------
Shared (all modalities):
  TaskName, SamplingFrequency, RecordingDuration

EEG/MEG/iEEG specific:
  EEGChannelCount, MEGChannelCount, ECOGChannelCount, SEEGChannelCount,
  EOGChannelCount, EMGChannelCount, MiscChannelCount, TriggerChannelCount

fMRI/BOLD specific (also from NIfTI header):
  RepetitionTime (TR), NumberOfVolumesDiscardedByScanner,
  NumberOfVolumesDiscardedByUser, SliceTiming (length → n_slices)
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from qortex.client.remote import NIfTIHeader, RemoteFileGateway, _pick_url
from qortex.core.entities import FileRecord, Manifest

log = logging.getLogger(__name__)

# BIDS sidecar fields that carry acquisition parameters
_SIDECAR_PARAM_FIELDS = {
    "SamplingFrequency", "RecordingDuration", "TaskName",
    "EEGChannelCount", "MEGChannelCount", "ECOGChannelCount", "SEEGChannelCount",
    "EOGChannelCount", "EMGChannelCount", "MiscChannelCount",
    "RepetitionTime", "NumberOfVolumesDiscardedByScanner",
    "NumberOfVolumesDiscardedByUser", "PowerLineFrequency",
}

# Modalities whose sidecars carry SamplingFrequency / RecordingDuration
_SIGNAL_MODALITIES = {"eeg", "meg", "ieeg", "fnirs"}
_FMRI_MODALITIES = {"bold", "fmri"}
_VOLUME_MODALITIES = {"t1w", "t2w", "dwi", "t2star", "flair", "pet"}


# ── Per-file acquisition record ───────────────────────────────────────────────

@dataclass
class AcquisitionParams:
    """Acquisition parameters for one BIDS signal file, from sidecar or header."""
    path: str
    modality: str | None
    subject: str | None
    session: str | None
    task: str | None
    run: str | None
    sfreq: float | None = None           # sampling frequency (Hz)
    recording_duration_s: float | None = None
    n_channels: int | None = None        # total signal channels
    tr_s: float | None = None            # fMRI repetition time (s)
    n_volumes: int | None = None         # fMRI volumes (from header or sidecar)
    n_volumes_discarded: int = 0         # dummy scans
    voxel_sizes_mm: tuple[float, ...] | None = None
    fmri_shape: tuple[int, ...] | None = None
    task_name: str | None = None
    source: str = "sidecar"             # "sidecar" | "nifti_header" | "estimated"

    @property
    def effective_duration_s(self) -> float | None:
        """Signal duration after discarding dummy scans."""
        if self.recording_duration_s is not None:
            return self.recording_duration_s
        if self.tr_s and self.n_volumes:
            eff_vols = max(0, self.n_volumes - self.n_volumes_discarded)
            return eff_vols * self.tr_s
        return None

    @property
    def n_windows(self) -> int | None:
        """Placeholder: computed by SignalBudget.estimate_windows()."""
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "modality": self.modality,
            "subject": self.subject,
            "task": self.task,
            "sfreq": self.sfreq,
            "recording_duration_s": round(self.recording_duration_s, 1) if self.recording_duration_s else None,
            "n_channels": self.n_channels,
            "tr_s": self.tr_s,
            "n_volumes": self.n_volumes,
            "effective_duration_s": round(self.effective_duration_s, 1) if self.effective_duration_s else None,
            "source": self.source,
        }


# ── Budget report ─────────────────────────────────────────────────────────────

@dataclass
class ModalityBudget:
    """Signal content summary for one modality."""
    modality: str
    n_files: int = 0
    n_subjects: int = 0
    total_duration_hours: float = 0.0
    mean_sfreq: float | None = None
    mean_n_channels: float | None = None
    mean_recording_min: float | None = None
    tasks: list[str] = field(default_factory=list)

    # Windowed sample estimates (populated by estimate_windows)
    n_windows_total: int = 0
    window_duration_s: float | None = None
    window_overlap: float = 0.0

    @property
    def total_hours(self) -> float:
        """Backward-compatible alias for ``total_duration_hours``."""
        return self.total_duration_hours

    @property
    def avg_sfreq(self) -> float:
        """Backward-compatible alias for ``mean_sfreq``.

        Older scenario scripts formatted this value numerically even for
        modalities such as fMRI where a sampling frequency is not defined in
        the sidecar summary. Return 0.0 for that display-only compatibility
        surface while preserving ``mean_sfreq=None`` in structured output.
        """
        return self.mean_sfreq if self.mean_sfreq is not None else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "n_files": self.n_files,
            "n_subjects": self.n_subjects,
            "total_hours": round(self.total_duration_hours, 2),
            "mean_sfreq_hz": round(self.mean_sfreq, 1) if self.mean_sfreq else None,
            "mean_channels": round(self.mean_n_channels, 1) if self.mean_n_channels else None,
            "mean_duration_min": round(self.mean_recording_min, 1) if self.mean_recording_min else None,
            "tasks": self.tasks,
            "n_windows_total": self.n_windows_total,
        }


@dataclass
class SignalBudget:
    """Acquisition parameter report for a dataset, built from remote sidecars.

    Answers questions like:
      - "How many hours of EEG does this dataset contain?"
      - "How many 2-second windows will I get?"
      - "Is there enough data to train a 3-class classifier?"
      - "Which subjects contribute the most data?"

    Attributes
    ----------
    dataset_id:
        The OpenNeuro dataset ID.
    acquisition_records:
        Per-file acquisition parameters.
    modality_budgets:
        Aggregated per-modality summaries.
    n_sidecars_fetched / n_sidecars_failed:
        Fetch success counts.
    adequacy_warnings:
        Actionable data adequacy observations.
    """
    dataset_id: str
    acquisition_records: list[AcquisitionParams] = field(default_factory=list)
    modality_budgets: dict[str, ModalityBudget] = field(default_factory=dict)
    n_sidecars_fetched: int = 0
    n_sidecars_failed: int = 0
    adequacy_warnings: list[str] = field(default_factory=list)

    @property
    def total_hours(self) -> float:
        return sum(b.total_duration_hours for b in self.modality_budgets.values())

    @property
    def n_subjects_with_signal(self) -> int:
        return len({r.subject for r in self.acquisition_records if r.subject})

    def estimate_windows(
        self,
        window_duration_s: float,
        overlap: float = 0.0,
        modality: str | None = None,
    ) -> dict[str, int]:
        """Estimate number of fixed-stride windows across the dataset.

        Parameters
        ----------
        window_duration_s:
            Window length in seconds.
        overlap:
            Fraction of window overlap (0–1). 0.5 = 50% overlap.
        modality:
            If set, only estimate for this modality.

        Returns
        -------
        dict[modality, n_windows]
        """
        step_s = window_duration_s * (1.0 - overlap)
        if step_s <= 0:
            raise ValueError(f"Overlap {overlap} too large for window {window_duration_s}s")

        result: dict[str, int] = {}
        for rec in self.acquisition_records:
            mod = rec.modality or "unknown"
            if modality and mod != modality:
                continue
            dur = rec.effective_duration_s
            if dur and dur >= window_duration_s:
                n = max(0, int((dur - window_duration_s) / step_s) + 1)
                result[mod] = result.get(mod, 0) + n

        # Update modality budgets
        for mod, n in result.items():
            if mod in self.modality_budgets:
                mb = self.modality_budgets[mod]
                mb.n_windows_total = n
                mb.window_duration_s = window_duration_s
                mb.window_overlap = overlap

        return result

    def per_subject_windows(
        self, window_duration_s: float, overlap: float = 0.0
    ) -> dict[str, int]:
        """Estimate windows per subject — identifies data-heavy outliers."""
        step_s = window_duration_s * (1.0 - overlap)
        counts: dict[str, int] = {}
        for rec in self.acquisition_records:
            subj = rec.subject or "unknown"
            dur = rec.effective_duration_s
            if dur and dur >= window_duration_s:
                n = max(0, int((dur - window_duration_s) / step_s) + 1)
                counts[subj] = counts.get(subj, 0) + n
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def minimum_download_for_n_windows(
        self,
        n_windows_target: int,
        window_duration_s: float,
        overlap: float = 0.0,
        modality: str | None = None,
    ) -> dict[str, Any]:
        """Find the minimum subjects needed to reach a window count target.

        Returns the sorted list of subjects (most data first) and cumulative
        windows, so you can plan a minimal selective download.
        """
        step_s = window_duration_s * (1.0 - overlap)
        per_subj: dict[str, int] = {}
        for rec in self.acquisition_records:
            mod = rec.modality or "unknown"
            if modality and mod != modality:
                continue
            subj = rec.subject or "unknown"
            dur = rec.effective_duration_s
            if dur and dur >= window_duration_s:
                n = max(0, int((dur - window_duration_s) / step_s) + 1)
                per_subj[subj] = per_subj.get(subj, 0) + n

        ranked = sorted(per_subj.items(), key=lambda x: -x[1])
        cumulative = 0
        selected: list[dict] = []
        for subj, n in ranked:
            cumulative += n
            selected.append({"subject": subj, "n_windows": n, "cumulative": cumulative})
            if cumulative >= n_windows_target:
                break

        return {
            "target_windows": n_windows_target,
            "subjects_needed": len(selected),
            "windows_achieved": cumulative,
            "subjects": selected,
            "coverage_fraction": min(1.0, cumulative / max(1, n_windows_target)),
        }

    def summary(self) -> str:
        lines = [
            f"Signal Budget — {self.dataset_id}",
            f"Sidecars fetched: {self.n_sidecars_fetched} ({self.n_sidecars_failed} failed)",
            f"Total signal: {self.total_hours:.2f} hours across {self.n_subjects_with_signal} subjects",
            "",
        ]
        for mod, mb in self.modality_budgets.items():
            lines.append(
                f"  {mod:12s}  {mb.total_duration_hours:.2f}h  "
                f"{mb.n_subjects}subs  "
                f"sfreq={mb.mean_sfreq:.0f}Hz  " if mb.mean_sfreq else f"  {mod:12s}  {mb.total_duration_hours:.2f}h  {mb.n_subjects}subs  "
            )
        if self.adequacy_warnings:
            lines += ["", "Adequacy warnings:"]
            for w in self.adequacy_warnings:
                lines.append(f"  • {w}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "total_hours": round(self.total_hours, 3),
            "n_subjects_with_signal": self.n_subjects_with_signal,
            "n_sidecars_fetched": self.n_sidecars_fetched,
            "n_sidecars_failed": self.n_sidecars_failed,
            "modality_budgets": {k: v.to_dict() for k, v in self.modality_budgets.items()},
            "adequacy_warnings": self.adequacy_warnings,
        }


# ── Estimator ─────────────────────────────────────────────────────────────────

class SignalBudgetEstimator:
    """Estimate signal content from remote JSON sidecars and NIfTI headers.

    Usage::

        estimator = SignalBudgetEstimator(gateway)
        budget = estimator.estimate(manifest)
        print(budget.summary())
        windows = budget.estimate_windows(window_duration_s=2.0, overlap=0.5)
    """

    def __init__(self, gateway: RemoteFileGateway | None = None) -> None:
        self._gateway = gateway or RemoteFileGateway()

    def estimate(
        self,
        manifest: Manifest,
        *,
        concurrency: int = 24,
        include_nifti_headers: bool = True,
        max_sidecars: int | None = None,
    ) -> SignalBudget:
        """Fetch sidecars + NIfTI headers concurrently and build budget.

        Parameters
        ----------
        include_nifti_headers:
            When True, fetches NIfTI headers (352 bytes each via Range) for
            fMRI/MRI files whose sidecars may not carry RepetitionTime.
        max_sidecars:
            Cap the number of sidecars to fetch (for very large datasets).
        """
        budget = SignalBudget(dataset_id=manifest.dataset_id)

        # ── Step 1: Build per-signal sidecar chains + flat URL map ───────
        sidecar_url_map, signal_chains = _build_sidecar_chains(manifest)

        if max_sidecars is not None:
            # Cap by limiting the number of signal files, not sidecar URLs,
            # so we preserve full chains for the files we do process.
            signal_chains = signal_chains[:max_sidecars]
            needed = {p for _, chain in signal_chains for p in chain}
            sidecar_url_map = {k: v for k, v in sidecar_url_map.items() if k in needed}

        log.info(
            "Fetching %d unique sidecars for %d signal files...",
            len(sidecar_url_map), len(signal_chains),
        )
        sidecar_results = self._gateway.batch_fetch_json(sidecar_url_map, concurrency=concurrency)

        # ── Step 2: Merge sidecar chain per signal file → AcquisitionParams
        acquisitions: list[AcquisitionParams] = []
        n_failed = 0
        nifti_to_fetch: dict[str, tuple[str, FileRecord]] = {}  # sig_path → (nifti_url, nifti_fr)

        nifti_index = _build_nifti_index(manifest) if include_nifti_headers else {}

        for sig_file, chain_paths in signal_chains:
            merged: dict = {}
            any_ok = False
            for spath in chain_paths:
                result = sidecar_results.get(spath)
                if isinstance(result, Exception) or result is None:
                    log.debug("Sidecar fetch failed %s: %s", spath, result)
                    n_failed += 1
                elif isinstance(result, dict):
                    merged.update(result)   # BIDS: later (more-specific) wins
                    any_ok = True

            if not any_ok:
                continue

            params = _parse_sidecar(sig_file, merged)

            # Queue NIfTI header fetch when fMRI sidecar missing TR / volumes
            if (
                include_nifti_headers
                and sig_file.modality in _FMRI_MODALITIES
                and (params.tr_s is None or params.n_volumes is None)
            ):
                key = (sig_file.subject, sig_file.session, sig_file.task, sig_file.run)
                nifti_fr = nifti_index.get(key)
                if nifti_fr and nifti_fr.urls:
                    nifti_to_fetch[sig_file.path] = (_pick_url(nifti_fr), nifti_fr)

            acquisitions.append(params)

        budget.n_sidecars_fetched = len(acquisitions)
        budget.n_sidecars_failed = n_failed

        # ── Step 3: NIfTI header fetches for fMRI files ───────────────────
        if nifti_to_fetch:
            log.info("Fetching %d NIfTI headers (352 bytes each)...", len(nifti_to_fetch))
            nifti_url_map = {k: v[0] for k, v in nifti_to_fetch.items()}
            header_results = _fetch_nifti_headers_concurrent(
                self._gateway, nifti_url_map, concurrency=concurrency
            )
            # Update acquisitions with header-derived params (keyed by signal file path)
            acq_by_path = {a.path: a for a in acquisitions}
            for sig_path, hdr in header_results.items():
                if isinstance(hdr, NIfTIHeader) and sig_path in acq_by_path:
                    a = acq_by_path[sig_path]
                    if a.tr_s is None and hdr.tr_s:
                        a.tr_s = hdr.tr_s
                    if a.n_volumes is None and hdr.n_volumes:
                        a.n_volumes = hdr.n_volumes
                    if hdr.shape:
                        a.fmri_shape = hdr.shape
                    if hdr.voxel_sizes_mm:
                        a.voxel_sizes_mm = hdr.voxel_sizes_mm
                    a.source = "nifti_header"

        budget.acquisition_records = acquisitions

        # ── Step 4: Aggregate into per-modality budgets ───────────────────
        _aggregate_budgets(budget)

        # ── Step 5: Adequacy warnings ─────────────────────────────────────
        budget.adequacy_warnings = _adequacy_warnings(budget)

        return budget


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_sidecar_chains(
    manifest: Manifest,
) -> tuple[dict[str, str], list[tuple[FileRecord, list[str]]]]:
    """Build data structures for per-signal-file BIDS sidecar inheritance.

    Returns
    -------
    url_map:
        ``{sidecar_path: cdn_url}`` — deduplicated across all signal files so
        each unique sidecar URL is fetched exactly once.
    signal_chains:
        ``[(signal_file, [sidecar_path, ...]), ...]`` — ordered most-general
        to most-specific. Used after fetching to merge params per signal file.
    """
    from qortex.manifest.sidecar import SidecarResolver
    from qortex.client.remote import _pick_url

    resolver = SidecarResolver(manifest.files)
    url_map: dict[str, str] = {}
    signal_chains: list[tuple[FileRecord, list[str]]] = []

    signal_files = [
        f for f in manifest.files
        if not f.is_dir
        and f.modality in (_SIGNAL_MODALITIES | _FMRI_MODALITIES)
        and f.extension not in (".json", ".tsv", ".csv", ".bvec", ".bval")
        and f.subject
    ]

    for sig_file in signal_files:
        chain = resolver.resolve(sig_file)
        chain_paths: list[str] = []
        for sidecar_fr in chain:
            if not sidecar_fr.urls:
                continue
            try:
                url = _pick_url(sidecar_fr)
                url_map[sidecar_fr.path] = url   # dedup: same path = same fetch
                chain_paths.append(sidecar_fr.path)
            except Exception:
                pass
        if chain_paths:
            signal_chains.append((sig_file, chain_paths))

    return url_map, signal_chains


def _build_nifti_index(manifest: Manifest) -> dict[tuple, FileRecord]:
    """Map BIDS entity key → NIfTI FileRecord for fMRI files."""
    index: dict[tuple, FileRecord] = {}
    for f in manifest.files:
        if f.is_dir or f.extension not in (".nii", ".nii.gz"):
            continue
        if f.modality in _FMRI_MODALITIES or f.suffix in ("bold", "cbv"):
            key = (f.subject, f.session, f.task, f.run)
            index[key] = f
    return index


def _parse_sidecar(fr: FileRecord, params: dict[str, Any]) -> AcquisitionParams:
    """Extract acquisition parameters from a parsed JSON sidecar dict."""
    sfreq = _to_float(params.get("SamplingFrequency"))
    duration = _to_float(params.get("RecordingDuration"))
    tr = _to_float(params.get("RepetitionTime"))

    # Channel count: sum all channel type counts
    ch_count_fields = [
        "EEGChannelCount", "MEGChannelCount", "ECOGChannelCount",
        "SEEGChannelCount", "EOGChannelCount",
    ]
    n_channels: int | None = None
    ch_total = sum(int(params[k]) for k in ch_count_fields if k in params and params[k])
    if ch_total > 0:
        n_channels = ch_total

    n_vols_disc = int(
        (_to_float(params.get("NumberOfVolumesDiscardedByScanner")) or 0)
        + (_to_float(params.get("NumberOfVolumesDiscardedByUser")) or 0)
    )

    return AcquisitionParams(
        path=fr.path,
        modality=fr.modality,
        subject=fr.subject,
        session=fr.session,
        task=fr.task,
        run=fr.run,
        sfreq=sfreq,
        recording_duration_s=duration,
        n_channels=n_channels,
        tr_s=tr,
        n_volumes_discarded=n_vols_disc,
        task_name=params.get("TaskName"),
        source="sidecar",
    )


def _fetch_nifti_headers_concurrent(
    gateway: RemoteFileGateway,
    url_map: dict[str, str],
    concurrency: int,
) -> dict[str, Any]:
    """Fetch NIfTI headers (Range requests) concurrently using _run_async."""
    from qortex.client.remote import (
        _GZIP_RANGE_BYTES,
        _async_batch_fetch,
        _decompress_nifti_header,
        _parse_nifti_header,
        _run_async,
    )

    # _async_batch_fetch calls fetch_fn(url, client, max_bytes) — accept and ignore max_bytes
    async def _fetch_header(url: str, client, max_bytes: int = _GZIP_RANGE_BYTES) -> NIfTIHeader:
        is_gz = ".gz" in url.lower()
        range_bytes = _GZIP_RANGE_BYTES if is_gz else 352
        r = await client.get(url, headers={"Range": f"bytes=0-{range_bytes - 1}"})
        raw = r.content
        header_bytes = _decompress_nifti_header(raw, url) if is_gz else raw
        return _parse_nifti_header(header_bytes, len(raw))

    return _run_async(
        _async_batch_fetch(
            url_map=url_map,
            fetch_fn=_fetch_header,
            concurrency=concurrency,
            cfg=gateway._cfg,
            max_bytes=_GZIP_RANGE_BYTES,
        )
    )


def _aggregate_budgets(budget: SignalBudget) -> None:
    modality_data: dict[str, dict] = {}

    for rec in budget.acquisition_records:
        mod = rec.modality or "unknown"
        if mod not in modality_data:
            modality_data[mod] = {
                "subjects": set(),
                "durations": [],
                "sfreqs": [],
                "channels": [],
                "tasks": set(),
            }
        d = modality_data[mod]
        if rec.subject:
            d["subjects"].add(rec.subject)
        dur = rec.effective_duration_s
        if dur:
            d["durations"].append(dur)
        if rec.sfreq:
            d["sfreqs"].append(rec.sfreq)
        if rec.n_channels:
            d["channels"].append(rec.n_channels)
        if rec.task:
            d["tasks"].add(rec.task)

    for mod, d in modality_data.items():
        total_dur_h = sum(d["durations"]) / 3600.0
        mb = ModalityBudget(
            modality=mod,
            n_files=sum(1 for r in budget.acquisition_records if (r.modality or "unknown") == mod),
            n_subjects=len(d["subjects"]),
            total_duration_hours=total_dur_h,
            mean_sfreq=statistics.mean(d["sfreqs"]) if d["sfreqs"] else None,
            mean_n_channels=statistics.mean(d["channels"]) if d["channels"] else None,
            mean_recording_min=statistics.mean(d["durations"]) / 60 if d["durations"] else None,
            tasks=sorted(d["tasks"]),
        )
        budget.modality_budgets[mod] = mb


def _adequacy_warnings(budget: SignalBudget) -> list[str]:
    warnings: list[str] = []

    if budget.total_hours < 1.0:
        warnings.append(
            f"Total signal is only {budget.total_hours * 60:.0f} minutes. "
            "Most EEG/MEG classifiers need >2 hours of data for reliable results."
        )

    for mod, mb in budget.modality_budgets.items():
        if mb.n_subjects < 10 and mod in _SIGNAL_MODALITIES:
            warnings.append(
                f"Modality '{mod}' has only {mb.n_subjects} subject(s). "
                "Subject-independent generalization will be very limited."
            )
        if mb.mean_sfreq and mb.mean_sfreq < 100 and mod == "eeg":
            warnings.append(
                f"Mean EEG sampling frequency is only {mb.mean_sfreq:.0f} Hz. "
                "Gamma-band (>40 Hz) features will be unavailable."
            )

    return warnings


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) and f > 0 else None
    except (TypeError, ValueError):
        return None
