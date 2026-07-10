"""ConversionPipeline — orchestrates load → window → split → write.

Key design decisions
--------------------
* **Streaming-first**: the pipeline never holds all loaded arrays in memory at
  once.  When split_spec is None the stream goes straight from loader →
  windower → writer.  When split_spec is set, a two-pass approach is used:
  pass-1 collects only lightweight metadata (subject, label, source) to assign
  splits; pass-2 reloads and streams to the writer with split labels attached.

* **Failure tracking**: every load / sample-extraction failure is counted and
  reported in ConversionResult.  Users can see exactly how many files were
  skipped and why, rather than getting a silent partial artifact.

* **Event-aligned windowing**: when window_spec.event_aligned=True and an
  events EventsRecord is available for the file, event_aligned_windows() is
  used; otherwise falls back to fixed_windows().
"""

from __future__ import annotations

import logging
import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from qortex.core.entities import (
    ArtifactManifest,
    ConversionResult,
    FileRecord,
    Manifest,
    SampleRecord,
)
from qortex.convert.formats import get_writer
from qortex.convert.provenance import build_provenance, save_provenance
from qortex.convert.splits import SplitSpec, apply_split
from qortex.convert.windows import WindowSpec, event_aligned_windows, fixed_windows
from qortex.parse._registry import LoaderRegistry

log = logging.getLogger(__name__)


@dataclass
class _LoadStats:
    n_loaded: int = 0
    n_skipped_metadata: int = 0
    n_skipped_no_loader: int = 0
    n_skipped_missing: int = 0
    n_failed: int = 0
    failed_files: list[str] = field(default_factory=list)


_NON_SAMPLE_BIDS_SUFFIXES = frozenset({
    "participants",
    "sessions",
    "scans",
    "channels",
    "electrodes",
    "coordsystem",
})


def _is_non_sample_metadata_file(file_rec: FileRecord) -> bool:
    """Return True for BIDS metadata tables that must not become ML samples."""
    if file_rec.is_essential:
        return True
    if file_rec.suffix in _NON_SAMPLE_BIDS_SUFFIXES:
        return True
    if file_rec.filename in {"participants.tsv", "participants.json", "sessions.tsv"}:
        return True
    return False


class ConversionPipeline:
    """End-to-end ETL: BIDS files → load → window → split → ML format.

    The pipeline is deliberately serial (no async) so it can run inside any
    Jupyter kernel or HPC job scheduler without event-loop conflicts.

    Parameters
    ----------
    manifest:
        Built by ``manifest.builder.ManifestBuilder``.
    data_dir:
        Root of the locally downloaded BIDS tree.
    output_dir:
        Destination for the converted artifact.
    output_format:
        ``"parquet"`` | ``"zarr"`` | ``"hdf5"`` | ``"webdataset"`` |
        ``"huggingface"`` | ``"tfrecord"``.
    window_spec:
        Fixed-stride or event-aligned window config.  ``None`` = full signal.
    split_spec:
        Train / val / test split strategy.  ``None`` = no split column set.
    shard_size:
        Samples per output shard for sharded formats.
    loader_registry:
        Override the global loader registry (useful for testing).
    skip_missing:
        If True (default), failed files are logged and counted but do not abort
        the pipeline.  Set False to raise on first failure.
    """

    def __init__(
        self,
        manifest: Manifest,
        data_dir: Path,
        output_dir: Path,
        output_format: str = "parquet",
        window_spec: WindowSpec | None = None,
        split_spec: SplitSpec | None = None,
        shard_size: int = 1000,
        loader_registry: LoaderRegistry | None = None,
        skip_missing: bool = True,
    ) -> None:
        self.manifest = manifest
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_format = output_format
        self.window_spec = window_spec
        self.split_spec = split_spec
        self.shard_size = shard_size
        self.skip_missing = skip_missing

        if loader_registry is None:
            self._registry = LoaderRegistry()
            self._registry.discover()
        else:
            self._registry = loader_registry

        self._stats = _LoadStats()

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> ConversionResult:
        t0 = time.monotonic()
        self._stats = _LoadStats()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = get_writer(self.output_format)

        if self.split_spec is not None:
            # Two-pass: collect metadata first, assign splits, stream with labels.
            n_samples, n_subjects, split_counts, source_files = (
                self._run_with_split(writer)
            )
        else:
            # Single streaming pass: load → window → write.
            n_samples, n_subjects, source_files = self._run_streaming(writer)
            split_counts = {}

        prov = build_provenance(
            dataset_id=self.manifest.dataset_id,
            snapshot=self.manifest.snapshot or "latest",
            operation="convert",
            output_path=self.output_dir,
            config={
                "output_format": self.output_format,
                "window_spec": vars(self.window_spec) if self.window_spec else None,
                "split_spec": vars(self.split_spec) if self.split_spec else None,
                "shard_size": self.shard_size,
            },
        )
        save_provenance(prov, self.output_dir)
        artifact_manifest = self._write_artifact_manifest(
            n_samples=n_samples,
            n_subjects=n_subjects,
            split_counts=split_counts,
            source_files=source_files,
        )

        elapsed = time.monotonic() - t0
        s = self._stats
        warnings: list[str] = []
        if s.n_failed:
            warnings.append(
                f"{s.n_failed} file(s) failed to load: {', '.join(s.failed_files[:5])}"
                + (" …" if len(s.failed_files) > 5 else "")
            )
        if s.n_skipped_no_loader:
            warnings.append(
                f"{s.n_skipped_no_loader} file(s) had no registered loader."
            )

        log.info(
            "ConversionPipeline finished: %d samples, %d subjects in %.1fs "
            "(%d failed, %d no-loader, %d missing-local)",
            n_samples, n_subjects, elapsed,
            s.n_failed, s.n_skipped_no_loader, s.n_skipped_missing,
        )

        return ConversionResult(
            output_format=self.output_format,
            output_path=self.output_dir,
            n_samples=n_samples,
            n_subjects=n_subjects,
            splits=split_counts,
            elapsed=elapsed,
            provenance=prov,
            artifact_manifest=artifact_manifest,
            warnings=warnings,
        )

    # ── Streaming run (no split) ──────────────────────────────────────────

    def _run_streaming(
        self, writer
    ) -> tuple[int, int, list[str]]:
        """Single-pass: stream load → window → write without buffering arrays."""
        sample_stream = self._sample_stream()
        windowed = self._apply_windows_streaming(sample_stream)

        subjects: set[str] = set()
        source_files: set[str] = set()
        n_samples = 0

        def _annotate(stream: Iterator[SampleRecord]) -> Iterator[SampleRecord]:
            nonlocal n_samples
            for s in stream:
                if s.subject:
                    subjects.add(s.subject)
                src = s.provenance.get("source_path") or s.provenance.get("source")
                if src:
                    source_files.add(src)
                n_samples += 1
                yield s

        writer.write(_annotate(windowed), self.output_dir, shard_size=self.shard_size)
        return n_samples, len(subjects), sorted(source_files)

    # ── Two-pass run (with split) ─────────────────────────────────────────

    def _run_with_split(
        self, writer
    ) -> tuple[int, int, dict[str, int], list[str]]:
        """Two-pass: pass-1 builds split map; pass-2 streams with split labels.

        The split key is a monotonic global sample index injected into each
        sample's provenance during pass-1 (``_sample_idx``).  Pass-2 assigns
        the same index in the same iteration order and looks it up in the
        split_map.  This is collision-proof even when multiple events share the
        same (source, onset, subject) triple — e.g. repeated baseline events,
        duplicate trial types, or multi-subject tables with identical onsets.
        """
        # Pass 1: collect lightweight metadata (no signal arrays stored).
        meta_samples: list[SampleRecord] = []
        for sample_idx, sample in enumerate(
            self._apply_windows_streaming(self._sample_stream())
        ):
            # Inject a deterministic, globally unique sample index.
            prov_with_idx = {**sample.provenance, "_sample_idx": sample_idx}
            meta = SampleRecord(
                data=None,
                label=sample.label,
                label_name=sample.label_name,
                subject=sample.subject,
                session=sample.session,
                task=sample.task,
                run=sample.run,
                modality=sample.modality,
                onset=sample.onset,
                duration=sample.duration,
                sfreq=sample.sfreq,
                provenance=prov_with_idx,
            )
            meta_samples.append(meta)

        total_pass1 = len(meta_samples)

        # Assign splits using lightweight metadata.
        train, val, test = apply_split(meta_samples, self.split_spec)  # type: ignore[arg-type]

        # Build collision-proof split map: sample_idx (int) → split label.
        split_map: dict[int, str] = {}
        for part, label in [(train, "train"), (val, "val"), (test, "test")]:
            for s in part:
                idx = s.provenance.get("_sample_idx")
                if idx is None:
                    # Defensive: should never happen — fall back to string key.
                    src = s.provenance.get("source_path") or s.provenance.get("source") or ""
                    fallback_key = hash(f"{src}|{s.onset or 0:.6f}|{s.subject or ''}|{id(s)}")
                    split_map[fallback_key] = label  # type: ignore[index]
                else:
                    split_map[idx] = label

        split_counts = {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        }
        log.info("Split assignment: %s (total pass-1 samples: %d)", split_counts, total_pass1)

        # Pass 2: re-stream with split labels injected.
        # The two passes are deterministic because _sample_stream() and
        # _apply_windows_streaming() are pure functions of the manifest +
        # local files; the iteration order is identical across both passes.
        subjects: set[str] = set()
        source_files: set[str] = set()
        n_samples = 0

        def _with_split(stream: Iterator[SampleRecord]) -> Iterator[SampleRecord]:
            nonlocal n_samples
            for sample_idx, s in enumerate(stream):
                s.split = split_map.get(sample_idx)
                src = s.provenance.get("source_path") or s.provenance.get("source") or ""
                if s.subject:
                    subjects.add(s.subject)
                if src:
                    source_files.add(src)
                n_samples += 1
                yield s

        windowed2 = self._apply_windows_streaming(self._sample_stream())
        writer.write(_with_split(windowed2), self.output_dir, shard_size=self.shard_size)
        return n_samples, len(subjects), split_counts, sorted(source_files)

    # ── Artifact manifest ─────────────────────────────────────────────────

    def _write_artifact_manifest(
        self,
        *,
        n_samples: int,
        n_subjects: int,
        split_counts: dict[str, int],
        source_files: list[str],
    ) -> ArtifactManifest:
        s = self._stats
        payload = {
            "dataset_id": self.manifest.dataset_id,
            "snapshot": self.manifest.snapshot,
            "format": self.output_format,
            "source_files": source_files,
            "window": vars(self.window_spec) if self.window_spec else {},
            "split": vars(self.split_spec) if self.split_spec else {},
        }
        artifact_id = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        manifest = ArtifactManifest(
            artifact_id=artifact_id,
            dataset_id=self.manifest.dataset_id,
            snapshot=self.manifest.snapshot,
            doi=self.manifest.doi,
            output_format=self.output_format,
            output_path=str(self.output_dir),
            n_samples=n_samples,
            n_subjects=n_subjects,
            splits=split_counts,
            source_files=source_files,
            window_config=payload["window"],
            split_config=payload["split"],
            data_schema={
                "sample": "qortex.core.entities.SampleRecord",
                "data": "array | table | mapping",
                "label": "any | null",
            },
        )
        manifest_dict = manifest.model_dump()
        manifest_dict["n_failed_files"] = s.n_failed
        manifest_dict["n_skipped_files"] = s.n_skipped_no_loader + s.n_skipped_missing
        manifest_dict["failed_files"] = s.failed_files
        (self.output_dir / "artifact_manifest.json").write_text(
            json.dumps(manifest_dict, indent=2, default=str),
            encoding="utf-8",
        )
        return manifest

    # ── Internal helpers ──────────────────────────────────────────────────

    def _sample_stream(self) -> Iterator[SampleRecord]:
        """Stream SampleRecords from all loadable files in the manifest."""
        for file_rec in self.manifest.files:
            if file_rec.is_dir:
                continue

            if _is_non_sample_metadata_file(file_rec):
                self._stats.n_skipped_metadata += 1
                log.debug("Skipping BIDS metadata file during conversion: %s", file_rec.path)
                continue

            local_path = self.data_dir / file_rec.path
            if not local_path.exists():
                self._stats.n_skipped_missing += 1
                log.debug("File not found locally, skipping: %s", file_rec.path)
                continue

            loader = self._registry.resolve(file_rec)
            if loader is None:
                self._stats.n_skipped_no_loader += 1
                log.debug("No loader for file: %s", file_rec.path)
                continue

            try:
                record = loader.load(file_rec, local_path)
            except Exception as exc:
                self._stats.n_failed += 1
                self._stats.failed_files.append(file_rec.path)
                log.warning("Load failed for %s: %s", file_rec.path, exc)
                if not self.skip_missing:
                    raise
                continue

            try:
                count = 0
                for sample in loader.to_sample_records(record):
                    self._stats.n_loaded += 1
                    count += 1
                    yield sample
                log.debug("Loaded %d samples from %s", count, file_rec.path)
            except Exception as exc:
                self._stats.n_failed += 1
                self._stats.failed_files.append(file_rec.path)
                log.warning("to_sample_records failed for %s: %s", file_rec.path, exc)
                if not self.skip_missing:
                    raise

    def _apply_windows_streaming(
        self, stream: Iterator[SampleRecord]
    ) -> Iterator[SampleRecord]:
        """Apply windowing to each sample as it arrives — no buffering.

        When spec.event_aligned=True, the events companion file is looked up
        from the manifest by BIDS entities (subject/session/task/run) and the
        events TSV is loaded on first access (cached per unique key).
        Falls back to fixed_windows() when no events file is found.
        """
        if self.window_spec is None:
            yield from stream
            return

        import numpy as np
        spec = self.window_spec

        if spec.event_aligned:
            yield from self._apply_event_aligned_streaming(stream, spec)
            return

        for sample in stream:
            if sample.data is None or sample.sfreq is None:
                yield sample
                continue
            arr = np.asarray(sample.data)
            if arr.ndim == 2:
                yield from fixed_windows(sample, spec)
            else:
                # 3D or 4D volumes — pass through; no fixed-stride windowing.
                yield sample

    def _apply_event_aligned_streaming(
        self, stream: Iterator[SampleRecord], spec: WindowSpec
    ) -> Iterator[SampleRecord]:
        """Event-aligned windowing with lazy per-key events loading."""
        from qortex.core.entities import FileRecord
        from qortex.convert.windows import event_aligned_windows, fixed_windows

        # Build BIDS-entity index: (subject, session, task, run) → events FileRecord
        events_index: dict[tuple, FileRecord] = {}
        for fr in self.manifest.files:
            if fr.is_dir or fr.suffix != "events":
                continue
            key = (fr.subject, fr.session, fr.task, fr.run)
            events_index[key] = fr

        # Cache of loaded EventsRecord keyed by events file path
        events_cache: dict[str, object] = {}

        for sample in stream:
            if sample.data is None or sample.sfreq is None:
                yield sample
                continue

            import numpy as np
            arr = np.asarray(sample.data)
            if arr.ndim != 2:
                yield sample
                continue

            # Look up events by BIDS entities matching the sample's provenance
            sub = sample.subject
            sess = sample.session
            task = sample.task
            run = sample.run

            events_fr = events_index.get((sub, sess, task, run))
            if events_fr is None:
                # Partial key fallback: try without session, then without run
                events_fr = (
                    events_index.get((sub, None, task, run))
                    or events_index.get((sub, sess, task, None))
                    or events_index.get((sub, None, task, None))
                )

            if events_fr is None:
                log.debug(
                    "No events file found for sample %s (sub=%s ses=%s task=%s run=%s); "
                    "falling back to fixed_windows.",
                    sample.provenance.get("source", ""),
                    sub, sess, task, run,
                )
                yield from fixed_windows(sample, spec)
                continue

            events_path = events_fr.path
            if events_path not in events_cache:
                local_events = self.data_dir / events_path
                if not local_events.exists():
                    log.debug("Events file not found locally: %s; falling back.", events_path)
                    yield from fixed_windows(sample, spec)
                    continue
                try:
                    from qortex.parse.behavior import BehaviorLoader
                    loader = BehaviorLoader()
                    events_cache[events_path] = loader.load(events_fr, local_events)
                except Exception as exc:
                    log.warning("Failed to load events %s: %s; falling back.", events_path, exc)
                    events_cache[events_path] = None

            events_record = events_cache[events_path]
            if events_record is None:
                yield from fixed_windows(sample, spec)
                continue

            windows = list(event_aligned_windows(sample, events_record, spec))
            if not windows:
                log.debug(
                    "event_aligned_windows produced 0 windows for %s "
                    "(window_duration=%.2fs may exceed signal length or all events OOB).",
                    sample.provenance.get("source", ""), spec.duration_s,
                )
            yield from windows
