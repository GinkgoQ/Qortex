"""ConversionPipeline — orchestrates load → window → split → write."""

from __future__ import annotations

import logging
import json
import hashlib
import time
from pathlib import Path
from typing import Iterator

from qortex.core.entities import (
    ArtifactManifest,
    ConversionResult,
    Manifest,
    SampleRecord,
)
from qortex.convert.formats import get_writer
from qortex.convert.provenance import build_provenance, save_provenance
from qortex.convert.splits import SplitSpec, apply_split
from qortex.convert.windows import WindowSpec, fixed_windows
from qortex.parse._registry import LoaderRegistry

log = logging.getLogger(__name__)


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
        Fixed-stride sliding window config.  ``None`` = no windowing (one
        SampleRecord per file, full signal).
    split_spec:
        Train / val / test split strategy.  ``None`` = no split column set.
    shard_size:
        Samples per output shard for sharded formats.
    loader_registry:
        Override the global loader registry (useful for testing).
    skip_missing:
        If True (default), silently skip files that have no loader or fail
        to load.  If False, raise on first failure.
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

    # ── Public API ────────────────────────────────────────────────────────

    def run(self) -> ConversionResult:
        t0 = time.monotonic()

        # 1. Load → (optional) window
        samples = list(self._load_all())
        log.info("Loaded %d samples from %s", len(samples), self.data_dir)

        # 2. Window
        if self.window_spec is not None:
            samples = list(self._apply_windows(samples))
            log.info("After windowing: %d samples", len(samples))

        # 3. Split
        split_counts: dict[str, int] = {}
        if self.split_spec is not None:
            train, val, test = apply_split(samples, self.split_spec)
            split_counts = {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            }
            samples = train + val + test
            log.info("Split: %s", split_counts)

        n_samples = len(samples)
        n_subjects = len({s.subject for s in samples if s.subject})
        source_files = sorted({
            src
            for s in samples
            for src in [
                s.provenance.get("source_path"),
                s.provenance.get("source"),
            ]
            if isinstance(src, str) and src
        })

        # 4. Write
        self.output_dir.mkdir(parents=True, exist_ok=True)
        writer = get_writer(self.output_format)
        writer.write(iter(samples), self.output_dir, shard_size=self.shard_size)

        # 5. Provenance
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
        log.info(
            "ConversionPipeline finished: %d samples, %d subjects in %.1fs",
            n_samples, n_subjects, elapsed,
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
        )

    def _write_artifact_manifest(
        self,
        *,
        n_samples: int,
        n_subjects: int,
        split_counts: dict[str, int],
        source_files: list[str],
    ) -> ArtifactManifest:
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
        (self.output_dir / "artifact_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    # ── Internal helpers ──────────────────────────────────────────────────

    def _load_all(self) -> Iterator[SampleRecord]:
        """Load each data file using its resolved loader.

        Lifecycle per file:
        1. resolve loader from registry
        2. call loader.load(file_rec, local_path)  → AnyRecord
        3. call loader.to_sample_records(record)   → Iterator[SampleRecord]
        """
        n_loaded = n_skipped = n_failed = 0

        for file_rec in self.manifest.files:
            if file_rec.is_dir:
                continue

            local_path = self.data_dir / file_rec.path
            if not local_path.exists():
                n_skipped += 1
                log.debug("File not found locally, skipping: %s", file_rec.path)
                continue

            loader = self._registry.resolve(file_rec)
            if loader is None:
                n_skipped += 1
                log.debug("No loader for file: %s", file_rec.path)
                continue

            try:
                record = loader.load(file_rec, local_path)
            except Exception as exc:
                if self.skip_missing:
                    n_failed += 1
                    log.warning("Load failed for %s: %s", file_rec.path, exc)
                    continue
                raise

            try:
                count = 0
                for sample in loader.to_sample_records(record):
                    yield sample
                    count += 1
                n_loaded += count
                log.debug("Loaded %d samples from %s", count, file_rec.path)
            except Exception as exc:
                if self.skip_missing:
                    n_failed += 1
                    log.warning("to_sample_records failed for %s: %s", file_rec.path, exc)
                    continue
                raise

        log.info(
            "_load_all: %d samples from %d files (%d skipped, %d failed)",
            n_loaded, n_loaded + n_skipped + n_failed, n_skipped, n_failed,
        )

    def _apply_windows(
        self, samples: list[SampleRecord]
    ) -> Iterator[SampleRecord]:
        assert self.window_spec is not None
        spec = self.window_spec
        for sample in samples:
            if sample.sfreq is not None and sample.data is not None:
                import numpy as np
                arr = np.asarray(sample.data)
                if arr.ndim == 2:
                    yield from fixed_windows(sample, spec)
                    continue
            yield sample
