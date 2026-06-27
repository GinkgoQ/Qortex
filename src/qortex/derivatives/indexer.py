"""BIDS derivative directory indexer.

Scans the ``derivatives/`` sub-tree of a downloaded BIDS dataset, auto-detects
pipeline directories, and builds a queryable index of derivative files linked
back to their raw-data sources via BIDS entity matching.

Supported pipelines (auto-detected by directory name pattern):
  * fmriprep      — preprocessed BOLD, confounds, brain masks
  * freesurfer    — surface reconstructions
  * mriqc         — image quality metrics (group_T1w.tsv, group_bold.tsv)
  * mne-bids      — MNE-BIDS pipeline outputs
  * smriprep      — structural MRI preprocessing
  * qsiprep       — diffusion preprocessing
  * fsl-*         — FSL pipeline family
  * custom        — any other directory under derivatives/

Design principles
-----------------
* No nibabel/numpy imports at module level — keeps import time low.
* All file discovery is filesystem-only; never downloads.
* Entity extraction is regex-based and mirrors the BIDS spec.
* QC metric tables are parsed lazily (only when qc_table() is called).
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# BIDS entity extraction regex: matches sub-XX, ses-XX, task-XX, run-XX, etc.
_BIDS_ENTITY_RE = re.compile(
    r"(?:^|_)"
    r"(sub|ses|task|run|acq|dir|space|res|echo|part|hemi|den|proc|split)"
    r"-([^_\.\s/]+)"
)

# Pipeline name → canonical identifier mapping
_PIPELINE_ALIASES: dict[str, str] = {
    "fmriprep": "fmriprep",
    "fmri_prep": "fmriprep",
    "freesurfer": "freesurfer",
    "fs": "freesurfer",
    "mriqc": "mriqc",
    "mne-bids-pipeline": "mne-bids",
    "mne_bids_pipeline": "mne-bids",
    "mne-bids": "mne-bids",
    "smriprep": "smriprep",
    "qsiprep": "qsiprep",
    "eddyqc": "qsiprep",
}

# Files we know carry QC metrics, keyed by pipeline
_QC_TABLE_PATTERNS: dict[str, list[str]] = {
    "mriqc": ["group_T1w.tsv", "group_bold.tsv", "group_dwi.tsv"],
    "fmriprep": [],   # per-run confounds only; no single group TSV
}

# Derivative roles inferred from filename patterns
_ROLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"_desc-confounds_timeseries\.tsv$"), "confounds"),
    (re.compile(r"_desc-brain_mask\.nii(\.gz)?$"), "brain_mask"),
    (re.compile(r"_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold\.nii(\.gz)?$"), "preproc_bold_mni"),
    (re.compile(r"_space-T1w_desc-preproc_bold\.nii(\.gz)?$"), "preproc_bold_native"),
    (re.compile(r"_desc-preproc_T1w\.nii(\.gz)?$"), "preproc_T1w"),
    (re.compile(r"_desc-preproc_dwi\.nii(\.gz)?$"), "preproc_dwi"),
    (re.compile(r"_hemi-(L|R)_desc-smoothwm_surf\.gii$"), "surface"),
    (re.compile(r"_aparc\+aseg\.mgz$"), "segmentation"),
    (re.compile(r"_dseg\.nii(\.gz)?$"), "dseg"),
    (re.compile(r"_stat-tstat_statmap\.nii(\.gz)?$"), "tstat"),
    (re.compile(r"_space-\w+_res-\w+_\w+\.nii(\.gz)?$"), "registered_volume"),
]


def _extract_entities(filename: str) -> dict[str, str]:
    """Extract BIDS key-value entities from a filename stem."""
    return {m.group(1): m.group(2) for m in _BIDS_ENTITY_RE.finditer(filename)}


def _infer_role(filename: str) -> str:
    for pattern, role in _ROLE_PATTERNS:
        if pattern.search(filename):
            return role
    return "derivative"


def _canonical_pipeline(name: str) -> str:
    lower = name.lower().replace("-", "_")
    for alias, canonical in _PIPELINE_ALIASES.items():
        if lower == alias.replace("-", "_"):
            return canonical
    return name.lower()


@dataclass
class DerivativeFile:
    """One file in the derivatives directory."""

    pipeline: str
    path: str                        # BIDS-relative from bids_root
    absolute_path: Path
    filename: str
    extension: str
    entities: dict[str, str]         # parsed BIDS entities (no "sub-" prefix)
    role: str                        # e.g. "confounds", "brain_mask", "preproc_bold_mni"
    raw_source: str | None = None    # BIDS-relative path of the raw counterpart, if linked

    @property
    def subject(self) -> str | None:
        return self.entities.get("sub")

    @property
    def session(self) -> str | None:
        return self.entities.get("ses")

    @property
    def task(self) -> str | None:
        return self.entities.get("task")

    @property
    def run(self) -> str | None:
        return self.entities.get("run")

    @property
    def size(self) -> int | None:
        try:
            return self.absolute_path.stat().st_size
        except OSError:
            return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": self.pipeline,
            "path": self.path,
            "filename": self.filename,
            "extension": self.extension,
            "entities": self.entities,
            "role": self.role,
            "raw_source": self.raw_source,
            "size": self.size,
        }


@dataclass
class PipelineInfo:
    """Metadata about one discovered derivative pipeline."""

    name: str
    canonical_name: str
    directory: Path
    n_files: int = 0
    n_subjects: int = 0
    has_qc_tables: bool = False
    qc_table_paths: list[Path] = field(default_factory=list)
    description_json: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "directory": str(self.directory),
            "n_files": self.n_files,
            "n_subjects": self.n_subjects,
            "has_qc_tables": self.has_qc_tables,
            "qc_table_paths": [str(p) for p in self.qc_table_paths],
        }


class DerivativeIndex:
    """Queryable index of all derivative files for one dataset.

    Built by ``DerivativeIndexer.build()``.  Read-only after construction.
    """

    def __init__(
        self,
        bids_root: Path,
        files: list[DerivativeFile],
        pipeline_infos: list[PipelineInfo],
    ) -> None:
        self._bids_root = bids_root
        self._files = files
        self._pipeline_infos = {pi.canonical_name: pi for pi in pipeline_infos}

        # Build secondary indices for O(1) lookup
        self._by_subject: dict[str, list[DerivativeFile]] = {}
        self._by_pipeline: dict[str, list[DerivativeFile]] = {}
        self._by_raw: dict[str, list[DerivativeFile]] = {}

        for df in files:
            if df.subject:
                self._by_subject.setdefault(f"sub-{df.subject}", []).append(df)
            self._by_pipeline.setdefault(df.pipeline, []).append(df)
            if df.raw_source:
                self._by_raw.setdefault(df.raw_source, []).append(df)

    @property
    def pipelines(self) -> list[str]:
        """Canonical names of detected derivative pipelines."""
        return sorted(self._pipeline_infos)

    @property
    def n_files(self) -> int:
        return len(self._files)

    def pipeline_info(self, name: str) -> PipelineInfo | None:
        return self._pipeline_infos.get(_canonical_pipeline(name))

    def for_subject(
        self,
        subject: str,
        *,
        pipeline: str | None = None,
        role: str | None = None,
    ) -> list[DerivativeFile]:
        """Return all derivative files for one subject.

        Parameters
        ----------
        subject:
            BIDS subject ID with or without ``sub-`` prefix.
        pipeline:
            Optional: restrict to this pipeline.
        role:
            Optional: restrict to files of this role
            (``"confounds"``, ``"brain_mask"``, ``"preproc_bold_mni"``, etc.).
        """
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"
        files = self._by_subject.get(subject, [])
        if pipeline:
            canon = _canonical_pipeline(pipeline)
            files = [f for f in files if f.pipeline == canon]
        if role:
            files = [f for f in files if f.role == role]
        return files

    def for_raw(self, raw_path: str) -> list[DerivativeFile]:
        """Return derivative files linked to a specific raw BIDS path."""
        return self._by_raw.get(raw_path, [])

    def for_pipeline(
        self,
        pipeline: str,
        *,
        role: str | None = None,
        subject: str | None = None,
    ) -> list[DerivativeFile]:
        """Return all files from one pipeline, with optional further filtering."""
        files = self._by_pipeline.get(_canonical_pipeline(pipeline), [])
        if role:
            files = [f for f in files if f.role == role]
        if subject:
            sub_raw = subject.removeprefix("sub-")
            files = [f for f in files if f.entities.get("sub") == sub_raw]
        return files

    def confounds(
        self,
        subject: str | None = None,
        session: str | None = None,
        task: str | None = None,
        run: str | None = None,
    ) -> list[DerivativeFile]:
        """Return fMRIPrep confound TSV files, with optional entity filters."""
        files = self._by_pipeline.get("fmriprep", [])
        files = [f for f in files if f.role == "confounds"]
        if subject:
            raw = subject.removeprefix("sub-")
            files = [f for f in files if f.entities.get("sub") == raw]
        if session:
            raw = session.removeprefix("ses-")
            files = [f for f in files if f.entities.get("ses") == raw]
        if task:
            files = [f for f in files if f.entities.get("task") == task]
        if run:
            files = [f for f in files if f.entities.get("run") == run]
        return files

    def qc_table(self, pipeline: str = "mriqc") -> Any:
        """Parse and return a QC metric table as a Polars DataFrame.

        For MRIQC, concatenates all group_*.tsv files found in the pipeline
        directory into a single DataFrame.

        Returns an empty DataFrame if no QC tables exist.
        """
        import polars as pl

        canon = _canonical_pipeline(pipeline)
        info = self._pipeline_infos.get(canon)
        if info is None or not info.qc_table_paths:
            log.warning(
                "No QC tables found for pipeline %r. "
                "Ensure MRIQC has been run and derivatives/ is present.",
                pipeline,
            )
            return pl.DataFrame()

        frames: list[pl.DataFrame] = []
        for tsv in info.qc_table_paths:
            try:
                df = pl.read_csv(str(tsv), separator="\t", null_values=["n/a", "N/A"])
                df = df.with_columns(
                    pl.lit(tsv.stem).alias("_source_table"),
                )
                frames.append(df)
            except Exception as exc:
                log.warning("Cannot parse QC table %s: %s", tsv, exc)

        if not frames:
            return pl.DataFrame()

        return pl.concat(frames, how="diagonal")

    def confound_summary(
        self,
        subject: str,
        session: str | None = None,
        task: str | None = None,
    ) -> dict[str, Any]:
        """Parse fMRIPrep confound TSVs and return summary statistics.

        Returns a dict with per-run statistics including mean/max framewise
        displacement, mean DVARS, and number of high-motion volumes.
        """
        import csv as _csv
        import math

        confound_files = self.confounds(subject=subject, session=session, task=task)
        if not confound_files:
            return {}

        summaries: list[dict[str, Any]] = []
        for cf in confound_files:
            try:
                with open(cf.absolute_path, newline="", encoding="utf-8") as fh:
                    rows = list(_csv.DictReader(fh, delimiter="\t"))
            except Exception as exc:
                log.warning("Cannot read confounds %s: %s", cf.path, exc)
                continue

            fd_col = next(
                (c for c in (rows[0] if rows else {}) if "framewise_displacement" in c.lower()),
                None,
            )
            dvars_col = next(
                (c for c in (rows[0] if rows else {}) if "std_dvars" in c.lower()),
                None,
            )

            fd_vals: list[float] = []
            dvars_vals: list[float] = []
            for row in rows:
                if fd_col:
                    raw = row.get(fd_col, "")
                    if raw and raw.lower() not in ("n/a", "na", "nan", ""):
                        try:
                            fd_vals.append(float(raw))
                        except ValueError:
                            pass
                if dvars_col:
                    raw = row.get(dvars_col, "")
                    if raw and raw.lower() not in ("n/a", "na", "nan", ""):
                        try:
                            dvars_vals.append(float(raw))
                        except ValueError:
                            pass

            fd_mean = sum(fd_vals) / len(fd_vals) if fd_vals else None
            fd_max = max(fd_vals) if fd_vals else None
            dvars_mean = sum(dvars_vals) / len(dvars_vals) if dvars_vals else None
            n_high_motion = sum(1 for v in fd_vals if v > 0.5)

            summaries.append({
                "path": cf.path,
                "task": cf.task,
                "run": cf.run,
                "n_volumes": len(rows),
                "fd_mean": round(fd_mean, 4) if fd_mean is not None else None,
                "fd_max": round(fd_max, 4) if fd_max is not None else None,
                "dvars_mean": round(dvars_mean, 4) if dvars_mean is not None else None,
                "n_high_motion_volumes": n_high_motion,
                "high_motion_fraction": (
                    round(n_high_motion / len(rows), 4) if rows else None
                ),
            })

        return {"subject": subject, "runs": summaries}

    def subjects(self, pipeline: str | None = None) -> list[str]:
        """Return all subjects (``sub-XX`` form) present in derivatives."""
        files = self._files
        if pipeline:
            files = self.for_pipeline(pipeline)
        subs = {f"sub-{f.entities['sub']}" for f in files if f.entities.get("sub")}
        return sorted(subs)

    def summary(self) -> str:
        lines = [
            f"Derivative Index — {self.n_files} files across {len(self.pipelines)} pipelines",
        ]
        for canon in self.pipelines:
            info = self._pipeline_infos[canon]
            qc_tag = " [QC tables]" if info.has_qc_tables else ""
            lines.append(
                f"  {canon:<20} {info.n_files:>5} files  "
                f"{info.n_subjects:>3} subjects{qc_tag}"
            )
        return "\n".join(lines)


class DerivativeIndexer:
    """Scan a BIDS tree's ``derivatives/`` folder and build a DerivativeIndex.

    Parameters
    ----------
    bids_root:
        Root of the downloaded BIDS tree.
    max_depth:
        Maximum directory recursion depth within each pipeline directory.
        Increase if the pipeline writes deeply nested outputs.
    """

    def __init__(self, bids_root: Path, *, max_depth: int = 6) -> None:
        self.bids_root = Path(bids_root).expanduser().resolve()
        self.max_depth = max_depth
        self._derivatives_root = self.bids_root / "derivatives"

    # ── Public API ────────────────────────────────────────────────────────

    @cached_property
    def index(self) -> DerivativeIndex:
        """Build and cache the derivative index (filesystem scan)."""
        return self.build()

    def build(self, *, pipelines: list[str] | None = None) -> DerivativeIndex:
        """Scan derivatives/ and return a fresh DerivativeIndex.

        Parameters
        ----------
        pipelines:
            Optional allowlist of pipeline names to index.  When None, all
            sub-directories of ``derivatives/`` are indexed.
        """
        if not self._derivatives_root.is_dir():
            log.info("No derivatives/ directory found at %s", self.bids_root)
            return DerivativeIndex(
                bids_root=self.bids_root,
                files=[],
                pipeline_infos=[],
            )

        pipeline_dirs = sorted(
            p for p in self._derivatives_root.iterdir() if p.is_dir()
        )
        if pipelines:
            allowed = {_canonical_pipeline(n) for n in pipelines}
            pipeline_dirs = [
                p for p in pipeline_dirs
                if _canonical_pipeline(p.name) in allowed
            ]

        all_files: list[DerivativeFile] = []
        pipeline_infos: list[PipelineInfo] = []

        # Also index raw BIDS files for cross-linking
        raw_paths = self._index_raw_paths()

        for pipe_dir in pipeline_dirs:
            canon = _canonical_pipeline(pipe_dir.name)
            log.debug("Indexing pipeline: %s (%s)", pipe_dir.name, canon)

            pipe_files = self._scan_pipeline(pipe_dir, canon)
            self._link_to_raw(pipe_files, raw_paths)

            # Pipeline info
            description = self._load_description(pipe_dir)
            qc_paths = self._find_qc_tables(pipe_dir, canon)
            subjects = {f.entities.get("sub") for f in pipe_files if f.entities.get("sub")}

            info = PipelineInfo(
                name=pipe_dir.name,
                canonical_name=canon,
                directory=pipe_dir,
                n_files=len(pipe_files),
                n_subjects=len(subjects),
                has_qc_tables=bool(qc_paths),
                qc_table_paths=qc_paths,
                description_json=description,
            )
            pipeline_infos.append(info)
            all_files.extend(pipe_files)

        log.info(
            "Derivative index built: %d files across %d pipelines",
            len(all_files), len(pipeline_infos),
        )
        return DerivativeIndex(
            bids_root=self.bids_root,
            files=all_files,
            pipeline_infos=pipeline_infos,
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _scan_pipeline(self, pipe_dir: Path, pipeline: str) -> list[DerivativeFile]:
        files: list[DerivativeFile] = []
        for path in self._walk(pipe_dir, depth=0):
            if path.is_dir():
                continue
            rel = path.relative_to(self.bids_root)
            ext = _compound_extension(path.name)
            entities = _extract_entities(path.name)
            role = _infer_role(path.name)
            files.append(DerivativeFile(
                pipeline=pipeline,
                path=str(rel),
                absolute_path=path,
                filename=path.name,
                extension=ext,
                entities=entities,
                role=role,
            ))
        return files

    def _walk(self, directory: Path, depth: int) -> Any:
        if depth > self.max_depth:
            return
        try:
            for item in sorted(directory.iterdir()):
                yield item
                if item.is_dir():
                    yield from self._walk(item, depth + 1)
        except PermissionError:
            log.warning("Permission denied reading %s", directory)

    def _link_to_raw(
        self,
        deriv_files: list[DerivativeFile],
        raw_paths: dict[str, str],
    ) -> None:
        """Set raw_source on each derivative file via entity matching."""
        for df in deriv_files:
            ents = df.entities
            sub = ents.get("sub")
            if not sub:
                continue
            candidate = self._match_raw(ents, raw_paths)
            if candidate:
                df.raw_source = candidate

    def _match_raw(
        self,
        entities: dict[str, str],
        raw_paths: dict[str, str],
    ) -> str | None:
        """Find the most specific raw file matching the derivative's entities."""
        sub = entities.get("sub")
        ses = entities.get("ses")
        task = entities.get("task")
        run = entities.get("run")
        acq = entities.get("acq")

        best: str | None = None
        best_score = -1

        for key, raw_path in raw_paths.items():
            key_ents = _extract_entities(key)
            if key_ents.get("sub") != sub:
                continue
            if ses and key_ents.get("ses") != ses:
                continue
            if task and key_ents.get("task") is not None and key_ents.get("task") != task:
                continue
            if run and key_ents.get("run") is not None and key_ents.get("run") != run:
                continue

            score = sum([
                key_ents.get("ses") == ses if ses else 0,
                key_ents.get("task") == task if task else 0,
                key_ents.get("run") == run if run else 0,
                key_ents.get("acq") == acq if acq else 0,
            ])
            if score > best_score:
                best_score = score
                best = raw_path

        return best

    def _index_raw_paths(self) -> dict[str, str]:
        """Build {filename_stem: bids_rel_path} for raw NIfTI files."""
        raw: dict[str, str] = {}
        for path in self.bids_root.glob("sub-*/**/*.nii*"):
            if "derivatives" in path.parts:
                continue
            rel = str(path.relative_to(self.bids_root))
            raw[path.name] = rel
        return raw

    def _load_description(self, pipe_dir: Path) -> dict[str, Any]:
        desc_path = pipe_dir / "dataset_description.json"
        if desc_path.is_file():
            try:
                import json
                return json.loads(desc_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _find_qc_tables(self, pipe_dir: Path, pipeline: str) -> list[Path]:
        patterns = _QC_TABLE_PATTERNS.get(pipeline, [])
        found: list[Path] = []
        for pattern in patterns:
            matches = list(pipe_dir.glob(f"**/{pattern}"))
            found.extend(sorted(matches))
        # Also look for any group_*.tsv at top level (generic)
        if not found:
            top_level = list(pipe_dir.glob("group_*.tsv"))
            found.extend(sorted(top_level))
        return found

    # ── Properties delegating to cached index ─────────────────────────────

    @property
    def pipelines(self) -> list[str]:
        return self.index.pipelines

    def for_subject(self, subject: str, **kwargs) -> list[DerivativeFile]:
        return self.index.for_subject(subject, **kwargs)

    def for_raw(self, raw_path: str) -> list[DerivativeFile]:
        return self.index.for_raw(raw_path)

    def qc_table(self, pipeline: str = "mriqc"):
        return self.index.qc_table(pipeline)

    def confound_summary(self, subject: str, **kwargs) -> dict[str, Any]:
        return self.index.confound_summary(subject, **kwargs)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compound_extension(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return ".nii.gz"
    if filename.endswith(".tar.gz"):
        return ".tar.gz"
    _, dot, ext = filename.rpartition(".")
    return f".{ext}" if dot else ""
