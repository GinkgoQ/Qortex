"""Tensor harmonization reporter.

Analyses a collection of ImageRecords (or raw NIfTI paths) to detect all spatial
incompatibilities that would cause silent data corruption or runtime crashes in
PyTorch/MONAI/TorchIO dataloaders:

  * Shape mismatch (batch collation crash)
  * Voxel size divergence (model learns wrong scale)
  * Affine orientation mismatch (axes-flipped input without warning)
  * Field-strength heterogeneity (distributional shift)
  * TR mismatch for fMRI (temporal alignment broken)
  * dtype inconsistency (precision loss during batching)

The reporter groups subjects into clusters sharing identical TensorSpecs,
identifies the majority-consensus spec, and emits structured recommendations
for resampling/reorientation targets.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qortex.core.entities import FileRecord, ImageRecord

log = logging.getLogger(__name__)

# Tolerance thresholds for "effectively identical" comparisons
_VOXEL_RTOL = 0.05   # 5 % relative tolerance on voxel dimensions
_AFFINE_DET_RTOL = 0.01  # determinant drift
_TR_ATOL = 0.01      # seconds absolute tolerance for TR


class IssueSeverity(str, Enum):
    CRITICAL = "critical"   # will crash batching
    WARNING  = "warning"    # will cause silent distributional shift
    INFO     = "info"       # informational divergence worth noting


@dataclass(frozen=True)
class TensorSpec:
    """Canonical spatial fingerprint of one volume.

    All fields are hashable so TensorSpec can be used as a dict key for
    grouping subjects into identical-spec clusters.
    """
    shape: tuple[int, ...]             # (x, y, z) or (x, y, z, t)
    voxel_size_mm: tuple[float, ...]   # (dx, dy, dz) rounded to 4 dp
    orientation: str                   # RAS / LAS / LPS etc.
    dtype: str                         # "float32", "int16", ...
    n_volumes: int | None              # None for 3-D, count for 4-D
    tr_s: float | None                 # Repetition time (fMRI only)
    field_strength_T: float | None     # 1.5 / 3.0 / 7.0 T etc.
    manufacturer: str | None

    @classmethod
    def from_record(cls, record: ImageRecord) -> "TensorSpec":
        shape = tuple(record.shape)
        vs = tuple(round(float(v), 4) for v in record.voxel_size)
        n_vol = record.n_volumes
        tr = record.tr
        fs = record.metadata.get("magnetic_field_strength")
        if fs is not None:
            try:
                fs = round(float(fs), 2)
            except (TypeError, ValueError):
                fs = None
        orientation = str(record.metadata.get("orientation", "unknown"))
        dtype = str(record.metadata.get("dtype", "unknown"))
        manufacturer = record.metadata.get("manufacturer")
        if isinstance(manufacturer, str):
            manufacturer = manufacturer.strip() or None
        return cls(
            shape=shape,
            voxel_size_mm=vs,
            orientation=orientation,
            dtype=dtype,
            n_volumes=n_vol,
            tr_s=round(float(tr), 4) if tr is not None else None,
            field_strength_T=fs,
            manufacturer=manufacturer,
        )

    @classmethod
    def from_path(cls, path: Path, sidecar: dict[str, Any] | None = None) -> "TensorSpec":
        """Construct from a local NIfTI path without an ImageRecord."""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("harmonize requires nibabel: pip install 'qortex[mri]'")

        img = nib.load(str(path))
        img_can = nib.as_closest_canonical(img)
        hdr = img_can.header
        shape = tuple(img_can.shape)
        zooms = tuple(round(float(z), 4) for z in hdr.get_zooms()[:3])
        try:
            ornt = nib.orientations.aff2axcodes(img_can.affine)
            orientation = "".join(ornt)
        except Exception:
            orientation = "unknown"
        dtype = str(img_can.get_data_dtype())
        n_vol = shape[3] if len(shape) == 4 else None

        sc = sidecar or {}
        tr = sc.get("RepetitionTime")
        fs = sc.get("MagneticFieldStrength")
        manufacturer = sc.get("Manufacturer") or None
        if fs is not None:
            try:
                fs = round(float(fs), 2)
            except (TypeError, ValueError):
                fs = None
        return cls(
            shape=shape,
            voxel_size_mm=zooms,
            orientation=orientation,
            dtype=dtype,
            n_volumes=n_vol,
            tr_s=round(float(tr), 4) if tr is not None else None,
            field_strength_T=fs,
            manufacturer=manufacturer,
        )

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.shape[:3]

    @property
    def fov_mm(self) -> tuple[float, ...]:
        return tuple(
            round(s * v, 2)
            for s, v in zip(self.spatial_shape, self.voxel_size_mm)
        )

    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.voxel_size_mm))

    def is_spatially_compatible(self, other: "TensorSpec", rtol: float = _VOXEL_RTOL) -> bool:
        """True when shape AND voxel size are within tolerance (batch-safe)."""
        if self.spatial_shape != other.spatial_shape:
            return False
        for a, b in zip(self.voxel_size_mm, other.voxel_size_mm):
            if abs(a - b) / max(abs(a), abs(b), 1e-9) > rtol:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "voxel_size_mm": list(self.voxel_size_mm),
            "fov_mm": list(self.fov_mm),
            "voxel_volume_mm3": self.voxel_volume_mm3(),
            "orientation": self.orientation,
            "dtype": self.dtype,
            "n_volumes": self.n_volumes,
            "tr_s": self.tr_s,
            "field_strength_T": self.field_strength_T,
            "manufacturer": self.manufacturer,
        }


@dataclass
class HarmonizationIssue:
    severity: IssueSeverity
    code: str
    message: str
    subjects: list[str] = field(default_factory=list)
    values: dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "subjects": self.subjects,
            "values": self.values,
            "recommendation": self.recommendation,
        }


@dataclass
class HarmonizationGroup:
    """Cluster of subjects sharing the same TensorSpec."""
    spec: TensorSpec
    subjects: list[str]
    n: int = 0
    is_consensus: bool = False   # True for the largest group

    def __post_init__(self) -> None:
        self.n = len(self.subjects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "subjects": self.subjects,
            "n": self.n,
            "is_consensus": self.is_consensus,
        }


class HarmonizationReport:
    """Full harmonization analysis result.

    Attributes
    ----------
    groups:
        Clusters of subjects grouped by identical TensorSpec.
    issues:
        Structured list of detected incompatibilities.
    consensus_spec:
        The TensorSpec held by the plurality of subjects.
    n_subjects:
        Total number of subjects analysed.
    harmonized_fraction:
        Fraction of subjects that match the consensus spec exactly.
    """

    def __init__(
        self,
        groups: list[HarmonizationGroup],
        issues: list[HarmonizationIssue],
        subject_spec_map: dict[str, TensorSpec],
    ) -> None:
        self.groups = groups
        self.issues = issues
        self.subject_spec_map = subject_spec_map
        self.n_subjects = len(subject_spec_map)

        # Largest group = consensus
        if groups:
            groups_sorted = sorted(groups, key=lambda g: g.n, reverse=True)
            groups_sorted[0].is_consensus = True
            self.consensus_spec: TensorSpec | None = groups_sorted[0].spec
            self.harmonized_fraction = groups_sorted[0].n / max(self.n_subjects, 1)
        else:
            self.consensus_spec = None
            self.harmonized_fraction = 0.0

    @property
    def critical_issues(self) -> list[HarmonizationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.CRITICAL]

    @property
    def warning_issues(self) -> list[HarmonizationIssue]:
        return [i for i in self.issues if i.severity == IssueSeverity.WARNING]

    @property
    def is_batch_safe(self) -> bool:
        """True when all subjects share the same spatial shape — no batch crash risk."""
        return len(self.critical_issues) == 0

    def outlier_subjects(self) -> list[str]:
        """Subjects NOT in the consensus group."""
        if self.consensus_spec is None:
            return []
        consensus_group = next(
            (g for g in self.groups if g.is_consensus), None
        )
        if consensus_group is None:
            return []
        consensus_set = set(consensus_group.subjects)
        return [s for s in self.subject_spec_map if s not in consensus_set]

    def resampling_target(self) -> TensorSpec | None:
        """Return the consensus spec — suitable input to a resampling transform.

        When subjects disagree on shape only (not voxel size), the consensus
        shape may not be meaningful; callers should check ``is_batch_safe`` first
        and decide whether to resample to the consensus or to a fixed atlas grid.
        """
        return self.consensus_spec

    def per_subject_table(self) -> list[dict[str, Any]]:
        """Flat table, one row per subject, with full TensorSpec columns."""
        rows: list[dict[str, Any]] = []
        for subject, spec in sorted(self.subject_spec_map.items()):
            row: dict[str, Any] = {"subject": subject}
            row.update(spec.to_dict())
            row["in_consensus"] = (spec == self.consensus_spec)
            rows.append(row)
        return rows

    def summary(self) -> str:
        lines = [
            f"Harmonization Report — {self.n_subjects} subjects",
            f"  Spec groups  : {len(self.groups)}",
            f"  Consensus    : {self.harmonized_fraction * 100:.1f}% of subjects",
            f"  Batch-safe   : {'YES' if self.is_batch_safe else 'NO — CRITICAL ISSUES PRESENT'}",
            f"  Critical     : {len(self.critical_issues)}",
            f"  Warnings     : {len(self.warning_issues)}",
        ]
        if self.consensus_spec:
            cs = self.consensus_spec
            lines += [
                "  Consensus spec:",
                f"    shape        : {list(cs.shape)}",
                f"    voxel_mm     : {list(cs.voxel_size_mm)}",
                f"    orientation  : {cs.orientation}",
                f"    dtype        : {cs.dtype}",
                f"    field (T)    : {cs.field_strength_T}",
            ]
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues:
                pfx = issue.severity.value.upper()
                sub_str = f" ({len(issue.subjects)} subjects)" if issue.subjects else ""
                lines.append(f"    [{pfx}] {issue.code}{sub_str}: {issue.message}")
                if issue.recommendation:
                    lines.append(f"      → {issue.recommendation}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_subjects": self.n_subjects,
            "n_groups": len(self.groups),
            "harmonized_fraction": round(self.harmonized_fraction, 4),
            "is_batch_safe": self.is_batch_safe,
            "consensus_spec": self.consensus_spec.to_dict() if self.consensus_spec else None,
            "groups": [g.to_dict() for g in self.groups],
            "issues": [i.to_dict() for i in self.issues],
            "per_subject": self.per_subject_table(),
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return out

    def to_dataframe(self):
        """Return per-subject table as a Polars DataFrame."""
        import polars as pl
        rows = self.per_subject_table()
        if not rows:
            return pl.DataFrame()
        # Normalise list columns to strings for Polars
        for row in rows:
            for k, v in list(row.items()):
                if isinstance(v, (list, tuple)):
                    row[k] = str(v)
        return pl.DataFrame(rows)

    def to_markdown(self, path: str | Path) -> Path:
        out = Path(path)
        lines = [
            "# Qortex Harmonization Report",
            "",
            f"| Metric | Value |",
            f"| --- | --- |",
            f"| Subjects analysed | {self.n_subjects} |",
            f"| Distinct spec groups | {len(self.groups)} |",
            f"| Consensus fraction | {self.harmonized_fraction * 100:.1f}% |",
            f"| Batch-safe | {'✅ Yes' if self.is_batch_safe else '❌ No'} |",
            f"| Critical issues | {len(self.critical_issues)} |",
            f"| Warnings | {len(self.warning_issues)} |",
            "",
        ]
        if self.issues:
            lines += ["## Issues", ""]
            lines += ["| Severity | Code | Message | Subjects |"]
            lines += ["| --- | --- | --- | --- |"]
            for issue in self.issues:
                lines.append(
                    f"| {issue.severity.value} | `{issue.code}` | {issue.message} | {len(issue.subjects)} |"
                )
            lines.append("")

        lines += ["## Spec Groups", ""]
        for g in sorted(self.groups, key=lambda g: g.n, reverse=True):
            tag = " *(consensus)*" if g.is_consensus else ""
            lines += [
                f"### {g.n} subjects — shape {list(g.spec.shape)}{tag}",
                f"- Voxel size: {list(g.spec.voxel_size_mm)} mm",
                f"- Orientation: {g.spec.orientation}",
                f"- dtype: {g.spec.dtype}",
                f"- Field strength: {g.spec.field_strength_T} T",
                f"- Subjects: {', '.join(sorted(g.subjects)[:10])}"
                + (f" … +{g.n - 10} more" if g.n > 10 else ""),
                "",
            ]
        out.write_text("\n".join(lines), encoding="utf-8")
        return out


class HarmonizationReporter:
    """Build HarmonizationReports from collections of ImageRecords or paths.

    Parameters
    ----------
    voxel_rtol:
        Relative tolerance for voxel-size comparison (default 5 %).
    field_strength_atol:
        Absolute tolerance for field-strength grouping in Tesla (default 0.1 T).
    check_dtype:
        When True (default) emit a WARNING when dtypes diverge.
    check_manufacturer:
        When True (default) emit an INFO when manufacturers diverge.
    """

    def __init__(
        self,
        *,
        voxel_rtol: float = _VOXEL_RTOL,
        field_strength_atol: float = 0.1,
        check_dtype: bool = True,
        check_manufacturer: bool = True,
    ) -> None:
        self.voxel_rtol = voxel_rtol
        self.field_strength_atol = field_strength_atol
        self.check_dtype = check_dtype
        self.check_manufacturer = check_manufacturer

    # ── Public entry points ───────────────────────────────────────────────

    def from_records(
        self,
        records: list[ImageRecord],
        *,
        subject_key: str = "subject",
    ) -> HarmonizationReport:
        """Analyse a list of already-loaded ImageRecords."""
        subject_spec: dict[str, TensorSpec] = {}
        for rec in records:
            sub = rec.file.entities.subject or rec.file.path
            spec = TensorSpec.from_record(rec)
            if sub in subject_spec:
                log.debug("Subject %s has multiple records — using first", sub)
            else:
                subject_spec[sub] = spec
        return self._build_report(subject_spec)

    def from_paths(
        self,
        paths: list[Path],
        *,
        sidecar_loader: Any | None = None,
    ) -> HarmonizationReport:
        """Analyse a list of local NIfTI paths.

        Parameters
        ----------
        sidecar_loader:
            Optional callable ``(path: Path) -> dict`` that returns the JSON
            sidecar for each NIfTI. When None, only file-level header info is used.
        """
        subject_spec: dict[str, TensorSpec] = {}
        for path in paths:
            # Infer subject from BIDS path convention
            sub = _subject_from_path(path)
            sidecar: dict[str, Any] = {}
            if sidecar_loader is not None:
                try:
                    sidecar = sidecar_loader(path) or {}
                except Exception as exc:
                    log.warning("Sidecar load failed for %s: %s", path, exc)
            try:
                spec = TensorSpec.from_path(path, sidecar)
            except Exception as exc:
                log.warning("Cannot inspect %s: %s — skipping", path, exc)
                continue
            if sub not in subject_spec:
                subject_spec[sub] = spec
        return self._build_report(subject_spec)

    def from_directory(
        self,
        bids_root: Path,
        *,
        datatype: str = "anat",
        suffix: str = "T1w",
        extension: str = ".nii.gz",
    ) -> HarmonizationReport:
        """Scan a local BIDS tree and analyse all matching files.

        Parameters
        ----------
        bids_root:
            Root of the BIDS tree (directory that contains ``sub-*`` folders).
        datatype:
            BIDS datatype folder (``"anat"`` | ``"func"`` | ``"dwi"`` | etc.).
        suffix:
            BIDS suffix, e.g. ``"T1w"``, ``"bold"``, ``"dwi"``.
        extension:
            File extension to match (default ``".nii.gz"``).
        """
        from qortex.parse._mne_utils import load_json_sidecar

        pattern = f"sub-*/{datatype}/*_{suffix}{extension}"
        paths = sorted(bids_root.glob(pattern))
        if not paths:
            # Also try session-level path
            pattern_ses = f"sub-*/ses-*/{datatype}/*_{suffix}{extension}"
            paths = sorted(bids_root.glob(pattern_ses))

        def _sidecar_loader(p: Path) -> dict[str, Any]:
            return load_json_sidecar(p)

        return self.from_paths(paths, sidecar_loader=_sidecar_loader)

    # ── Internal ─────────────────────────────────────────────────────────

    def _build_report(
        self,
        subject_spec: dict[str, TensorSpec],
    ) -> HarmonizationReport:
        if not subject_spec:
            return HarmonizationReport(groups=[], issues=[], subject_spec_map={})

        # Group by exact TensorSpec equality
        spec_subjects: dict[TensorSpec, list[str]] = {}
        for sub, spec in subject_spec.items():
            spec_subjects.setdefault(spec, []).append(sub)

        groups = [
            HarmonizationGroup(spec=spec, subjects=subs)
            for spec, subs in spec_subjects.items()
        ]

        issues = self._detect_issues(subject_spec, groups)
        return HarmonizationReport(
            groups=groups,
            issues=issues,
            subject_spec_map=subject_spec,
        )

    def _detect_issues(
        self,
        subject_spec: dict[str, TensorSpec],
        groups: list[HarmonizationGroup],
    ) -> list[HarmonizationIssue]:
        issues: list[HarmonizationIssue] = []
        all_specs = list(subject_spec.values())

        # ── Shape mismatch ────────────────────────────────────────────────
        shape_counter: Counter[tuple[int, ...]] = Counter(
            s.spatial_shape for s in all_specs
        )
        if len(shape_counter) > 1:
            consensus_shape, consensus_n = shape_counter.most_common(1)[0]
            outlier_subs = [
                sub for sub, sp in subject_spec.items()
                if sp.spatial_shape != consensus_shape
            ]
            shape_values = {
                str(shape): n for shape, n in shape_counter.items()
            }
            issues.append(HarmonizationIssue(
                severity=IssueSeverity.CRITICAL,
                code="SHAPE_MISMATCH",
                message=(
                    f"{len(shape_counter)} distinct spatial shapes detected. "
                    f"Consensus {list(consensus_shape)} ({consensus_n} subjects), "
                    f"{len(outlier_subs)} subjects diverge."
                ),
                subjects=outlier_subs,
                values=shape_values,
                recommendation=(
                    "Resample all volumes to the consensus shape using a "
                    "spatial resampling transform (e.g. monai.transforms.Resized "
                    "or torchio.transforms.Resample) before batching."
                ),
            ))

        # ── Voxel size mismatch ───────────────────────────────────────────
        voxel_subs_by_vs: dict[str, list[str]] = {}
        for sub, sp in subject_spec.items():
            key = str(sp.voxel_size_mm)
            voxel_subs_by_vs.setdefault(key, []).append(sub)

        if len(voxel_subs_by_vs) > 1:
            all_vs = [sp.voxel_size_mm for sp in all_specs]
            mean_vs = tuple(
                round(float(np.mean([v[i] for v in all_vs])), 4)
                for i in range(min(len(v) for v in all_vs))
            )
            max_pct = max(
                abs(v[i] - mean_vs[i]) / max(abs(mean_vs[i]), 1e-9) * 100
                for sp in all_specs
                for i, v in [(slice(None), sp.voxel_size_mm)]
                for i in range(len(mean_vs))
            )
            outlier_vox_subs = [
                sub for sub, sp in subject_spec.items()
                if any(
                    abs(sp.voxel_size_mm[i] - mean_vs[i]) / max(abs(mean_vs[i]), 1e-9) > self.voxel_rtol
                    for i in range(len(mean_vs))
                )
            ]
            if outlier_vox_subs:
                issues.append(HarmonizationIssue(
                    severity=IssueSeverity.WARNING,
                    code="VOXEL_SIZE_DIVERGENCE",
                    message=(
                        f"{len(outlier_vox_subs)} subjects deviate > {self.voxel_rtol * 100:.0f}% "
                        f"from mean voxel size {list(mean_vs)} mm. "
                        f"Max divergence: {max_pct:.1f}%."
                    ),
                    subjects=outlier_vox_subs,
                    values={"mean_voxel_mm": list(mean_vs), "max_divergence_pct": round(max_pct, 2)},
                    recommendation=(
                        "Standardise voxel size to the consensus or a target isotropic "
                        "resolution (e.g. 1 mm³) using Resample before feature extraction."
                    ),
                ))

        # ── Orientation mismatch ──────────────────────────────────────────
        orient_counter: Counter[str] = Counter(sp.orientation for sp in all_specs)
        if len(orient_counter) > 1:
            consensus_orient = orient_counter.most_common(1)[0][0]
            outlier_orient_subs = [
                sub for sub, sp in subject_spec.items()
                if sp.orientation != consensus_orient
            ]
            issues.append(HarmonizationIssue(
                severity=IssueSeverity.CRITICAL,
                code="ORIENTATION_MISMATCH",
                message=(
                    f"Mixed orientations: {dict(orient_counter)}. "
                    f"{len(outlier_orient_subs)} subjects differ from "
                    f"consensus {consensus_orient!r}."
                ),
                subjects=outlier_orient_subs,
                values=dict(orient_counter),
                recommendation=(
                    "Apply nibabel.as_closest_canonical() or "
                    "torchio.transforms.ToCanonical() during data loading to "
                    "enforce RAS orientation before any downstream processing."
                ),
            ))

        # ── TR mismatch for fMRI ──────────────────────────────────────────
        tr_values = [
            (sub, sp.tr_s)
            for sub, sp in subject_spec.items()
            if sp.tr_s is not None
        ]
        if len(tr_values) > 1:
            trs = [v for _, v in tr_values]
            tr_range = max(trs) - min(trs)
            if tr_range > _TR_ATOL:
                tr_by_val: dict[str, list[str]] = {}
                for sub, tr in tr_values:
                    tr_by_val.setdefault(str(tr), []).append(sub)
                non_consensus_tr_subs = []
                if tr_by_val:
                    consensus_tr = max(tr_by_val, key=lambda k: len(tr_by_val[k]))
                    non_consensus_tr_subs = [
                        sub for sub, tr in tr_values
                        if abs(tr - float(consensus_tr)) > _TR_ATOL
                    ]
                issues.append(HarmonizationIssue(
                    severity=IssueSeverity.CRITICAL,
                    code="TR_MISMATCH",
                    message=(
                        f"Repetition times span {min(trs):.3f}–{max(trs):.3f} s "
                        f"(range {tr_range:.3f} s). Temporal features computed "
                        f"across subjects will be misaligned."
                    ),
                    subjects=non_consensus_tr_subs,
                    values={"tr_values": dict(Counter(str(v) for _, v in tr_values))},
                    recommendation=(
                        "Select only subjects with a common TR, or resample the "
                        "temporal dimension to a common sampling grid after loading."
                    ),
                ))

        # ── n_volumes mismatch ────────────────────────────────────────────
        vol_counts = Counter(
            sp.n_volumes for sp in all_specs if sp.n_volumes is not None
        )
        if len(vol_counts) > 1:
            consensus_vols = vol_counts.most_common(1)[0][0]
            outlier_vol_subs = [
                sub for sub, sp in subject_spec.items()
                if sp.n_volumes is not None and sp.n_volumes != consensus_vols
            ]
            issues.append(HarmonizationIssue(
                severity=IssueSeverity.WARNING,
                code="VOLUME_COUNT_MISMATCH",
                message=(
                    f"{len(vol_counts)} distinct volume counts: "
                    f"{dict(vol_counts)}. "
                    f"Consensus is {consensus_vols} volumes."
                ),
                subjects=outlier_vol_subs,
                values={"volume_counts": {str(k): v for k, v in vol_counts.items()}},
                recommendation=(
                    "Truncate all runs to the consensus volume count, or build "
                    "separate cohort groups per run-length."
                ),
            ))

        # ── dtype mismatch ────────────────────────────────────────────────
        if self.check_dtype:
            dtype_counter: Counter[str] = Counter(sp.dtype for sp in all_specs)
            if len(dtype_counter) > 1:
                outlier_dtype_subs = [
                    sub for sub, sp in subject_spec.items()
                    if sp.dtype != dtype_counter.most_common(1)[0][0]
                ]
                issues.append(HarmonizationIssue(
                    severity=IssueSeverity.WARNING,
                    code="DTYPE_MISMATCH",
                    message=(
                        f"Mixed storage dtypes: {dict(dtype_counter)}. "
                        f"Implicit casting during batching may silently "
                        f"alter precision or sign."
                    ),
                    subjects=outlier_dtype_subs,
                    values=dict(dtype_counter),
                    recommendation=(
                        "Cast all volumes to float32 during loading (e.g. "
                        "img.get_fdata(dtype=np.float32)) to ensure uniform precision."
                    ),
                ))

        # ── Field strength mismatch ───────────────────────────────────────
        fs_values = [
            (sub, sp.field_strength_T)
            for sub, sp in subject_spec.items()
            if sp.field_strength_T is not None
        ]
        if len(fs_values) > 1:
            unique_fs = sorted({v for _, v in fs_values})
            if len(unique_fs) > 1:
                fs_range = max(unique_fs) - min(unique_fs)
                if fs_range > self.field_strength_atol:
                    fs_counter: Counter[float] = Counter(v for _, v in fs_values)
                    consensus_fs = fs_counter.most_common(1)[0][0]
                    outlier_fs_subs = [
                        sub for sub, fs in fs_values
                        if abs(fs - consensus_fs) > self.field_strength_atol
                    ]
                    issues.append(HarmonizationIssue(
                        severity=IssueSeverity.WARNING,
                        code="FIELD_STRENGTH_HETEROGENEITY",
                        message=(
                            f"Field strengths: {unique_fs} T. "
                            f"{len(outlier_fs_subs)} subjects differ > "
                            f"{self.field_strength_atol} T from consensus "
                            f"{consensus_fs} T."
                        ),
                        subjects=outlier_fs_subs,
                        values={"field_strengths_T": {str(k): v for k, v in fs_counter.items()}},
                        recommendation=(
                            "Restrict cohort to a single field strength, or apply "
                            "ComBat harmonization (neuroCombat) to remove "
                            "scanner-induced intensity bias before modelling."
                        ),
                    ))

        # ── Manufacturer heterogeneity ────────────────────────────────────
        if self.check_manufacturer:
            mfr_values = [
                sp.manufacturer for sp in all_specs if sp.manufacturer
            ]
            if mfr_values:
                mfr_counter: Counter[str] = Counter(mfr_values)
                if len(mfr_counter) > 1:
                    consensus_mfr = mfr_counter.most_common(1)[0][0]
                    outlier_mfr_subs = [
                        sub for sub, sp in subject_spec.items()
                        if sp.manufacturer and sp.manufacturer != consensus_mfr
                    ]
                    issues.append(HarmonizationIssue(
                        severity=IssueSeverity.INFO,
                        code="MULTI_MANUFACTURER",
                        message=(
                            f"Multiple scanner manufacturers: "
                            f"{dict(mfr_counter)}. "
                            f"{len(outlier_mfr_subs)} subjects differ from "
                            f"consensus {consensus_mfr!r}."
                        ),
                        subjects=outlier_mfr_subs,
                        values=dict(mfr_counter),
                        recommendation=(
                            "Consider stratified analysis by manufacturer, or "
                            "apply ComBat to mitigate vendor-related signal bias."
                        ),
                    ))

        return issues


# ── Helpers ───────────────────────────────────────────────────────────────────

def _subject_from_path(path: Path) -> str:
    """Extract 'sub-XX' from a BIDS-style path, or fall back to the filename stem."""
    for part in path.parts:
        if part.startswith("sub-"):
            return part
    return path.stem
