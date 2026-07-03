"""Multi-dataset cohort builder.

CohortBuilder assembles a harmonized subject pool from multiple OpenNeuro
datasets.  It operates in two phases:

  Phase 1 — Discovery (remote, no download):
    • Fetches manifests for all specified datasets via the OpenNeuro API.
    • Parses participants.tsv for demographic filters (age, sex, diagnosis).
    • Applies structural filters (modality, field strength, n_subjects minimum).
    • Optionally queries the live OpenNeuro catalog for additional datasets
      matching free-text criteria.

  Phase 2 — Assembly (pure metadata):
    • Merges qualified subjects across datasets into a unified CohortManifest.
    • Applies inter-dataset harmonization checks (optional).
    • Emits structured per-subject records (CohortSubject) with provenance.

No data is downloaded.  The cohort manifest can be passed to export.MONAIExporter
or export.TorchIOExporter, or used as the subject list for ``Dataset.download()``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_BIDS_ENTITY_RE = re.compile(r"(?:^|_)(sub|ses|task|run|acq|dir)-([^_\.\s]+)")


@dataclass
class CohortSubject:
    """One participant drawn into a cohort from a specific dataset."""

    subject_id: str             # BIDS ID e.g. "sub-01"
    dataset_id: str
    snapshot: str
    # Demographics (from participants.tsv when available)
    age: float | None = None
    sex: str | None = None       # "M" | "F" | other
    group: str | None = None     # e.g. "control" / "patient"
    diagnosis: str | None = None
    handedness: str | None = None
    # Imaging metadata (from sidecar when fetched via inspect)
    field_strength_T: float | None = None
    manufacturer: str | None = None
    modalities: list[str] = field(default_factory=list)
    # Arbitrary extra columns from participants.tsv
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "dataset_id": self.dataset_id,
            "snapshot": self.snapshot,
            "age": self.age,
            "sex": self.sex,
            "group": self.group,
            "diagnosis": self.diagnosis,
            "handedness": self.handedness,
            "field_strength_T": self.field_strength_T,
            "manufacturer": self.manufacturer,
            "modalities": self.modalities,
            **self.extra,
        }


@dataclass
class CohortDatasetEntry:
    """Per-dataset summary within a cohort."""

    dataset_id: str
    snapshot: str
    doi: str | None
    n_subjects_total: int
    n_subjects_selected: int
    n_subjects_excluded: int
    exclusion_reasons: dict[str, int]   # reason → count
    modalities: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "snapshot": self.snapshot,
            "doi": self.doi,
            "n_subjects_total": self.n_subjects_total,
            "n_subjects_selected": self.n_subjects_selected,
            "n_subjects_excluded": self.n_subjects_excluded,
            "exclusion_reasons": self.exclusion_reasons,
            "modalities": self.modalities,
        }


class CohortManifest:
    """The result of ``CohortBuilder.build()``.

    Contains all passing subjects with full provenance and per-dataset stats.
    Provides export connectors to MONAI / TorchIO and Polars DataFrames.
    """

    def __init__(
        self,
        subjects: list[CohortSubject],
        dataset_entries: list[CohortDatasetEntry],
        filters_applied: list[str],
        built_at: datetime,
    ) -> None:
        self.subjects = subjects
        self.dataset_entries = dataset_entries
        self.filters_applied = filters_applied
        self.built_at = built_at

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def n_datasets(self) -> int:
        return len(self.dataset_entries)

    @property
    def dataset_ids(self) -> list[str]:
        return sorted({s.dataset_id for s in self.subjects})

    @property
    def subject_ids_by_dataset(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for s in self.subjects:
            result.setdefault(s.dataset_id, []).append(s.subject_id)
        return result

    # ── Queries ───────────────────────────────────────────────────────────

    def subjects_for_dataset(self, dataset_id: str) -> list[CohortSubject]:
        return [s for s in self.subjects if s.dataset_id == dataset_id]

    def filter_by_sex(self, sex: str) -> "CohortManifest":
        filtered = [s for s in self.subjects if s.sex and s.sex.upper() == sex.upper()]
        return CohortManifest(
            subjects=filtered,
            dataset_entries=self.dataset_entries,
            filters_applied=[*self.filters_applied, f"sex={sex}"],
            built_at=self.built_at,
        )

    def filter_by_age(self, min_age: float, max_age: float) -> "CohortManifest":
        filtered = [
            s for s in self.subjects
            if s.age is not None and min_age <= s.age <= max_age
        ]
        return CohortManifest(
            subjects=filtered,
            dataset_entries=self.dataset_entries,
            filters_applied=[*self.filters_applied, f"age=[{min_age},{max_age}]"],
            built_at=self.built_at,
        )

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"CohortManifest — {self.n_subjects} subjects across {self.n_datasets} datasets",
            f"  Built at   : {self.built_at.isoformat()}",
            f"  Filters    : {', '.join(self.filters_applied) or 'none'}",
            "",
            "  Per-dataset:",
        ]
        for entry in sorted(self.dataset_entries, key=lambda e: e.dataset_id):
            lines.append(
                f"    {entry.dataset_id:<14} "
                f"{entry.n_subjects_selected:>4}/{entry.n_subjects_total} subjects  "
                f"modalities: {', '.join(entry.modalities)}"
            )
        if len(self.subjects) > 0:
            age_vals = [s.age for s in self.subjects if s.age is not None]
            sex_counts: dict[str, int] = {}
            for s in self.subjects:
                if s.sex:
                    sex_counts[s.sex.upper()] = sex_counts.get(s.sex.upper(), 0) + 1
            lines.append("")
            if age_vals:
                lines.append(
                    f"  Age range  : {min(age_vals):.1f}–{max(age_vals):.1f} "
                    f"(mean {sum(age_vals)/len(age_vals):.1f})"
                )
            if sex_counts:
                lines.append(f"  Sex dist.  : {sex_counts}")
        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_subjects": self.n_subjects,
            "n_datasets": self.n_datasets,
            "built_at": self.built_at.isoformat(),
            "filters_applied": self.filters_applied,
            "datasets": [e.to_dict() for e in self.dataset_entries],
            "subjects": [s.to_dict() for s in self.subjects],
        }

    def to_json(self, path: str | Path, *, indent: int = 2) -> Path:
        out = Path(path)
        out.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return out

    def subject_table(self) -> Any:
        """Return a Polars DataFrame with one row per cohort subject."""
        import polars as pl
        rows = [s.to_dict() for s in self.subjects]
        if not rows:
            return pl.DataFrame()
        for row in rows:
            for k, v in list(row.items()):
                if isinstance(v, (list, tuple)):
                    row[k] = ", ".join(str(x) for x in v)
        return pl.DataFrame(rows)

    def dataset_table(self) -> Any:
        """Return a Polars DataFrame with one row per dataset."""
        import polars as pl
        rows = [e.to_dict() for e in self.dataset_entries]
        if not rows:
            return pl.DataFrame()
        for row in rows:
            for k, v in list(row.items()):
                if isinstance(v, (list, tuple, dict)):
                    row[k] = str(v)
        return pl.DataFrame(rows)

    # ── Export connectors ─────────────────────────────────────────────────

    def export_monai(
        self,
        output_dir: Path,
        bids_roots: dict[str, Path],
        **kwargs: Any,
    ) -> list[Path]:
        """Export each dataset's subjects as separate MONAI datalist JSONs.

        Parameters
        ----------
        output_dir:
            Parent directory for outputs.  Each dataset gets a sub-folder.
        bids_roots:
            Mapping ``{dataset_id: local_bids_root_path}``.
        **kwargs:
            Forwarded to ``MONAIExporter.export()``.

        Returns
        -------
        list[Path]
            Written ``dataset.json`` paths per dataset.
        """
        from qortex.export.monai import MONAIExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for ds_id, sub_list in self.subject_ids_by_dataset.items():
            bids_root = bids_roots.get(ds_id)
            if bids_root is None:
                log.warning("No bids_root provided for %s — skipping export", ds_id)
                continue
            ds_out = output_dir / ds_id
            ds_out.mkdir(exist_ok=True)
            exp = MONAIExporter(bids_root=bids_root)
            path = exp.export(ds_out, **kwargs)
            written.append(path)

        return written

    def export_torchio(
        self,
        output_dir: Path,
        bids_roots: dict[str, Path],
        **kwargs: Any,
    ) -> list[Path]:
        """Export each dataset's subjects as TorchIO manifest JSONs."""
        from qortex.export.torchio import TorchIOExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        for ds_id, sub_list in self.subject_ids_by_dataset.items():
            bids_root = bids_roots.get(ds_id)
            if bids_root is None:
                log.warning("No bids_root provided for %s — skipping export", ds_id)
                continue
            ds_out = output_dir / ds_id
            ds_out.mkdir(exist_ok=True)
            exp = TorchIOExporter(bids_root=bids_root)
            path = exp.export(ds_out, **kwargs)
            written.append(path)

        return written


# ── Demographic filter spec (internal) ───────────────────────────────────────

@dataclass
class _ModalityRequirement:
    modality: str           # e.g. "mri", "eeg"
    datatype: str | None    # e.g. "anat"
    suffix: str | None      # e.g. "T1w"


class CohortBuilder:
    """Fluent builder for multi-dataset neuroimaging cohorts.

    All filter methods return ``self`` for chaining.  Call ``build()`` to
    execute all discovery and filtering steps.

    Dataset sources (at least one required before ``build()``):
      * ``add_dataset(dataset_id)``         — pin one dataset by ID
      * ``add_live_search(query, ...)``     — query OpenNeuro and add matches

    Subject-level filters (all optional, all ANDed together):
      * ``require_modality(modality, ...)`` — must have this imaging data
      * ``min_subjects_per_dataset(n)``     — drop datasets below this count
      * ``age_range(min, max)``             — participants.tsv age filter
      * ``sex(value)``                      — participants.tsv sex filter
      * ``diagnosis(value)``                — participants.tsv diagnosis filter
      * ``scanner_field_strength(T, ...)``  — sidecar field strength filter
    """

    def __init__(self, token: str | None = None) -> None:
        self._token = token
        self._dataset_ids: list[str] = []
        self._live_searches: list[dict[str, Any]] = []
        self._modality_requirements: list[_ModalityRequirement] = []
        self._min_subjects: int = 1
        self._age_min: float | None = None
        self._age_max: float | None = None
        self._sex_filter: str | None = None
        self._diagnosis_filter: str | None = None
        self._group_filter: str | None = None
        self._field_strength: float | None = None
        self._field_strength_tol: float = 0.25
        self._run_harmonization: bool = False
        self._filters_log: list[str] = []
        self._api_delay: float = 0.5   # courtesy delay between API calls

    # ── Dataset sources ───────────────────────────────────────────────────

    def add_dataset(self, dataset_id: str, snapshot: str | None = None) -> "CohortBuilder":
        """Add a specific OpenNeuro dataset to the cohort."""
        entry = dataset_id if snapshot is None else f"{dataset_id}:{snapshot}"
        if dataset_id not in [d.split(":")[0] for d in self._dataset_ids]:
            self._dataset_ids.append(entry)
        return self

    def add_datasets(self, *dataset_ids: str) -> "CohortBuilder":
        """Add multiple dataset IDs at once."""
        for ds_id in dataset_ids:
            self.add_dataset(ds_id)
        return self

    def add_live_search(
        self,
        query: str | None = None,
        *,
        modality: str | None = None,
        task: str | None = None,
        min_subjects: int = 1,
        limit: int = 20,
    ) -> "CohortBuilder":
        """Add datasets discovered via live OpenNeuro catalog search.

        Results are fetched when ``build()`` is called.

        Parameters
        ----------
        query:
            Free-text query (matched against dataset name, description, tasks).
        modality:
            Filter by modality label in OpenNeuro catalog.
        min_subjects:
            Only add datasets with at least this many subjects.
        limit:
            Maximum number of datasets to add from this search.
        """
        self._live_searches.append({
            "query": query,
            "modality": modality,
            "task": task,
            "min_subjects": min_subjects,
            "limit": limit,
        })
        return self

    # ── Subject filters ───────────────────────────────────────────────────

    def require_modality(
        self,
        modality: str,
        *,
        datatype: str | None = None,
        suffix: str | None = None,
    ) -> "CohortBuilder":
        """Require each included subject to have data for this modality.

        Parameters
        ----------
        modality:
            e.g. ``"mri"``, ``"eeg"``, ``"fmri"``, ``"dwi"``
        datatype:
            Optional BIDS datatype (e.g. ``"anat"``).
        suffix:
            Optional BIDS suffix (e.g. ``"T1w"``).
        """
        self._modality_requirements.append(
            _ModalityRequirement(modality=modality, datatype=datatype, suffix=suffix)
        )
        self._filters_log.append(
            f"modality={modality}" + (f"/{datatype}" if datatype else "")
            + (f"_{suffix}" if suffix else "")
        )
        return self

    def min_subjects_per_dataset(self, n: int) -> "CohortBuilder":
        """Exclude entire datasets that have fewer than ``n`` passing subjects."""
        self._min_subjects = n
        self._filters_log.append(f"min_subjects_per_dataset={n}")
        return self

    def age_range(self, min_age: float, max_age: float) -> "CohortBuilder":
        """Restrict cohort to subjects within this age range (inclusive)."""
        self._age_min = min_age
        self._age_max = max_age
        self._filters_log.append(f"age=[{min_age},{max_age}]")
        return self

    def sex(self, value: str) -> "CohortBuilder":
        """Filter by sex (case-insensitive; values: ``"M"``, ``"F"``, etc.)."""
        self._sex_filter = value.upper()
        self._filters_log.append(f"sex={value}")
        return self

    def diagnosis(self, value: str) -> "CohortBuilder":
        """Filter by diagnosis column in participants.tsv."""
        self._diagnosis_filter = value
        self._filters_log.append(f"diagnosis={value}")
        return self

    def group(self, value: str) -> "CohortBuilder":
        """Filter by group column in participants.tsv (e.g. ``"control"``)."""
        self._group_filter = value
        self._filters_log.append(f"group={value}")
        return self

    def scanner_field_strength(
        self,
        tesla: float,
        *,
        tolerance: float = 0.25,
    ) -> "CohortBuilder":
        """Restrict to subjects scanned at a specific field strength.

        Parameters
        ----------
        tesla:
            Target field strength in Tesla (e.g. 1.5, 3.0, 7.0).
        tolerance:
            Maximum absolute deviation from ``tesla`` in Tesla.
        """
        self._field_strength = tesla
        self._field_strength_tol = tolerance
        self._filters_log.append(f"field_strength={tesla}T±{tolerance}T")
        return self

    def with_harmonization_check(self) -> "CohortBuilder":
        """Run tensor harmonization analysis and log any critical mismatches."""
        self._run_harmonization = True
        self._filters_log.append("harmonization_check=True")
        return self

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self) -> CohortManifest:
        """Execute discovery, filtering, and assembly.

        Fetches manifests and participants.tsv from the OpenNeuro API for all
        specified datasets.  Returns a CohortManifest with all qualifying subjects.

        Raises
        ------
        RuntimeError
            When no dataset sources have been specified.
        """
        if not self._dataset_ids and not self._live_searches:
            raise RuntimeError(
                "No datasets specified. Call add_dataset() or add_live_search() first."
            )

        all_dataset_ids = list(self._dataset_ids)

        # Phase 1a: Live search expansion
        if self._live_searches:
            extra_ids = self._run_live_searches()
            for ds_id in extra_ids:
                if ds_id not in [d.split(":")[0] for d in all_dataset_ids]:
                    all_dataset_ids.append(ds_id)
            log.info("Live search added %d datasets", len(extra_ids))

        # Phase 1b: Fetch manifests + participants
        all_subjects: list[CohortSubject] = []
        dataset_entries: list[CohortDatasetEntry] = []

        for ds_entry in all_dataset_ids:
            parts = ds_entry.split(":", 1)
            ds_id = parts[0]
            snapshot_pin = parts[1] if len(parts) > 1 else None

            log.info("Processing dataset %s ...", ds_id)
            try:
                subjects, entry = self._process_dataset(ds_id, snapshot_pin)
            except Exception as exc:
                log.warning("Failed to process %s: %s — skipping", ds_id, exc)
                continue

            if entry.n_subjects_selected < self._min_subjects:
                log.info(
                    "Dataset %s has only %d passing subjects < min %d — excluded",
                    ds_id, entry.n_subjects_selected, self._min_subjects,
                )
                # Preserve the real per-subject exclusion reasons computed by
                # _process_dataset (e.g. "missing_required_modality",
                # demographic filters) rather than overwriting them with a
                # single generic "below_min_subjects" — that discarded the
                # actual cause and made every exclusion look identical.
                reasons = dict(entry.exclusion_reasons) if entry.exclusion_reasons else {}
                reasons["below_min_subjects_threshold"] = entry.n_subjects_selected
                exclusion_entry = CohortDatasetEntry(
                    dataset_id=ds_id,
                    snapshot=entry.snapshot,
                    doi=entry.doi,
                    n_subjects_total=entry.n_subjects_total,
                    n_subjects_selected=0,
                    n_subjects_excluded=entry.n_subjects_total,
                    exclusion_reasons=reasons,
                    modalities=entry.modalities,
                )
                dataset_entries.append(exclusion_entry)
                continue

            all_subjects.extend(subjects)
            dataset_entries.append(entry)

            # Courtesy delay between API calls
            if self._api_delay > 0 and ds_id != all_dataset_ids[-1].split(":")[0]:
                time.sleep(self._api_delay)

        log.info(
            "Cohort built: %d subjects across %d datasets",
            len(all_subjects), len(dataset_entries),
        )
        return CohortManifest(
            subjects=all_subjects,
            dataset_entries=dataset_entries,
            filters_applied=list(self._filters_log),
            built_at=datetime.now(timezone.utc),
        )

    # ── Private ───────────────────────────────────────────────────────────

    def _run_live_searches(self) -> list[str]:
        """Execute all registered live searches and return discovered dataset IDs."""
        from qortex.catalog.search import DatasetQuery

        found_ids: list[str] = []
        for search_spec in self._live_searches:
            q = DatasetQuery()
            if search_spec.get("query"):
                q.containing(search_spec["query"])
            if search_spec.get("modality"):
                q.modality(search_spec["modality"])
            if search_spec.get("task"):
                q.task(search_spec["task"])
            q.limit(search_spec.get("limit", 20))
            if search_spec.get("min_subjects"):
                q.min_subjects(search_spec["min_subjects"])
            try:
                results = q.live(token=self._token, sync_local=True)
                for r in results:
                    ds_id = r.get("dataset_id")
                    if ds_id and ds_id not in found_ids:
                        found_ids.append(ds_id)
            except Exception as exc:
                log.warning("Live search failed: %s", exc)
        return found_ids

    def _process_dataset(
        self,
        dataset_id: str,
        snapshot_pin: str | None,
    ) -> tuple[list[CohortSubject], CohortDatasetEntry]:
        """Fetch manifest + participants for one dataset and apply all filters."""
        from qortex.client.graphql import OpenNeuroClient
        from qortex.manifest.builder import ManifestBuilder

        client = OpenNeuroClient(token=self._token)
        builder = ManifestBuilder()

        try:
            if snapshot_pin:
                snap_ref = client.get_snapshot(dataset_id, snapshot_pin)
            else:
                snap_ref = client.get_latest_snapshot(dataset_id)

            snap_ref, raw_files = client.get_files(dataset_id, snap_ref.tag)
            manifest = builder.build(dataset_id, snap_ref, raw_files)
        finally:
            client.close()

        # Parse participants.tsv (from manifest file list — may need remote fetch)
        participants_data = self._fetch_participants_tsv(manifest, dataset_id, snap_ref.tag)

        # Determine available subjects
        all_subjects_in_manifest = manifest.summary.subjects
        n_total = len(all_subjects_in_manifest)

        passing_subjects: list[CohortSubject] = []
        exclusion_reasons: dict[str, int] = {}

        for sub_raw in all_subjects_in_manifest:
            sub_id = f"sub-{sub_raw}" if not sub_raw.startswith("sub-") else sub_raw
            demographics = participants_data.get(sub_id, participants_data.get(sub_raw, {}))

            # Apply demographic filters
            reject_reason = self._apply_demographic_filters(demographics)
            if reject_reason:
                exclusion_reasons[reject_reason] = exclusion_reasons.get(reject_reason, 0) + 1
                continue

            # Apply modality requirement filters
            if self._modality_requirements:
                has_required = self._check_modality_requirements(manifest, sub_raw)
                if not has_required:
                    exclusion_reasons["missing_required_modality"] = (
                        exclusion_reasons.get("missing_required_modality", 0) + 1
                    )
                    continue

            cohort_sub = CohortSubject(
                subject_id=sub_id,
                dataset_id=dataset_id,
                snapshot=snap_ref.tag,
                age=_safe_float(demographics.get("age")),
                sex=_normalise_sex(demographics.get("sex")),
                group=demographics.get("group"),
                diagnosis=demographics.get("diagnosis"),
                handedness=demographics.get("handedness"),
                modalities=manifest.subjects_with_modality(
                    self._modality_requirements[0].modality
                    if self._modality_requirements else "mri"
                ) and list(manifest.summary.modalities) or list(manifest.summary.modalities),
                extra={k: v for k, v in demographics.items()
                       if k not in ("age", "sex", "group", "diagnosis", "handedness",
                                    "participant_id")},
            )
            passing_subjects.append(cohort_sub)

        entry = CohortDatasetEntry(
            dataset_id=dataset_id,
            snapshot=snap_ref.tag,
            doi=manifest.doi,
            n_subjects_total=n_total,
            n_subjects_selected=len(passing_subjects),
            n_subjects_excluded=n_total - len(passing_subjects),
            exclusion_reasons=exclusion_reasons,
            modalities=list(manifest.summary.modalities),
        )
        return passing_subjects, entry

    def _fetch_participants_tsv(
        self,
        manifest: Any,
        dataset_id: str,
        snapshot: str,
    ) -> dict[str, dict[str, str]]:
        """Parse participants.tsv into {subject_id: demographics_dict}."""
        participants_file = manifest.get_file("participants.tsv")
        if participants_file is None:
            return {}

        url = participants_file.urls[0] if participants_file.urls else None
        if not url:
            return {}

        try:
            from qortex.client.remote import RemoteFileGateway
            with RemoteFileGateway() as remote:
                content = remote.fetch_text(url)
        except Exception as exc:
            log.debug("Cannot fetch participants.tsv for %s: %s", dataset_id, exc)
            return {}

        return _parse_participants_tsv(content)

    def _apply_demographic_filters(self, demographics: dict[str, str]) -> str | None:
        """Return the first failing filter reason, or None if all pass."""
        if self._age_min is not None or self._age_max is not None:
            age = _safe_float(demographics.get("age"))
            if age is None:
                return "age_missing"
            if self._age_min is not None and age < self._age_min:
                return "age_below_min"
            if self._age_max is not None and age > self._age_max:
                return "age_above_max"

        if self._sex_filter:
            sex = _normalise_sex(demographics.get("sex"))
            if sex is None:
                return "sex_missing"
            if sex.upper() != self._sex_filter.upper():
                return "sex_mismatch"

        if self._diagnosis_filter:
            diag = str(demographics.get("diagnosis", "")).strip()
            if diag.lower() not in (
                self._diagnosis_filter.lower(), "n/a", ""
            ) and self._diagnosis_filter.lower() != diag.lower():
                return "diagnosis_mismatch"

        if self._group_filter:
            grp = str(demographics.get("group", "")).strip()
            if grp.lower() != self._group_filter.lower():
                return "group_mismatch"

        return None

    def _check_modality_requirements(self, manifest: Any, subject_raw: str) -> bool:
        """Return True if subject has all required modality files."""
        # manifest.summary.subjects (the source of subject_raw in the caller's
        # loop) yields "sub-XX"-prefixed IDs, but FileRecord.entities.subject
        # — and therefore Manifest.filter(subjects=...) — matches against the
        # bare ID ("XX"). Passing the prefixed form through silently matched
        # zero files for every subject, excluding entire real datasets with
        # a misleading "missing_required_modality" reason.
        subject_bare = subject_raw.removeprefix("sub-")
        for req in self._modality_requirements:
            files = manifest.filter(
                subjects=[subject_bare],
                modalities=[req.modality] if req.modality else None,
                datatypes=[req.datatype] if req.datatype else None,
                include_shared=False,
            )
            if req.suffix:
                files = [f for f in files if f.suffix == req.suffix]
            if not files:
                return False
        return True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_participants_tsv(content: str) -> dict[str, dict[str, str]]:
    """Parse a participants.tsv string into {participant_id: row_dict}."""
    lines = content.strip().splitlines()
    if not lines:
        return {}
    reader = csv.DictReader(lines, delimiter="\t")
    result: dict[str, dict[str, str]] = {}
    for row in reader:
        sub = row.get("participant_id", "").strip()
        if not sub:
            continue
        if not sub.startswith("sub-"):
            sub = f"sub-{sub}"
        result[sub] = {k: v.strip() for k, v in row.items()}
    return result


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(str(value).strip())
        return v if v == v else None  # NaN check
    except (ValueError, TypeError):
        return None


def _normalise_sex(value: Any) -> str | None:
    if not value:
        return None
    s = str(value).strip().upper()
    if not s or s in ("N/A", "NA", "NAN", ""):
        return None
    if s in ("M", "MALE"):
        return "M"
    if s in ("F", "FEMALE"):
        return "F"
    return s
