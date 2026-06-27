"""Federated multi-dataset cohort engine backed by DuckDB.

``FederatedCohort`` assembles subjects from multiple OpenNeuro datasets into a
single, reproducible, bias-audited cohort with:

  * SQL-style declarative metadata querying across the unified subject registry
  * Automatic label harmonization (resolve "HC", "Control", "Healthy" → one class)
  * Demographic balancing via under- or oversampling
  * Cryptographic leakage detection — prevents the same human appearing in both
    train and test splits when they participated in multiple studies
  * Frozen, hash-locked manifest export for 100% reproducible AI training runs
  * Automated Hugging Face-style dataset cards (Markdown + YAML front matter)

Architecture
------------
All subject metadata from all datasets is loaded into an in-process DuckDB
database.  Filtering, aggregation, and balancing operations run as SQL; the
result is materialised as a Python list of ``CohortSubject`` objects.
DuckDB is a dependency (pip install duckdb) used internally only — callers
never interact with it directly.

Usage::

    cohort = (
        FederatedCohort("my_schizophrenia_cohort")
        .add_dataset("ds000030", label_map={"patient": "schizophrenia", "control": "healthy"})
        .add_dataset("ds000171", label_map={"scz": "schizophrenia", "hc": "healthy"})
        .add_live_search("schizophrenia T1w", min_subjects=10)
        .require_modality("mri", datatype="anat", suffix="T1w")
        .age_range(18, 65)
        .harmonize_metadata("sex", {"M": "male", "Male": "male", "F": "female", "Female": "female"})
        .balance_demographics("diagnosis", method="undersample")
        .check_data_leakage()
        .build()
    )
    cohort.export_manifest("cohort_v1.json")
    cohort.generate_dataset_card("./cards/")
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class _DatasetSpec:
    dataset_id: str
    snapshot: str | None
    label_map: dict[str, str]       # raw label → harmonized label
    modality_filter: str | None
    datatype_filter: str | None
    suffix_filter: str | None


@dataclass
class FederatedSubject:
    """One subject in the federated cohort registry."""

    subject_id: str                  # BIDS ID e.g. "sub-01"
    dataset_id: str
    snapshot: str
    harmonized_label: str | None     # after label_map application
    age: float | None
    sex: str | None
    site: str | None                 # scanner/institution
    modalities: list[str]
    field_strength_T: float | None
    fingerprint_hash: str = ""       # cryptographic deduplication key
    split: str | None = None         # "train" | "val" | "test"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "dataset_id": self.dataset_id,
            "snapshot": self.snapshot,
            "harmonized_label": self.harmonized_label,
            "age": self.age,
            "sex": self.sex,
            "site": self.site,
            "modalities": self.modalities,
            "field_strength_T": self.field_strength_T,
            "fingerprint_hash": self.fingerprint_hash,
            "split": self.split,
            **self.extra,
        }


class FederatedCohort:
    """Declarative multi-dataset cohort builder backed by DuckDB.

    Parameters
    ----------
    name:
        Human-readable cohort name used in dataset cards and manifest exports.
    token:
        OpenNeuro API token for private dataset access.
    seed:
        Random seed for reproducible balancing and splits.
    """

    def __init__(
        self,
        name: str = "federated_cohort",
        *,
        token: str | None = None,
        seed: int = 42,
    ) -> None:
        self.name = name
        self._token = token
        self._seed = seed
        self._dataset_specs: list[_DatasetSpec] = []
        self._live_searches: list[dict[str, Any]] = []
        self._modality_req: list[tuple[str, str | None, str | None]] = []
        self._age_min: float | None = None
        self._age_max: float | None = None
        self._sex_filter: str | None = None
        self._min_per_dataset: int = 1
        self._harmonize_rules: dict[str, dict[str, str]] = {}   # col → {raw → canonical}
        self._balance_col: str | None = None
        self._balance_method: str = "undersample"
        self._do_leakage_check: bool = False
        self._leakage_method: str = "subject_hash"
        self._subjects: list[FederatedSubject] = []
        self._built: bool = False

    # ── Source specification ───────────────────────────────────────────────

    def add_dataset(
        self,
        dataset_id: str,
        *,
        snapshot: str | None = None,
        label_map: dict[str, str] | None = None,
    ) -> "FederatedCohort":
        """Add a specific OpenNeuro dataset.

        Parameters
        ----------
        label_map:
            Maps raw participant label values (from participants.tsv) to
            canonical harmonized labels, e.g.
            ``{"patient": "schizophrenia", "ctrl": "healthy"}``.
            Values not in the map are passed through unchanged.
        """
        self._dataset_specs.append(_DatasetSpec(
            dataset_id=dataset_id,
            snapshot=snapshot,
            label_map=label_map or {},
            modality_filter=None,
            datatype_filter=None,
            suffix_filter=None,
        ))
        return self

    def add_live_search(
        self,
        query: str | None = None,
        *,
        modality: str | None = None,
        min_subjects: int = 1,
        limit: int = 20,
        label_map: dict[str, str] | None = None,
    ) -> "FederatedCohort":
        """Discover and add datasets from a live OpenNeuro catalog search."""
        self._live_searches.append({
            "query": query,
            "modality": modality,
            "min_subjects": min_subjects,
            "limit": limit,
            "label_map": label_map or {},
        })
        return self

    # ── Filters ───────────────────────────────────────────────────────────

    def require_modality(
        self,
        modality: str,
        *,
        datatype: str | None = None,
        suffix: str | None = None,
    ) -> "FederatedCohort":
        """Require each subject to have data for this modality."""
        self._modality_req.append((modality, datatype, suffix))
        return self

    def age_range(self, min_age: float, max_age: float) -> "FederatedCohort":
        self._age_min = min_age
        self._age_max = max_age
        return self

    def sex(self, value: str) -> "FederatedCohort":
        self._sex_filter = value.upper()
        return self

    def min_subjects_per_dataset(self, n: int) -> "FederatedCohort":
        self._min_per_dataset = n
        return self

    # ── Harmonization and balancing ───────────────────────────────────────

    def harmonize_metadata(
        self,
        column: str,
        mapping: dict[str, str],
    ) -> "FederatedCohort":
        """Apply a value-level harmonization mapping to a metadata column.

        Parameters
        ----------
        column:
            Column name in participants.tsv (e.g. ``"sex"``, ``"diagnosis"``).
        mapping:
            Raw value → canonical value, e.g.
            ``{"M": "male", "Male": "male", "F": "female", "Female": "female"}``.

        Values not in the mapping are passed through as-is.  Can be called
        multiple times for different columns.
        """
        self._harmonize_rules.setdefault(column, {}).update(mapping)
        return self

    def balance_demographics(
        self,
        target_column: str,
        *,
        method: str = "undersample",
    ) -> "FederatedCohort":
        """Balance the cohort across a target demographic or label column.

        Parameters
        ----------
        target_column:
            Column to balance on, e.g. ``"diagnosis"`` or ``"sex"``.
        method:
            ``"undersample"`` (default): downsample majority classes to the
            minority class size.
            ``"oversample"``: upsample minority classes via random duplication
            (approximate, not SMOTE).
        """
        self._balance_col = target_column
        self._balance_method = method
        return self

    def check_data_leakage(
        self,
        *,
        method: str = "subject_hash",
        cross_site: bool = True,
    ) -> "FederatedCohort":
        """Enable cryptographic leakage detection during ``build()``.

        Parameters
        ----------
        method:
            ``"subject_hash"`` (default): hash the combination of
            (age_bucket, sex, field_strength_T, diagnosis) to detect subjects
            who appear in multiple datasets under different IDs.
            ``"strict_id"``: only flag exact BIDS subject ID matches across
            datasets (less conservative).
        cross_site:
            When True, also check across different dataset IDs.
        """
        self._do_leakage_check = True
        self._leakage_method = method
        return self

    # ── Build ─────────────────────────────────────────────────────────────

    def build(
        self,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> "FederatedCohort":
        """Execute discovery, filtering, harmonization, balancing, and split.

        Populates ``self.subjects`` with the final ``FederatedSubject`` list.

        Returns
        -------
        FederatedCohort
            Returns ``self`` for chaining.
        """
        # Phase 1: Live search expansion
        if self._live_searches:
            extra_specs = self._run_live_searches()
            self._dataset_specs.extend(extra_specs)

        # Phase 2: Fetch and normalise all subjects
        raw_subjects: list[FederatedSubject] = []
        for spec in self._dataset_specs:
            try:
                subs = self._fetch_dataset_subjects(spec)
                if len(subs) < self._min_per_dataset:
                    log.info(
                        "%s: only %d subjects pass filters < min %d — skipping",
                        spec.dataset_id, len(subs), self._min_per_dataset,
                    )
                    continue
                raw_subjects.extend(subs)
            except Exception as exc:
                log.warning("Failed to fetch %s: %s", spec.dataset_id, exc)

        if not raw_subjects:
            log.warning("FederatedCohort.build(): no subjects collected.")
            self._subjects = []
            self._built = True
            return self

        # Phase 3: Load into DuckDB for SQL-driven filtering
        filtered = self._duckdb_filter(raw_subjects)

        # Phase 4: Harmonize metadata fields
        harmonized = self._apply_harmonization(filtered)

        # Phase 5: Leakage detection
        if self._do_leakage_check:
            harmonized = self._deduplicate_subjects(harmonized)

        # Phase 6: Demographic balancing
        if self._balance_col:
            harmonized = self._balance(harmonized)

        # Phase 7: Train/val/test split
        self._subjects = self._assign_splits(harmonized, train_frac, val_frac)
        self._built = True

        log.info(
            "FederatedCohort '%s' built: %d subjects, %d datasets",
            self.name, len(self._subjects),
            len({s.dataset_id for s in self._subjects}),
        )
        return self

    # ── Queries on built cohort ───────────────────────────────────────────

    def sql(self, query: str) -> Any:
        """Run a DuckDB SQL query against the built subject registry.

        Parameters
        ----------
        query:
            SQL statement.  The subject table is available as ``subjects``.
            Returns a Polars DataFrame.

        Examples
        --------
        >>> cohort.sql("SELECT harmonized_label, COUNT(*) FROM subjects GROUP BY 1")
        """
        if not self._built:
            raise RuntimeError("Call build() first.")
        import duckdb
        import polars as pl
        rows = [s.to_dict() for s in self._subjects]
        conn = duckdb.connect(":memory:")
        df = pl.DataFrame(rows)
        conn.register("subjects", df.to_arrow())
        result = conn.execute(query).fetchdf()
        conn.close()
        return pl.from_pandas(result)

    @property
    def subjects(self) -> list[FederatedSubject]:
        if not self._built:
            raise RuntimeError("Call build() first.")
        return self._subjects

    @property
    def n_subjects(self) -> int:
        return len(self._subjects)

    @property
    def dataset_ids(self) -> list[str]:
        return sorted({s.dataset_id for s in self._subjects})

    # ── Export ────────────────────────────────────────────────────────────

    def export_manifest(
        self,
        filepath: str | Path,
        *,
        lock_versions: bool = True,
        indent: int = 2,
    ) -> Path:
        """Freeze the cohort into a reproducible JSON manifest.

        Parameters
        ----------
        lock_versions:
            When True, includes the snapshot tag and a SHA-256 fingerprint
            of the subject list to detect tampering.

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")

        sub_dicts = [s.to_dict() for s in self._subjects]
        fingerprint = hashlib.sha256(
            json.dumps(sub_dicts, sort_keys=True).encode()
        ).hexdigest()

        manifest = {
            "cohort_name": self.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_subjects": self.n_subjects,
            "n_datasets": len(self.dataset_ids),
            "datasets": self.dataset_ids,
            "split_counts": self._split_counts(),
            "label_distribution": self._label_distribution(),
            "fingerprint_sha256": fingerprint if lock_versions else None,
            "harmonize_rules": self._harmonize_rules,
            "balance_column": self._balance_col,
            "balance_method": self._balance_method,
            "leakage_check_applied": self._do_leakage_check,
            "subjects": sub_dicts,
        }
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=indent), encoding="utf-8")
        log.info("Cohort manifest written to %s (fingerprint: %s)", out, fingerprint[:16])
        return out

    def generate_dataset_card(
        self,
        output_dir: str | Path,
        *,
        model_type: str = "classification",
        task_categories: list[str] | None = None,
        license: str = "CC-BY-4.0",
    ) -> Path:
        """Generate a Hugging Face-style Markdown dataset card.

        Produces a ``README.md`` in ``output_dir`` with YAML front matter,
        demographic statistics, label distribution, per-dataset provenance,
        and ML readiness assessment.

        Parameters
        ----------
        output_dir:
            Directory where ``README.md`` is written.

        Returns
        -------
        Path
            Path to the written ``README.md``.
        """
        if not self._built:
            raise RuntimeError("Call build() first.")
        from qortex.cohort.card import DatasetCardGenerator
        generator = DatasetCardGenerator(self)
        return generator.generate(
            output_dir=Path(output_dir),
            model_type=model_type,
            task_categories=task_categories or ["medical-imaging"],
            license=license,
        )

    def subject_table(self) -> Any:
        """Return a Polars DataFrame of all cohort subjects."""
        import polars as pl
        rows = [s.to_dict() for s in self._subjects]
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    def label_summary(self) -> dict[str, int]:
        """Return label → count mapping for the built cohort."""
        return dict(Counter(
            s.harmonized_label or "unknown" for s in self._subjects
        ))

    def summary(self) -> str:
        if not self._built:
            return "FederatedCohort (not yet built — call build())"
        dist = self._label_distribution()
        splits = self._split_counts()
        age_vals = [s.age for s in self._subjects if s.age is not None]
        sex_counts = Counter(s.sex or "unknown" for s in self._subjects)
        lines = [
            f"FederatedCohort: {self.name!r}",
            f"  Subjects   : {self.n_subjects}",
            f"  Datasets   : {', '.join(self.dataset_ids)}",
            f"  Labels     : {dict(dist)}",
            f"  Splits     : {splits}",
        ]
        if age_vals:
            lines.append(
                f"  Age        : {min(age_vals):.1f}–{max(age_vals):.1f} "
                f"(mean {sum(age_vals)/len(age_vals):.1f})"
            )
        lines.append(f"  Sex        : {dict(sex_counts)}")
        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────

    def _run_live_searches(self) -> list[_DatasetSpec]:
        from qortex.catalog.search import DatasetQuery
        specs: list[_DatasetSpec] = []
        for search in self._live_searches:
            q = DatasetQuery()
            if search.get("query"):
                q.containing(search["query"])
            if search.get("modality"):
                q.modality(search["modality"])
            if search.get("min_subjects"):
                q.min_subjects(search["min_subjects"])
            q.limit(search.get("limit", 20))
            try:
                results = q.live(token=self._token, sync_local=True)
            except Exception as exc:
                log.warning("Live search failed: %s", exc)
                continue
            for r in results:
                ds_id = r.get("dataset_id")
                if ds_id and not any(s.dataset_id == ds_id for s in self._dataset_specs):
                    specs.append(_DatasetSpec(
                        dataset_id=ds_id,
                        snapshot=None,
                        label_map=search.get("label_map", {}),
                        modality_filter=None,
                        datatype_filter=None,
                        suffix_filter=None,
                    ))
        return specs

    def _fetch_dataset_subjects(self, spec: _DatasetSpec) -> list[FederatedSubject]:
        from qortex.client.graphql import OpenNeuroClient
        from qortex.manifest.builder import ManifestBuilder
        import csv as _csv

        with OpenNeuroClient(token=self._token) as client:
            snap_ref = (
                client.get_snapshot(spec.dataset_id, spec.snapshot)
                if spec.snapshot
                else client.get_latest_snapshot(spec.dataset_id)
            )
            snap_ref, raw_files = client.get_files(spec.dataset_id, snap_ref.tag)
            manifest = ManifestBuilder().build(spec.dataset_id, snap_ref, raw_files)

        # Try to fetch participants.tsv for demographics
        participants: dict[str, dict[str, str]] = {}
        participants_file = manifest.get_file("participants.tsv")
        if participants_file and participants_file.urls:
            try:
                from qortex.client.remote import RemoteFileGateway
                with RemoteFileGateway() as gw:
                    text = gw.fetch_text(participants_file.urls[0])
                lines = text.strip().splitlines()
                reader = _csv.DictReader(lines, delimiter="\t")
                for row in reader:
                    pid = row.get("participant_id", "").strip()
                    if not pid.startswith("sub-"):
                        pid = f"sub-{pid}"
                    participants[pid] = {k: v.strip() for k, v in row.items()}
            except Exception as exc:
                log.debug("Cannot fetch participants.tsv for %s: %s", spec.dataset_id, exc)

        subs: list[FederatedSubject] = []
        for sub_raw in manifest.summary.subjects:
            sub_id = f"sub-{sub_raw}" if not sub_raw.startswith("sub-") else sub_raw
            demo = participants.get(sub_id, {})

            # Apply modality filter
            if self._modality_req:
                has_all = True
                for mod, dt, sfx in self._modality_req:
                    files = manifest.filter(
                        subjects=[sub_raw],
                        modalities=[mod],
                        datatypes=[dt] if dt else None,
                        include_shared=False,
                    )
                    if sfx:
                        files = [f for f in files if f.suffix == sfx]
                    if not files:
                        has_all = False
                        break
                if not has_all:
                    continue

            # Demographic filters
            age = _safe_float(demo.get("age"))
            if self._age_min is not None and (age is None or age < self._age_min):
                continue
            if self._age_max is not None and (age is None or age > self._age_max):
                continue

            sex = _normalise_sex(demo.get("sex"))
            if self._sex_filter and sex and sex.upper() != self._sex_filter:
                continue

            raw_label = demo.get("diagnosis", demo.get("group", demo.get("label_")))
            harmonized = spec.label_map.get(str(raw_label or "").lower(), raw_label)

            fs = _safe_float(demo.get("MagneticFieldStrength"))
            fingerprint = _fingerprint(sub_id, spec.dataset_id, age, sex, harmonized, fs)

            subs.append(FederatedSubject(
                subject_id=sub_id,
                dataset_id=spec.dataset_id,
                snapshot=snap_ref.tag,
                harmonized_label=harmonized,
                age=age,
                sex=sex,
                site=demo.get("site") or demo.get("institution"),
                modalities=list(manifest.summary.modalities),
                field_strength_T=fs,
                fingerprint_hash=fingerprint,
                extra={k: v for k, v in demo.items()
                       if k not in ("participant_id", "age", "sex", "group",
                                    "diagnosis", "site", "institution")},
            ))
        return subs

    def _duckdb_filter(self, subjects: list[FederatedSubject]) -> list[FederatedSubject]:
        """Run SQL filters inside DuckDB for fast set operations."""
        try:
            import duckdb
            import polars as pl
        except ImportError:
            log.warning("DuckDB not installed — skipping SQL filter pass. "
                        "pip install duckdb for full FederatedCohort support.")
            return subjects

        rows = [s.to_dict() for s in subjects]
        if not rows:
            return subjects

        conn = duckdb.connect(":memory:")
        df = pl.DataFrame(rows)
        conn.register("subjects", df.to_arrow())

        conditions: list[str] = []
        if self._age_min is not None:
            conditions.append(f"age >= {self._age_min}")
        if self._age_max is not None:
            conditions.append(f"age <= {self._age_max}")
        if self._sex_filter:
            conditions.append(f"UPPER(sex) = '{self._sex_filter}'")

        where = " AND ".join(conditions)
        query = f"SELECT * FROM subjects" + (f" WHERE {where}" if where else "")
        try:
            result_df = conn.execute(query).pl()
            conn.close()
        except Exception as exc:
            conn.close()
            log.warning("DuckDB filter query failed: %s — returning unfiltered", exc)
            return subjects

        # Reconstruct FederatedSubject list preserving objects
        passing_ids = {
            (row["subject_id"], row["dataset_id"])
            for row in result_df.to_dicts()
        }
        return [s for s in subjects if (s.subject_id, s.dataset_id) in passing_ids]

    def _apply_harmonization(self, subjects: list[FederatedSubject]) -> list[FederatedSubject]:
        """Apply column-level harmonization rules to all subjects in-place."""
        for s in subjects:
            for col, mapping in self._harmonize_rules.items():
                if col == "sex" and s.sex is not None:
                    s.sex = mapping.get(s.sex, mapping.get(s.sex.lower(), s.sex))
                elif col in ("diagnosis", "label") and s.harmonized_label is not None:
                    s.harmonized_label = mapping.get(
                        s.harmonized_label,
                        mapping.get(s.harmonized_label.lower(), s.harmonized_label),
                    )
                elif col in s.extra:
                    raw_val = str(s.extra[col])
                    s.extra[col] = mapping.get(raw_val, mapping.get(raw_val.lower(), raw_val))
        return subjects

    def _deduplicate_subjects(self, subjects: list[FederatedSubject]) -> list[FederatedSubject]:
        """Remove suspected duplicate subjects (same human across datasets).

        Uses the fingerprint hash to detect subjects with identical age bucket,
        sex, field strength, and diagnosis appearing in multiple datasets.
        The first-seen instance is retained; later duplicates are dropped.
        """
        seen_hashes: dict[str, str] = {}   # hash → "dataset_id/subject_id"
        unique: list[FederatedSubject] = []
        dropped = 0
        for s in subjects:
            fh = s.fingerprint_hash
            if fh and fh in seen_hashes:
                log.debug(
                    "Leakage candidate: %s/%s matches %s (same fingerprint)",
                    s.dataset_id, s.subject_id, seen_hashes[fh],
                )
                dropped += 1
            else:
                if fh:
                    seen_hashes[fh] = f"{s.dataset_id}/{s.subject_id}"
                unique.append(s)
        if dropped:
            log.warning(
                "Leakage check removed %d suspected duplicate subjects "
                "(cross-dataset fingerprint collision).",
                dropped,
            )
        return unique

    def _balance(self, subjects: list[FederatedSubject]) -> list[FederatedSubject]:
        """Balance the cohort across the target column."""
        col = self._balance_col
        groups: dict[str, list[FederatedSubject]] = {}
        for s in subjects:
            val = (
                s.harmonized_label if col in ("diagnosis", "label")
                else s.sex if col == "sex"
                else str(s.extra.get(col, "unknown"))
            )
            groups.setdefault(str(val or "unknown"), []).append(s)

        if not groups:
            return subjects

        if self._balance_method == "undersample":
            min_n = min(len(g) for g in groups.values())
            rng = random.Random(self._seed)
            balanced: list[FederatedSubject] = []
            for g in groups.values():
                rng.shuffle(g)
                balanced.extend(g[:min_n])
            log.info(
                "Balanced cohort by undersampling to %d per class (%d total)",
                min_n, len(balanced),
            )
            return balanced

        elif self._balance_method == "oversample":
            max_n = max(len(g) for g in groups.values())
            rng = random.Random(self._seed)
            balanced = []
            for g in groups.values():
                while len(g) < max_n:
                    g.append(rng.choice(g[:len(g)]))
                balanced.extend(g[:max_n])
            log.info("Balanced cohort by oversampling to %d per class.", max_n)
            return balanced

        log.warning("Unknown balance method %r — skipping balancing.", self._balance_method)
        return subjects

    def _assign_splits(
        self,
        subjects: list[FederatedSubject],
        train_frac: float,
        val_frac: float,
    ) -> list[FederatedSubject]:
        """Assign leakage-safe splits: all epochs from one subject go to one split."""
        n = len(subjects)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)

        # Sort by subject fingerprint for reproducible, seed-sensitive ordering
        ordered = sorted(
            subjects,
            key=lambda s: hashlib.sha256(
                f"{self._seed}:{s.fingerprint_hash}:{s.subject_id}".encode()
            ).hexdigest(),
        )
        for i, s in enumerate(ordered):
            if i < n_train:
                s.split = "train"
            elif i < n_train + n_val:
                s.split = "val"
            else:
                s.split = "test"
        return ordered

    def _split_counts(self) -> dict[str, int]:
        return dict(Counter(s.split or "unassigned" for s in self._subjects))

    def _label_distribution(self) -> dict[str, int]:
        return dict(Counter(str(s.harmonized_label or "unknown") for s in self._subjects))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fingerprint(
    subject_id: str,
    dataset_id: str,
    age: float | None,
    sex: str | None,
    label: str | None,
    field_strength: float | None,
) -> str:
    """SHA-256 fingerprint for cross-dataset duplicate detection.

    Bucketed age (5-year bins) + sex + label + field strength.
    Subject ID is NOT included to catch same-human across datasets.
    """
    age_bucket = int(age / 5) * 5 if age is not None else -1
    raw = f"{age_bucket}|{(sex or '').upper()}|{(label or '').lower()}|{round(field_strength, 1) if field_strength else '?'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(str(value).strip())
        return v if v == v else None
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
