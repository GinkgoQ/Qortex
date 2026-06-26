"""Dataset Selector — rank OpenNeuro datasets by fitness for a research goal.

A researcher wanting to train an EEG motor imagery classifier needs:
  - At least 20 subjects
  - MEG or EEG modality
  - Motor imagery task
  - At least 2 labeled classes with >50 trials each
  - Open license (CC0 or CC-BY)
  - Reasonable dataset size (<10 GB)

Expressing this as code should be natural::

    from qortex.inspect.selector import ResearchGoal, DatasetSelector

    goal = ResearchGoal(
        modality="eeg",
        task_keywords=["motor", "imagery", "mi"],
        min_subjects=20,
        min_trials_per_class=50,
        min_n_classes=2,
        license_must_be_open=True,
        max_size_gb=10.0,
    )

    selector = DatasetSelector()
    ranking = selector.find(goal, limit=10)
    for rank in ranking:
        print(rank.summary_line())

This module combines:
  1. Local catalog search (fast, offline) — coarse pre-filter
  2. OpenNeuro API rich metadata — subject count, engagement, paper DOI
  3. LabelLandscape analysis — class count, imbalance, coverage
  4. SignalBudget estimation — total signal hours, min windows achievable

Design: lazily escalate from catalog → API → remote events depending on
how many candidates survive each filter. Most workloads never touch the
remote events layer if the catalog filter is restrictive enough.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ── Research goal specification ───────────────────────────────────────────────

@dataclass
class ResearchGoal:
    """Typed specification of a machine learning research objective.

    All constraints are optional — leave any at None to skip that filter.
    The selector scores each dataset against each constraint and returns a
    ranked list with per-dimension explanations.

    Parameters
    ----------
    modality:
        Primary signal modality: "eeg", "meg", "ieeg", "fnirs", "bold", "t1w".
    task_keywords:
        The dataset's tasks must contain at least one of these keywords
        (case-insensitive substring match).
    min_subjects:
        Minimum number of subjects with the target modality.
    min_trials_per_class:
        Minimum events of each class across the whole dataset.
    min_n_classes:
        Minimum distinct trial_type labels.
    max_imbalance_ratio:
        Maximum allowed max_class / min_class event count ratio.
    min_recording_hours:
        Minimum total signal hours for the target modality.
    max_size_gb:
        Maximum total dataset size in GB.
    license_must_be_open:
        If True, only datasets with CC0, CC-BY, or equivalent are considered.
    species:
        Filter by species string (e.g. "homo sapiens", "mus musculus").
    min_downloads:
        Minimum community download count (proxy for data quality / usability).
    min_bids_version:
        Minimum BIDS specification version (e.g. "1.4.0").
    data_processed_ok:
        If False (default), prefer raw data. If True, preprocessed also ok.
    """
    modality: str | None = None
    task_keywords: list[str] = field(default_factory=list)
    min_subjects: int | None = None
    min_trials_per_class: int | None = None
    min_n_classes: int | None = None
    max_imbalance_ratio: float | None = None
    min_recording_hours: float | None = None
    max_size_gb: float | None = None
    license_must_be_open: bool = False
    species: str | None = None
    min_downloads: int | None = None
    min_bids_version: str | None = None
    data_processed_ok: bool = True

    def describe(self) -> str:
        parts = []
        if self.modality:
            parts.append(f"modality={self.modality}")
        if self.task_keywords:
            parts.append(f"task∈{self.task_keywords}")
        if self.min_subjects:
            parts.append(f"subjects≥{self.min_subjects}")
        if self.min_n_classes:
            parts.append(f"classes≥{self.min_n_classes}")
        if self.min_trials_per_class:
            parts.append(f"trials/class≥{self.min_trials_per_class}")
        if self.max_size_gb:
            parts.append(f"size≤{self.max_size_gb}GB")
        if self.license_must_be_open:
            parts.append("open-license")
        return "ResearchGoal(" + ", ".join(parts) + ")"


_OPEN_LICENSES = {
    "cc0", "cc0-1.0", "pddl", "cc-by", "cc-by-4.0", "cc-by-3.0",
    "odc-by", "mit", "apache-2.0",
}


# ── Per-dataset fitness score ─────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score and explanation for one fitness dimension."""
    name: str
    score: float           # 0–1 where 1 = fully meets requirement
    weight: float          # importance multiplier
    met: bool              # True if the hard constraint is satisfied
    value: Any = None      # actual observed value
    target: Any = None     # required value
    note: str = ""

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class DatasetFitness:
    """Multi-dimensional fitness assessment for one dataset against a goal.

    Attributes
    ----------
    dataset_id:
        OpenNeuro accession number.
    total_score:
        Weighted sum of dimension scores, normalised to 0–100.
    dimensions:
        Per-dimension breakdown for transparency.
    hard_fail:
        List of constraints that disqualify this dataset entirely.
    recommendation:
        One-line human-readable verdict.
    rich_info:
        The full RichDatasetInfo (if available — set when API was queried).
    """
    dataset_id: str
    total_score: float = 0.0
    dimensions: list[DimensionScore] = field(default_factory=list)
    hard_fail: list[str] = field(default_factory=list)
    recommendation: str = ""
    rich_info: Any = None   # RichDatasetInfo | None

    @property
    def is_viable(self) -> bool:
        return len(self.hard_fail) == 0

    @property
    def grade(self) -> str:
        if not self.is_viable:
            return "F"
        if self.total_score >= 80:
            return "A"
        if self.total_score >= 65:
            return "B"
        if self.total_score >= 50:
            return "C"
        if self.total_score >= 35:
            return "D"
        return "F"

    def summary_line(self) -> str:
        status = "✓" if self.is_viable else "✗"
        dims = " | ".join(
            f"{d.name}={d.score:.2f}" for d in self.dimensions[:4]
        )
        return (
            f"{status} {self.dataset_id:15s}  score={self.total_score:.1f}/100 [{self.grade}]  "
            f"{dims}  — {self.recommendation}"
        )

    def report(self) -> str:
        lines = [
            f"Dataset: {self.dataset_id}",
            f"Fitness: {self.total_score:.1f}/100 [{self.grade}]",
        ]
        if self.hard_fail:
            lines.append(f"DISQUALIFIED: {'; '.join(self.hard_fail)}")
        lines.append("Dimension breakdown:")
        for d in self.dimensions:
            met_str = "✓" if d.met else "✗"
            lines.append(
                f"  {met_str} {d.name:30s}  score={d.score:.2f}  "
                f"(observed={d.value}, target={d.target})  {d.note}"
            )
        lines.append(f"Verdict: {self.recommendation}")
        return "\n".join(lines)


# ── Selector ──────────────────────────────────────────────────────────────────

class DatasetSelector:
    """Rank OpenNeuro datasets by fitness for a machine learning research goal.

    The selector works in three lazily-escalating tiers:

    Tier 1 (always runs) — local catalog:
        Fast structured filter: modality, min_subjects, max_size.
        License and min_downloads are NOT enforced here (catalog index may
        lack the field); they are scored and enforced in Tier 2 API scoring.

    Tier 2 (optional, when catalog candidates pass Tier 1) — OpenNeuro API:
        Fetches ``RichDatasetInfo`` (1 API call per dataset) for engagement
        metrics, BIDS version, demographics, study design.

    Tier 3 (optional, for top-K candidates) — remote events:
        Runs ``LabelLandscapeAnalyzer`` to validate class count, imbalance,
        and per-class trial counts. This takes a few seconds per dataset.

    Parameters
    ----------
    token:
        Optional OpenNeuro API token for private datasets.
    config:
        Override global QortexConfig.
    """

    def __init__(
        self,
        token: str | None = None,
        config: Any = None,
    ) -> None:
        from qortex.core.config import get_config
        self._cfg = config or get_config()
        self._token = token

    def rank(
        self,
        dataset_ids: list[str],
        goal: ResearchGoal,
        *,
        tier2_api: bool = True,
        tier3_events: bool = False,
        tier3_top_k: int = 5,
        max_events_files: int = 50,
    ) -> list[DatasetFitness]:
        """Score and rank a list of known dataset IDs against a goal.

        Parameters
        ----------
        dataset_ids:
            List of OpenNeuro accession numbers to evaluate.
        goal:
            The research objective.
        tier2_api:
            If True, fetch RichDatasetInfo from API for each candidate.
        tier3_events:
            If True, run remote label landscape analysis for top-K candidates.
        tier3_top_k:
            Number of candidates to run Tier 3 on (most expensive).
        max_events_files:
            Cap events files fetched per dataset in Tier 3.

        Returns
        -------
        list[DatasetFitness]
            Sorted by total_score descending. Disqualified datasets are
            listed last with grade "F".
        """
        from qortex.client.graphql import OpenNeuroClient

        results: list[DatasetFitness] = []

        with OpenNeuroClient(token=self._token, config=self._cfg) as client:
            for ds_id in dataset_ids:
                fitness = DatasetFitness(dataset_id=ds_id)

                if tier2_api:
                    try:
                        rich = client.get_dataset_rich(ds_id)
                        fitness.rich_info = rich
                    except Exception as exc:
                        log.warning("API fetch failed for %s: %s", ds_id, exc)
                        rich = None
                else:
                    rich = None

                _score_dataset(fitness, goal, rich_info=rich)
                results.append(fitness)

        # Tier 3: remote events analysis for top viable candidates
        if tier3_events:
            viable = [r for r in results if r.is_viable]
            viable.sort(key=lambda r: -r.total_score)
            top = viable[:tier3_top_k]
            if top:
                self._run_tier3(top, goal, max_events_files=max_events_files)

        # Sort: viable first (by score), then disqualified
        results.sort(key=lambda r: (0 if r.is_viable else 1, -r.total_score))
        return results

    def find(
        self,
        goal: ResearchGoal,
        *,
        limit: int = 10,
        catalog_limit: int = 200,
        tier2_api: bool = True,
        tier3_events: bool = False,
        catalog_path: Any = None,
        include_failed: bool = False,
    ) -> list[DatasetFitness]:
        """Search the local catalog and rank results by goal fitness.

        Uses Tier 1 (catalog) as a fast pre-filter, then Tier 2 (API) and
        optionally Tier 3 (remote events) for the top candidates.

        Parameters
        ----------
        catalog_limit:
            How many catalog candidates to evaluate via API (Tier 2).
        include_failed:
            If True, also return disqualified datasets after viable ones
            (useful for debugging why datasets were rejected). Default False.

        Notes
        -----
        Tier 1 filters on modality, min_subjects, and max_size_gb only.
        License and min_downloads are enforced in Tier 2 API scoring, not here,
        because the local catalog index may not reliably carry those fields.
        Use ``rank()`` directly when you already know the dataset IDs — it is
        the diagnostic API and always returns all results including failed ones.
        """
        from qortex.catalog.search import DatasetQuery

        query = DatasetQuery(catalog_path)

        if goal.modality:
            query.modality(goal.modality)
        if goal.min_subjects is not None:
            query.min_subjects(goal.min_subjects)
        if goal.max_size_gb is not None:
            query.max_size_gb(goal.max_size_gb)

        query.limit(catalog_limit)
        candidates = query.fetch()

        dataset_ids = [row["dataset_id"] for row in candidates if row.get("dataset_id")]
        log.info("DatasetSelector: %d catalog candidates for %s", len(dataset_ids), goal.describe())

        if not dataset_ids:
            return []

        ranked = self.rank(
            dataset_ids,
            goal,
            tier2_api=tier2_api,
            tier3_events=tier3_events,
            tier3_top_k=5,
        )
        if include_failed:
            return ranked[:limit]
        return [r for r in ranked if r.is_viable][:limit]

    def _run_tier3(
        self,
        fitness_list: list[DatasetFitness],
        goal: ResearchGoal,
        max_events_files: int,
    ) -> None:
        """Augment fitness scores with remote label landscape data."""
        from qortex.client.graphql import OpenNeuroClient
        from qortex.client.remote import RemoteFileGateway
        from qortex.inspect.label_landscape import LabelLandscapeAnalyzer
        from qortex.manifest.builder import ManifestBuilder

        gateway = RemoteFileGateway(config=self._cfg)
        analyzer = LabelLandscapeAnalyzer(gateway)
        builder = ManifestBuilder()

        with OpenNeuroClient(token=self._token, config=self._cfg) as client:
            for fitness in fitness_list:
                try:
                    snap_ref, raw_files = client.get_files(fitness.dataset_id)
                    manifest = builder.build(fitness.dataset_id, snap_ref, raw_files)
                    landscape = analyzer.analyze(
                        manifest,
                        max_events_files=max_events_files,
                    )
                    _augment_with_landscape(fitness, goal, landscape)
                    log.debug(
                        "Tier 3 for %s: %d classes, imbalance=%.2f",
                        fitness.dataset_id,
                        landscape.n_classes,
                        landscape.imbalance_ratio or 0,
                    )
                except Exception as exc:
                    log.warning("Tier 3 failed for %s: %s", fitness.dataset_id, exc)


# ── Scoring engine ────────────────────────────────────────────────────────────

def _score_dataset(
    fitness: DatasetFitness,
    goal: ResearchGoal,
    rich_info: Any,   # RichDatasetInfo | None
) -> None:
    """Populate fitness dimensions from RichDatasetInfo (Tier 2 scoring)."""
    dims: list[DimensionScore] = []

    # ── Modality match ────────────────────────────────────────────────────
    if goal.modality:
        modalities = (rich_info.modalities if rich_info else []) or []
        modality_match = any(
            goal.modality.lower() in m.lower() or m.lower() in goal.modality.lower()
            for m in modalities
        )
        dims.append(DimensionScore(
            name="modality",
            score=1.0 if modality_match else 0.0,
            weight=3.0,
            met=modality_match,
            value=modalities,
            target=goal.modality,
            note="" if modality_match else "modality not found in dataset",
        ))
        if not modality_match:
            fitness.hard_fail.append(f"modality '{goal.modality}' not present")

    # ── Task keyword match ────────────────────────────────────────────────
    if goal.task_keywords:
        tasks = (rich_info.tasks if rich_info else []) or []
        task_text = " ".join(tasks).lower()
        matched = any(kw.lower() in task_text for kw in goal.task_keywords)
        dims.append(DimensionScore(
            name="task_keywords",
            score=1.0 if matched else 0.0,
            weight=2.5,
            met=matched,
            value=tasks,
            target=goal.task_keywords,
            note="" if matched else f"none of {goal.task_keywords} found in tasks",
        ))
        if not matched:
            fitness.hard_fail.append(f"task keywords {goal.task_keywords} not found")

    # ── Subject count ─────────────────────────────────────────────────────
    if goal.min_subjects is not None:
        snap = rich_info.latest_snapshot_summary if rich_info else None
        n_subj = snap.n_subjects if snap else None
        if n_subj is not None:
            ratio = n_subj / goal.min_subjects
            score = min(1.0, ratio)
            met = n_subj >= goal.min_subjects
        else:
            score, met = 0.5, True  # unknown — don't disqualify
        dims.append(DimensionScore(
            name="subject_count",
            score=score,
            weight=2.0,
            met=met,
            value=n_subj,
            target=goal.min_subjects,
            note=f"{n_subj} subjects" if n_subj else "subject count unknown",
        ))
        if not met and n_subj is not None:
            fitness.hard_fail.append(f"only {n_subj} subjects (need {goal.min_subjects})")

    # ── Size constraint ───────────────────────────────────────────────────
    if goal.max_size_gb is not None:
        snap = rich_info.latest_snapshot_summary if rich_info else None
        size_gb = snap.total_size_gb if snap else None
        if size_gb is not None:
            score = 1.0 if size_gb <= goal.max_size_gb else max(0.0, 1.0 - (size_gb - goal.max_size_gb) / goal.max_size_gb)
            met = size_gb <= goal.max_size_gb
        else:
            score, met = 0.7, True
        dims.append(DimensionScore(
            name="size",
            score=score,
            weight=1.0,
            met=met,
            value=round(size_gb, 2) if size_gb else None,
            target=goal.max_size_gb,
            note=f"{size_gb:.1f} GB" if size_gb else "size unknown",
        ))

    # ── License ───────────────────────────────────────────────────────────
    if goal.license_must_be_open:
        lic = (rich_info.license if rich_info else None) or ""
        is_open = lic.lower().replace(" ", "-") in _OPEN_LICENSES or lic.lower().startswith("cc")
        dims.append(DimensionScore(
            name="open_license",
            score=1.0 if is_open else 0.0,
            weight=1.5,
            met=is_open,
            value=lic or None,
            target="CC0/CC-BY/MIT",
            note="" if is_open else f"license '{lic}' may not be open",
        ))
        if not is_open and lic:
            fitness.hard_fail.append(f"license '{lic}' not in open-license allowlist")

    # ── Species ───────────────────────────────────────────────────────────
    if goal.species:
        ds_species = (rich_info.species if rich_info else None) or ""
        match = goal.species.lower() in ds_species.lower() or not ds_species
        dims.append(DimensionScore(
            name="species",
            score=1.0 if match else 0.0,
            weight=1.0,
            met=match,
            value=ds_species or None,
            target=goal.species,
        ))

    # ── Engagement (proxy for data quality) ───────────────────────────────
    if rich_info:
        eng = rich_info.engagement
        pop = eng.popularity_score
        dims.append(DimensionScore(
            name="community_engagement",
            score=min(1.0, pop / 50.0),
            weight=0.5,
            met=True,
            value={"downloads": eng.downloads, "stars": eng.stars},
            note=f"popularity={pop:.1f}/100",
        ))

    # ── Downloads min ─────────────────────────────────────────────────────
    if goal.min_downloads is not None and rich_info:
        dl = rich_info.engagement.downloads
        met = dl >= goal.min_downloads
        dims.append(DimensionScore(
            name="min_downloads",
            score=min(1.0, dl / goal.min_downloads),
            weight=0.5,
            met=met,
            value=dl,
            target=goal.min_downloads,
        ))
        if not met:
            fitness.hard_fail.append(f"only {dl} downloads (need {goal.min_downloads})")

    # ── BIDS version ──────────────────────────────────────────────────────
    if goal.min_bids_version and rich_info:
        snap = rich_info.latest_snapshot_summary
        bids_ver = snap.bids_version if snap else None
        if bids_ver:
            try:
                met = _version_gte(bids_ver, goal.min_bids_version)
            except Exception:
                met = True  # can't compare — don't disqualify
        else:
            met = True
        dims.append(DimensionScore(
            name="bids_version",
            score=1.0 if met else 0.5,
            weight=0.3,
            met=met,
            value=bids_ver,
            target=goal.min_bids_version,
        ))

    fitness.dimensions = dims

    # Compute total score (normalise by total possible weight)
    total_weight = sum(d.weight for d in dims)
    if total_weight > 0:
        weighted_sum = sum(d.weighted_score for d in dims)
        fitness.total_score = round(100.0 * weighted_sum / total_weight, 1)
    else:
        fitness.total_score = 0.0

    # Recommendation
    if fitness.hard_fail:
        fitness.recommendation = "Disqualified: " + "; ".join(fitness.hard_fail[:2])
    elif fitness.total_score >= 80:
        fitness.recommendation = "Excellent fit. Recommended for download."
    elif fitness.total_score >= 60:
        fitness.recommendation = "Good fit. Check task alignment before downloading."
    elif fitness.total_score >= 40:
        fitness.recommendation = "Partial fit. Review gaps before committing to download."
    else:
        fitness.recommendation = "Poor fit. Consider other datasets."


def _augment_with_landscape(
    fitness: DatasetFitness,
    goal: ResearchGoal,
    landscape: Any,  # LabelLandscape
) -> None:
    """Add Tier 3 dimensions from LabelLandscape to fitness."""
    dims = fitness.dimensions

    if goal.min_n_classes is not None:
        met = landscape.n_classes >= goal.min_n_classes
        dims.append(DimensionScore(
            name="n_classes",
            score=min(1.0, landscape.n_classes / max(1, goal.min_n_classes)),
            weight=2.5,
            met=met,
            value=landscape.n_classes,
            target=goal.min_n_classes,
        ))
        if not met:
            fitness.hard_fail.append(
                f"only {landscape.n_classes} classes (need {goal.min_n_classes})"
            )

    if goal.min_trials_per_class is not None:
        min_count = min(landscape.class_counts.values()) if landscape.class_counts else 0
        met = min_count >= goal.min_trials_per_class
        dims.append(DimensionScore(
            name="trials_per_class",
            score=min(1.0, min_count / max(1, goal.min_trials_per_class)),
            weight=2.0,
            met=met,
            value=min_count,
            target=goal.min_trials_per_class,
        ))

    if goal.max_imbalance_ratio is not None and landscape.imbalance_ratio is not None:
        met = landscape.imbalance_ratio <= goal.max_imbalance_ratio
        score = min(1.0, goal.max_imbalance_ratio / max(1, landscape.imbalance_ratio))
        dims.append(DimensionScore(
            name="class_balance",
            score=score,
            weight=1.5,
            met=met,
            value=round(landscape.imbalance_ratio, 2),
            target=goal.max_imbalance_ratio,
        ))

    # Recompute total score
    total_weight = sum(d.weight for d in dims)
    if total_weight > 0:
        fitness.total_score = round(100.0 * sum(d.weighted_score for d in dims) / total_weight, 1)

    # Update recommendation
    if fitness.hard_fail:
        fitness.recommendation = "Disqualified after events analysis: " + "; ".join(fitness.hard_fail[:2])
    elif fitness.total_score >= 80:
        fitness.recommendation = "Excellent fit (events verified). Recommended."
    elif fitness.total_score >= 60:
        fitness.recommendation = "Good fit (events verified)."
    else:
        fitness.recommendation = "Partial fit — review class distribution."


def _version_gte(v1: str, v2: str) -> bool:
    """Return True if version string v1 >= v2 (simple dot-comparison)."""
    def _parts(v: str):
        return [int(x) for x in v.lstrip("v").split(".")[:3]]
    try:
        return _parts(v1) >= _parts(v2)
    except Exception:
        return True
