"""Typed OpenNeuro GraphQL client.

All queries use the ``variables`` field of the GraphQL POST body — never
string interpolation — so special characters in IDs/tags are handled safely.

Public surface
--------------
OpenNeuroClient
    .get_dataset(dataset_id)              → DatasetRef  (lightweight)
    .get_dataset_rich(dataset_id)         → RichDatasetInfo (full metadata)
    .get_snapshots(dataset_id)            → list[SnapshotRef]
    .get_snapshot(dataset_id, tag)        → SnapshotRef
    .get_latest_snapshot(dataset_id)      → SnapshotRef
    .get_snapshot_summary(dataset_id, tag)→ SnapshotSummary (API-level, no file tree)
    .get_files(dataset_id, tag=None)      → (SnapshotRef, list[dict])
    .search_datasets(...)                 → list[DatasetRef]
    .search_datasets_rich(...)            → list[RichDatasetInfo]

Design notes
------------
* The OpenNeuro GraphQL API exposes richer metadata than we previously queried:
  - ``snapshot.summary`` gives subjects, sessions, tasks, modalities, and
    **subjectMetadata** (age, sex, group) directly — no participants.tsv download.
  - ``snapshot.hexsha`` is a content hash for reliable cache invalidation.
  - ``dataset.analytics`` provides views and download counts.
  - ``snapshot.description`` includes Funding, References, BIDSVersion.
  - ``stars`` and ``followers`` are list types (count via len()).
* All of this is fetched without touching the file tree or downloading data.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from qortex.client.auth import resolve_token
from qortex.client.transport import SyncTransport
from qortex.core.config import QortexConfig, get_config
from qortex.core.entities import DatasetRef, SnapshotRef
from qortex.core.exceptions import (
    APIError,
    DatasetNotFoundError,
    ManifestError,
    SnapshotNotFoundError,
)

log = logging.getLogger(__name__)

# ── Rich result types ─────────────────────────────────────────────────────────

@dataclass
class SubjectDemographic:
    """Demographics for one participant, sourced directly from the API."""
    participant_id: str
    age: int | None = None
    sex: str | None = None
    group: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "age": self.age,
            "sex": self.sex,
            "group": self.group,
        }


@dataclass
class SnapshotSummary:
    """API-level BIDS summary for one snapshot — no file tree required."""
    dataset_id: str
    tag: str
    hexsha: str | None
    subjects: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    total_files: int = 0
    total_size_bytes: int = 0
    data_processed: bool = False
    subject_demographics: list[SubjectDemographic] = field(default_factory=list)
    bids_version: str | None = None
    license: str | None = None
    funding: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    ethics_approvals: list[str] = field(default_factory=list)
    how_to_acknowledge: str | None = None

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def n_sessions(self) -> int:
        return len(self.sessions)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / 1e9

    def demographics_dataframe(self):
        """Return subject demographics as a Polars DataFrame."""
        import polars as pl
        if not self.subject_demographics:
            return pl.DataFrame(schema={"participant_id": pl.Utf8, "age": pl.Int32, "sex": pl.Utf8, "group": pl.Utf8})
        return pl.DataFrame([d.to_dict() for d in self.subject_demographics])

    def age_stats(self) -> dict[str, Any] | None:
        """Return age distribution statistics from API-sourced demographics."""
        ages = [d.age for d in self.subject_demographics if d.age is not None]
        if not ages:
            return None
        return {
            "n": len(ages),
            "mean": round(sum(ages) / len(ages), 1),
            "min": min(ages),
            "max": max(ages),
            "sex_distribution": _count_values(d.sex for d in self.subject_demographics if d.sex),
            "group_distribution": _count_values(d.group for d in self.subject_demographics if d.group),
        }


@dataclass
class DatasetEngagement:
    """Community engagement metrics from the OpenNeuro platform."""
    views: int = 0
    downloads: int = 0
    stars: int = 0
    followers: int = 0

    @property
    def popularity_score(self) -> float:
        """Composite popularity score (0-100). Higher = more community adoption."""
        # Weighted: downloads matter most (actual use), then views, stars, followers
        score = (
            min(self.downloads / 500.0, 40.0)
            + min(self.views / 50_000.0, 30.0)
            + min(self.stars / 25.0, 20.0)
            + min(self.followers / 10.0, 10.0)
        )
        return round(score, 1)


@dataclass
class RichDatasetInfo:
    """Full OpenNeuro dataset metadata, sourced from the API without file tree access.

    This is the richer alternative to ``DatasetRef``. Use it for inspection,
    dataset selection, and metadata-first workflows.
    """
    id: str
    name: str | None = None
    description: str | None = None
    doi: str | None = None
    license: str | None = None
    authors: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    species: str | None = None
    senior_author: str | None = None
    study_domain: str | None = None
    study_design: str | None = None
    study_longitudinal: str | None = None
    associated_paper_doi: str | None = None
    openneuro_paper_doi: str | None = None
    grant_funder: str | None = None
    grant_id: str | None = None
    data_processed: bool = False
    publish_date: str | None = None
    created: str | None = None
    engagement: DatasetEngagement = field(default_factory=DatasetEngagement)
    latest_snapshot_tag: str | None = None
    latest_snapshot_hexsha: str | None = None
    latest_snapshot_summary: SnapshotSummary | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataset_ref(self) -> DatasetRef:
        """Downcast to the lightweight DatasetRef for backwards compatibility."""
        return DatasetRef(
            id=self.id,
            name=self.name,
            description=self.description,
            doi=self.doi,
            license=self.license,
            authors=self.authors,
            modalities=self.modalities,
            tasks=self.tasks,
            raw_metadata=self.raw_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        snap = self.latest_snapshot_summary
        return {
            "id": self.id,
            "name": self.name,
            "doi": self.doi,
            "license": self.license,
            "authors": self.authors,
            "modalities": self.modalities,
            "tasks": self.tasks,
            "species": self.species,
            "senior_author": self.senior_author,
            "study_domain": self.study_domain,
            "study_design": self.study_design,
            "associated_paper_doi": self.associated_paper_doi,
            "publish_date": self.publish_date,
            "engagement": {
                "views": self.engagement.views,
                "downloads": self.engagement.downloads,
                "stars": self.engagement.stars,
                "followers": self.engagement.followers,
                "popularity_score": self.engagement.popularity_score,
            },
            "latest_snapshot": {
                "tag": self.latest_snapshot_tag,
                "hexsha": self.latest_snapshot_hexsha,
                "n_subjects": snap.n_subjects if snap else None,
                "n_sessions": snap.n_sessions if snap else None,
                "n_tasks": snap.n_tasks if snap else None,
                "subjects": snap.subjects if snap else [],
                "sessions": snap.sessions if snap else [],
                "tasks": snap.tasks if snap else [],
                "total_files": snap.total_files if snap else None,
                "total_size_gb": round(snap.total_size_gb, 3) if snap else None,
                "bids_version": snap.bids_version if snap else None,
                "funding": snap.funding if snap else [],
                "references": snap.references if snap else [],
                "ethics_approvals": snap.ethics_approvals if snap else [],
                "how_to_acknowledge": snap.how_to_acknowledge if snap else None,
                "data_processed": snap.data_processed if snap else False,
            } if self.latest_snapshot_tag else None,
        }


# ── Query definitions ─────────────────────────────────────────────────────────

_Q_DATASET = """
query GetDataset($datasetId: ID!) {
    dataset(id: $datasetId) {
        id
        metadata {
            datasetName
            modalities
            tasksCompleted
        }
        latestSnapshot {
            id
            tag
            created
            size
            description {
                Name
                License
                DatasetDOI
                Authors
            }
        }
    }
}
"""

_Q_DATASET_RICH = """
query GetDatasetRich($datasetId: ID!) {
    dataset(id: $datasetId) {
        id
        created
        publishDate
        metadata {
            datasetName
            modalities
            tasksCompleted
            dataProcessed
            species
            associatedPaperDOI
            openneuroPaperDOI
            seniorAuthor
            studyDomain
            studyDesign
            studyLongitudinal
            grantFunderName
            grantIdentifier
        }
        analytics {
            views
            downloads
        }
        stars { userId }
        followers { userId }
        latestSnapshot {
            id
            tag
            hexsha
            created
            size
            description {
                Name
                License
                DatasetDOI
                Authors
                BIDSVersion
                Funding
                ReferencesAndLinks
                HowToAcknowledge
                EthicsApprovals
            }
            summary {
                modalities
                sessions
                subjects
                tasks
                size
                totalFiles
                dataProcessed
                subjectMetadata {
                    participantId
                    age
                    sex
                    group
                }
            }
        }
    }
}
"""

_Q_DATASET_README = """
query GetDatasetReadme($datasetId: ID!) {
    dataset(id: $datasetId) {
        latestSnapshot {
            tag
            readme
        }
    }
}
"""

_Q_SNAPSHOT_README = """
query GetSnapshotReadme($datasetId: ID!, $tag: String!) {
    snapshot(datasetId: $datasetId, tag: $tag) {
        tag
        readme
    }
}
"""

_Q_SNAPSHOT_ISSUES = """
query GetSnapshotIssues($datasetId: ID!, $tag: String!) {
    snapshot(datasetId: $datasetId, tag: $tag) {
        tag
        issues {
            severity
            key
            reason
            files { id name path }
            helpUrl
        }
    }
}
"""

_Q_SNAPSHOTS = """
query GetSnapshots($datasetId: ID!) {
    dataset(id: $datasetId) {
        snapshots {
            id
            tag
            created
            size
            hexsha
        }
    }
}
"""

_Q_SNAPSHOT_SUMMARY = """
query GetSnapshotSummary($datasetId: ID!, $tag: String!) {
    snapshot(datasetId: $datasetId, tag: $tag) {
        id
        tag
        hexsha
        created
        size
        description {
            Name
            License
            DatasetDOI
            Authors
            BIDSVersion
            Funding
            ReferencesAndLinks
            HowToAcknowledge
            EthicsApprovals
        }
        summary {
            modalities
            sessions
            subjects
            tasks
            size
            totalFiles
            dataProcessed
            subjectMetadata {
                participantId
                age
                sex
                group
            }
        }
    }
}
"""

_Q_SNAPSHOT_FILES = """
query GetSnapshotFiles($datasetId: ID!, $tag: String!) {
    snapshot(datasetId: $datasetId, tag: $tag) {
        id
        tag
        description {
            Name
            DatasetDOI
            License
            Authors
        }
        files(recursive: true) {
            id
            filename
            urls
            size
            directory
        }
    }
}
"""

_Q_LATEST_FILES = """
query GetLatestFiles($datasetId: ID!) {
    dataset(id: $datasetId) {
        latestSnapshot {
            id
            tag
            hexsha
            description {
                Name
                DatasetDOI
                License
                Authors
            }
            files(recursive: true) {
                id
                filename
                urls
                size
                directory
            }
        }
    }
}
"""

_Q_SEARCH = """
query SearchDatasets($first: Int, $after: String) {
    datasets(first: $first, after: $after) {
        edges {
            node {
                id
                publishDate
                metadata {
                    datasetName
                    modalities
                    tasksCompleted
                    species
                    seniorAuthor
                    studyDomain
                }
                analytics {
                    views
                    downloads
                }
                stars { userId }
                latestSnapshot {
                    tag
                    hexsha
                    size
                    description {
                        License
                        DatasetDOI
                        Authors
                    }
                    summary {
                        subjects
                        sessions
                        tasks
                        modalities
                        totalFiles
                        dataProcessed
                    }
                }
            }
        }
        pageInfo {
            hasNextPage
            endCursor
        }
    }
}
"""

# ── Client ────────────────────────────────────────────────────────────────────

class OpenNeuroClient:
    """Synchronous typed GraphQL client for the OpenNeuro API.

    Manages a persistent HTTP connection pool for the session lifetime.
    Use as a context manager for automatic cleanup::

        with OpenNeuroClient() as client:
            info = client.get_dataset_rich("ds000117")
            print(info.engagement.downloads)
    """

    def __init__(
        self,
        token: str | None = None,
        config: QortexConfig | None = None,
    ) -> None:
        self._cfg = config or get_config()
        self._token = resolve_token(token)
        self._transport = SyncTransport(self._cfg)

    # ── Internal ──────────────────────────────────────────────────────────

    def _query(
        self,
        query: str,
        variables: dict[str, Any] | None = None,
        timeout: float | None = None,
        *,
        partial_ok: bool = False,
    ) -> dict[str, Any]:
        """Execute a GQL query and return the ``data`` dict.

        ``partial_ok``: per the GraphQL spec a response may carry BOTH ``data``
        and ``errors`` — a nullable field that failed (e.g. one dataset with a
        broken ``latestSnapshot``) nulls only that field while every sibling in
        the same response stays valid. For collection queries (the catalog
        sweep over all ~1.8k datasets) a handful of such field errors must NOT
        discard the whole page; when ``partial_ok`` is set we return the
        ``data`` that did resolve and let the caller skip the null nodes.
        Single-entity lookups keep the strict default: any error is fatal.
        """
        cookies: dict[str, str] = {}
        if self._token:
            cookies["accessToken"] = self._token

        response = self._transport.post(
            self._cfg.gql_endpoint,
            json={"query": query, "variables": variables or {}},
            cookies=cookies,
            timeout=timeout,
        )

        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise APIError(
                f"OpenNeuro returned non-JSON response (HTTP {response.status_code})",
                status_code=response.status_code,
            ) from exc

        if "errors" in body:
            # Partial success (data present + field errors) is fatal only for
            # strict single-entity callers; a collection sweep tolerates it.
            if not (partial_ok and body.get("data") is not None):
                self._raise_gql_error(body["errors"], variables or {})

        data = body.get("data")
        if data is None:
            raise APIError("GraphQL response contained no 'data' field.")
        return data

    @staticmethod
    def _raise_gql_error(errors: list[dict], variables: dict) -> None:
        msg = errors[0].get("message", "Unknown GraphQL error")
        dataset_id = variables.get("datasetId", "")
        tag = variables.get("tag", "")

        if "does not exist" in msg or "not found" in msg.lower():
            if tag:
                raise SnapshotNotFoundError(dataset_id, tag)
            if dataset_id:
                raise DatasetNotFoundError(dataset_id)

        if "do not have access" in msg or "Unauthorized" in msg:
            from qortex.core.exceptions import AuthError
            raise AuthError(
                f"Access denied: {msg}. "
                f"Run `qortex login` to configure your API token."
            )

        raise APIError(f"GraphQL error: {msg}")

    # ── Public API — lightweight ───────────────────────────────────────────

    def get_dataset(self, dataset_id: str) -> DatasetRef:
        """Return lightweight metadata for a dataset (fast, 1 round-trip)."""
        data = self._query(_Q_DATASET, {"datasetId": dataset_id},
                           timeout=self._cfg.metadata_timeout)
        node = data.get("dataset")
        if node is None:
            raise DatasetNotFoundError(dataset_id)
        meta = node.get("metadata") or {}
        return DatasetRef(
            id=node["id"],
            name=meta.get("datasetName"),
            description=((node.get("latestSnapshot") or {}).get("description") or {}).get("Name"),
            doi=_normalise_doi(((node.get("latestSnapshot") or {}).get("description") or {}).get("DatasetDOI")),
            license=((node.get("latestSnapshot") or {}).get("description") or {}).get("License"),
            authors=(((node.get("latestSnapshot") or {}).get("description") or {}).get("Authors") or []),
            modalities=meta.get("modalities") or [],
            tasks=meta.get("tasksCompleted") or [],
            raw_metadata=meta,
        )

    # ── Public API — rich metadata ────────────────────────────────────────

    def get_dataset_rich(self, dataset_id: str) -> RichDatasetInfo:
        """Return the full OpenNeuro dataset profile in a single API call.

        This is the primary entry point for metadata-first workflows.
        Includes community engagement, API-level subject demographics, BIDS
        summary, acquisition metadata, and full description fields — all
        without downloading or iterating the file tree.
        """
        data = self._query(_Q_DATASET_RICH, {"datasetId": dataset_id},
                           timeout=self._cfg.metadata_timeout)
        node = data.get("dataset")
        if node is None:
            raise DatasetNotFoundError(dataset_id)

        meta = node.get("metadata") or {}
        analytics = node.get("analytics") or {}
        snap = node.get("latestSnapshot") or {}
        desc = snap.get("description") or {}
        snap_summary = snap.get("summary") or {}

        engagement = DatasetEngagement(
            views=int(analytics.get("views") or 0),
            downloads=int(analytics.get("downloads") or 0),
            stars=len(node.get("stars") or []),
            followers=len(node.get("followers") or []),
        )

        summary = _parse_snapshot_summary(
            dataset_id=dataset_id,
            snap=snap,
            desc=desc,
            snap_summary=snap_summary,
        ) if snap else None

        return RichDatasetInfo(
            id=dataset_id,
            name=meta.get("datasetName"),
            description=desc.get("Name"),
            doi=_normalise_doi(desc.get("DatasetDOI")),
            license=desc.get("License"),
            authors=desc.get("Authors") or [],
            modalities=meta.get("modalities") or [],
            tasks=meta.get("tasksCompleted") or [],
            species=meta.get("species"),
            senior_author=meta.get("seniorAuthor"),
            study_domain=meta.get("studyDomain"),
            study_design=meta.get("studyDesign"),
            study_longitudinal=meta.get("studyLongitudinal"),
            associated_paper_doi=_normalise_doi(meta.get("associatedPaperDOI")),
            openneuro_paper_doi=_normalise_doi(meta.get("openneuroPaperDOI")),
            grant_funder=meta.get("grantFunderName"),
            grant_id=meta.get("grantIdentifier"),
            data_processed=bool(meta.get("dataProcessed")),
            publish_date=node.get("publishDate"),
            created=node.get("created"),
            engagement=engagement,
            latest_snapshot_tag=snap.get("tag"),
            latest_snapshot_hexsha=snap.get("hexsha"),
            latest_snapshot_summary=summary,
            raw_metadata=meta,
        )

    def get_snapshot_summary(
        self, dataset_id: str, tag: str
    ) -> SnapshotSummary:
        """Return API-level BIDS summary for a specific snapshot.

        This is faster than fetching the full file tree: it uses the
        ``snapshot.summary`` field which OpenNeuro pre-computes and caches.
        Includes subject demographics (age, sex, group), session list,
        task list, file count, and total size — all without the file tree.
        """
        data = self._query(
            _Q_SNAPSHOT_SUMMARY,
            {"datasetId": dataset_id, "tag": tag},
            timeout=self._cfg.metadata_timeout,
        )
        snap = data.get("snapshot")
        if snap is None:
            raise SnapshotNotFoundError(dataset_id, tag)

        desc = snap.get("description") or {}
        snap_summary = snap.get("summary") or {}
        return _parse_snapshot_summary(
            dataset_id=dataset_id,
            snap=snap,
            desc=desc,
            snap_summary=snap_summary,
        )

    def get_snapshots(self, dataset_id: str) -> list[SnapshotRef]:
        """Return all available snapshot tags, newest first."""
        data = self._query(_Q_SNAPSHOTS, {"datasetId": dataset_id},
                           timeout=self._cfg.metadata_timeout)
        node = data.get("dataset")
        if node is None:
            raise DatasetNotFoundError(dataset_id)
        snaps = node.get("snapshots") or []
        result = [
            SnapshotRef(
                dataset_id=dataset_id,
                tag=s["tag"],
                id=s["id"],
                created=s.get("created"),
                size=s.get("size"),
                hexsha=s.get("hexsha"),
            )
            for s in snaps
        ]
        # Sort newest first (tags are typically semver-like)
        result.sort(key=lambda s: s.created or "", reverse=True)
        return result

    def count_datasets(self) -> int:
        """Total number of datasets on OpenNeuro — one cheap round-trip.

        The GraphQL connection reports its full size via ``pageInfo.count``
        without paging the edges, so the catalog sweep can learn the target
        up front (show progress against a real denominator, and pre-compute
        every page's offset cursor for concurrent fetching)."""
        data = self._query(
            "query { datasets(first: 1) { pageInfo { count } } }",
            timeout=self._cfg.metadata_timeout,
        )
        return int(((data.get("datasets") or {}).get("pageInfo") or {}).get("count") or 0)

    def get_snapshot(self, dataset_id: str, tag: str) -> SnapshotRef:
        """Return a specific snapshot ref (validates that the tag exists)."""
        available = self.get_snapshots(dataset_id)
        tags = {s.tag: s for s in available}
        if tag not in tags:
            raise SnapshotNotFoundError(dataset_id, tag, list(tags))
        return tags[tag]

    def get_latest_snapshot(self, dataset_id: str) -> SnapshotRef:
        """Return the latest published snapshot."""
        data = self._query(_Q_DATASET, {"datasetId": dataset_id},
                           timeout=self._cfg.metadata_timeout)
        node = data.get("dataset")
        if node is None:
            raise DatasetNotFoundError(dataset_id)
        latest = node.get("latestSnapshot")
        if latest is None:
            raise ManifestError(
                f"Dataset {dataset_id!r} has no published snapshots yet."
            )
        return SnapshotRef(
            dataset_id=dataset_id,
            tag=latest["tag"],
            id=latest["id"],
            created=latest.get("created"),
            size=latest.get("size"),
        )

    def get_files(
        self,
        dataset_id: str,
        tag: str | None = None,
    ) -> tuple[SnapshotRef, list[dict]]:
        """Fetch the recursive file tree for a snapshot.

        This fetches all file records (path, size, URLs, directory flag).
        For lightweight metadata-only inspection, prefer
        ``get_snapshot_summary()`` which avoids the file tree entirely.

        Parameters
        ----------
        tag:
            Snapshot tag.  ``None`` → latest published snapshot.

        Returns
        -------
        (snapshot_ref, raw_file_dicts)
        """
        if tag is None:
            data = self._query(_Q_LATEST_FILES, {"datasetId": dataset_id},
                               timeout=self._cfg.metadata_timeout)
            snap_raw = data["dataset"]["latestSnapshot"]
        else:
            data = self._query(_Q_SNAPSHOT_FILES,
                               {"datasetId": dataset_id, "tag": tag},
                               timeout=self._cfg.metadata_timeout)
            snap_raw = data["snapshot"]

        desc = snap_raw.get("description") or {}
        snap = SnapshotRef(
            dataset_id=dataset_id,
            tag=snap_raw["tag"],
            id=snap_raw["id"],
            doi=_normalise_doi(desc.get("DatasetDOI")),
            hexsha=snap_raw.get("hexsha"),
        )
        raw_files: list[dict] = snap_raw.get("files") or []
        return snap, raw_files

    def get_readme(self, dataset_id: str, tag: str | None = None) -> str | None:
        """Return the README text for a dataset snapshot.

        The README is the description shown on the OpenNeuro dataset page.
        Returns ``None`` if the dataset has no README file.

        Parameters
        ----------
        tag:
            Snapshot tag. ``None`` → latest published snapshot.
        """
        if tag is None:
            data = self._query(_Q_DATASET_README, {"datasetId": dataset_id},
                               timeout=self._cfg.metadata_timeout)
            snap_raw = (data.get("dataset") or {}).get("latestSnapshot") or {}
        else:
            data = self._query(_Q_SNAPSHOT_README,
                               {"datasetId": dataset_id, "tag": tag},
                               timeout=self._cfg.metadata_timeout)
            snap_raw = data.get("snapshot") or {}
        return snap_raw.get("readme") or None

    def get_validation_issues(
        self, dataset_id: str, tag: str
    ) -> list[dict[str, Any]]:
        """Return BIDS validation issues for a snapshot.

        Each entry is a dict with keys: ``severity``, ``key``, ``reason``,
        ``files`` (list of dicts with id/name/path), and ``helpUrl``.
        Severity is ``"error"`` or ``"warning"``.

        Returns an empty list if the snapshot has no issues or the API does
        not support the issues field for this dataset.
        """
        try:
            data = self._query(_Q_SNAPSHOT_ISSUES,
                               {"datasetId": dataset_id, "tag": tag},
                               timeout=self._cfg.metadata_timeout)
            snap = data.get("snapshot") or {}
            return snap.get("issues") or []
        except APIError:
            return []

    # ── Search ────────────────────────────────────────────────────────────

    def search_datasets(
        self,
        *,
        modality: str | None = None,
        task: str | None = None,
        limit: int = 100,
    ) -> list[DatasetRef]:
        """Return datasets matching the given filters (client-side filtering)."""
        refs = self._search_raw(modality=modality, task=task, limit=limit)
        return [r.to_dataset_ref() for r in refs]

    def search_datasets_rich(
        self,
        *,
        modality: str | None = None,
        task: str | None = None,
        min_subjects: int | None = None,
        min_downloads: int | None = None,
        species: str | None = None,
        data_processed: bool | None = None,
        limit: int = 100,
        sort_by: str = "downloads",
    ) -> list[RichDatasetInfo]:
        """Rich dataset search with engagement sorting and demographic filtering.

        This is the advanced search path — it returns ``RichDatasetInfo``
        objects with community engagement, API-level subject counts, BIDS
        metadata and demographics from the API summary.

        Parameters
        ----------
        sort_by:
            ``"downloads"``, ``"views"``, ``"stars"``, ``"subjects"``,
            ``"size"``, or ``"recent"``.
        """
        refs = self._search_raw(modality=modality, task=task, limit=min(limit * 3, 300))

        # Apply additional filters
        if min_subjects is not None:
            refs = [r for r in refs
                    if (r.latest_snapshot_summary and
                        r.latest_snapshot_summary.n_subjects >= min_subjects)]
        if min_downloads is not None:
            refs = [r for r in refs if r.engagement.downloads >= min_downloads]
        if species is not None:
            refs = [r for r in refs
                    if (r.species or "").lower() == species.lower()]
        if data_processed is not None:
            refs = [r for r in refs if r.data_processed == data_processed]

        # Sort
        sort_key = {
            "downloads": lambda r: r.engagement.downloads,
            "views": lambda r: r.engagement.views,
            "stars": lambda r: r.engagement.stars,
            "subjects": lambda r: (r.latest_snapshot_summary.n_subjects
                                   if r.latest_snapshot_summary else 0),
            "size": lambda r: (r.latest_snapshot_summary.total_size_bytes
                               if r.latest_snapshot_summary else 0),
            "recent": lambda r: r.publish_date or r.created or "",
            "popularity": lambda r: r.engagement.popularity_score,
        }.get(sort_by, lambda r: r.engagement.downloads)

        refs.sort(key=sort_key, reverse=True)
        return refs[:limit]

    def _search_raw(
        self,
        modality: str | None,
        task: str | None,
        limit: int,
    ) -> list[RichDatasetInfo]:
        """Paginate through the datasets endpoint and apply client-side filters."""
        refs: list[RichDatasetInfo] = []
        cursor: str | None = None
        fetched = 0

        while fetched < limit:
            batch = min(100, limit - fetched)
            variables: dict[str, Any] = {"first": batch}
            if cursor:
                variables["after"] = cursor

            try:
                data = self._query(_Q_SEARCH, variables,
                                   timeout=self._cfg.metadata_timeout)
            except APIError as exc:
                log.warning("Search page failed: %s", exc)
                break

            edges = (data.get("datasets") or {}).get("edges") or []
            page_info = (data.get("datasets") or {}).get("pageInfo") or {}

            for edge in edges:
                node = edge.get("node") or {}
                info = _parse_search_node(node)
                if info is None:
                    continue

                mods = info.modalities
                tasks = info.tasks
                if modality and modality.lower() not in [m.lower() for m in mods]:
                    continue
                if task and task.lower() not in [t.lower() for t in tasks]:
                    continue
                refs.append(info)

            fetched += len(edges)
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

        return refs

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "OpenNeuroClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Shared, connection-pooled client ──────────────────────────────────────────
#
# ``OpenNeuroClient`` holds a persistent ``httpx.Client`` (its ``SyncTransport``)
# and is safe to share across threads — httpx.Client is documented thread-safe,
# and every GraphQL call goes through the retrying ``SyncTransport.post``. The
# expensive part of a *cold* call is the TLS handshake to the OpenNeuro API;
# constructing a fresh client per request (as call sites used to) throws that
# warm connection away every time. A long-lived server (the Atlas console API,
# a notebook session, a batch job) should build the client once and reuse it so
# HTTP keep-alive amortizes the handshake across every subsequent query. This
# mirrors ``qortex.client.remote.get_shared_gateway`` for byte-range reads.
_shared_client: "OpenNeuroClient | None" = None
_shared_client_lock = threading.Lock()


def get_shared_client() -> "OpenNeuroClient":
    """Return the process-wide token-less ``OpenNeuroClient``, creating it once.

    Use for anonymous metadata reads that dominate interactive workloads. When a
    per-user token is required, construct a dedicated ``OpenNeuroClient(token=...)``
    instead — the shared instance is deliberately token-less so one user's
    credentials never leak into another's cached connection.
    """
    global _shared_client
    if _shared_client is None:
        with _shared_client_lock:
            if _shared_client is None:
                _shared_client = OpenNeuroClient()
    return _shared_client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    return doi.removeprefix("doi:")


def _parse_snapshot_summary(
    *,
    dataset_id: str,
    snap: dict,
    desc: dict,
    snap_summary: dict,
) -> SnapshotSummary:
    sub_meta_raw = snap_summary.get("subjectMetadata") or []
    subject_demographics = [
        SubjectDemographic(
            participant_id=m.get("participantId") or "",
            age=m.get("age"),
            sex=m.get("sex"),
            group=m.get("group"),
        )
        for m in sub_meta_raw
        if m.get("participantId")
    ]

    funding = desc.get("Funding") or []
    refs = desc.get("ReferencesAndLinks") or []
    ethics = desc.get("EthicsApprovals") or []

    return SnapshotSummary(
        dataset_id=dataset_id,
        tag=snap.get("tag") or "",
        hexsha=snap.get("hexsha"),
        subjects=snap_summary.get("subjects") or [],
        sessions=snap_summary.get("sessions") or [],
        tasks=snap_summary.get("tasks") or [],
        modalities=snap_summary.get("modalities") or [],
        total_files=int(snap_summary.get("totalFiles") or 0),
        total_size_bytes=int(snap_summary.get("size") or snap.get("size") or 0),
        data_processed=bool(snap_summary.get("dataProcessed")),
        subject_demographics=subject_demographics,
        bids_version=desc.get("BIDSVersion"),
        license=desc.get("License"),
        funding=funding if isinstance(funding, list) else [funding] if funding else [],
        references=refs if isinstance(refs, list) else [refs] if refs else [],
        ethics_approvals=ethics if isinstance(ethics, list) else [ethics] if ethics else [],
        how_to_acknowledge=desc.get("HowToAcknowledge"),
    )


def _parse_search_node(node: dict) -> RichDatasetInfo | None:
    dataset_id = node.get("id")
    if not dataset_id:
        return None

    meta = node.get("metadata") or {}
    analytics = node.get("analytics") or {}
    snap = node.get("latestSnapshot") or {}
    desc = snap.get("description") or {}
    snap_summary = snap.get("summary") or {}

    engagement = DatasetEngagement(
        views=int(analytics.get("views") or 0),
        downloads=int(analytics.get("downloads") or 0),
        stars=len(node.get("stars") or []),
    )

    summary = _parse_snapshot_summary(
        dataset_id=dataset_id,
        snap=snap,
        desc=desc,
        snap_summary=snap_summary,
    ) if snap else None

    return RichDatasetInfo(
        id=dataset_id,
        name=meta.get("datasetName"),
        doi=_normalise_doi(desc.get("DatasetDOI")),
        license=desc.get("License"),
        authors=desc.get("Authors") or [],
        modalities=meta.get("modalities") or [],
        tasks=meta.get("tasksCompleted") or [],
        species=meta.get("species"),
        senior_author=meta.get("seniorAuthor"),
        study_domain=meta.get("studyDomain"),
        data_processed=bool(snap_summary.get("dataProcessed")),
        publish_date=node.get("publishDate"),
        engagement=engagement,
        latest_snapshot_tag=snap.get("tag"),
        latest_snapshot_hexsha=snap.get("hexsha"),
        latest_snapshot_summary=summary,
        raw_metadata=meta,
    )


def _count_values(values) -> dict[str, int]:
    """Count occurrences of each value in an iterable."""
    counts: dict[str, int] = {}
    for v in values:
        counts[str(v)] = counts.get(str(v), 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))
