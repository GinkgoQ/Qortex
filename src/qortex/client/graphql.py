"""Typed OpenNeuro GraphQL client.

All queries use the ``variables`` field of the GraphQL POST body — never
string interpolation — so special characters in IDs/tags are handled safely.

Public surface:
    OpenNeuroClient
        .get_dataset(dataset_id)          → DatasetRef
        .get_snapshots(dataset_id)        → list[SnapshotRef]
        .get_snapshot(dataset_id, tag)    → SnapshotRef
        .get_latest_snapshot(dataset_id)  → SnapshotRef
        .get_files(dataset_id, tag=None)  → (SnapshotRef, list[dict])
        .search_datasets(...)             → list[DatasetRef]
"""

from __future__ import annotations

import json
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

_Q_SNAPSHOTS = """
query GetSnapshots($datasetId: ID!) {
    dataset(id: $datasetId) {
        snapshots {
            id
            tag
            created
            size
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
                metadata {
                    datasetName
                    modalities
                    tasksCompleted
                }
                latestSnapshot {
                    tag
                    size
                    description {
                        License
                        DatasetDOI
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
    """Synchronous typed GraphQL client for the OpenNeuro API."""

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
    ) -> dict[str, Any]:
        """Execute a GQL query and return the ``data`` dict."""
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

    # ── Public API ────────────────────────────────────────────────────────

    def get_dataset(self, dataset_id: str) -> DatasetRef:
        """Return lightweight metadata for a dataset."""
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
            doi=_normalise_doi((node.get("latestSnapshot") or {}).get("description", {}).get("DatasetDOI")),
            license=(node.get("latestSnapshot") or {}).get("description", {}).get("License"),
            authors=((node.get("latestSnapshot") or {}).get("description") or {}).get("Authors") or [],
            modalities=meta.get("modalities") or [],
            tasks=meta.get("tasksCompleted") or [],
            raw_metadata=meta,
        )

    def get_snapshots(self, dataset_id: str) -> list[SnapshotRef]:
        """Return all available snapshot tags for a dataset."""
        data = self._query(_Q_SNAPSHOTS, {"datasetId": dataset_id},
                           timeout=self._cfg.metadata_timeout)
        node = data.get("dataset")
        if node is None:
            raise DatasetNotFoundError(dataset_id)
        snaps = node.get("snapshots") or []
        return [
            SnapshotRef(
                dataset_id=dataset_id,
                tag=s["tag"],
                id=s["id"],
                created=s.get("created"),
                size=s.get("size"),
            )
            for s in snaps
        ]

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

        Parameters
        ----------
        tag:
            Snapshot tag.  ``None`` → latest published snapshot.

        Returns
        -------
        (snapshot_ref, raw_file_dicts)
            Raw file dicts are passed to ManifestBuilder for BIDS parsing.
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
        )
        raw_files: list[dict] = snap_raw.get("files") or []
        return snap, raw_files

    def search_datasets(
        self,
        *,
        modality: str | None = None,
        task: str | None = None,
        limit: int = 100,
    ) -> list[DatasetRef]:
        """Return datasets matching the given filters.

        Note: the OpenNeuro GQL API has limited server-side filtering; this
        method fetches up to *limit* datasets and applies client-side filters.
        """
        refs: list[DatasetRef] = []
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
            except APIError:
                break

            edges = data.get("datasets", {}).get("edges") or []
            page_info = data.get("datasets", {}).get("pageInfo") or {}

            for edge in edges:
                node = edge.get("node") or {}
                meta = node.get("metadata") or {}
                mods = meta.get("modalities") or []
                tasks = meta.get("tasksCompleted") or []

                if modality and modality.lower() not in [m.lower() for m in mods]:
                    continue
                if task and task.lower() not in [t.lower() for t in tasks]:
                    continue

                snap = node.get("latestSnapshot") or {}
                desc = snap.get("description") or {}
                refs.append(DatasetRef(
                    id=node["id"],
                    name=meta.get("datasetName"),
                    doi=_normalise_doi(desc.get("DatasetDOI")),
                    license=desc.get("License"),
                    modalities=mods,
                    tasks=tasks,
                    raw_metadata=meta,
                ))

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise_doi(doi: str | None) -> str | None:
    if doi is None:
        return None
    return doi.removeprefix("doi:")
