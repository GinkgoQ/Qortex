"""DICOMweb / PACS source adapter (WADO-RS + QIDO-RS).

Queries a DICOMweb endpoint for study/series metadata, then streams
pixel data as assembled QortexVolume objects.  Authentication is read
from ``spec.extra["auth"]``.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    QortexVolume,
    SourceProfile,
    WarningItem,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)

_DICOM_CONTENT_TYPE = "application/octet-stream"


class DICOMWebAdapter(SourceAdapter):
    """Source adapter for a DICOMweb (WADO-RS / QIDO-RS) endpoint.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with:
        - ``path``: base URL of the DICOMweb server (e.g. ``https://host/wado/rs``)
        - ``query``: dict with ``study_uid`` and/or ``series_uid``
        - ``extra["auth"]``: ``{"type": "bearer", "token": "..."}`` or
          ``{"type": "basic", "username": "...", "password": "..."}``
    """

    def __init__(self, spec: SourceSpec, *, window_spec: WindowSpec | None = None) -> None:
        if not spec.path:
            raise ValueError("DICOMWebAdapter requires spec.path (DICOMweb base URL)")
        self._base_url = spec.path.rstrip("/")
        self._spec = spec
        self._query = spec.query or {}
        self._auth = (spec.extra or {}).get("auth", {})
        self._window_spec = window_spec

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        session = self._make_session()
        series_meta = self._fetch_series_metadata(session)

        modality = "unknown"
        n_instances = 0
        rows, cols = 0, 0

        if series_meta:
            first = series_meta[0]
            modality = first.get("00080060", {}).get("Value", ["unknown"])[0].lower()
            n_instances = len(series_meta)
            rows_tag = first.get("00280010", {}).get("Value", [0])
            cols_tag = first.get("00280011", {}).get("Value", [0])
            rows = rows_tag[0] if rows_tag else 0
            cols = cols_tag[0] if cols_tag else 0

        return SourceProfile(
            source_id=f"dicomweb:{self._base_url}",
            source_type="dicomweb",
            modality=modality,
            n_channels=1,
            sampling_rate_hz=None,
            spatial_shape=(n_instances, rows, cols),
            dtype="float32",
            axis_convention=AxisConvention.spatial_zyx,
            path=self._base_url,
            evidence={
                "modality": EvidenceStatus.confirmed if modality != "unknown" else EvidenceStatus.missing,
                "spatial_shape": EvidenceStatus.inferred,
            },
        )

    def read_batch(self) -> list[QortexData]:
        return list(self.stream())

    def stream(self) -> Iterator[QortexData]:
        session = self._make_session()
        study_uid = self._query.get("study_uid", "")
        series_uid = self._query.get("series_uid", "")

        if not study_uid:
            log.warning("DICOMWebAdapter: no study_uid in spec.query; yielding nothing")
            return

        # List instances in the series
        instances_url = (
            f"{self._base_url}/studies/{study_uid}"
            f"/series/{series_uid}/instances"
            if series_uid
            else f"{self._base_url}/studies/{study_uid}/instances"
        )
        resp = session.get(instances_url, headers={"Accept": "application/json"})
        resp.raise_for_status()
        instances = resp.json()

        slices = []
        for inst in instances:
            sop_uid = inst.get("00080018", {}).get("Value", [None])[0]
            if not sop_uid:
                continue
            pixel_url = (
                f"{self._base_url}/studies/{study_uid}"
                f"/series/{series_uid}/instances/{sop_uid}/frames/1"
            )
            pixel_resp = session.get(pixel_url, headers={"Accept": "application/octet-stream"})
            pixel_resp.raise_for_status()
            arr = np.frombuffer(pixel_resp.content, dtype=np.uint16).astype(np.float32)
            rows = inst.get("00280010", {}).get("Value", [arr.shape[0]])[0]
            cols = inst.get("00280011", {}).get("Value", [arr.shape[0]])[0]
            if arr.size == rows * cols:
                arr = arr.reshape(rows, cols)
            slices.append(arr)

        if slices:
            volume = np.stack(slices, axis=0)
            yield QortexVolume(
                data=volume,
                shape=volume.shape,
                axes=["z", "y", "x"],
                dtype="float32",
                units="HU",
                affine=None,
                voxel_sizes_mm=None,
                coordinate_frame="patient_lps",
                source_provenance={
                    "source_type": "dicomweb",
                    "base_url": self._base_url,
                    "study_uid": study_uid,
                    "series_uid": series_uid,
                },
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_session(self):
        try:
            import requests
        except ImportError:
            raise ImportError(
                "DICOMweb support requires requests. "
                "Install with: pip install requests"
            )
        session = requests.Session()
        auth_type = self._auth.get("type", "")
        if auth_type == "bearer":
            session.headers["Authorization"] = f"Bearer {self._auth.get('token', '')}"
        elif auth_type == "basic":
            from requests.auth import HTTPBasicAuth
            session.auth = HTTPBasicAuth(
                self._auth.get("username", ""), self._auth.get("password", "")
            )
        return session

    def _fetch_series_metadata(self, session) -> list[dict]:
        study_uid = self._query.get("study_uid", "")
        series_uid = self._query.get("series_uid", "")
        if not study_uid:
            return []
        url = (
            f"{self._base_url}/studies/{study_uid}/series/{series_uid}/instances"
            if series_uid
            else f"{self._base_url}/studies/{study_uid}/series"
        )
        try:
            resp = session.get(url, headers={"Accept": "application/json"}, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.warning("DICOMWeb metadata fetch failed: %s", exc)
            return []
