"""FastAPI service exposing the Qortex catalog and download APIs."""

from __future__ import annotations

from typing import Any, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel as _PydanticModel
except ImportError:
    raise ImportError(
        "Qortex console requires FastAPI: pip install qortex[dashboard]"
    )


app = FastAPI(
    title="Qortex API",
    description="Qortex by GinkgoQ — OpenNeuro dataset catalog and download service.",
    version="0.1.0",
)


# ── Models ────────────────────────────────────────────────────────────────────

class DownloadRequest(_PydanticModel):
    dataset_id: str
    snapshot: Optional[str] = None
    subjects: Optional[list[str]] = None
    tasks: Optional[list[str]] = None
    modalities: Optional[list[str]] = None
    include_derivatives: bool = False   # mirrors SelectionSpec.include_derivatives
    output_dir: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/catalog/search")
async def catalog_search(
    q: Optional[str] = Query(None, description="Free-text search"),
    modality: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    author: Optional[str] = Query(None),
    license: Optional[str] = Query(None),
    min_subjects: Optional[int] = Query(None),
    max_size_gb: Optional[float] = Query(None),
    has_events: Optional[bool] = Query(None),
    has_derivatives: Optional[bool] = Query(None),
    limit: int = Query(20, le=200),
) -> list[dict[str, Any]]:
    from qortex.catalog.search import search
    return search(
        query=q,
        modality=modality,
        task=task,
        author=author,
        license=license,
        min_subjects=min_subjects,
        max_size_gb=max_size_gb,
        has_events=has_events,
        has_derivatives=has_derivatives,
        limit=limit,
    )


@app.get("/catalog/{dataset_id}")
async def catalog_get(dataset_id: str) -> dict[str, Any]:
    from qortex.catalog.index import CatalogIndex
    from qortex.core.config import get_config
    cfg = get_config()
    idx = CatalogIndex(cfg.cache_dir / "catalog" / "catalog.duckdb")
    result = idx.get(dataset_id)
    idx.close()
    if result is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not in local catalog.")
    return result


@app.post("/catalog/refresh")
async def catalog_refresh_endpoint(max_pages: int = 40) -> dict[str, Any]:
    from qortex.catalog.refresh import refresh
    n = refresh(max_pages=max_pages, progress=False)
    return {"datasets_indexed": n}


@app.post("/catalog/refresh/{dataset_id}")
async def catalog_refresh_dataset_endpoint(
    dataset_id: str,
    deep: bool = Query(True),
) -> dict[str, Any]:
    from qortex.catalog.refresh import refresh_dataset

    return refresh_dataset(dataset_id, include_file_summary=deep)


@app.get("/dataset/{dataset_id}/manifest")
async def get_manifest(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.client.graphql import OpenNeuroClient
    from qortex.manifest.builder import ManifestBuilder

    client = OpenNeuroClient()
    builder = ManifestBuilder()
    if snapshot:
        snap_ref = client.get_snapshot(dataset_id, snapshot)
    else:
        snap_ref = client.get_latest_snapshot(dataset_id)

    snap_ref, raw_files = client.get_files(dataset_id, snap_ref.tag)
    manifest = builder.build(dataset_id, snap_ref, raw_files)

    s = manifest.summary
    return {
        "dataset_id": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "n_files": s.file_count,
        "n_subjects": s.n_subjects,
        "total_size": s.total_size,
        "modalities": s.modalities,
        "has_events": s.has_events,
        "has_derivatives": s.has_derivatives,
    }


@app.get("/dataset/{dataset_id}/eda")
async def get_eda(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.client.graphql import OpenNeuroClient
    from qortex.eda.report import EDAEngine
    from qortex.manifest.builder import ManifestBuilder

    client = OpenNeuroClient()
    builder = ManifestBuilder()
    if snapshot:
        snap_ref = client.get_snapshot(dataset_id, snapshot)
    else:
        snap_ref = client.get_latest_snapshot(dataset_id)

    snap_ref, raw_files = client.get_files(dataset_id, snap_ref.tag)
    manifest = builder.build(dataset_id, snap_ref, raw_files)

    engine = EDAEngine(manifest)
    report = engine.run()
    q = report.quality

    return {
        "dataset_id": dataset_id,
        "snapshot": manifest.snapshot,
        "bids_score": q.bids_score,
        "ml_readiness_score": q.ml_readiness_score,
        "loadability_score": q.loadability_score,
        "issues": q.issues,
        "risks": q.risks,
        "modality_summaries": {
            m: {"n_files": ms.n_files, "n_subjects": ms.n_subjects, "total_size_mb": ms.total_size / 1e6}
            for m, ms in report.modality_summaries.items()
        },
    }
