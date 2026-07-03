"""Qortex Atlas console API.

A FastAPI service that exposes real Qortex library calls over HTTP for the
Qortex Atlas web UI. Every route below calls genuine, unmodified Qortex
functions against the live OpenNeuro GraphQL/CDN endpoints (or the local
DuckDB catalog cache) — nothing here fabricates data. Where Qortex can
answer a question without downloading anything (remote header reads, byte
-range signal epochs, events-TSV scans), this API exposes that directly so
the UI can show real evidence before a single byte of a dataset is
downloaded.

Run with:  qortex dashboard   (installs via ``pip install qortex[dashboard]``)
or directly:  uvicorn qortex.console.api:app --port 8420
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Optional

try:
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel as _PydanticModel
except ImportError:
    raise ImportError(
        "Qortex console requires FastAPI: pip install qortex[dashboard]"
    )

from qortex.console import atlas_jobs, atlas_models, atlas_timing
from qortex.console.atlas_cache import TTLCache
from qortex.console.atlas_evidence import build_evidence, mlreadiness_dims
from qortex.console.atlas_serialize import to_jsonable
from qortex.core.exceptions import (
    APIError,
    AuthError,
    DatasetNotFoundError,
    NetworkError,
    QortexError,
    RateLimitError,
    SnapshotNotFoundError,
)

app = FastAPI(
    title="Qortex Atlas API",
    description="Qortex by GinkgoQ — real OpenNeuro/BIDS dataset intelligence for Qortex Atlas.",
    version="0.2.0",
)

# The Atlas frontend is a static SPA served from a different origin/port —
# allow it in during local development. Tighten this to a specific origin
# list before exposing the API beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error mapping ─────────────────────────────────────────────────────────────

async def call(fn, *args, **kwargs):
    """Run a blocking Qortex call off the event loop and map its exceptions
    onto the right HTTP status. Every Qortex call in this module (dataset
    downloads, GraphQL round-trips, remote byte-range reads) is synchronous
    and network-bound — never call these inline in an ``async def`` route.
    """
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except (DatasetNotFoundError, SnapshotNotFoundError) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except (APIError, NetworkError) as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except QortexError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001 - last resort: never leak a bare 500 traceback to the UI
        raise HTTPException(
            status_code=502,
            detail=f"{type(e).__name__}: {e} (this specific file/dataset may not support this operation)",
        ) from e


# ── Shared helpers ────────────────────────────────────────────────────────────

_MANIFEST_CACHE_TTL_S = 120.0  # a workspace visit fires several tab loads in quick
                               # succession, each needing the same file tree — cache
                               # briefly rather than re-fetching OpenNeuro every time
# Coalescing TTL cache: when several tabs mount at once on a cold dataset, the
# first request fetches while the rest wait and share its result, instead of
# every tab firing its own OpenNeuro round-trip (see atlas_cache.TTLCache).
_MANIFEST_CACHE = TTLCache(ttl=_MANIFEST_CACHE_TTL_S, maxsize=128)


def _manifest_for(dataset_id: str, snapshot: str | None = None, token: str | None = None):
    def _build():
        from qortex.client.graphql import OpenNeuroClient, get_shared_client
        from qortex.manifest.builder import ManifestBuilder

        # Reuse one warm, keep-alive HTTP connection for anonymous reads (the
        # interactive default) instead of paying a fresh TLS handshake to
        # OpenNeuro on every dataset. A token-scoped call gets its own client so
        # credentials never ride a shared connection.
        client = get_shared_client() if token is None else OpenNeuroClient(token=token)
        # One GraphQL round-trip, not two: get_files(tag=None) resolves the
        # latest snapshot *and* returns its file tree in a single query. Passing
        # an explicit tag also fetches files directly — the previous
        # get_snapshot()/get_latest_snapshot() pre-call was a redundant hop.
        snap_ref, raw_files = client.get_files(dataset_id, snapshot)
        return ManifestBuilder().build(dataset_id, snap_ref, raw_files)

    return _MANIFEST_CACHE.get_or_compute((dataset_id, snapshot), _build)


def _find_file(manifest, *, subject: str | None, session: str | None = None,
                task: str | None = None, run: str | None = None,
                extensions: tuple[str, ...] = (), suffix: str | None = None):
    """Locate the first FileRecord matching BIDS entities + extension, from an
    already-fetched Manifest (no extra network call)."""
    for f in manifest.files:
        if f.is_dir:
            continue
        if subject and f.entities.subject != subject.replace("sub-", ""):
            continue
        if session and f.entities.session != session.replace("ses-", ""):
            continue
        if task and f.entities.task != task:
            continue
        if run and f.entities.run != run:
            continue
        if extensions and f.extension not in extensions:
            continue
        if suffix and f.suffix != suffix:
            continue
        return f
    return None


# ── Health / store status ────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/store/status")
async def store_status() -> dict[str, Any]:
    """Real counts from the local DuckDB/SQLite catalog cache — the only
    'local store' in this system. Opening a dataset workspace never depends
    on this cache; it is purely a fast pre-filter for Explore search."""
    def _status() -> dict[str, Any]:
        from qortex.catalog.index import CatalogIndex
        from qortex.core.config import get_config

        cfg = get_config()
        db_path = cfg.cache_dir / "catalog" / "catalog.duckdb"
        if not db_path.exists():
            return {"n_datasets": 0, "n_profiled": 0, "db_path": str(db_path), "exists": False}
        idx = CatalogIndex(db_path)
        try:
            all_rows = idx.search(limit=100_000)
            n_profiled = sum(1 for r in all_rows if (r.get("n_files") or 0) > 0)
            return {
                "n_datasets": len(all_rows),
                "n_profiled": n_profiled,
                "db_path": str(db_path),
                "exists": True,
            }
        finally:
            idx.close()

    return await call(_status)


# ── Catalog search (local cache) + facets ────────────────────────────────────

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

    return await call(
        search, query=q, modality=modality, task=task, author=author, license=license,
        min_subjects=min_subjects, max_size_gb=max_size_gb, has_events=has_events,
        has_derivatives=has_derivatives, limit=limit,
    )


@app.get("/catalog/facets")
async def catalog_facets(limit: int = Query(50, le=200)) -> dict[str, Any]:
    from qortex.catalog.search import facets as facets_fn

    return await call(facets_fn, limit=limit)


@app.get("/catalog/{dataset_id}")
async def catalog_get(dataset_id: str, auto_refresh: bool = Query(True)) -> dict[str, Any]:
    from qortex.catalog.index import CatalogIndex
    from qortex.catalog.refresh import refresh_dataset
    from qortex.core.config import get_config

    def _get() -> dict[str, Any] | None:
        cfg = get_config()
        idx = CatalogIndex(cfg.cache_dir / "catalog" / "catalog.duckdb")
        try:
            return idx.get(dataset_id)
        finally:
            idx.close()

    result = await call(_get)
    if result is None:
        if not auto_refresh:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not in local catalog.")
        # Opening a dataset the user found via live search should still work —
        # transparently profile + cache it rather than 404ing.
        result = await call(refresh_dataset, dataset_id, include_file_summary=True)
    return result


@app.post("/catalog/refresh")
async def catalog_refresh_endpoint(max_pages: int = Query(10, le=200)) -> dict[str, Any]:
    from qortex.catalog.refresh import refresh

    n = await call(refresh, max_pages=max_pages, progress=False)
    return {"datasets_indexed": n}


@app.post("/catalog/refresh/{dataset_id}")
async def catalog_refresh_dataset_endpoint(dataset_id: str, deep: bool = Query(True)) -> dict[str, Any]:
    from qortex.catalog.refresh import refresh_dataset

    return await call(refresh_dataset, dataset_id, include_file_summary=deep)


# ── Hybrid search: local catalog ∪ live OpenNeuro, ranked, provenance-tagged ─
# This is the "advanced search" surface — local cache results answer
# instantly; live OpenNeuro results are merged in and clearly marked so the
# UI never conflates a fast local guess with a fresh remote fact.

@app.get("/search/hybrid")
async def search_hybrid(
    q: Optional[str] = Query(None),
    modality: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    min_subjects: Optional[int] = Query(None),
    max_size_gb: Optional[float] = Query(None),
    license: Optional[str] = Query(None),
    has_events: Optional[bool] = Query(None),
    include_live: bool = Query(True, description="Also query the live OpenNeuro API and merge results"),
    limit: int = Query(30, le=200),
) -> dict[str, Any]:
    from qortex.catalog.search import search as local_search

    def _run() -> dict[str, Any]:
        local = local_search(
            query=q, modality=modality, task=task, min_subjects=min_subjects,
            max_size_gb=max_size_gb, license=license, has_events=has_events, limit=limit,
        )
        for row in local:
            row["_source"] = "local"
        seen = {r["dataset_id"] for r in local}
        live: list[dict[str, Any]] = []
        if include_live:
            from qortex.catalog.search import live_search
            # live_search's `limit` bounds how many raw OpenNeuro nodes are
            # *scanned* before client-side modality/task filtering, not how
            # many matches come back — a filtered search needs a much larger
            # scan pool than the number of results actually wanted.
            scan_limit = max(limit * 20, 300) if (modality or task) else limit
            try:
                # sync_local=False: caching live results into the local
                # DuckDB catalog is a deliberate, explicit action (Settings
                # → "Refresh catalog", or POST /catalog/refresh) — CatalogIndex
                # currently costs ~100-400ms per row across its 5 tables, so
                # syncing on every interactive search made results appear to
                # hang for 10s+. The results below are shown either way;
                # syncing them is a side effect this endpoint doesn't need.
                remote = live_search(modality=modality, task=task, query=q, limit=scan_limit, sync_local=False)
            except QortexError:
                remote = []
            remote = remote[:limit]
            for row in remote:
                rid = row.get("dataset_id") or row.get("id")
                if rid in seen:
                    continue
                row["_source"] = "live"
                row["dataset_id"] = rid
                live.append(row)
                seen.add(rid)
        return {"local": local, "live": live, "total": len(local) + len(live)}

    return await call(_run)


# ── Search engine v2: compiler -> {structured, BM25, semantic} -> RRF fusion
# -> evidence-partitioned filtering -> negative-space summary. See
# qortex.search (and qortex-atlas-search-engine.md) for the full design. This
# is a genuine multi-method engine, distinct from /search/hybrid above (which
# is only a local-cache ∪ live-OpenNeuro *source* merge, not method fusion).

_SEARCH_ENGINE: Any = None


def _get_search_engine():
    global _SEARCH_ENGINE
    if _SEARCH_ENGINE is None:
        from qortex.search.engine import SearchEngine

        _SEARCH_ENGINE = SearchEngine()
    return _SEARCH_ENGINE


@app.get("/search/engine")
async def search_engine(
    q: Optional[str] = Query(None, description="Free text; parsed by the deterministic query compiler"),
    modality: Optional[str] = Query(None),
    min_subjects: Optional[int] = Query(None),
    max_size_gb: Optional[float] = Query(None),
    license_open: Optional[bool] = Query(None),
    has_events: Optional[bool] = Query(None),
    include_unknown_evidence: bool = Query(True, description="Keep unresolved-evidence datasets in results (never silently drop them)"),
    deep: bool = Query(False, description="Also run the DatasetFitness structural re-rank over the shortlist (may call the live OpenNeuro API)"),
    limit: int = Query(20, le=200),
) -> dict[str, Any]:
    engine = _get_search_engine()

    def _run() -> dict[str, Any]:
        response = engine.search(
            q,
            modality=modality,
            min_subjects=min_subjects,
            max_size_gb=max_size_gb,
            license_open=license_open,
            has_events=has_events,
            include_unknown_evidence=include_unknown_evidence,
            deep=deep,
            limit=limit,
        )
        return {
            "results": to_jsonable(response.results),
            "plan": to_jsonable(response.plan),
            "negative_space": to_jsonable(response.negative_space),
            "timings_ms": response.timings_ms,
        }

    return await call(_run)


@app.post("/search/engine/refresh")
async def search_engine_refresh() -> dict[str, Any]:
    """Rebuild the lexical (BM25) and semantic (LSA) indexes from the current
    catalog state. Call after ``/catalog/refresh``; cheap to call often —
    both indexes hash their input and skip re-embedding an unchanged corpus."""
    engine = _get_search_engine()
    return await call(engine.refresh_indexes)


# ── Goal-based ranking (the Explore "Goal Builder") ─────────────────────────

class GoalBody(_PydanticModel):
    modality: Optional[str] = None
    task_keywords: list[str] = []
    min_subjects: Optional[int] = None
    min_trials_per_class: Optional[int] = None
    min_n_classes: Optional[int] = None
    max_imbalance_ratio: Optional[float] = None
    min_recording_hours: Optional[float] = None
    max_size_gb: Optional[float] = None
    license_must_be_open: bool = False
    species: Optional[str] = None
    dataset_ids: Optional[list[str]] = None  # if given, rank only these; else search+rank
    tier3_events: bool = False
    limit: int = 10
    # DatasetSelector.find()'s Tier 2 makes one live OpenNeuro API call per
    # catalog candidate — the default catalog_limit=200 upstream is fine for
    # a batch script but far too slow (minutes) for an interactive UI, so
    # Atlas caps it much lower by default.
    catalog_limit: int = 20


@app.post("/goal/find")
async def goal_find(body: GoalBody = Body(...)) -> list[dict[str, Any]]:
    from qortex.inspect.selector import DatasetSelector, ResearchGoal

    goal = ResearchGoal(
        modality=body.modality, task_keywords=body.task_keywords or [],
        min_subjects=body.min_subjects, min_trials_per_class=body.min_trials_per_class,
        min_n_classes=body.min_n_classes, max_imbalance_ratio=body.max_imbalance_ratio,
        min_recording_hours=body.min_recording_hours, max_size_gb=body.max_size_gb,
        license_must_be_open=body.license_must_be_open, species=body.species,
    )
    selector = DatasetSelector()

    def _run():
        if body.dataset_ids:
            return selector.rank(body.dataset_ids, goal, tier3_events=body.tier3_events)
        return selector.find(goal, limit=body.limit, catalog_limit=body.catalog_limit, tier3_events=body.tier3_events)

    fitness_list = await call(_run)
    return [to_jsonable(f) for f in fitness_list]


# ── Dataset workspace ─────────────────────────────────────────────────────────

_PROFILE_CACHE = TTLCache(ttl=_MANIFEST_CACHE_TTL_S, maxsize=128)


@app.get("/dataset/{dataset_id}/profile")
async def dataset_profile(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    level: str = Query("summary", pattern="^(summary|manifest|deep)$"),
) -> dict[str, Any]:
    """DatasetInspector.inspect() does its own OpenNeuro file-tree fetch,
    independent of the /readiness /doctor /plan manifest cache above — and
    the Atlas workspace re-requests the profile on every single tab
    navigation (each tab is a full SPA route remount). Without a cache here,
    clicking through Overview → Evidence → Readiness on one large dataset
    triggers three redundant multi-second OpenNeuro fetches for identical data."""
    key = (dataset_id, snapshot, level)
    hit = _PROFILE_CACHE.peek(key)
    if hit is not None:
        # Warm hit — answer without even hopping to the threadpool.
        return hit

    from qortex.inspect.dataset import DatasetInspector

    def _compute() -> dict[str, Any]:
        inspector = DatasetInspector()
        profile = inspector.inspect(dataset_id, tag=snapshot, level=level)
        out = to_jsonable(profile.as_dict())
        if level == "deep":
            landscape = getattr(profile, "_label_landscape", None)
            budget = getattr(profile, "_signal_budget", None)
            if landscape is not None:
                out["label_landscape"] = to_jsonable(landscape)
            if budget is not None:
                out["signal_budget"] = to_jsonable(budget)
        out["readiness_dims"] = mlreadiness_dims(out)
        return out

    # Coalesce concurrent misses for the same (dataset, snapshot, level) so a
    # burst of tab loads triggers exactly one inspect(), not one per tab.
    return await call(_PROFILE_CACHE.get_or_compute, key, _compute)


@app.get("/dataset/{dataset_id}/manifest")
async def dataset_manifest(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    modality: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    limit: int = Query(500, le=5000),
    offset: int = Query(0),
) -> dict[str, Any]:
    def _run() -> dict[str, Any]:
        manifest = _manifest_for(dataset_id, snapshot)
        files = manifest.filter(
            subjects=[subject] if subject else None,
            modalities=[modality] if modality else None,
            tasks=[task] if task else None,
        )
        page = files[offset: offset + limit]
        return {
            "dataset_id": manifest.dataset_id,
            "snapshot": manifest.snapshot,
            "summary": to_jsonable(manifest.summary),
            "total_matching": len(files),
            "offset": offset,
            "limit": limit,
            "files": to_jsonable(page),
        }

    return await call(_run)


@app.get("/dataset/{dataset_id}/readiness")
async def dataset_readiness(dataset_id: str, snapshot: Optional[str] = Query(None)) -> dict[str, Any]:
    from qortex.check.readiness import compute_readiness
    from qortex.decision import can_train as can_train_fn

    def _run() -> dict[str, Any]:
        with atlas_timing.timed("readiness", dataset_id):
            manifest = _manifest_for(dataset_id, snapshot)
            readiness = compute_readiness(manifest)
            ct = can_train_fn(manifest)
            evidence = build_evidence(
                dataset_id=dataset_id,
                manifest_summary=to_jsonable(manifest.summary),
                readiness=readiness,
                can_train=ct,
            )
        return {
            "dataset_id": dataset_id,
            "snapshot": manifest.snapshot,
            "readiness": to_jsonable(readiness),
            "can_train": to_jsonable(ct),
            "evidence": evidence.as_dict(),
        }

    return await call(_run)


@app.get("/timing/estimate")
async def timing_estimate(operation: str = Query(...), key: str = Query("*")) -> dict[str, Any]:
    """A real ETA, built from this machine's own observed history for
    *operation* (optionally scoped to one dataset via *key*) — never a
    fabricated percentage or a network-speed guess. Returns
    ``has_estimate: false`` until at least one real run has been timed."""
    return atlas_timing.estimate(operation, key)


@app.get("/dataset/{dataset_id}/doctor")
async def dataset_doctor(dataset_id: str, snapshot: Optional[str] = Query(None)) -> dict[str, Any]:
    from qortex.decision import doctor as doctor_fn

    def _run():
        manifest = _manifest_for(dataset_id, snapshot)
        return doctor_fn(manifest)

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/label-landscape")
async def dataset_label_landscape(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    max_events_files: int = Query(60, le=500),
) -> dict[str, Any]:
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        return ds.label_landscape(max_events_files=max_events_files)

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/signal-budget")
async def dataset_signal_budget(dataset_id: str, snapshot: Optional[str] = Query(None)) -> dict[str, Any]:
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        return ds.signal_budget()

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/participants")
async def dataset_participants(dataset_id: str, snapshot: Optional[str] = Query(None)) -> dict[str, Any]:
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        df = ds.participants()
        return {"columns": df.columns, "rows": df.to_dicts()}

    return await call(_run)


@app.get("/dataset/{dataset_id}/events")
async def dataset_events(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    subject: Optional[str] = Query(None),
    session: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    run: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        df = ds.events(subject=subject, session=session, task=task, run=run)
        return {"columns": df.columns, "rows": df.to_dicts()}

    return await call(_run)


@app.get("/dataset/{dataset_id}/preview")
async def dataset_preview(
    dataset_id: str,
    path: str = Query(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Table/JSON preview (TSV/CSV/JSON) — see the sibling /nifti-info,
    /nifti-slice.png, and /eeg-preview routes for binary formats, which
    ``FilePreview`` itself does not parse."""
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        return ds.preview(path)

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/sidecar")
async def dataset_sidecar(
    dataset_id: str,
    path: str = Query(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        return ds.sidecar(path)

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/nifti-info")
async def dataset_nifti_info(
    dataset_id: str,
    path: str = Query(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Real NIfTI header, read via a single HTTP range request (~352 bytes) —
    zero bytes of volume data transferred."""
    from qortex import Dataset

    def _run():
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        return ds.nifti_info(path)

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/nifti-slice.png")
async def dataset_nifti_slice_png(
    dataset_id: str,
    subject: str = Query(...),
    modality: str = Query("T1w"),
    session: Optional[str] = Query(None),
    run: Optional[str] = Query(None),
    axis: int = Query(2, ge=0, le=2),
    slice_index: Optional[int] = Query(None),
    time_index: int = Query(0),
    snapshot: Optional[str] = Query(None),
):
    """A real anatomical slice, decoded from bytes fetched via HTTP range
    requests against the OpenNeuro CDN — the full NIfTI volume is never
    downloaded. Percentile-windowed and PNG-encoded server-side."""
    import numpy as np
    from PIL import Image

    from qortex import Dataset

    def _run() -> bytes:
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        arr = ds.stream_slice(
            subject=subject, modality=modality, session=session, run=run,
            axis=axis, slice_index=slice_index, time_index=time_index,
        )
        arr = np.asarray(arr, dtype=np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255
        img = Image.fromarray(np.rot90(arr.astype(np.uint8)))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    png_bytes = await call(_run)
    return Response(content=png_bytes, media_type="image/png")


# International 10-20/10-10 EEG electrode nomenclature — the real, standard
# naming convention used across essentially every clinical/research EEG
# system regardless of manufacturer. Some consumer headsets (this was found
# on a real Emotiv EPOC export) multiplex non-neural channels — packet
# timestamps, a sample counter, an interpolation flag, per-electrode contact-
# quality ("CQ_*") channels, battery level, marker/event channels — into the
# same EDF as the real electrodes, all mislabeled with the same "uV" unit
# and "electrode" transducer string as the real ones, so unit/transducer
# metadata can't distinguish them. The channel *label* is the only reliable
# signal, and matching it against the real 10-20/10-10 standard generalizes
# to any EEG BIDS dataset, not just this one.
_EEG_ELECTRODE_RE = re.compile(
    r"^(Fp|AF|FT|FC|TP|CP|PO|F|C|T|P|O|A|M|I)(z|\d{1,2})$", re.IGNORECASE
)


@app.get("/dataset/{dataset_id}/eeg-preview")
async def dataset_eeg_preview(
    dataset_id: str,
    subject: str = Query(...),
    session: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    run: Optional[str] = Query(None),
    tmin: float = Query(0.0, ge=0),
    tmax: float = Query(4.0, gt=0),
    max_channels: int = Query(8, le=64),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Real, physical-unit EEG/MEG samples for a short window, decoded from
    an EDF/BDF file via HTTP byte-range requests — only the bytes for this
    exact time window are ever transferred. Formats without a remote-signal
    reader (.set/.fif/.vhdr) honestly report as unsupported rather than
    fabricating a waveform."""
    from qortex.stream import EDFStreamer

    def _run() -> dict[str, Any]:
        manifest = _manifest_for(dataset_id, snapshot)
        f = _find_file(manifest, subject=subject, session=session, task=task, run=run,
                        extensions=(".edf", ".bdf"))
        if f is None:
            return {"supported": False,
                    "reason": "No .edf/.bdf recording found for these entities. "
                              "Remote signal streaming currently supports EDF/BDF only; "
                              ".set/.fif/.vhdr require a full download to preview."}
        if not f.urls:
            return {"supported": False, "reason": "Matched file has no download URL in this snapshot."}
        streamer = EDFStreamer(f.urls[0])
        hdr = streamer.header()
        max_t = min(tmax, hdr.duration_s or tmax)
        all_channels = streamer.channel_info()
        electrodes = [c for c in all_channels if _EEG_ELECTRODE_RE.match(c["label"])]
        # Fall back to the unfiltered list if nothing matched the 10-20/10-10
        # pattern — some legitimate systems use other real montages (e.g.
        # high-density nets with numeric-only labels) and this must never
        # silently return zero channels for those.
        selected = electrodes if electrodes else all_channels
        channels = [c["label"] for c in selected[:max_channels]]
        epoch = streamer.get_epoch(tmin, max_t, channels=channels)
        return {
            "supported": True,
            "path": f.path,
            "sfreq": hdr.sampling_rates[0] if hdr.sampling_rates else None,
            "duration_s": hdr.duration_s,
            "channels": channels,
            "n_channels_total": len(all_channels),
            "n_channels_excluded": len(all_channels) - len(electrodes) if electrodes else 0,
            "tmin": tmin, "tmax": max_t,
            "series": [row.tolist() for row in epoch],
        }

    return await call(_run)


class PlanBody(_PydanticModel):
    preset: str = "label-check"  # validate | label-check | smoke-train | full-train
    modality: Optional[str] = None
    subjects: Optional[list[str]] = None


_PRESET_SPECS: dict[str, dict[str, Any]] = {
    "validate": {"metadata_only": True},
    "label-check": {"metadata_only": True, "event_complete": True},
    "smoke-train": {"loadable_only": True},
    "full-train": {},
}


@app.post("/dataset/{dataset_id}/plan")
async def dataset_plan(
    dataset_id: str,
    body: PlanBody = Body(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """A real, explainable ``DownloadPlan`` — the exact file list, byte
    estimate, and per-file ``SelectionReason`` Qortex would use for a real
    download, computed via ``Dataset.download(dry_run=True)`` (no bytes
    transferred)."""
    from qortex import Dataset

    spec = dict(_PRESET_SPECS.get(body.preset, {}))
    if body.modality:
        spec["modalities"] = [body.modality]
    if body.preset == "smoke-train" and body.subjects is None:
        spec["subjects"] = None  # planner will still cap to first loadable recording via loadable_only

    def _run():
        with atlas_timing.timed("plan", dataset_id):
            ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
            result = ds.download(dry_run=True, subjects=body.subjects, **spec)
        return result.plan

    plan = await call(_run)
    plan_json = to_jsonable(plan)
    plan_json["command"] = (
        f"qortex download {dataset_id} --goal {body.preset}"
        + (f" --modality {body.modality}" if body.modality else "")
    )
    return plan_json


class DownloadBody(_PydanticModel):
    preset: str = "label-check"
    modality: Optional[str] = None
    subjects: Optional[list[str]] = None
    output_dir: Optional[str] = None


@app.post("/dataset/{dataset_id}/download")
async def dataset_download(
    dataset_id: str,
    body: DownloadBody = Body(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Submit a REAL download as a background job. Returns immediately with a
    job id — poll GET /jobs/{id}. Presets are capped to metadata/small-subject
    selections by design (validate/label-check/smoke-train); full-train will
    genuinely transfer the whole dataset, so the UI should confirm that with
    the user before calling this preset."""
    from qortex import Dataset

    spec = dict(_PRESET_SPECS.get(body.preset, {}))
    if body.modality:
        spec["modalities"] = [body.modality]
    output_dir = Path(body.output_dir) if body.output_dir else None

    def _run(on_progress=None):
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        result = ds.download(subjects=body.subjects, output_dir=output_dir, on_progress=on_progress, **spec)
        return to_jsonable(result)

    job = atlas_jobs.submit(f"Download {dataset_id} ({body.preset})", _run, report_progress=True)
    return atlas_jobs.to_public(job)


@app.get("/dataset/{dataset_id}/content-status")
async def dataset_content_status(
    dataset_id: str,
    local_path: str = Query(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.decision import content_status as content_status_fn

    def _run():
        manifest = _manifest_for(dataset_id, snapshot)
        return content_status_fn(Path(local_path), manifest=manifest)

    return to_jsonable(await call(_run))


@app.get("/dataset/{dataset_id}/compatibility")
async def dataset_compatibility(
    dataset_id: str,
    model_id: Optional[str] = Query(None),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Build a real ``SourceProfile`` from remotely-gathered signal-budget
    evidence (no download) and run it through the unmodified
    ``CompatibilityEngine`` against one or all catalog model contracts."""
    from qortex import Dataset
    from qortex.neuroai.compatibility import CompatibilityEngine
    from qortex.neuroai.contracts import EvidenceStatus, SourceProfile

    def _run() -> dict[str, Any]:
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        manifest = ds.manifest()
        budget = ds.signal_budget()
        # Prefer an actual neural-signal modality over incidental BIDS
        # datatypes (e.g. "behavior", "phenotype") that a raw modalities list
        # may list first with no meaningful ordering.
        signal_priority = ["eeg", "meg", "ieeg", "bold", "dwi", "t1w", "t2w"]
        mods = manifest.summary.modalities
        modality = next((m for m in signal_priority if m in mods), next(iter(mods), None))
        mb = budget.modality_budgets.get(modality) if modality else None
        source = SourceProfile(
            source_id=dataset_id,
            source_type="bids",
            modality=modality,
            n_channels=getattr(mb, "n_channels_mode", None) if mb else None,
            sampling_rate_hz=getattr(mb, "sampling_rate_mode", None) if mb else None,
            n_subjects=manifest.summary.n_subjects,
            available_suffixes=manifest.summary.suffixes,
            evidence_status=EvidenceStatus.inferred if mb else EvidenceStatus.unknown,
        )
        engine = CompatibilityEngine()
        model_ids = [model_id] if model_id else list(atlas_models.MODEL_CATALOG.keys())
        reports = []
        for mid in model_ids:
            model = atlas_models.MODEL_CATALOG.get(mid)
            if model is None:
                continue
            reports.append(engine.check(source, model))
        return {"source": to_jsonable(source), "reports": to_jsonable(reports)}

    return await call(_run)


@app.get("/dataset/{dataset_id}/compare/{other_id}")
async def dataset_compare(
    dataset_id: str, other_id: str,
    snapshot_a: Optional[str] = Query(None), snapshot_b: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.inspect.dataset import DatasetInspector

    def _run():
        return DatasetInspector().compare(dataset_id, other_id, tag_a=snapshot_a, tag_b=snapshot_b)

    return to_jsonable(await call(_run))


# ── Cohort composition ────────────────────────────────────────────────────────

class CohortBody(_PydanticModel):
    dataset_ids: list[str]
    require_modality: Optional[str] = None
    min_subjects_per_dataset: int = 1
    run_harmonization: bool = True


@app.post("/cohort/compose")
async def cohort_compose(body: CohortBody = Body(...)) -> dict[str, Any]:
    from qortex.cohort import CohortBuilder

    def _run():
        builder = CohortBuilder()
        for did in body.dataset_ids:
            builder.add_dataset(did)
        if body.require_modality:
            builder.require_modality(body.require_modality)
        builder.min_subjects_per_dataset(body.min_subjects_per_dataset)
        if body.run_harmonization:
            builder.with_harmonization_check()
        return builder.build().to_dict()

    return to_jsonable(await call(_run))


# ── Model catalog ──────────────────────────────────────────────────────────

@app.get("/models")
async def list_models() -> list[dict[str, Any]]:
    return atlas_models.list_models()


# ── Jobs ──────────────────────────────────────────────────────────────────

@app.get("/jobs")
async def jobs_list() -> list[dict[str, Any]]:
    return [atlas_jobs.to_public(j) for j in atlas_jobs.list_jobs()]


@app.get("/jobs/{job_id}")
async def jobs_get(job_id: str) -> dict[str, Any]:
    job = atlas_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}.")
    out = atlas_jobs.to_public(job)
    if job.status == "done":
        out["result"] = to_jsonable(job.result)
    return out
