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
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

try:
    from fastapi import Body, FastAPI, HTTPException, Query
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response
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

log = logging.getLogger(__name__)

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
    except FileNotFoundError as e:
        # The Qortex facade raises builtin FileNotFoundError for "this path
        # isn't in the manifest / this dataset has no events file / no URL for
        # this file" (see qortex/__init__.py). Those are honest 404s for the
        # requested resource — not the upstream/gateway failure the generic
        # handler below reports. Without this, an Events tab on a dataset that
        # legitimately has no events, or a Preview of a path not in the
        # snapshot, surfaced a scary "502 ... may not support this operation"
        # instead of a clean not-found.
        raise HTTPException(status_code=404, detail=str(e)) from e
    except (ValueError, KeyError) as e:
        # Unsatisfiable/malformed request parameters (a BIDS entity or axis the
        # file doesn't have, a bad enum) — a client 400, not a 502.
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


@app.get("/cache/inventory")
async def cache_inventory() -> dict[str, object]:
    """Measure the persistent cache directories used by this installation."""
    from qortex.console.cache_inventory import cache_inventory as inspect_cache

    return await call(inspect_cache)


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
    # Ceiling raised well past OpenNeuro's full corpus (~1.8k datasets) so the
    # Datasets browse surface can render the whole local catalog, not an
    # arbitrary 200-row slice of it. Default stays small for typeahead callers.
    limit: int = Query(20, le=5000),
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


@app.get("/catalog/count")
async def catalog_count() -> dict[str, Any]:
    """The real number of datasets on OpenNeuro (one cheap GraphQL round-trip),
    plus how many are already cached locally — so the UI can show an honest
    'X of N' target before any sweep starts.

    Registered BEFORE ``/catalog/{dataset_id}`` on purpose: FastAPI matches
    routes in definition order, so the literal path must win over the
    parameterized one or 'count' gets read as a dataset id."""
    from qortex.catalog.index import CatalogIndex
    from qortex.client.graphql import get_shared_client
    from qortex.core.config import get_config

    def _counts() -> dict[str, Any]:
        total = get_shared_client().count_datasets()
        idx = CatalogIndex(get_config().cache_dir / "catalog" / "catalog.duckdb")
        try:
            cached = idx.count()
        finally:
            idx.close()
        return {"total": total, "cached": cached}

    return await call(_counts)


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
async def catalog_refresh_endpoint(max_pages: int = Query(40, le=200)) -> dict[str, Any]:
    from qortex.catalog.refresh import refresh

    n = await call(refresh, max_pages=max_pages, progress=False)
    return {"datasets_indexed": n}


@app.post("/catalog/refresh/start")
async def catalog_refresh_start(max_pages: int = Query(40, le=200)) -> dict[str, Any]:
    """Kick off a full catalog sweep as a background job and return immediately.

    Count-first: the total is fetched up front so the client can render a real
    progress bar, and the sweep fetches pages concurrently (offset cursors) —
    the whole ~1.8k-dataset refresh lands in well under a minute. Poll
    ``/jobs/{id}`` for live ``progress`` and the final ``datasets_indexed``."""
    from qortex.catalog.refresh import refresh
    from qortex.client.graphql import get_shared_client

    total = await call(lambda: get_shared_client().count_datasets())
    job = atlas_jobs.submit(
        "Refresh full catalog from OpenNeuro",
        refresh, max_pages=max_pages, progress=False, workers=8,
        report_progress=True,
    )
    return {"job_id": job.id, "total": total}


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
                # Skip rows with no resolvable id rather than emitting a result
                # with dataset_id=None (which the UI can't open or dedupe) —
                # matches _fetch_live_supplement's handling below.
                if not rid or rid in seen:
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
    include_live: bool = Query(
        False,
        description="Also fetch live OpenNeuro results not yet in the local catalog — the local "
        "engine (BM25/semantic/RRF/evidence) never ranks these, they are appended, tagged "
        "_source='live', so a query is never limited to whatever has already been indexed locally.",
    ),
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
        live_results: list[dict[str, Any]] = []
        if include_live:
            live_results = _fetch_live_supplement(response, q=q, modality=modality, limit=limit)
        return {
            "results": to_jsonable(response.results),
            "live_results": to_jsonable(live_results),
            "plan": to_jsonable(response.plan),
            "negative_space": to_jsonable(response.negative_space),
            "timings_ms": response.timings_ms,
        }

    return await call(_run)


def _fetch_live_supplement(
    response: Any, *, q: Optional[str], modality: Optional[str], limit: int
) -> list[dict[str, Any]]:
    """Datasets from the live OpenNeuro API not already covered by the local
    engine's results — explicitly unranked by BM25/semantic/RRF (there is no
    local structural/lexical/semantic signal for a dataset that was never
    indexed), so these are appended after, never interleaved into
    ``fused_score`` order, and the frontend must label them distinctly rather
    than implying they went through the same ranking."""
    from qortex.catalog.search import live_search

    local_ids = {r.dataset_id for r in response.results}
    modality_constraint = response.plan.hard.get("modality")
    # A modality constraint's `value` is a set; guard against it being empty so
    # this can never IndexError (which the generic handler would turn into a
    # confusing 502 for what is really a successful, if unconstrained, search).
    live_modality = modality or (
        sorted(modality_constraint.value)[0]
        if modality_constraint and modality_constraint.value
        else None
    )
    try:
        remote = live_search(query=q, modality=live_modality, limit=max(30, limit * 3), sync_local=False)
    except QortexError:
        return []
    out: list[dict[str, Any]] = []
    for row in remote:
        rid = row.get("dataset_id") or row.get("id")
        if not rid or rid in local_ids:
            continue
        row["dataset_id"] = rid
        row["_source"] = "live"
        out.append(row)
        local_ids.add(rid)
        if len(out) >= limit:
            break
    return out


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
async def dataset_readiness(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    modality: Optional[str] = Query(None),
    target: Optional[str] = Query(None),
    label_column: Optional[str] = Query(None, min_length=1, max_length=128),
    label_missing: Literal["drop", "keep", "error"] = Query("drop"),
    split_strategy: Literal["subject", "subject_session", "recording"] = Query("subject"),
) -> dict[str, Any]:
    from qortex.check.readiness import compute_readiness
    from qortex.core.config import get_config
    from qortex.core.entities import LabelPolicy
    from qortex.decision import can_train as can_train_fn
    from qortex.lake.layout import LakeLayout

    def _run() -> dict[str, Any]:
        with atlas_timing.timed("readiness", dataset_id):
            manifest = _manifest_for(dataset_id, snapshot)
            data_root = LakeLayout(get_config()).data_dir(dataset_id, manifest.snapshot)
            local_path = data_root if data_root.is_dir() else None
            policy = LabelPolicy(column=label_column.strip(), missing=label_missing) if label_column else None
            readiness = compute_readiness(
                manifest,
                local_path=local_path,
                label_policy=policy,
                modality=modality,
            )
            ct = can_train_fn(
                manifest,
                modality=modality,
                target=target,
                local_path=local_path,
                label_policy=policy,
                split_strategy=split_strategy,
            )
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
            "local_label_evidence": {
                "data_root_present": local_path is not None,
                "policy": to_jsonable(policy) if policy else None,
            },
            "evidence": evidence.as_dict(),
        }

    return await call(_run)


@app.get("/dataset/{dataset_id}/validation")
async def dataset_validation(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Return snapshot validation issues published by OpenNeuro.

    OpenNeuro exposes issue findings, not the validator's full passed-check
    inventory. The response therefore leaves passed counts and tool version
    unknown instead of deriving them from an absence of issues.
    """
    from qortex.client.graphql import get_shared_client

    def _run() -> dict[str, Any]:
        manifest = _manifest_for(dataset_id, snapshot)
        issues = get_shared_client().get_validation_issues(dataset_id, manifest.snapshot)
        groups: dict[str, dict[str, Any]] = {}
        counts = {"error": 0, "warning": 0}
        for issue in issues:
            severity = str(issue.get("severity") or "warning").lower()
            if severity not in counts:
                severity = "warning"
            counts[severity] += 1
            key = str(issue.get("key") or "unclassified")
            group = groups.setdefault(key, {
                "key": key,
                "severity": severity,
                "reason": issue.get("reason"),
                "help_url": issue.get("helpUrl"),
                "occurrences": 0,
                "files": [],
            })
            group["occurrences"] += 1
            for file_info in issue.get("files") or []:
                path = file_info.get("path") or file_info.get("name")
                if path and path not in group["files"] and len(group["files"]) < 20:
                    group["files"].append(path)
            if severity == "error":
                group["severity"] = "error"
        ordered = sorted(
            groups.values(),
            key=lambda item: (item["severity"] != "error", item["key"]),
        )
        return {
            "dataset_id": dataset_id,
            "snapshot": manifest.snapshot,
            "source": "OpenNeuro snapshot validation issues",
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "validator": {
                "name": "BIDS Validator",
                "version": None,
                "version_evidence": "not exposed by the OpenNeuro snapshot issues API",
            },
            "coverage": {
                "passed_checks": None,
                "passed_checks_evidence": "OpenNeuro publishes issues only; absence is not counted as passed checks",
                "issue_occurrences": len(issues),
                **counts,
            },
            "issues": ordered,
        }

    return await call(_run)


@app.post("/dataset/{dataset_id}/validation/local/start")
async def dataset_local_validation_start(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Run the installed official BIDS Validator over bytes present locally."""
    import secrets

    from qortex.console.local_validation import run_local_bids_validation
    from qortex.core.config import get_config
    from qortex.lake.layout import LakeLayout

    manifest = _manifest_for(dataset_id, snapshot)
    layout = LakeLayout(get_config())
    data_dir = layout.data_dir(dataset_id, manifest.snapshot)
    run_id = f"bv-{secrets.token_hex(6)}"
    output_dir = layout.reports_dir(dataset_id, manifest.snapshot) / "bids-validator" / run_id

    def _run() -> dict[str, Any]:
        result = run_local_bids_validation(manifest, data_dir, output_dir)
        return {"run_id": run_id, **result}

    job = atlas_jobs.submit(f"Validate local {dataset_id}@{manifest.snapshot}", _run)
    return {"job_id": job.id, "run_id": run_id, "snapshot": manifest.snapshot}


@app.get(
    "/dataset/{dataset_id}/validation/local/runs/{snapshot}/{run_id}/artifacts/{artifact_name}"
)
async def dataset_local_validation_artifact(
    dataset_id: str,
    snapshot: str,
    run_id: str,
    artifact_name: str,
) -> FileResponse:
    """Serve one report artifact from a bounded local-validator run."""
    from qortex.console.local_validation import resolve_validation_artifact
    from qortex.core.config import get_config
    from qortex.lake.layout import LakeLayout

    if re.fullmatch(r"bv-[a-f0-9]{12}", run_id) is None:
        raise HTTPException(status_code=400, detail="invalid validation run id")
    manifest = _manifest_for(dataset_id, snapshot)
    output_dir = (
        LakeLayout(get_config()).reports_dir(dataset_id, manifest.snapshot)
        / "bids-validator"
        / run_id
    )
    try:
        artifact = await call(resolve_validation_artifact, output_dir, artifact_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="validation artifact not found") from exc
    return FileResponse(artifact, filename=artifact.name, media_type="application/json")


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
    from qortex.eda.participants import (
        ParticipantRecord,
        ParticipantsTable,
        summarize_demographics,
    )

    def _run():
        manifest = _manifest_for(dataset_id, snapshot)
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=manifest)
        df = ds.participants()
        rows = df.to_dicts()
        sidecar: dict[str, Any] = {}
        try:
            candidate = next(
                (record.path for record in manifest.files if record.path.endswith("participants.json")),
                None,
            )
            if candidate:
                preview = ds.preview(candidate)
                sidecar = preview.get("data", preview) if isinstance(preview, dict) else {}
        except (FileNotFoundError, ValueError, TypeError):
            sidecar = {}

        summary = None
        if "age" in df.columns and "sex" in df.columns:
            table = ParticipantsTable(
                columns=list(df.columns),
                records=[
                    ParticipantRecord(
                        participant_id=str(row.get("participant_id", "")),
                        values={
                            key: "" if value is None else str(value)
                            for key, value in row.items()
                            if key != "participant_id"
                        },
                    )
                    for row in rows
                ],
                sidecar=sidecar,
            )
            summary = summarize_demographics(table)
        return {"columns": df.columns, "rows": rows, "demographics": summary}

    return await call(_run)


@app.get("/dataset/{dataset_id}/coverage")
async def dataset_coverage(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    from qortex.eda.coverage import observed_coverage_report

    return await call(
        observed_coverage_report,
        _manifest_for(dataset_id, snapshot),
        offset=offset,
        limit=limit,
    )


class CoverageSelectorBody(_PydanticModel):
    session: Optional[str] = None
    task: Optional[str] = None
    run: Optional[str] = None
    modality: str
    suffix: str


class CoverageExpectationBody(_PydanticModel):
    id: Optional[str] = None
    selector: CoverageSelectorBody
    expected_subjects: list[str]


class CoverageDesignBody(_PydanticModel):
    expectations: list[CoverageExpectationBody]
    offset: int = 0
    limit: int = 100


@app.post("/dataset/{dataset_id}/coverage/evaluate-design")
async def dataset_coverage_design(
    dataset_id: str,
    body: CoverageDesignBody = Body(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.eda.coverage import evaluate_coverage_expectations

    try:
        return await call(
            evaluate_coverage_expectations,
            _manifest_for(dataset_id, snapshot),
            [item.model_dump() for item in body.expectations],
            offset=body.offset,
            limit=body.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class AnnotationSaveBody(_PydanticModel):
    model_config = {"extra": "forbid"}
    source_path: str
    annotation_id: Optional[str] = None
    expected_revision: Optional[int] = None
    payload: dict[str, Any]


@app.get("/dataset/{dataset_id}/annotations")
async def dataset_annotations(
    dataset_id: str,
    snapshot: str = Query(...),
    source_path: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.console.annotation_store import list_annotations

    manifest = _manifest_for(dataset_id, snapshot)
    if source_path is not None and not any(record.path == source_path for record in manifest.files):
        raise HTTPException(status_code=400, detail="source_path is not in this immutable snapshot manifest")
    return await call(list_annotations, dataset_id, manifest.snapshot, source_path=source_path)


@app.post("/dataset/{dataset_id}/annotations")
async def save_dataset_annotation(
    dataset_id: str,
    body: AnnotationSaveBody = Body(...),
    snapshot: str = Query(...),
) -> dict[str, Any]:
    from qortex.console.annotation_store import save_annotation

    manifest = _manifest_for(dataset_id, snapshot)
    record = next((item for item in manifest.files if item.path == body.source_path), None)
    if record is None:
        raise HTTPException(status_code=400, detail="source_path is not in this immutable snapshot manifest")
    if record.extension not in {".nii", ".nii.gz"}:
        raise HTTPException(status_code=400, detail="viewer annotations require a NIfTI source")
    source = {
        "path": record.path,
        "filename": record.filename,
        "size_bytes": record.size,
        "checksum": record.checksum,
        "checksum_algorithm": "md5" if record.checksum else None,
        "urls": record.urls,
        "snapshot": manifest.snapshot,
    }
    try:
        # Keep the domain conflict visible here: the generic ``call`` wrapper
        # intentionally maps unknown runtime failures to 502, while an
        # optimistic revision mismatch is an HTTP resource conflict.
        return await run_in_threadpool(
            save_annotation,
            dataset_id=dataset_id,
            snapshot=manifest.snapshot,
            source=source,
            payload=body.payload,
            annotation_id=body.annotation_id,
            expected_revision=body.expected_revision,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/dataset/{dataset_id}/annotations/{annotation_id}")
async def get_dataset_annotation(
    dataset_id: str,
    annotation_id: str,
    snapshot: str = Query(...),
    revision: Optional[int] = Query(None, ge=1),
) -> dict[str, Any]:
    from qortex.console.annotation_store import load_annotation

    manifest = _manifest_for(dataset_id, snapshot)
    try:
        return await call(load_annotation, dataset_id, manifest.snapshot, annotation_id, revision=revision)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _local_bold_context(dataset_id: str, snapshot: str | None, path: str | None) -> dict[str, Any]:
    from qortex.core.config import get_config
    from qortex.lake.layout import LakeLayout

    manifest = _manifest_for(dataset_id, snapshot)
    candidate_paths = sorted(
        record.path for record in manifest.files
        if not record.is_dir and record.suffix == "bold" and record.extension in {".nii", ".nii.gz"}
    )
    if path is not None and path not in candidate_paths:
        raise HTTPException(status_code=400, detail="path is not a BOLD NIfTI in this snapshot manifest")
    data_root = LakeLayout(get_config()).data_dir(dataset_id, manifest.snapshot).resolve()
    available = [
        candidate for candidate in candidate_paths
        if (data_root / candidate).resolve().is_relative_to(data_root) and (data_root / candidate).is_file()
    ]
    selected = path or (available[0] if available else None)
    return {
        "manifest": manifest,
        "candidate_paths": candidate_paths,
        "available": available,
        "selected": selected,
        "local_file": (data_root / selected).resolve() if selected in available else None,
    }


@app.get("/dataset/{dataset_id}/fmri-qc")
async def dataset_fmri_qc(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    max_frames: int = Query(500, ge=2, le=2000),
    fd_threshold_mm: float = Query(0.5, ge=0.0),
    dvars_threshold: Optional[float] = Query(None, ge=0.0),
) -> dict[str, Any]:
    """Compute real framewise BOLD QC from a locally downloaded 4-D NIfTI."""
    from qortex.visualize.volume import VolumeViewer

    context = _local_bold_context(dataset_id, snapshot, path)
    manifest = context["manifest"]
    candidate_paths = context["candidate_paths"]
    available = context["available"]
    selected = context["selected"]
    if selected is None or selected not in available:
        return {
            "dataset_id": dataset_id,
            "snapshot": manifest.snapshot,
            "available": False,
            "selected_path": selected,
            "available_paths": available,
            "manifest_bold_count": len(candidate_paths),
            "reason": (
                "The selected BOLD file is not downloaded locally. Full framewise QC requires consecutive image volumes; remote header or slice ranges are insufficient."
                if selected
                else "No locally downloaded BOLD NIfTI is available for full framewise QC."
            ),
        }

    local_file = context["local_file"]

    def _run() -> dict[str, Any]:
        viewer = VolumeViewer(local_file, modality="fmri")
        report = viewer.fmri_qc_report(
            max_frames=max_frames,
            fd_threshold_mm=fd_threshold_mm,
            dvars_threshold=dvars_threshold,
        )
        return {
            "dataset_id": dataset_id,
            "snapshot": manifest.snapshot,
            "available": True,
            "selected_path": selected,
            "available_paths": available,
            "manifest_bold_count": len(candidate_paths),
            "report": report,
        }

    return await call(_run)


class PersistentFmriQcBody(_PydanticModel):
    path: Optional[str] = None
    max_frames: int = 500
    fd_threshold_mm: float = 0.5
    dvars_threshold: Optional[float] = None


@app.post("/dataset/{dataset_id}/fmri-qc/runs")
async def start_persistent_fmri_qc(
    dataset_id: str,
    body: PersistentFmriQcBody = Body(default=PersistentFmriQcBody()),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    from qortex.console.fmri_qc_runs import run_persistent_fmri_qc

    if not 2 <= body.max_frames <= 2000:
        raise HTTPException(status_code=400, detail="max_frames must be in [2, 2000]")
    if body.fd_threshold_mm < 0 or (body.dvars_threshold is not None and body.dvars_threshold < 0):
        raise HTTPException(status_code=400, detail="QC thresholds must be non-negative")
    context = _local_bold_context(dataset_id, snapshot, body.path)
    selected = context["selected"]
    if selected is None or context["local_file"] is None:
        raise HTTPException(status_code=409, detail="No selected locally downloaded BOLD NIfTI is available")
    job = atlas_jobs.submit(
        f"Persist fMRI QC for {dataset_id}:{selected}",
        run_persistent_fmri_qc,
        dataset_id=dataset_id,
        snapshot=context["manifest"].snapshot,
        source_path=selected,
        local_file=context["local_file"],
        max_frames=body.max_frames,
        fd_threshold_mm=body.fd_threshold_mm,
        dvars_threshold=body.dvars_threshold,
        report_progress=True,
    )
    return {"job_id": job.id, "dataset_id": dataset_id, "snapshot": context["manifest"].snapshot, "path": selected}


@app.get("/fmri-qc/runs/{run_id}")
async def get_persistent_fmri_qc(run_id: str) -> dict[str, Any]:
    from qortex.console.fmri_qc_runs import load_fmri_qc_run

    return await call(load_fmri_qc_run, run_id)


@app.get("/fmri-qc/runs/{run_id}/artifacts/{artifact}")
async def get_persistent_fmri_qc_artifact(run_id: str, artifact: str) -> FileResponse:
    from qortex.console.fmri_qc_runs import fmri_qc_artifact_path

    path = await call(fmri_qc_artifact_path, run_id, artifact)
    media_types = {".json": "application/json", ".csv": "text/csv", ".gz": "application/gzip"}
    return FileResponse(path, filename=path.name, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


@app.get("/dataset/{dataset_id}/signal-analysis")
async def dataset_signal_analysis(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
    path: Optional[str] = Query(None),
    duration_seconds: float = Query(20.0, gt=0.0, le=120.0),
    max_channels: int = Query(32, ge=2, le=64),
    connectivity_threshold: float = Query(0.35, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Run bounded Neuroclassic analytics on a locally downloaded signal file."""
    from qortex.core.config import get_config
    from qortex.eda.signal_analysis import analyze_signal_file
    from qortex.lake.layout import LakeLayout

    manifest = _manifest_for(dataset_id, snapshot)
    candidate_records = sorted(
        (
            record
            for record in manifest.files
            if not record.is_dir
            and record.suffix in {"eeg", "meg", "ieeg"}
            and record.extension in {".edf", ".bdf", ".fif", ".set", ".vhdr"}
        ),
        key=lambda record: record.path,
    )
    records_by_path = {record.path: record for record in candidate_records}
    if path is not None and path not in records_by_path:
        raise HTTPException(status_code=400, detail="path is not an MNE-readable signal file in this snapshot manifest")

    data_root = LakeLayout(get_config()).data_dir(dataset_id, manifest.snapshot).resolve()
    available_records = [
        record
        for record in candidate_records
        if (data_root / record.path).resolve().is_relative_to(data_root)
        and (data_root / record.path).is_file()
    ]
    if path is not None:
        available_records = [record for record in available_records if record.path == path]

    def _run() -> dict[str, Any]:
        load_errors: list[dict[str, str]] = []
        for record in available_records:
            local_file = (data_root / record.path).resolve()
            try:
                report = analyze_signal_file(
                    local_file,
                    file_record=record,
                    duration_seconds=duration_seconds,
                    max_channels=max_channels,
                    connectivity_threshold=connectivity_threshold,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                load_errors.append({"path": record.path, "error": f"{type(exc).__name__}: {exc}"})
                continue
            return {
                "dataset_id": dataset_id,
                "snapshot": manifest.snapshot,
                "available": True,
                "selected_path": record.path,
                "downloaded_paths": [item.path for item in available_records],
                "manifest_signal_count": len(candidate_records),
                "unreadable_downloads": load_errors,
                "report": report,
            }
        return {
            "dataset_id": dataset_id,
            "snapshot": manifest.snapshot,
            "available": False,
            "selected_path": path,
            "downloaded_paths": [item.path for item in available_records],
            "manifest_signal_count": len(candidate_records),
            "unreadable_downloads": load_errors[:20],
            "reason": (
                "Downloaded signal files were present but none could be parsed as complete recordings."
                if available_records
                else "No locally downloaded MNE-readable EEG, MEG, or iEEG recording is available."
            ),
        }

    return await call(_run)


class ConversionBody(_PydanticModel):
    paths: list[str]
    output_format: str = "parquet"
    shard_size: int = 1000


@app.get("/dataset/{dataset_id}/conversion/options")
async def dataset_conversion_options(
    dataset_id: str,
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """List locally present loader-backed sources and installed output writers."""
    from qortex.console.conversion_runs import conversion_options
    from qortex.core.config import get_config
    from qortex.lake.layout import LakeLayout

    manifest = _manifest_for(dataset_id, snapshot)
    data_dir = LakeLayout(get_config()).data_dir(dataset_id, manifest.snapshot)
    return await call(conversion_options, manifest, data_dir)


@app.post("/dataset/{dataset_id}/conversion/start")
async def dataset_conversion_start(
    dataset_id: str,
    body: ConversionBody = Body(...),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Start a strict conversion over explicit locally downloaded paths."""
    import secrets

    from qortex.console.conversion_runs import conversion_capabilities, run_conversion
    from qortex.core.config import get_config
    from qortex.lake.layout import LakeLayout

    if not 1 <= len(body.paths) <= 100:
        raise HTTPException(status_code=400, detail="paths must contain between 1 and 100 files")
    if not 1 <= body.shard_size <= 100_000:
        raise HTTPException(status_code=400, detail="shard_size must be in [1, 100000]")
    if len(body.paths) != len(set(body.paths)):
        raise HTTPException(status_code=400, detail="paths must not contain duplicates")
    formats = {item["name"]: item for item in conversion_capabilities()["formats"]}
    capability = formats.get(body.output_format)
    if capability is None:
        raise HTTPException(status_code=400, detail=f"unsupported output format: {body.output_format}")
    if not capability["available"]:
        missing = ", ".join(capability["missing_packages"])
        raise HTTPException(
            status_code=409,
            detail=f"{body.output_format} requires missing packages: {missing}",
        )
    manifest = _manifest_for(dataset_id, snapshot)
    layout = LakeLayout(get_config())
    data_dir = layout.data_dir(dataset_id, manifest.snapshot)
    run_id = f"cv-{secrets.token_hex(6)}"
    output_dir = layout.exports_dir(dataset_id, manifest.snapshot) / run_id

    def _run() -> dict[str, Any]:
        result = run_conversion(
            manifest,
            data_dir,
            output_dir,
            paths=body.paths,
            output_format=body.output_format,
            shard_size=body.shard_size,
        )
        return {"run_id": run_id, **result}

    job = atlas_jobs.submit(
        f"Convert {dataset_id}@{manifest.snapshot} to {body.output_format}",
        _run,
    )
    return {"job_id": job.id, "run_id": run_id, "snapshot": manifest.snapshot}


@app.get("/dataset/{dataset_id}/conversion/runs/{snapshot}/{run_id}/artifacts/{artifact_path:path}")
async def dataset_conversion_artifact(
    dataset_id: str,
    snapshot: str,
    run_id: str,
    artifact_path: str,
) -> FileResponse:
    """Serve one inventoried conversion artifact from its bounded run directory."""
    import mimetypes

    from qortex.console.conversion_runs import resolve_conversion_artifact
    from qortex.core.config import get_config
    from qortex.lake.layout import LakeLayout

    if re.fullmatch(r"cv-[a-f0-9]{12}", run_id) is None:
        raise HTTPException(status_code=400, detail="invalid conversion run id")
    manifest = _manifest_for(dataset_id, snapshot)
    output_dir = LakeLayout(get_config()).exports_dir(dataset_id, manifest.snapshot) / run_id
    try:
        artifact = await call(resolve_conversion_artifact, output_dir, artifact_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="conversion artifact not found") from exc
    media_type = mimetypes.guess_type(artifact.name)[0] or "application/octet-stream"
    return FileResponse(artifact, filename=artifact.name, media_type=media_type)


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
    zero bytes of volume data transferred. Uses NiftiStreamer's header parser
    (the same one /nifti-slice-data uses), not Dataset.nifti_info()'s simpler
    gateway-level parser — that one never computed an affine at all, which
    silently made `axis_codes` below impossible to fill in until this switch.
    Adds `axis_codes` (e.g. ["L","A","S"]) when nibabel is available: which
    real-world anatomical direction each voxel axis increases toward,
    computed from this file's own affine matrix via nibabel's standard
    `aff2axcodes` — a real, derived fact (not a convention-based guess), used
    for the Viewer Lab's plane edge labels. Omitted entirely (never guessed)
    if nibabel isn't installed."""

    def _run():
        streamer, _file_record = _resolve_nifti_streamer(dataset_id, snapshot, path)
        hdr = streamer.header()
        result = hdr.to_dict()
        try:
            import nibabel as nib
            import numpy as np

            result["axis_codes"] = list(nib.aff2axcodes(np.asarray(hdr.affine)))
        except Exception:
            pass
        return result

    return await call(_run)


def _guess_modality(suffix: str) -> str:
    """BIDS suffix -> the coarse modality bucket qortex.visualize._colors'
    preset tables are keyed on (mri/fmri/dwi/pet/ct)."""
    s = (suffix or "").lower()
    if s in {"bold", "cbv", "sbref"}:
        return "fmri"
    if s == "dwi":
        return "dwi"
    if s == "pet":
        return "pet"
    return "mri"


def _presets_for_modality(modality: str, arr: Any) -> list[dict[str, Any]]:
    """Every named window preset applicable to this modality, each resolved
    to concrete (vmin, vmax) against *this* array — computed fresh per
    request (a handful of percentile passes over one already-fetched 2D
    slice, not the whole volume) so presets reflect this file's real
    intensity distribution rather than a generic guess."""
    from qortex.visualize._colors import (
        CT_PRESETS,
        FMRI_PRESETS,
        MR_PRESETS,
        PET_PRESETS,
        auto_window,
    )

    table = {"mri": MR_PRESETS, "dwi": MR_PRESETS, "fmri": FMRI_PRESETS, "pet": PET_PRESETS, "ct": CT_PRESETS}
    presets = table.get(modality, MR_PRESETS)
    out = []
    for name, preset in presets.items():
        vmin, vmax = auto_window(arr, modality=modality, preset=preset)
        out.append({"name": name, "vmin": vmin, "vmax": vmax, "colormap": preset.colormap})
    return out


def _resolve_nifti_streamer(dataset_id: str, snapshot: Optional[str], path: str):
    """Resolve any BIDS-relative NIfTI path in the manifest to a live
    ``NiftiStreamer`` + its ``FileRecord`` — the same path-based resolution
    ``/nifti-info`` already uses (``Dataset.nifti_info``), factored out so
    the slice-data/slice-png routes below can view *any* NIfTI file a user
    clicks in the file browser, not just the canonical subject/modality
    lookup ``Dataset.stream_slice`` performs."""
    from qortex.client.remote import _pick_url

    manifest = _manifest_for(dataset_id, snapshot)
    target = next((f for f in manifest.files if f.path == path), None)
    if target is None:
        raise DatasetNotFoundError(f"Path {path!r} not found in manifest")
    url = _pick_url(target)
    if not url:
        raise DatasetNotFoundError(f"No download URL available for {path!r}")
    return _streamer_for(url), target


# Reuse one NiftiStreamer per file URL across requests. The Viewer fires many
# small requests against the same file (scrub the Z slider, flip planes, switch
# presets, step through time) and previously built a *fresh* streamer each time
# — re-fetching the ~128 KB header and, for a .nii.gz, re-decompressing the whole
# volume on every single slice. Sharing the instance lets its header cache and
# decompressed-volume LRU (see NiftiStreamer._volume_array) actually pay off
# across the request stream, which is where the viewer's latency really lives.
# TTL-bounded so idle files release their cached volumes.
_STREAMER_CACHE = TTLCache(ttl=_MANIFEST_CACHE_TTL_S, maxsize=16)


def _streamer_for(url: str):
    from qortex.stream import NiftiStreamer

    return _STREAMER_CACHE.get_or_compute((url,), lambda: NiftiStreamer(url))


@app.get("/dataset/{dataset_id}/nifti-slice-data")
async def dataset_nifti_slice_data(
    dataset_id: str,
    path: str = Query(..., description="BIDS-relative path to a .nii/.nii.gz file, from /dataset/{id}/manifest"),
    axis: int = Query(2, ge=0, le=2, description="0=first spatial axis, 1=second, 2=third (see NiftiStreamer.get_slice)"),
    slice_index: Optional[int] = Query(None),
    time_index: int = Query(0, ge=0),
    histogram: bool = Query(False, description="Also return a real intensity histogram of this slice (counts + bin edges + min/max/mean/std) for the windowing panel — computed from the already-decoded pixels, no extra fetch"),
    histogram_bins: int = Query(128, ge=8, le=512),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Raw calibrated intensities for one 2D slice, base64-encoded float32 —
    for genuine client-side windowing/leveling: drag-to-adjust contrast with
    zero network round trips per adjustment, the standard PACS interaction
    pattern. Complements ``/nifti-slice.png`` (which bakes one fixed contrast
    into a PNG server-side); this hands the client the real numbers plus
    every applicable clinical preset resolved against this exact slice, so
    the UI can offer instant preset switching and free dragging without
    ever re-fetching pixel data — only ``axis``/``slice_index``/``time_index``
    changes require a new request, exactly like scrubbing to a new frame in
    any PACS viewer."""
    import base64

    import numpy as np

    def _run() -> dict[str, Any]:
        import time

        from qortex.console.stream_telemetry import stream_telemetry

        streamer, file_record = _resolve_nifti_streamer(dataset_id, snapshot, path)
        hdr = streamer.header()
        idx = slice_index if slice_index is not None else hdr.spatial_shape[axis] // 2
        # Two genuinely different operations, timed separately: axis=2 on an
        # uncompressed .nii takes the single-Range-request fast path
        # (NiftiStreamer._fetch_slice_contiguous); every other case still
        # fetches the whole volume server-side. Mixing their durations into
        # one estimate would produce a median that's honest for neither —
        # the Viewer Lab's per-plane ETA depends on knowing which one applies.
        is_fast_path = axis == 2 and not path.lower().endswith(".gz")
        operation = "nifti_slice_axial_fast" if is_fast_path else "nifti_slice_full_volume"
        cache_before = streamer.stream_stats()
        started = time.perf_counter()
        with atlas_timing.timed(operation, dataset_id):
            arr = streamer.get_slice(axis=axis, index=idx, t=time_index, dtype=np.float32)
        elapsed_seconds = time.perf_counter() - started
        cache_after = streamer.stream_stats()
        stream_telemetry.record({
            "operation": operation,
            "dataset_id": dataset_id,
            "path": path,
            "axis": axis,
            "slice_index": idx,
            "time_index": time_index,
            "elapsed_seconds": elapsed_seconds,
            "response_data_bytes": int(arr.nbytes),
            "cache_hits_delta": int(cache_after.get("hits", 0) - cache_before.get("hits", 0)),
            "cache_misses_delta": int(cache_after.get("misses", 0) - cache_before.get("misses", 0)),
            "cache_hit_bytes_delta": int(cache_after.get("hit_bytes", 0) - cache_before.get("hit_bytes", 0)),
            "cache_bytes_inserted_delta": int(cache_after.get("bytes_inserted", 0) - cache_before.get("bytes_inserted", 0)),
            "decoded_volume_hits_delta": int(cache_after.get("volume_hits", 0) - cache_before.get("volume_hits", 0)),
            "decoded_volume_misses_delta": int(cache_after.get("volume_misses", 0) - cache_before.get("volume_misses", 0)),
            "decoded_volume_resident_bytes": int(cache_after.get("volume_resident_bytes", 0)),
            "source_kind": "local" if getattr(streamer, "_is_local", False) else "remote",
        })
        modality = _guess_modality(file_record.suffix)
        data_b64 = base64.b64encode(np.ascontiguousarray(arr, dtype=np.float32).tobytes()).decode("ascii")
        from qortex.visualize._colors import auto_window

        auto_vmin, auto_vmax = auto_window(arr, modality=modality, suffix=file_record.suffix or "")
        return {
            "path": path,
            "axis": axis,
            "slice_index": idx,
            "time_index": time_index,
            "shape": list(arr.shape),
            "dtype": "float32",
            "data_b64": data_b64,
            "auto_vmin": auto_vmin,
            "auto_vmax": auto_vmax,
            "voxel_size_mm": list(hdr.voxel_sizes_mm),
            "spatial_shape": list(hdr.spatial_shape),
            "n_volumes": hdr.n_volumes,
            "is_4d": hdr.is_4d,
            "modality": modality,
            "suffix": file_record.suffix,
            "presets": _presets_for_modality(modality, arr),
            "is_fast_path": is_fast_path,
            "histogram": _intensity_histogram(arr, histogram_bins) if histogram else None,
        }

    return await call(_run)


@app.get("/stream/telemetry")
async def stream_telemetry_report(
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Return bounded measurements from actual NIfTI slice-data requests."""
    from qortex.console.stream_telemetry import stream_telemetry

    return stream_telemetry.report(limit=limit)


def _intensity_histogram(arr: Any, bins: int) -> dict[str, Any] | None:
    """A real intensity histogram of one already-decoded slice — the numbers a
    PACS windowing panel needs to draw its distribution curve and place the
    window/level handles, computed from pixels already in hand (no extra fetch).

    Non-finite voxels (NaN/inf, common in statistical and masked maps) are
    excluded so they never collapse the range; an all-non-finite or constant
    slice returns ``None`` rather than a degenerate single-bin histogram."""
    import numpy as np

    flat = np.asarray(arr, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return None
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return None
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    return {
        "counts": counts.astype(int).tolist(),
        "bin_edges": edges.astype(float).tolist(),
        "min": lo,
        "max": hi,
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "n_finite": int(finite.size),
        "n_nonfinite": int(flat.size - finite.size),
    }


@app.get("/dataset/{dataset_id}/nifti-projection-data")
async def dataset_nifti_projection_data(
    dataset_id: str,
    path: str = Query(..., description="BIDS-relative path to a .nii/.nii.gz file"),
    axis: int = Query(2, ge=0, le=2),
    method: str = Query("mip", pattern="^(mip|minip|mean)$", description="mip=max intensity, minip=min intensity, mean=average, projected through the whole volume along this axis"),
    time_index: int = Query(0, ge=0),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """A real intensity projection through the *entire* volume along one
    axis — Maximum Intensity Projection (the standard angiography/PET
    review view), Minimum Intensity Projection, or a mean projection. There
    is no way to compute this from a single slice; it always requires the
    full volume server-side, so it is never eligible for the axial fast
    path a plain slice fetch gets. Returns the exact same (shape, data_b64,
    presets, …) contract as /nifti-slice-data — same shape-per-axis
    convention (axis=2 -> (nx,ny), etc.) — so the Viewer Lab renders a
    projection through its existing slice-rendering pipeline, just fed a
    max/min/mean array instead of one indexed slice."""
    import base64

    import numpy as np

    def _run() -> dict[str, Any]:
        streamer, file_record = _resolve_nifti_streamer(dataset_id, snapshot, path)
        with atlas_timing.timed("nifti_projection", dataset_id):
            vol = streamer.get_volume(t=time_index, dtype=np.float32)
        if method == "mip":
            proj = vol.max(axis=axis)
        elif method == "minip":
            proj = vol.min(axis=axis)
        else:
            proj = vol.mean(axis=axis)
        modality = _guess_modality(file_record.suffix)
        data_b64 = base64.b64encode(np.ascontiguousarray(proj, dtype=np.float32).tobytes()).decode("ascii")
        from qortex.visualize._colors import auto_window

        auto_vmin, auto_vmax = auto_window(proj, modality=modality, suffix=file_record.suffix or "")
        return {
            "path": path,
            "axis": axis,
            "method": method,
            "time_index": time_index,
            "shape": list(proj.shape),
            "dtype": "float32",
            "data_b64": data_b64,
            "auto_vmin": auto_vmin,
            "auto_vmax": auto_vmax,
            "modality": modality,
            "suffix": file_record.suffix,
            "presets": _presets_for_modality(modality, proj),
        }

    return await call(_run)


@app.get("/colormaps")
async def colormaps() -> dict[str, Any]:
    """(256, 3) uint8 LUTs for gray/hot/plasma/RdBu_r, base64-encoded (768
    bytes each) — fetched once and cached client-side so the Viewer Lab's
    client-side colormap rendering (see ``/nifti-slice-data``) matches
    exactly what ``qortex.visualize`` would produce server-side, rather than
    reimplementing (and risking a visual mismatch with) the same colormaps
    in JavaScript."""
    import base64

    from qortex.visualize._colors import get_lut

    def _run() -> dict[str, Any]:
        return {
            name: base64.b64encode(get_lut(name).tobytes()).decode("ascii")
            for name in ("gray", "hot", "plasma", "RdBu_r")
        }

    return await call(_run)


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
    preset: Optional[str] = Query(None, description="Named window preset (e.g. 't1w', 'bold', 'stat') — see /colormaps"),
    vmin: Optional[float] = Query(None, description="Explicit window low bound; overrides preset/auto-window"),
    vmax: Optional[float] = Query(None, description="Explicit window high bound"),
    colormap: Optional[str] = Query(None, description="gray|hot|plasma|RdBu_r; defaults to the preset's/modality's suggestion"),
    snapshot: Optional[str] = Query(None),
):
    """A real anatomical/functional slice, decoded from bytes fetched via
    HTTP range requests against the OpenNeuro CDN — the full NIfTI volume is
    never downloaded. Windowed via ``qortex.visualize._colors`` (the same
    clinical preset/LUT machinery Qortex's own report renderer uses), not a
    hardcoded fixed-percentile clip — previously this endpoint always did a
    1st/99th-percentile grayscale clip regardless of modality, with no
    client control at all. For interactive client-side windowing (drag to
    adjust, no round trip per change), use ``/nifti-slice-data`` instead —
    this endpoint remains for simple/non-interactive uses (thumbnails,
    static links, embedding in reports), and stays backward compatible: no
    new params required, existing callers get modality-aware auto-windowing
    instead of the old fixed clip, which is a strict improvement."""
    import numpy as np
    from PIL import Image

    from qortex import Dataset
    from qortex.visualize._colors import (
        apply_window,
        auto_window,
        colormap_for_modality,
        get_lut,
    )

    def _run() -> bytes:
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=_manifest_for(dataset_id, snapshot))
        arr = ds.stream_slice(
            subject=subject, modality=modality, session=session, run=run,
            axis=axis, slice_index=slice_index, time_index=time_index,
        )
        arr = np.asarray(arr, dtype=np.float32)
        detected_modality = _guess_modality(modality)
        if vmin is not None and vmax is not None:
            lo, hi = vmin, vmax
        else:
            lo, hi = auto_window(arr, modality=detected_modality, suffix=modality, preset=preset)
        indices = np.clip((apply_window(arr, lo, hi) * 255).astype(np.uint8), 0, 255)
        lut = get_lut(colormap or colormap_for_modality(detected_modality, modality))
        rgb = lut[np.rot90(indices)]
        img = Image.fromarray(rgb, mode="RGB")
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
#
# The optional `-<reference site>` suffix matters just as much as the base
# pattern: clinical/PSG recordings are routinely stored *referentially*
# ("Fp1-M2", "C3-A1" — a mastoid or ear reference), which is a real, standard
# montage convention, not a malformed label. Without matching it, a genuine
# PSG file (real case found: a sleep-EEG BIDS dataset labeling every scalp
# channel this way) had every single electrode excluded and dumped in with
# actual timestamp/counter channels — the exact miscategorization this regex
# exists to prevent, just from the opposite direction.
_EEG_ELECTRODE_RE = re.compile(
    r"^(Fp|AF|FT|FC|TP|CP|PO|F|C|T|P|O|A|M|I)(z|\d{1,2})"
    r"(-(Fp|AF|FT|FC|TP|CP|PO|F|C|T|P|O|A|M|I)(z|\d{1,2}))?$",
    re.IGNORECASE,
)


@app.get("/dataset/{dataset_id}/eeg-preview")
async def dataset_eeg_preview(
    dataset_id: str,
    subject: Optional[str] = Query(None, description="Required unless `path` is given directly"),
    session: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    run: Optional[str] = Query(None),
    path: Optional[str] = Query(None, description="BIDS-relative path to a specific .edf/.bdf file — overrides subject/session/task/run lookup, for viewing any file a user clicks directly in the file browser"),
    tmin: float = Query(0.0, ge=0),
    tmax: float = Query(4.0, gt=0),
    max_channels: int = Query(8, le=64),
    channels: Optional[str] = Query(None, description="Comma-separated explicit channel labels — overrides the automatic 10-20/10-10 selection, for a real channel picker"),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Real, physical-unit EEG/MEG samples for a short window, decoded from
    an EDF/BDF file via HTTP byte-range requests — only the bytes for this
    exact time window are ever transferred. Formats without a remote-signal
    reader (.set/.fif/.vhdr) honestly report as unsupported rather than
    fabricating a waveform. Always returns ``all_channels`` (full
    ``channel_info``: unit, physical/digital range, prefilter, transducer)
    regardless of which subset is actually plotted, so a UI can build a real
    channel picker instead of only ever seeing whatever the server happened
    to auto-select."""
    from qortex.stream import EDFStreamer

    def _run() -> dict[str, Any]:
        manifest = _manifest_for(dataset_id, snapshot)
        if path:
            f = next((rec for rec in manifest.files if rec.path == path), None)
        else:
            if not subject:
                return {"supported": False, "reason": "Either `path` or `subject` is required."}
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

        if channels:
            requested = {c.strip() for c in channels.split(",") if c.strip()}
            selected = [c for c in all_channels if c["label"] in requested]
            n_excluded = len(all_channels) - len(selected)
        else:
            electrodes = [c for c in all_channels if _EEG_ELECTRODE_RE.match(c["label"])]
            # Fall back to the unfiltered list if nothing matched the 10-20/
            # 10-10 pattern — some legitimate systems use other real montages
            # (e.g. high-density nets with numeric-only labels) and this must
            # never silently return zero channels for those.
            selected = (electrodes if electrodes else all_channels)[:max_channels]
            n_excluded = len(all_channels) - len(electrodes) if electrodes else 0

        selected_labels = [c["label"] for c in selected]
        epoch = streamer.get_epoch(tmin, max_t, channels=selected_labels) if selected_labels else []
        return {
            "supported": True,
            "path": f.path,
            "sfreq": hdr.sampling_rates[0] if hdr.sampling_rates else None,
            "duration_s": hdr.duration_s,
            "channels": selected_labels,
            "all_channels": all_channels,
            "n_channels_total": len(all_channels),
            "n_channels_excluded": n_excluded,
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


# ── Compatibility source-profile cache ───────────────────────────────────────
# The expensive part of a compatibility check is building the SourceProfile:
# a remote signal-budget scan (one JSON sidecar GET per signal file, plus NIfTI
# header Range reads) that took 40–90+ s on large datasets. Two facts make it
# cacheable and boundable without losing correctness:
#
#  * A snapshot is immutable — the same (dataset, snapshot) can never yield a
#    different SourceProfile, so disk-cache it forever and share it across
#    server restarts.
#  * The engine consumes only per-modality *modes* (n_channels_mode,
#    sampling_rate_mode). Acquisition parameters are near-homogeneous within a
#    BIDS dataset, so a bounded sample of sidecars produces the same modes as
#    an exhaustive sweep. The profile is already reported as
#    EvidenceStatus.inferred either way — sampling does not weaken the claim.
#
# In-memory single-flight sits in front of the disk file so a cold dataset hit
# by several tabs at once still triggers exactly one scan.
_COMPAT_SCAN_MAX_SIDECARS = 64
_COMPAT_CACHE = TTLCache(ttl=3600.0, maxsize=256)
_COMPAT_DISK_LOCK = threading.Lock()


def _compat_disk_path() -> "Path":
    from qortex.core.config import get_config

    return get_config().cache_dir / "catalog" / "compat_profiles.json"


def _compat_disk_load() -> dict[str, Any]:
    try:
        with open(_compat_disk_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _compat_disk_store(key: str, profile: dict[str, Any]) -> None:
    # Atomic replace under a lock: concurrent scans of different datasets must
    # not interleave partial writes or clobber each other's entries.
    with _COMPAT_DISK_LOCK:
        path = _compat_disk_path()
        data = _compat_disk_load()
        data[key] = profile
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            tmp.replace(path)
        except OSError:
            log.warning("compat profile cache write failed for %s", key, exc_info=True)


def _compat_source_profile(dataset_id: str, snapshot: Optional[str]) -> dict[str, Any]:
    """Return the SourceProfile field dict for (dataset, resolved snapshot),
    scanning remotely only on a true cold miss."""
    from qortex import Dataset
    from qortex.neuroai.contracts import EvidenceStatus

    manifest = _manifest_for(dataset_id, snapshot)
    key = f"{dataset_id}@{manifest.snapshot}"

    # Prefer an actual neural-signal modality over incidental BIDS datatypes
    # (e.g. "behavior", "phenotype") that a raw modalities list may list first
    # with no meaningful ordering. Decided from the manifest *before* the scan
    # so the scan can be tailored to the one modality the engine will consume.
    signal_priority = ["eeg", "meg", "ieeg", "bold", "dwi", "t1w", "t2w"]
    mods = manifest.summary.modalities
    modality = next((m for m in signal_priority if m in mods), next(iter(mods), None))
    # NIfTI header Range reads (for RepetitionTime) dominate the scan — tens of
    # seconds — but only carry information for MRI-family modalities. When the
    # engine's chosen modality is electrophysiology (eeg/meg/ieeg), every
    # parameter it needs is in the JSON sidecars, so the header fetches are pure
    # dead weight and are skipped. That alone turns a 50–90 s scan into ~6 s.
    _MRI_FAMILY = {"bold", "dwi", "t1w", "t2w", "fmri", "mri"}
    needs_nifti = modality in _MRI_FAMILY

    def _build() -> dict[str, Any]:
        disk = _compat_disk_load()
        cached = disk.get(key)
        if isinstance(cached, dict):
            return cached
        ds = Dataset(dataset_id, snapshot=snapshot, manifest=manifest)
        budget = ds.signal_budget(
            max_sidecars=_COMPAT_SCAN_MAX_SIDECARS,
            include_nifti_headers=needs_nifti,
        )
        mb = budget.modality_budgets.get(modality) if modality else None
        profile = {
            "source_id": dataset_id,
            "source_type": "bids",
            "modality": modality,
            "n_channels": getattr(mb, "n_channels_mode", None) if mb else None,
            "sampling_rate_hz": getattr(mb, "sampling_rate_mode", None) if mb else None,
            "n_subjects": manifest.summary.n_subjects,
            "available_suffixes": list(manifest.summary.suffixes or []),
            "evidence_status": (EvidenceStatus.inferred if mb else EvidenceStatus.unknown).value,
        }
        _compat_disk_store(key, profile)
        return profile

    return _COMPAT_CACHE.get_or_compute(key, _build)


@app.get("/dataset/{dataset_id}/compatibility")
async def dataset_compatibility(
    dataset_id: str,
    model_id: Optional[str] = Query(None),
    snapshot: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Build a real ``SourceProfile`` from remotely-gathered signal-budget
    evidence (no download) and run it through the unmodified
    ``CompatibilityEngine`` against one or all catalog model contracts.
    The profile is cached per immutable (dataset, snapshot) — see
    ``_compat_source_profile`` — so only the first request pays the scan."""
    from qortex.neuroai.compatibility import CompatibilityEngine
    from qortex.neuroai.contracts import SourceProfile

    def _run() -> dict[str, Any]:
        source = SourceProfile(**_compat_source_profile(dataset_id, snapshot))
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


class CohortComparisonVariableBody(_PydanticModel):
    column: str
    kind: str


class CohortComparisonBody(_PydanticModel):
    dataset_ids: list[str]
    variables: list[CohortComparisonVariableBody]
    snapshots: Optional[dict[str, str]] = None
    alpha: float = 0.05


@app.post("/cohort/compare-participants")
async def cohort_compare_participants(body: CohortComparisonBody = Body(...)) -> dict[str, Any]:
    """Compare explicitly typed participant variables between two real datasets."""
    from qortex import Dataset
    from qortex.neuroclassic.cohort_comparison import compare_participant_cohorts

    if len(body.dataset_ids) != 2 or len(set(body.dataset_ids)) != 2:
        raise HTTPException(status_code=400, detail="Exactly two distinct dataset_ids are required")
    if not body.variables:
        raise HTTPException(status_code=400, detail="At least one comparison variable is required")
    if not 0.0 < body.alpha < 1.0:
        raise HTTPException(status_code=400, detail="alpha must be in (0, 1)")
    for variable in body.variables:
        if variable.kind not in {"numeric", "categorical"}:
            raise HTTPException(status_code=400, detail=f"Variable {variable.column!r} kind must be numeric or categorical")

    def _run() -> dict[str, Any]:
        cohorts: dict[str, list[dict[str, Any]]] = {}
        sources = []
        required_columns = {variable.column for variable in body.variables}
        for dataset_id in body.dataset_ids:
            requested_snapshot = (body.snapshots or {}).get(dataset_id)
            manifest = _manifest_for(dataset_id, requested_snapshot)
            dataset = Dataset(dataset_id, snapshot=manifest.snapshot, manifest=manifest)
            frame = dataset.participants()
            missing_columns = sorted(required_columns - set(frame.columns))
            if missing_columns:
                raise ValueError(f"{dataset_id} participants.tsv lacks columns {missing_columns}")
            cohorts[dataset_id] = frame.to_dicts()
            record = next((item for item in manifest.files if item.path == "participants.tsv"), None)
            sources.append({
                "dataset_id": dataset_id,
                "snapshot": manifest.snapshot,
                "path": record.path if record else "participants.tsv",
                "size_bytes": record.size if record else None,
                "checksum_md5": record.checksum if record else None,
                "rows": len(cohorts[dataset_id]),
                "columns": list(frame.columns),
            })
        return compare_participant_cohorts(
            cohorts,
            variables=[{"column": item.column, "kind": item.kind} for item in body.variables],
            alpha=body.alpha,
            sources=sources,
        )

    try:
        return to_jsonable(await call(_run))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── Model catalog ──────────────────────────────────────────────────────────

@app.get("/models")
async def list_models() -> list[dict[str, Any]]:
    return atlas_models.list_models()


@app.get("/models/status")
async def model_runtime_status() -> dict[str, Any]:
    return atlas_models.runtime_summary()


@app.get("/models/execution-profiles")
async def model_execution_profiles() -> list[dict[str, Any]]:
    from qortex.console.model_execution import list_model_execution_profiles

    return list_model_execution_profiles()


class ModelExecutionBody(_PydanticModel):
    profile_id: str
    parameters: Optional[dict[str, Any]] = None


@app.post("/models/execute-public")
async def execute_public_model_profile(body: ModelExecutionBody = Body(...)) -> dict[str, Any]:
    from qortex.console.model_execution import get_model_execution_profile, run_model_execution_profile

    try:
        profile = get_model_execution_profile(body.profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job = atlas_jobs.submit(
        profile["display_name"],
        run_model_execution_profile,
        body.profile_id,
        parameters=body.parameters or {},
        report_progress=True,
    )
    return {
        "job_id": job.id,
        "profile_id": body.profile_id,
        "result_contract": profile["result_contract"],
    }


@app.get("/models/cache")
async def model_cache_inventory() -> dict[str, Any]:
    from qortex.console.model_cache_control import model_cache_inventory as inspect_model_cache

    return await call(inspect_model_cache)


class ModelCacheRemovalBody(_PydanticModel):
    confirmation_sha256: str


@app.delete("/models/cache/{model_id}")
async def remove_model_cache_artifact(
    model_id: str,
    body: ModelCacheRemovalBody = Body(...),
) -> dict[str, Any]:
    from qortex.console.model_cache_control import move_model_artifact_to_trash

    try:
        return await call(
            move_model_artifact_to_trash,
            model_id,
            confirmation_sha256=body.confirmation_sha256,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class PublicBratsValidationBody(_PydanticModel):
    case_id: str = "BraTS-GLI-00000-000"
    device: str = "auto"


@app.post("/models/brats/validate-public")
async def start_public_brats_validation(
    body: PublicBratsValidationBody = Body(default=PublicBratsValidationBody()),
) -> dict[str, Any]:
    """Run pinned pretrained MONAI weights on a real public BraTS case."""
    from qortex.neuroai.public_validation import run_public_brats_validation

    if body.device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be auto, cpu, or cuda")
    job = atlas_jobs.submit(
        f"Validate pretrained BraTS model on {body.case_id}",
        run_public_brats_validation,
        case_id=body.case_id,
        device=body.device,
        report_progress=True,
    )
    return {"job_id": job.id, "case_id": body.case_id}


@app.get("/models/brats/runs/{run_id}")
async def get_public_brats_validation(run_id: str) -> dict[str, Any]:
    from qortex.neuroai.public_validation import load_public_brats_run

    return await call(load_public_brats_run, run_id)


@app.get("/models/brats/runs/{run_id}/artifacts/{artifact}")
async def get_public_brats_artifact(run_id: str, artifact: str) -> FileResponse:
    from qortex.neuroai.public_validation import public_brats_artifact_path

    path = await call(public_brats_artifact_path, run_id, artifact)
    media_type = "application/json" if path.suffix == ".json" else "application/gzip"
    return FileResponse(path, filename=path.name, media_type=media_type)


class PublicDetectionValidationBody(_PydanticModel):
    image_id: int = 397133
    device: str = "auto"
    score_threshold: float = 0.5
    iou_threshold: float = 0.5


@app.post("/models/detection/validate-public")
async def start_public_detection_validation(
    body: PublicDetectionValidationBody = Body(default=PublicDetectionValidationBody()),
) -> dict[str, Any]:
    """Run pinned pretrained Torchvision weights on a real COCO validation image."""
    from qortex.neuroai.public_detection import run_public_detection_validation

    if body.device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be auto, cpu, or cuda")
    if not 0.0 < body.score_threshold < 1.0:
        raise HTTPException(status_code=400, detail="score_threshold must be in (0, 1)")
    if not 0.0 < body.iou_threshold <= 1.0:
        raise HTTPException(status_code=400, detail="iou_threshold must be in (0, 1]")
    job = atlas_jobs.submit(
        f"Validate pretrained object detector on COCO val2017 image {body.image_id}",
        run_public_detection_validation,
        image_id=body.image_id,
        device=body.device,
        score_threshold=body.score_threshold,
        iou_threshold=body.iou_threshold,
        report_progress=True,
    )
    return {"job_id": job.id, "image_id": body.image_id}


@app.get("/models/detection/runs/{run_id}")
async def get_public_detection_validation(run_id: str) -> dict[str, Any]:
    from qortex.neuroai.public_detection import load_public_detection_run

    return await call(load_public_detection_run, run_id)


@app.get("/models/detection/runs/{run_id}/artifacts/{artifact}")
async def get_public_detection_artifact(run_id: str, artifact: str) -> FileResponse:
    from qortex.neuroai.public_detection import public_detection_artifact_path

    path = await call(public_detection_artifact_path, run_id, artifact)
    media_types = {".json": "application/json", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    return FileResponse(path, filename=path.name, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


class PublicRoiConnectivityBody(_PydanticModel):
    max_frames: int = 168
    fd_threshold_mm: float = 0.5
    std_dvars_threshold: Optional[float] = None
    connectivity_threshold: float = 0.3


@app.post("/analysis/roi-connectivity/validate-public")
async def start_public_roi_connectivity_validation(
    body: PublicRoiConnectivityBody = Body(default=PublicRoiConnectivityBody()),
) -> dict[str, Any]:
    """Validate MNI atlas extraction and ROI connectivity on public BOLD data."""
    from qortex.neuroclassic.public_roi_connectivity import run_public_roi_connectivity

    if not 20 <= body.max_frames <= 168:
        raise HTTPException(status_code=400, detail="max_frames must be in [20, 168]")
    if body.fd_threshold_mm < 0 or (
        body.std_dvars_threshold is not None and body.std_dvars_threshold < 0
    ):
        raise HTTPException(status_code=400, detail="scrubbing thresholds must be non-negative")
    if not 0 < body.connectivity_threshold < 1:
        raise HTTPException(status_code=400, detail="connectivity_threshold must be in (0, 1)")
    job = atlas_jobs.submit(
        "Validate Schaefer-100 ROI connectivity on public MNI BOLD",
        run_public_roi_connectivity,
        max_frames=body.max_frames,
        fd_threshold_mm=body.fd_threshold_mm,
        std_dvars_threshold=body.std_dvars_threshold,
        connectivity_threshold=body.connectivity_threshold,
        report_progress=True,
    )
    return {"job_id": job.id, "dataset_id": "development_fmri", "atlas_id": "Schaefer2018_100Parcels_7Networks"}


@app.get("/analysis/roi-connectivity/runs/{run_id}")
async def get_public_roi_connectivity_validation(run_id: str) -> dict[str, Any]:
    from qortex.neuroclassic.public_roi_connectivity import load_public_roi_connectivity_run

    return await call(load_public_roi_connectivity_run, run_id)


@app.get("/analysis/roi-connectivity/runs/{run_id}/artifacts/{artifact}")
async def get_public_roi_connectivity_artifact(run_id: str, artifact: str) -> FileResponse:
    from qortex.neuroclassic.public_roi_connectivity import public_roi_connectivity_artifact_path

    path = await call(public_roi_connectivity_artifact_path, run_id, artifact)
    media_types = {".json": "application/json", ".csv": "text/csv", ".png": "image/png", ".gz": "application/gzip"}
    return FileResponse(path, filename=path.name, media_type=media_types.get(path.suffix.lower(), "application/octet-stream"))


# ── Jobs ──────────────────────────────────────────────────────────────────

@app.get("/runs/persistent")
async def persistent_runs(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    """List artifact-backed scientific and conversion runs across restarts."""
    from qortex.console.run_inventory import persistent_run_inventory

    return await call(persistent_run_inventory, limit=limit)

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
