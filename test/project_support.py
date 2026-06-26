"""Shared helpers for Qortex scenario projects.

Import with:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from project_support import require, DATASET_ID, real_manifest, ...
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import qortex


DATASET_ID = os.environ.get("QORTEX_REAL_TEST_DATASET", "ds000001")
SNAPSHOT = os.environ.get("QORTEX_REAL_TEST_SNAPSHOT") or None
_SHARED_METADATA_ENV = "QORTEX_REAL_METADATA_ROOT"
_SHARED_ARTIFACT_ENV = "QORTEX_REAL_ARTIFACT_ROOT"


# ── Assertion ─────────────────────────────────────────────────────────────────

def require(condition: bool, message: str) -> None:
    """Raise RuntimeError with message if condition is False."""
    if not condition:
        raise RuntimeError(f"ASSERTION FAILED: {message}")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"ASSERTION FAILED [{label}]: expected {expected!r}, got {actual!r}")


def require_type(value: Any, typ: type, label: str) -> None:
    if not isinstance(value, typ):
        raise RuntimeError(
            f"ASSERTION FAILED [{label}]: expected type {typ.__name__}, got {type(value).__name__}"
        )


def require_gt(actual: Any, threshold: Any, label: str) -> None:
    if not (actual > threshold):
        raise RuntimeError(f"ASSERTION FAILED [{label}]: expected > {threshold}, got {actual!r}")


def require_in(value: Any, collection: Any, label: str) -> None:
    if value not in collection:
        raise RuntimeError(f"ASSERTION FAILED [{label}]: {value!r} not in {collection!r}")


# ── Output helpers ────────────────────────────────────────────────────────────

def print_kv(label: str, value: Any) -> None:
    """Print a key-value pair or a flat dict."""
    if isinstance(value, dict):
        width = max((len(str(k)) for k in value), default=0)
        print(f"\n  {label}:")
        for k, v in value.items():
            print(f"    {str(k).ljust(width)} : {v}")
    else:
        print(f"  {label}: {value}")


def print_rows(label: str, rows: list[dict[str, Any]], *, limit: int = 8) -> None:
    """Print a list of dicts as a table."""
    print(f"\n  {label}:")
    if not rows:
        print("    (no rows)")
        return
    rows = rows[:limit]
    columns = list(rows[0])
    widths = {
        col: min(70, max(len(col), *(len(_cell(r.get(col, ""))) for r in rows)))
        for col in columns
    }
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(f"    {header}")
    print(f"    {sep}")
    for row in rows:
        cells = " | ".join(_cell(row.get(col, "")).ljust(widths[col])[:widths[col]] for col in columns)
        print(f"    {cells}")
    if len(rows) < limit:
        pass
    else:
        print(f"    ... (showing {limit} of more)")


def section(title: str) -> None:
    print(f"\n--- {title} ---", flush=True)


def banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}", flush=True)


def passed(name: str) -> None:
    print(f"\n[PASS] {name}", flush=True)


def fail(message: str) -> None:
    print(f"\n[FAIL] {message}", file=sys.stderr, flush=True)
    sys.exit(1)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def real_dataset() -> qortex.Dataset:
    return qortex.Dataset(DATASET_ID, snapshot=SNAPSHOT)


def real_manifest():
    """Return (Dataset, Manifest) for the test dataset."""
    ds = real_dataset()
    return ds, ds.manifest()


def real_metadata_root() -> tuple[Any, qortex.Dataset, Path]:
    """Return (cleanup_ctx, Dataset, local_path) with metadata downloaded.

    Reuses an existing download from QORTEX_REAL_METADATA_ROOT when set, so
    multiple scenario projects share one download across a full suite run.
    """
    shared = os.environ.get(_SHARED_METADATA_ENV)
    if shared:
        ctx = _NoCleanup()
        root = Path(shared).expanduser().resolve() / DATASET_ID
    else:
        ctx = tempfile.TemporaryDirectory()
        root = Path(ctx.name) / DATASET_ID

    ds = real_dataset()
    if not _metadata_tree_ready(root):
        root.parent.mkdir(parents=True, exist_ok=True)
        result = ds.download_metadata(output_dir=root, max_size_gb=0.2)
        require(result.success, f"metadata download failed: {result.report()}")
        require(result.plan.n_files > 0, "metadata download plan was empty")
    else:
        ds._data_dir = root
    return ctx, ds, root


def artifact_dir(root: Path, name: str) -> Path:
    """Return a writable directory for artifact output.

    Respects QORTEX_REAL_ARTIFACT_ROOT so the suite runner can place all
    artifacts in one shared location.
    """
    configured = os.environ.get(_SHARED_ARTIFACT_ENV)
    base = Path(configured) if configured else root.parent / "artifacts"
    path = base.expanduser().resolve() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Manifest search helpers ───────────────────────────────────────────────────

def first_events_file(manifest):
    for f in manifest.files:
        if f.suffix == "events" and f.extension == ".tsv":
            return f
    raise RuntimeError(f"No events.tsv in manifest for {manifest.dataset_id}")


def first_metadata_table(manifest):
    for name in ("participants.tsv", "sessions.tsv"):
        f = manifest.get_file(name)
        if f is not None:
            return f
    for f in manifest.files:
        if f.extension in {".tsv", ".csv"} and not f.is_dir:
            return f
    raise RuntimeError(f"No metadata table in manifest for {manifest.dataset_id}")


def primary_recording_with_events(manifest):
    from qortex.manifest.graph import ManifestGraph
    for rec in ManifestGraph(manifest).recordings():
        if rec.companions.events is not None and rec.primary.urls:
            return rec
    raise RuntimeError("No event-complete recording in manifest")


def downloaded_events_file(root: Path) -> Path:
    for path in sorted(root.rglob("*_events.tsv")):
        return path
    raise RuntimeError(f"No downloaded events.tsv under {root}")


# ── Split helpers ─────────────────────────────────────────────────────────────

def split_subject_rows(train, val, test) -> list[dict[str, Any]]:
    rows = []
    for split_name, samples in (("train", train), ("val", val), ("test", test)):
        subjects = sorted({s.subject for s in samples if s.subject})
        rows.append({
            "split": split_name,
            "subjects": ", ".join(subjects),
            "n_subjects": len(subjects),
            "n_samples": len(samples),
        })
    return rows


# ── Internal ──────────────────────────────────────────────────────────────────

def _cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _metadata_tree_ready(root: Path) -> bool:
    return (
        root.exists()
        and (root / "dataset_description.json").exists()
        and any(root.rglob("*_events.tsv"))
    )


class _NoCleanup:
    name = ""
    def cleanup(self) -> None:
        pass
