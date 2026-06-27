"""Manifest builder — converts raw OpenNeuro API file records into a Manifest.

The builder is the only place where raw API dicts are touched; everything
downstream works with typed ``FileRecord`` and ``Manifest`` objects.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from qortex.core.entities import (
    BIDSEntities,
    FileRecord,
    Manifest,
    ManifestSummary,
    SnapshotRef,
)
from qortex.core.exceptions import ManifestError
from qortex.manifest.bids import (
    extract_datatype,
    infer_modality,
    parse_filename,
    sidecar_group_key,
    _extract_extension,
)


class ManifestBuilder:
    """Build a ``Manifest`` from raw API file dicts."""

    def build(
        self,
        dataset_id: str,
        snapshot_ref: SnapshotRef,
        raw_files: list[dict],
    ) -> Manifest:
        if not raw_files:
            raise ManifestError(
                f"OpenNeuro returned an empty file list for "
                f"{dataset_id!r} (snapshot {snapshot_ref.tag!r}). "
                f"The snapshot may be empty or the API request may have failed."
            )

        records = [self._parse_record(raw) for raw in raw_files]

        # Attach sidecar group keys (cheap — no I/O)
        records = self._assign_sidecar_groups(records)

        summary = self._build_summary(records)

        return Manifest(
            dataset_id=dataset_id,
            snapshot=snapshot_ref.tag,
            doi=snapshot_ref.doi,
            files=records,
            summary=summary,
            built_at=datetime.now(timezone.utc),
        )

    # ── Record parsing ────────────────────────────────────────────────────

    def _parse_record(self, raw: dict) -> FileRecord:
        path: str = raw.get("filename") or raw.get("path") or ""
        filename = PurePosixPath(path).name
        extension = _extract_extension(filename)
        is_dir = bool(raw.get("directory", False))

        parsed = parse_filename(filename) if filename else {}
        extra: dict = parsed.pop("_extra", {})
        suffix: str = parsed.get("suffix") or ""
        datatype = extract_datatype(path) if not is_dir else None
        modality = infer_modality(datatype, suffix) if not is_dir else None

        entities = BIDSEntities(
            subject=parsed.get("subject"),
            session=parsed.get("session"),
            task=parsed.get("task"),
            run=parsed.get("run"),
            acquisition=parsed.get("acquisition"),
            direction=parsed.get("direction"),
            space=parsed.get("space"),
            resolution=parsed.get("resolution"),
            echo=parsed.get("echo"),
            part=parsed.get("part"),
            hemisphere=parsed.get("hemisphere"),
            density=parsed.get("density"),
            processing=parsed.get("processing"),
            split=parsed.get("split"),
            extra=extra,
        )

        urls = raw.get("urls") or []
        if isinstance(urls, str):
            urls = [urls]

        return FileRecord(
            id=raw.get("id") or path,
            path=path,
            filename=filename,
            extension=extension,
            size=raw.get("size"),
            urls=[u for u in urls if u],
            is_dir=is_dir,
            datatype=datatype,
            suffix=suffix or None,
            modality=modality,
            entities=entities,
        )

    # ── Sidecar groups ────────────────────────────────────────────────────

    def _assign_sidecar_groups(self, records: list[FileRecord]) -> list[FileRecord]:
        updated = []
        for rec in records:
            if rec.is_dir:
                updated.append(rec)
                continue
            key = sidecar_group_key(rec.path)
            updated.append(rec.model_copy(update={"sidecar_group": key}))
        return updated

    # ── Summary ───────────────────────────────────────────────────────────

    def _build_summary(self, records: list[FileRecord]) -> ManifestSummary:
        non_dir = [r for r in records if not r.is_dir]
        dirs = [r for r in records if r.is_dir]

        subjects: set[str] = set()
        sessions: set[str] = set()
        tasks: set[str] = set()
        modalities: set[str] = set()
        datatypes: set[str] = set()
        suffixes: set[str] = set()
        total_size = 0
        size_known = True

        for f in non_dir:
            if f.entities.subject:
                subjects.add(f"sub-{f.entities.subject}")
            if f.entities.session:
                sessions.add(f"ses-{f.entities.session}")
            if f.entities.task:
                tasks.add(f.entities.task)
            if f.modality:
                modalities.add(f.modality)
            if f.datatype:
                datatypes.add(f.datatype)
            if f.suffix:
                suffixes.add(f.suffix)
            if f.size is not None:
                total_size += f.size
            else:
                size_known = False

        filenames = {f.filename for f in non_dir}

        return ManifestSummary(
            subjects=sorted(subjects),
            sessions=sorted(sessions),
            tasks=sorted(tasks),
            modalities=sorted(modalities),
            datatypes=sorted(datatypes),
            suffixes=sorted(suffixes),
            file_count=len(non_dir),
            dir_count=len(dirs),
            total_size=total_size,
            total_size_known=size_known,
            has_derivatives=any(f.path.startswith("derivatives/") for f in non_dir),
            has_bidsignore=".bidsignore" in filenames,
            has_events=any(f.suffix == "events" for f in non_dir),
            has_participants_tsv="participants.tsv" in filenames,
        )


# ── Persistence helpers ───────────────────────────────────────────────────────

def save_manifest(manifest: Manifest, directory: Path) -> None:
    """Persist manifest to *directory* as Parquet + JSON sidecar.

    The Parquet file stores the full file list for fast programmatic access.
    The JSON sidecar stores summary + metadata for human inspection.

    All BIDSEntities fields (including acquisition, direction, space,
    resolution, echo, part, hemisphere, density, processing, split, and
    all non-standard extras) are preserved verbatim via entities_json so that
    reloaded manifests are bit-for-bit equivalent to freshly built ones.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # ── Parquet (file records) ────────────────────────────────────────────
    rows = []
    for f in manifest.files:
        row = f.model_dump()
        # Store the full BIDSEntities model as JSON — zero field loss.
        # The individual flat columns (sub/ses/task/run) are kept as
        # fast-filter shortcuts for downstream SQL/DuckDB queries.
        ent = row.pop("entities", {})
        row["sub"] = ent.get("subject")
        row["ses"] = ent.get("session")
        row["task"] = ent.get("task")
        row["run"] = ent.get("run")
        row["entities_json"] = json.dumps(ent, ensure_ascii=False)
        rows.append(row)

    _write_rows_parquet(rows, directory / "manifest.parquet")

    # ── JSON sidecar (summary + header) ──────────────────────────────────
    meta = {
        "dataset_id": manifest.dataset_id,
        "snapshot": manifest.snapshot,
        "doi": manifest.doi,
        "built_at": manifest.built_at.isoformat(),
        "summary": manifest.summary.model_dump(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )


def load_manifest(directory: Path) -> Manifest:
    """Load a previously saved manifest from *directory*."""
    json_path = directory / "manifest.json"
    parquet_path = directory / "manifest.parquet"

    if not json_path.exists() or not parquet_path.exists():
        raise ManifestError(
            f"Manifest not found in {directory}. "
            f"Call Dataset.manifest() to build it."
        )

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    summary = ManifestSummary(**meta["summary"])

    records: list[FileRecord] = []
    for row in _read_rows_parquet(parquet_path):
        entities = _deserialize_entities(row)
        # Remove all entity-related columns before passing to FileRecord
        for col in ("sub", "ses", "task", "run", "entities_json", "entities_extra", "entities"):
            row.pop(col, None)
        records.append(FileRecord(**row, entities=entities))

    return Manifest(
        dataset_id=meta["dataset_id"],
        snapshot=meta["snapshot"],
        doi=meta.get("doi"),
        files=records,
        summary=summary,
        built_at=datetime.fromisoformat(meta["built_at"]),
    )


def _deserialize_entities(row: dict) -> BIDSEntities:
    """Reconstruct a BIDSEntities from a Parquet row, handling both legacy
    and current serialisation formats.

    Current format (v ≥ 0.3): ``entities_json`` holds the full model_dump().
    Legacy format (v < 0.3): individual ``sub``/``ses``/``task``/``run`` columns
    plus ``entities_extra`` for non-standard entities.  Loaded manifests from
    old caches are transparently upgraded; a fresh save will write entities_json.
    """
    if "entities_json" in row and row["entities_json"]:
        try:
            raw = json.loads(row["entities_json"])
            # BIDSEntities.model_dump() stores "extra" as a nested dict — reconstruct directly.
            return BIDSEntities(**{k: v for k, v in raw.items() if k in BIDSEntities.model_fields})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # corrupt entry — fall through to legacy reconstruction

    # Legacy: reconstruct from individual columns
    return BIDSEntities(
        subject=row.get("sub"),
        session=row.get("ses"),
        task=row.get("task"),
        run=row.get("run"),
        extra=json.loads(row.get("entities_extra", "{}") or "{}"),
    )


def _write_rows_parquet(rows: list[dict], path: Path) -> None:
    try:
        import polars as pl
    except ImportError:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ManifestError(
                "Saving manifests requires either polars or pyarrow."
            ) from exc
        pq.write_table(pa.Table.from_pylist(rows), path)
    else:
        pl.DataFrame(rows, infer_schema_length=len(rows)).write_parquet(path)


def _read_rows_parquet(path: Path) -> list[dict]:
    try:
        import polars as pl
    except ImportError:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ManifestError(
                "Loading saved manifests requires either polars or pyarrow."
            ) from exc
        return pq.read_table(path).to_pylist()
    return pl.read_parquet(path).to_dicts()
