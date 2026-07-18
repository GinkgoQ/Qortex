"""Versioned, source-bound viewer annotation documents."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._+-]+$")
_LOCK = threading.RLock()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnnotationMeasurement(_StrictModel):
    id: str = Field(min_length=1, max_length=128)
    kind: Literal["distance", "roi"]
    distance_mm: float | None = Field(default=None, ge=0)
    start_voxel: list[int] | None = Field(default=None, min_length=3, max_length=3)
    end_voxel: list[int] | None = Field(default=None, min_length=3, max_length=3)
    mean: float | None = None
    std: float | None = Field(default=None, ge=0)
    min: float | None = None
    max: float | None = None
    voxel_count: int | None = Field(default=None, ge=0)
    volume_mm3: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_kind_contract(self) -> "AnnotationMeasurement":
        if self.kind == "distance" and self.distance_mm is None:
            raise ValueError("distance measurement requires distance_mm")
        if self.kind == "roi" and (self.start_voxel is None or self.end_voxel is None):
            raise ValueError("ROI measurement requires start_voxel and end_voxel")
        for name in ("distance_mm", "mean", "std", "min", "max", "volume_mm3"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class AnnotationBookmark(_StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    voxel: list[int] = Field(min_length=3, max_length=3)
    world_mm: list[float] = Field(min_length=3, max_length=3)
    frame: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def finite_world_coordinates(self) -> "AnnotationBookmark":
        if not all(math.isfinite(value) for value in self.world_mm):
            raise ValueError("bookmark world_mm values must be finite")
        return self


class AnnotationLayer(_StrictModel):
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=200)
    kind: Literal["measurements", "rois", "bookmarks"]
    visible: bool = True
    color: str | None = Field(default=None, max_length=32)
    item_ids: list[str] = Field(default_factory=list, max_length=10_000)


class AnnotationViewport(_StrictModel):
    crosshair_voxel: list[int] = Field(min_length=3, max_length=3)
    crosshair_world_mm: list[float] = Field(min_length=3, max_length=3)
    frame: int = Field(default=0, ge=0)
    layout: Literal["axial", "coronal", "sagittal", "single", "row", "grid", "mpr", "render"] = "grid"
    cal_min: float
    cal_max: float

    @model_validator(mode="after")
    def finite_viewport(self) -> "AnnotationViewport":
        values = [*self.crosshair_world_mm, self.cal_min, self.cal_max]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("viewport coordinates and calibration must be finite")
        if self.cal_min >= self.cal_max:
            raise ValueError("viewport cal_min must be less than cal_max")
        return self


class AnnotationPayload(_StrictModel):
    title: str = Field(min_length=1, max_length=200)
    layers: list[AnnotationLayer] = Field(default_factory=list, max_length=50)
    measurements: list[AnnotationMeasurement] = Field(default_factory=list, max_length=10_000)
    bookmarks: list[AnnotationBookmark] = Field(default_factory=list, max_length=1_000)
    viewport: AnnotationViewport

    @model_validator(mode="after")
    def validate_layer_references(self) -> "AnnotationPayload":
        measurement_ids = [item.id for item in self.measurements]
        bookmark_ids = [item.id for item in self.bookmarks]
        if len(set(measurement_ids)) != len(measurement_ids):
            raise ValueError("measurement ids must be unique")
        if len(set(bookmark_ids)) != len(bookmark_ids):
            raise ValueError("bookmark ids must be unique")
        layer_ids = [item.id for item in self.layers]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("layer ids must be unique")
        allowed_by_kind = {
            "measurements": {item.id for item in self.measurements if item.kind == "distance"},
            "rois": {item.id for item in self.measurements if item.kind == "roi"},
            "bookmarks": set(bookmark_ids),
        }
        referenced: set[str] = set()
        for layer in self.layers:
            unknown = set(layer.item_ids) - allowed_by_kind[layer.kind]
            if unknown:
                raise ValueError(f"layer {layer.id!r} references incompatible or unknown items {sorted(unknown)}")
            duplicate = referenced & set(layer.item_ids)
            if duplicate:
                raise ValueError(f"items may belong to only one layer; duplicates {sorted(duplicate)}")
            referenced.update(layer.item_ids)
        return self


def _safe_component(value: str, name: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(f"{name} contains unsupported characters")
    return value


def _root(store_root: Path | str | None) -> Path:
    return (Path(store_root) if store_root else Path.home() / ".qortex" / "annotations").resolve()


def _document_dir(root: Path, dataset_id: str, snapshot: str, annotation_id: str) -> Path:
    for value, name in ((dataset_id, "dataset_id"), (snapshot, "snapshot"), (annotation_id, "annotation_id")):
        _safe_component(value, name)
    path = (root / dataset_id / snapshot / annotation_id).resolve()
    if root not in path.parents:
        raise ValueError("annotation path escapes the configured store")
    return path


@contextmanager
def _store_lock(root: Path) -> Iterator[None]:
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".lock"
    with _LOCK, lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def save_annotation(
    *,
    dataset_id: str,
    snapshot: str,
    source: dict[str, Any],
    payload: dict[str, Any],
    annotation_id: str | None = None,
    expected_revision: int | None = None,
    store_root: Path | str | None = None,
) -> dict[str, Any]:
    validated = AnnotationPayload.model_validate(payload)
    root = _root(store_root)
    identifier = annotation_id or uuid.uuid4().hex
    doc_dir = _document_dir(root, dataset_id, snapshot, identifier)
    with _store_lock(root):
        head_path = doc_dir / "head.json"
        current = json.loads(head_path.read_text(encoding="utf-8")) if head_path.is_file() else None
        current_revision = int(current["revision"]) if current else 0
        if expected_revision is not None and expected_revision != current_revision:
            raise RuntimeError(
                f"annotation revision conflict: expected {expected_revision}, current {current_revision}"
            )
        if current is not None:
            current_source = current.get("source", {})
            identity_fields = ("path", "size_bytes", "checksum", "snapshot")
            if any(current_source.get(field) != source.get(field) for field in identity_fields):
                raise ValueError("annotation source identity cannot change across revisions")
        revision = current_revision + 1
        created_at = current.get("created_at") if current else datetime.now(timezone.utc).isoformat()
        document = {
            "schema": "qortex.viewer-annotation/v1",
            "annotation_id": identifier,
            "dataset_id": dataset_id,
            "snapshot": snapshot,
            "revision": revision,
            "created_at": created_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            **validated.model_dump(mode="json"),
        }
        version_path = doc_dir / "versions" / f"{revision:08d}.json"
        if version_path.exists():
            raise FileExistsError(f"annotation version {revision} already exists")
        _atomic_json(version_path, document)
        try:
            _atomic_json(head_path, document)
        except Exception:
            version_path.unlink(missing_ok=True)
            raise
    return document


def load_annotation(
    dataset_id: str,
    snapshot: str,
    annotation_id: str,
    *,
    revision: int | None = None,
    store_root: Path | str | None = None,
) -> dict[str, Any]:
    root = _root(store_root)
    doc_dir = _document_dir(root, dataset_id, snapshot, annotation_id)
    path = doc_dir / "head.json" if revision is None else doc_dir / "versions" / f"{revision:08d}.json"
    resolved = path.resolve()
    if doc_dir not in resolved.parents or not resolved.is_file():
        raise FileNotFoundError(f"No annotation {annotation_id!r} revision {revision!r}")
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Annotation {resolved} is not a JSON object")
    return document


def list_annotations(
    dataset_id: str,
    snapshot: str,
    *,
    source_path: str | None = None,
    store_root: Path | str | None = None,
) -> dict[str, Any]:
    root = _root(store_root)
    _safe_component(dataset_id, "dataset_id")
    _safe_component(snapshot, "snapshot")
    parent = (root / dataset_id / snapshot).resolve()
    if root not in parent.parents:
        raise ValueError("annotation path escapes the configured store")
    rows: list[dict[str, Any]] = []
    if parent.is_dir():
        for path in parent.glob("*/head.json"):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                if source_path is not None and document.get("source", {}).get("path") != source_path:
                    continue
                rows.append({
                    key: document.get(key)
                    for key in ("annotation_id", "title", "revision", "created_at", "updated_at", "source")
                } | {
                    "measurement_count": len(document.get("measurements", [])),
                    "bookmark_count": len(document.get("bookmarks", [])),
                    "layer_count": len(document.get("layers", [])),
                })
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return {"dataset_id": dataset_id, "snapshot": snapshot, "source_path": source_path, "annotations": rows}


__all__ = [
    "AnnotationPayload",
    "list_annotations",
    "load_annotation",
    "save_annotation",
]
