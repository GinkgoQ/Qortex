"""Manifest diff — compare two snapshots of the same dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from qortex.core.entities import FileRecord, Manifest


@dataclass(frozen=True)
class FileDiff:
    path: str
    change: Literal["added", "removed", "modified"]
    old: FileRecord | None = None
    new: FileRecord | None = None

    @property
    def size_delta(self) -> int:
        old_sz = self.old.size or 0 if self.old else 0
        new_sz = self.new.size or 0 if self.new else 0
        return new_sz - old_sz


@dataclass
class ManifestDiff:
    dataset_id: str
    from_snapshot: str
    to_snapshot: str
    added: list[FileRecord] = field(default_factory=list)
    removed: list[FileRecord] = field(default_factory=list)
    modified: list[FileDiff] = field(default_factory=list)

    @property
    def n_added(self) -> int:
        return len(self.added)

    @property
    def n_removed(self) -> int:
        return len(self.removed)

    @property
    def n_modified(self) -> int:
        return len(self.modified)

    @property
    def size_delta(self) -> int:
        added_sz = sum(f.size or 0 for f in self.added)
        removed_sz = sum(f.size or 0 for f in self.removed)
        modified_delta = sum(d.size_delta for d in self.modified)
        return added_sz - removed_sz + modified_delta

    def summary(self) -> str:
        return (
            f"Snapshot diff  {self.from_snapshot} → {self.to_snapshot}\n"
            f"  Added    : {self.n_added} files\n"
            f"  Removed  : {self.n_removed} files\n"
            f"  Modified : {self.n_modified} files\n"
            f"  Size Δ   : {self.size_delta / 1e6:+.1f} MB"
        )


def diff_manifests(old: Manifest, new: Manifest) -> ManifestDiff:
    """Compute file-level diff between two manifests of the same dataset."""
    if old.dataset_id != new.dataset_id:
        raise ValueError(
            f"Cannot diff manifests from different datasets: "
            f"{old.dataset_id!r} vs {new.dataset_id!r}"
        )

    old_by_path = {f.path: f for f in old.files if not f.is_dir}
    new_by_path = {f.path: f for f in new.files if not f.is_dir}

    old_paths = set(old_by_path)
    new_paths = set(new_by_path)

    added = [new_by_path[p] for p in sorted(new_paths - old_paths)]
    removed = [old_by_path[p] for p in sorted(old_paths - new_paths)]
    modified: list[FileDiff] = []

    for path in sorted(old_paths & new_paths):
        o = old_by_path[path]
        n = new_by_path[path]
        if _file_changed(o, n):
            modified.append(FileDiff(path=path, change="modified", old=o, new=n))

    return ManifestDiff(
        dataset_id=old.dataset_id,
        from_snapshot=old.snapshot,
        to_snapshot=new.snapshot,
        added=added,
        removed=removed,
        modified=modified,
    )


def _file_changed(old: FileRecord, new: FileRecord) -> bool:
    if old.size is not None and new.size is not None and old.size != new.size:
        return True
    if old.checksum and new.checksum and old.checksum != new.checksum:
        return True
    return False
