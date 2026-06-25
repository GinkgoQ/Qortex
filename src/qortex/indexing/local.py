"""Local BIDS index and OpenNeuro manifest reconciliation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.core.entities import LocalFileRecord, LocalIndexReport, Manifest
from qortex.manifest.bids import _extract_extension, parse_filename


class LocalBIDSIndexer:
    """Index a local BIDS directory and compare it with a remote manifest."""

    def __init__(self, *, use_pybids: bool = True) -> None:
        self.use_pybids = use_pybids

    def index(
        self,
        dataset_path: str | Path,
        *,
        manifest: Manifest | None = None,
        include_dirs: bool = False,
    ) -> LocalIndexReport:
        root = Path(dataset_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Local BIDS path does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Local BIDS path is not a directory: {root}")

        records = self._index_with_pybids(root) if self.use_pybids else None
        if records is None:
            records = self._index_filesystem(root, include_dirs=include_dirs)

        file_records = [record for record in records if not record.is_dir]
        dir_records = [record for record in records if record.is_dir]
        report = LocalIndexReport(
            dataset_path=str(root),
            n_files=len(file_records),
            n_dirs=len(dir_records),
            indexed_files=records,
        )
        if manifest is None:
            return report
        return self._reconcile(report, manifest)

    def _index_with_pybids(self, root: Path) -> list[LocalFileRecord] | None:
        try:
            from bids import BIDSLayout
        except ImportError:
            return None

        layout = BIDSLayout(str(root), validate=False, derivatives=True)
        paths = sorted(Path(path) for path in layout.get(return_type="filename"))
        records: list[LocalFileRecord] = []
        for path in paths:
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            stat = path.stat()
            try:
                entities = layout.parse_file_entities(str(path))
            except Exception:
                entities = _parse_entities(path.name)
            records.append(
                LocalFileRecord(
                    path=rel,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    extension=_extract_extension(path.name),
                    entities=dict(entities),
                )
            )
        return records

    def _index_filesystem(
        self,
        root: Path,
        *,
        include_dirs: bool,
    ) -> list[LocalFileRecord]:
        records: list[LocalFileRecord] = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if _is_qortex_internal(rel):
                continue
            if path.is_dir():
                if include_dirs:
                    stat = path.stat()
                    records.append(
                        LocalFileRecord(
                            path=rel,
                            size=0,
                            mtime=stat.st_mtime,
                            is_dir=True,
                            extension=None,
                            entities={},
                        )
                    )
                continue
            stat = path.stat()
            records.append(
                LocalFileRecord(
                    path=rel,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    extension=_extract_extension(path.name),
                    entities=_parse_entities(path.name),
                )
            )
        return records

    def _reconcile(
        self,
        report: LocalIndexReport,
        manifest: Manifest,
    ) -> LocalIndexReport:
        local_by_path = {
            record.path: record
            for record in report.indexed_files
            if not record.is_dir
        }
        remote_by_path = {
            file.path: file
            for file in manifest.files
            if not file.is_dir
        }
        local_paths = set(local_by_path)
        remote_paths = set(remote_by_path)
        missing = sorted(remote_paths - local_paths)
        extra = sorted(
            path for path in local_paths - remote_paths
            if not _is_qortex_internal(path)
        )
        size_mismatches = []
        for path in sorted(local_paths & remote_paths):
            expected = remote_by_path[path].size
            if expected is not None and local_by_path[path].size != expected:
                size_mismatches.append(path)

        return report.model_copy(
            update={
                "missing_remote": missing,
                "extra_local": extra,
                "size_mismatches": size_mismatches,
            }
        )


def index_local_bids(
    dataset_path: str | Path,
    *,
    manifest: Manifest | None = None,
    include_dirs: bool = False,
    use_pybids: bool = True,
) -> LocalIndexReport:
    """Convenience wrapper around :class:`LocalBIDSIndexer`."""
    return LocalBIDSIndexer(use_pybids=use_pybids).index(
        dataset_path,
        manifest=manifest,
        include_dirs=include_dirs,
    )


def _parse_entities(filename: str) -> dict[str, Any]:
    parsed = parse_filename(filename)
    parsed.pop("_extra", None)
    return {key: value for key, value in parsed.items() if value is not None}


def _is_qortex_internal(path: str) -> bool:
    return path == ".qortex" or path.startswith(".qortex/")
