"""Small bounded previews of local or remote OpenNeuro files."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import httpx

from qortex.core.config import get_config
from qortex.core.entities import FilePreview, FileRecord, Manifest
from qortex.core.exceptions import DatasetNotDownloadedError, DownloadError

# A fresh httpx.Client() per preview call means a full new TCP connection
# and TLS handshake every time — for a UI where a user clicks through many
# small files in the same dataset (the BIDS explorer's exact use pattern),
# that's a full handshake repeated per click against the *same* CDN host,
# instead of reusing one. One process-wide client with keep-alive lets
# httpx reuse the connection, which is where most of the wall-clock time on
# a small byte-range read actually goes.
_shared_client: httpx.Client | None = None


def _get_shared_client() -> httpx.Client:
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.Client(
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40, keepalive_expiry=60.0),
        )
    return _shared_client


def preview_file(
    manifest: Manifest,
    path: str,
    *,
    local_path: str | Path | None = None,
    n_rows: int = 5,
    max_bytes: int = 64_000,
    timeout_s: float | None = None,
) -> FilePreview:
    """Return a small preview without forcing a full dataset download."""
    file = manifest.get_file(path)
    if file is None:
        raise FileNotFoundError(f"File not found in manifest: {path}")

    if local_path is not None:
        candidate = Path(local_path).expanduser().resolve() / file.path
        if candidate.exists():
            data = _read_local_prefix(candidate, max_bytes)
            return _build_preview(
                manifest=manifest,
                file=file,
                data=data,
                source="local",
                n_rows=n_rows,
                content_type=None,
            )

    data, content_type = _read_remote_prefix(
        file,
        max_bytes=max_bytes,
        timeout_s=timeout_s,
    )
    return _build_preview(
        manifest=manifest,
        file=file,
        data=data,
        source="remote",
        n_rows=n_rows,
        content_type=content_type,
    )


def preview_metadata(
    manifest: Manifest,
    *,
    local_path: str | Path | None = None,
    n_rows: int = 5,
    max_files: int | None = None,
) -> list[FilePreview]:
    """Preview essential BIDS metadata and sidecar tables."""
    metadata_files = [
        file for file in manifest.files
        if not file.is_dir
        and (
            file.is_essential
            or file.extension in {".json", ".tsv", ".csv", ".bvec", ".bval"}
        )
    ]
    if max_files is not None:
        metadata_files = metadata_files[:max_files]
    previews: list[FilePreview] = []
    for file in metadata_files:
        try:
            previews.append(
                preview_file(
                    manifest,
                    file.path,
                    local_path=local_path,
                    n_rows=n_rows,
                )
            )
        except Exception:
            continue
    return previews


def _read_local_prefix(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as f:
        return f.read(max_bytes)


def _read_remote_prefix(
    file: FileRecord,
    *,
    max_bytes: int,
    timeout_s: float | None,
) -> tuple[bytes, str | None]:
    if not file.urls:
        raise DownloadError(file.path, "", "No download URL available for preview.")
    cfg = get_config()
    timeout = timeout_s or cfg.metadata_timeout
    headers = {"Range": f"bytes=0-{max(0, max_bytes - 1)}"}
    client = _get_shared_client()
    with client.stream("GET", file.urls[0], headers=headers, timeout=timeout) as response:
        if response.is_error:
            raise DownloadError(
                file.path,
                file.urls[0],
                f"preview GET returned HTTP {response.status_code}",
            )
        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            remaining = max_bytes - total
            if remaining <= 0:
                break
            chunks.append(chunk[:remaining])
            total += min(len(chunk), remaining)
            if total >= max_bytes:
                break
        return b"".join(chunks), response.headers.get("content-type")


def _build_preview(
    *,
    manifest: Manifest,
    file: FileRecord,
    data: bytes,
    source: str,
    n_rows: int,
    content_type: str | None,
) -> FilePreview:
    text = data.decode("utf-8", errors="replace")
    rows: list[dict] = []
    columns: list[str] = []
    if file.extension in {".tsv", ".csv"}:
        sep = "\t" if file.extension == ".tsv" else ","
        rows, columns = _parse_rows(text, sep=sep, n_rows=n_rows)
    elif file.extension == ".json":
        text = _format_json_preview(text)
    truncated = file.size is not None and len(data) < file.size
    return FilePreview(
        dataset_id=manifest.dataset_id,
        snapshot=manifest.snapshot,
        path=file.path,
        source=source,  # type: ignore[arg-type]
        bytes_read=len(data),
        truncated=truncated,
        columns=columns,
        rows=rows,
        text=text if not rows else None,
        content_type=content_type,
    )


def _parse_rows(text: str, *, sep: str, n_rows: int) -> tuple[list[dict], list[str]]:
    reader = csv.DictReader(io.StringIO(text), delimiter=sep)
    columns = reader.fieldnames or []
    rows: list[dict] = []
    for row in reader:
        rows.append(dict(row))
        if len(rows) >= n_rows:
            break
    return rows, columns


def _format_json_preview(text: str) -> str:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(data, indent=2, sort_keys=True)


def require_downloaded_path(local_path: str | Path | None, dataset_id: str, snapshot: str | None) -> Path:
    if local_path is None:
        raise DatasetNotDownloadedError(dataset_id, snapshot)
    root = Path(local_path).expanduser().resolve()
    if not root.exists():
        raise DatasetNotDownloadedError(dataset_id, snapshot)
    return root
