"""Build and persist ProvenanceRecord for conversion runs."""

from __future__ import annotations

import json
from pathlib import Path

from qortex.core.entities import ProvenanceRecord
from qortex._version import __version__


def build_provenance(
    dataset_id: str,
    snapshot: str,
    operation: str,
    output_path: str | Path | None = None,
    config: dict | None = None,
    source_files: list[str] | None = None,
    doi: str | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        qortex_version=__version__,
        dataset_id=dataset_id,
        snapshot=snapshot,
        doi=doi,
        operation=operation,  # type: ignore[arg-type]
        config=config or {},
        source_files=source_files or [],
        output_path=str(output_path) if output_path else None,
    )


def save_provenance(record: ProvenanceRecord, output_dir: Path) -> Path:
    """Persist a ProvenanceRecord as JSON alongside the converted artifact."""
    path = output_dir / "qortex_provenance.json"
    payload = {
        "qortex_version": record.qortex_version,
        "created_at": record.created_at.isoformat(),
        "dataset_id": record.dataset_id,
        "snapshot": record.snapshot,
        "doi": record.doi,
        "operation": record.operation,
        "config": record.config,
        "source_files": record.source_files,
        "output_path": record.output_path,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_provenance(output_dir: Path) -> ProvenanceRecord:
    """Load a previously saved ProvenanceRecord."""
    path = output_dir / "qortex_provenance.json"
    if not path.exists():
        raise FileNotFoundError(f"No provenance record found in {output_dir}")
    data = json.loads(path.read_text())
    return ProvenanceRecord(**data)
