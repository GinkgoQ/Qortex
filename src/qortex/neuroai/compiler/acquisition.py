"""Source acquisition planning for NeuroAI compilation."""

from __future__ import annotations

from pathlib import Path

from qortex.core.entities import FileRecord, Manifest
from qortex.manifest.bids import _extract_extension, extract_datatype, infer_modality, parse_entities, parse_filename
from qortex.manifest.graph import ManifestGraph
from qortex.neuroai.contracts import BaseModel, EvidenceStatus, Field


class AcquisitionPlan(BaseModel):
    source: str
    source_type: str
    required_download: bool
    estimated_download_gb: float | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    n_recordings: int | None = None
    required_files: list[str] = Field(default_factory=list)
    companion_files: list[str] = Field(default_factory=list)


def build_acquisition_plan(
    *,
    source: str,
    source_type: str,
    local_size_bytes: int | None,
    max_download_gb: float | None,
) -> AcquisitionPlan:
    if source_type.startswith("local_"):
        return AcquisitionPlan(
            source=source,
            source_type=source_type,
            required_download=False,
            estimated_download_gb=0.0,
            evidence_status=EvidenceStatus.confirmed,
        )

    plan = AcquisitionPlan(
        source=source,
        source_type=source_type,
        required_download=True,
        estimated_download_gb=(local_size_bytes / 1e9) if local_size_bytes is not None else None,
        evidence_status=EvidenceStatus.unknown,
        notes=["Remote source size is not known without manifest inspection; no download is performed by compile."],
    )
    if (
        max_download_gb is not None
        and plan.estimated_download_gb is not None
        and plan.estimated_download_gb > max_download_gb
    ):
        plan.blockers.append(
            f"Estimated download {plan.estimated_download_gb:.3f} GB exceeds limit {max_download_gb:.3f} GB."
        )
    return plan


def build_local_companion_plan(
    source_path: str,
    *,
    modality_filter: str | None = None,
) -> AcquisitionPlan:
    """Compute the real minimum-file acquisition set for a local BIDS directory.

    Walks the tree, builds real ``FileRecord``s via the same BIDS parsers the
    manifest layer uses, runs the actual ``ManifestGraph`` companion-closure
    machinery over them (so EEGLAB .set/.fdt pairing, channels/events/sidecar
    resolution, etc. all apply), and reports the minimal set of files a
    recording actually needs to load.
    """
    root = Path(source_path)
    if not root.is_dir():
        return AcquisitionPlan(
            source=source_path,
            source_type="local_bids_directory",
            required_download=False,
            estimated_download_gb=0.0,
            evidence_status=EvidenceStatus.unknown,
            notes=[f"Local BIDS path does not exist or is not a directory: {source_path}"],
        )

    files: list[FileRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        extension = _extract_extension(path.name)
        entities = parse_entities(path.name)
        datatype = extract_datatype(rel)
        suffix = parse_filename(path.name).get("suffix") or None
        modality = infer_modality(datatype, suffix)
        files.append(
            FileRecord(
                id=rel,
                path=rel,
                filename=path.name,
                extension=extension,
                size=path.stat().st_size,
                datatype=datatype,
                suffix=suffix,
                modality=modality,
                entities=entities,
            )
        )

    manifest = Manifest(dataset_id=root.name or "local", snapshot="local", files=files)
    graph = ManifestGraph(manifest)
    recordings = graph.recordings()
    if modality_filter is not None:
        recordings = [r for r in recordings if r.modality == modality_filter]

    if not recordings:
        return AcquisitionPlan(
            source=source_path,
            source_type="local_bids_directory",
            required_download=False,
            estimated_download_gb=0.0,
            evidence_status=EvidenceStatus.confirmed,
            n_recordings=0,
            notes=["No recognizable BIDS recordings found in local directory."],
        )

    required: dict[str, FileRecord] = {}
    companions: dict[str, FileRecord] = {}
    for recording in recordings:
        required[recording.primary.path] = recording.primary
        for companion in recording.companions.files:
            required[companion.path] = companion
            companions[companion.path] = companion

    return AcquisitionPlan(
        source=source_path,
        source_type="local_bids_directory",
        required_download=False,
        estimated_download_gb=0.0,
        evidence_status=EvidenceStatus.confirmed,
        n_recordings=len(recordings),
        required_files=sorted(required),
        companion_files=sorted(companions),
    )


__all__ = ["AcquisitionPlan", "build_acquisition_plan", "build_local_companion_plan"]
