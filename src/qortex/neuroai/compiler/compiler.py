"""Qortex NeuroAI compiler.

The compiler is an offline planner over existing NeuroAI registry contracts.
It does not download datasets, fetch weights, or execute models. Its job is
to produce a truthful execution plan that separates runnable candidates from
blocked, plan-only, and unavailable entries.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from qortex.neuroai.compiler.acquisition import build_acquisition_plan
from qortex.neuroai.compiler.candidates import build_candidates
from qortex.neuroai.compiler.evidence import EvidenceGraph
from qortex.neuroai.compiler.request import CompilationRequest
from qortex.neuroai.compiler.result import CompilationResult, SourceProfileSummary
from qortex.neuroai.contracts import EvidenceStatus


class NeuroAICompiler:
    """Compile a source/task request into an auditable execution plan."""

    def compile(self, request: CompilationRequest) -> CompilationResult:
        source_profile = profile_source(request.source)
        acquisition_plan = build_acquisition_plan(
            source=request.source,
            source_type=source_profile.source_type,
            local_size_bytes=source_profile.size_bytes,
            max_download_gb=request.max_download_gb,
        )
        evidence_graph = _evidence_graph(source_profile)

        from qortex.neuroai.models import zoo as _zoo  # noqa: F401
        from qortex.neuroai.models.zoo.registry import list_entries

        entries = list_entries(task=request.task)
        for entry in entries:
            evidence_graph.add_node(
                node_id=f"zoo:{entry.id}",
                kind="zoo_entry",
                status=entry.evidence_status,
                source=entry.source_url,
                value={
                    "id": entry.id,
                    "provider": entry.provider,
                    "execution_mode": entry.execution_mode.value,
                    "qortex_status": entry.qortex_status,
                    "task": list(entry.task),
                    "modality": list(entry.modality),
                },
            )
            evidence_graph.add_node(
                node_id=f"license:{entry.id}",
                kind="license",
                status=entry.license.evidence_status,
                source=entry.license.url,
                value={
                    "name": entry.license.name,
                    "commercial_use": entry.license.commercial_use,
                    "redistribution_allowed": entry.license.redistribution_allowed,
                    "requires_registration": entry.license.requires_registration,
                },
            )
            evidence_graph.add_node(
                node_id=f"security:{entry.id}",
                kind="security_policy",
                status=entry.evidence_status,
                value=entry.security.model_dump(mode="json"),
            )
            evidence_graph.add_node(
                node_id=f"runtime:{entry.id}",
                kind="runtime_status",
                status=entry.evidence_status,
                value=entry.qortex_status,
            )
            evidence_graph.add_edge(f"zoo:{entry.id}", f"license:{entry.id}", "requires")
            evidence_graph.add_edge(f"zoo:{entry.id}", f"security:{entry.id}", "requires")
            evidence_graph.add_edge(f"zoo:{entry.id}", f"runtime:{entry.id}", "requires")
        candidates = build_candidates(
            entries=entries,
            source_profile=source_profile,
            task=request.task,
            device=request.device,
            max_vram_gb=request.max_vram_gb,
            accept_unknown_license_risk=request.accept_unknown_license_risk,
            allow_remote_code=request.allow_remote_code,
            require_open_license=request.require_open_license,
        )
        if not request.include_plan_only:
            candidates = [candidate for candidate in candidates if candidate.runnable]

        return CompilationResult.build(
            request=request,
            source_profile=source_profile,
            evidence_graph=evidence_graph,
            acquisition_plan=acquisition_plan,
            candidates=candidates,
        )


def compile_neuroai(request: CompilationRequest) -> CompilationResult:
    """Functional wrapper for one-shot compilation."""

    return NeuroAICompiler().compile(request)


def profile_source(source: str) -> SourceProfileSummary:
    """Inspect a local source path without loading biomedical payload arrays."""

    path = Path(source)
    if path.exists():
        if path.is_file():
            size = path.stat().st_size
            return SourceProfileSummary(
                source=source,
                source_type="local_file",
                exists=True,
                size_bytes=size,
                sha256=_sha256_file(path),
                modality=_modality_from_path(path),
                available_suffixes=[_suffix(path)],
                evidence_status=EvidenceStatus.confirmed,
            )
        size = _directory_size(path)
        suffixes = sorted({_suffix(item) for item in path.rglob("*") if item.is_file()})
        return SourceProfileSummary(
            source=source,
            source_type="local_bids_directory" if (path / "dataset_description.json").exists() else "local_directory",
            exists=True,
            size_bytes=size,
            modality=_modality_from_suffixes(suffixes),
            available_suffixes=suffixes,
            evidence_status=EvidenceStatus.confirmed,
        )

    return SourceProfileSummary(
        source=source,
        source_type="remote_or_catalog_source",
        exists=False,
        evidence_status=EvidenceStatus.unknown,
        notes=["Source is not a local path visible to the compiler; compile does not fetch remote manifests."],
    )


def _evidence_graph(source_profile: SourceProfileSummary) -> EvidenceGraph:
    graph = EvidenceGraph()
    graph.add_node(
        node_id="source",
        kind="source_profile",
        status=source_profile.evidence_status,
        source=source_profile.source,
        value=source_profile.model_dump(mode="json"),
    )
    return graph


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _suffix(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".nii.gz"):
        return "nii.gz"
    suffix = path.suffix.lower().lstrip(".")
    return suffix or path.name.lower()


def _modality_from_path(path: Path) -> str | None:
    suffix = _suffix(path)
    if suffix in {"nii", "nii.gz", "mgz", "mgh"}:
        return "mri"
    if suffix in {"dcm", "dicom"}:
        return "dicom"
    if suffix in {"edf", "bdf", "fif", "vhdr", "set"}:
        return "eeg"
    return None


def _modality_from_suffixes(suffixes: list[str]) -> str | None:
    modalities = {
        modality
        for suffix in suffixes
        for modality in [_modality_from_path(Path(f"file.{suffix}"))]
        if modality is not None
    }
    if len(modalities) == 1:
        return next(iter(modalities))
    return None


__all__ = ["NeuroAICompiler", "compile_neuroai", "profile_source"]
