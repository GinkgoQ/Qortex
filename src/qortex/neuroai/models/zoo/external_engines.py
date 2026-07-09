"""External CLI neuroimaging engines (design spec section 12.2), following
the exact ExternalEngineContract pattern Phase 1 established for
external.totalsegmentator (zoo/seed_examples.py). These are file-based CLI
tools, not in-process tensor models -- entries never carry an
input_contract, only an external_engine_contract, per spec section 8.2.
"""

from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    LicenseInfo,
    SecurityPolicy,
    ZooEntry,
    ZooEntryType,
)


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def _engine_entry(
    engine: str,
    display_name: str,
    executable: str,
    source_url: str,
    modality: str,
    supported_tasks: list[str],
    output_file_types: list[str] | None = None,
) -> ZooEntry:
    return ZooEntry(
        id=f"external.{engine}",
        display_name=display_name,
        entry_type=ZooEntryType.external_engine,
        provider="external_cli",
        execution_mode=ExecutionMode.external_cli,
        source_url=source_url,
        modality=[modality],
        task=supported_tasks,
        external_engine_contract=ExternalEngineContract(
            engine=engine,
            executable=executable,
            input_file_types=["nifti"],
            output_file_types=output_file_types or ["nifti"],
            supported_modalities=[modality],
            supported_tasks=supported_tasks,
            command_builder=f"_build_{engine}_command",
            output_manifest_supported=False,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=_unlicensed(),
        security=SecurityPolicy(executable_names=[executable]),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_if_executable_available",
        priority="P0",
    )


def register_all() -> None:
    register(_engine_entry(
        "synthseg", "SynthSeg", "mri_synthseg",
        "https://github.com/BBillot/SynthSeg",
        "mri", ["whole_brain_segmentation"],
    ))
    register(_engine_entry(
        "synthstrip", "SynthStrip", "mri_synthstrip",
        "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/",
        "mri", ["skull_stripping"],
    ))
    register(_engine_entry(
        "hdbet", "HD-BET", "hd-bet",
        "https://github.com/MIC-DKFZ/HD-BET",
        "mri", ["skull_stripping"],
    ))
    register(_engine_entry(
        "fastsurfer", "FastSurfer", "run_fastsurfer.sh",
        "https://github.com/Deep-MI/FastSurfer",
        "mri", ["whole_brain_segmentation", "cortical_parcellation"],
        output_file_types=["nifti", "directory"],
    ))
    register(_engine_entry(
        "tractseg", "TractSeg", "TractSeg",
        "https://github.com/MIC-DKFZ/TractSeg",
        "dwi", ["white_matter_tract_segmentation"],
    ))


__all__ = ["register_all"]
