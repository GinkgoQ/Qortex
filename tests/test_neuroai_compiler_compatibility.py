"""Deep source-evidence-vs-model-contract compatibility evaluation.

These test the real `_compatibility()` function with real ZooEntry input
contracts and real SourceProfileSummary header evidence — proving the
compiler now evaluates channel count, sampling rate, orientation, and voxel
spacing against model requirements and emits specific required transforms,
rather than deciding on modality alone.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract
from qortex.neuroai.compiler.candidates import _compatibility
from qortex.neuroai.compiler.result import SourceProfileSummary
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    LicenseInfo,
    ZooEntry,
    ZooEntryType,
)


def _eeg_entry(*, n_channels=None, sampling_rate_hz=None) -> ZooEntry:
    return ZooEntry(
        id="test.eeg_model",
        display_name="EEG model",
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/m",
        modality=["eeg"],
        task=["classification"],
        input_contract=InputContract(
            modality="eeg",
            axis_convention=AxisConvention.batch_channels_time,
            n_channels=n_channels,
            sampling_rate_hz=sampling_rate_hz,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def _ct_entry(*, orientation_axis=None, voxel=None) -> ZooEntry:
    return ZooEntry(
        id="test.ct_model",
        display_name="CT model",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url="https://example.org/m",
        modality=["ct"],
        task=["segmentation"],
        input_contract=InputContract(
            modality="ct",
            axis_convention=orientation_axis or AxisConvention.channels_first,
            voxel_sizes_mm=voxel,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    )


def _eeg_source(*, n_channels=None, sampling_rate_hz=None) -> SourceProfileSummary:
    return SourceProfileSummary(
        source="/tmp/x.fif", source_type="local_file", exists=True, modality="eeg",
        n_channels=n_channels, sampling_rate_hz=sampling_rate_hz,
        evidence_status=EvidenceStatus.confirmed,
    )


def _ct_source(*, orientation=None, voxel=None) -> SourceProfileSummary:
    return SourceProfileSummary(
        source="/tmp/x.nii.gz", source_type="local_file", exists=True, modality="ct",
        orientation=orientation, voxel_sizes_mm=voxel,
        evidence_status=EvidenceStatus.confirmed,
    )


def test_exact_channel_and_rate_match_is_plain_compatible():
    proof = _compatibility(
        _eeg_entry(n_channels=32, sampling_rate_hz=256.0),
        _eeg_source(n_channels=32, sampling_rate_hz=256.0),
    )
    assert proof.status == "compatible"
    assert proof.required_transforms == []


def test_more_source_channels_needs_channel_select_transform():
    proof = _compatibility(
        _eeg_entry(n_channels=32, sampling_rate_hz=256.0),
        _eeg_source(n_channels=64, sampling_rate_hz=256.0),
    )
    assert proof.status == "compatible_with_transforms"
    t = [x for x in proof.required_transforms if x["transform"] == "select_channels"]
    assert t and t[0]["from"] == 64 and t[0]["to"] == 32


def test_fewer_source_channels_is_a_hard_incompatible():
    proof = _compatibility(
        _eeg_entry(n_channels=64, sampling_rate_hz=256.0),
        _eeg_source(n_channels=32, sampling_rate_hz=256.0),
    )
    assert proof.status == "incompatible"
    assert any("cannot be synthesized" in b for b in proof.blockers)


def test_sampling_rate_mismatch_needs_resample_transform():
    proof = _compatibility(
        _eeg_entry(n_channels=32, sampling_rate_hz=256.0),
        _eeg_source(n_channels=32, sampling_rate_hz=512.0),
    )
    assert proof.status == "compatible_with_transforms"
    t = [x for x in proof.required_transforms if x["transform"] == "resample"]
    assert t and t[0]["from"] == 512.0 and t[0]["to"] == 256.0


def test_orientation_mismatch_needs_reorient_transform():
    proof = _compatibility(
        _ct_entry(orientation_axis=AxisConvention.RAS),
        _ct_source(orientation="LAS"),
    )
    assert proof.status == "compatible_with_transforms"
    t = [x for x in proof.required_transforms if x["transform"] == "reorient"]
    assert t and t[0]["from"] == "LAS" and t[0]["to"] == "RAS"


def test_voxel_spacing_mismatch_needs_resample_spatial_transform():
    proof = _compatibility(
        _ct_entry(voxel=(1.0, 1.0, 1.0)),
        _ct_source(voxel=(2.0, 2.0, 2.0)),
    )
    assert proof.status == "compatible_with_transforms"
    t = [x for x in proof.required_transforms if x["transform"] == "resample_spatial"]
    assert t and t[0]["to"] == [1.0, 1.0, 1.0]


def test_channels_first_axis_is_not_treated_as_an_orientation():
    # A model whose axis_convention is "channels_first" (not a 3-letter
    # anatomical orientation) must NOT produce a spurious reorient transform
    # against a source with a real orientation code.
    proof = _compatibility(
        _ct_entry(orientation_axis=AxisConvention.channels_first),
        _ct_source(orientation="RAS"),
    )
    assert proof.status == "compatible"
    assert not any(t["transform"] == "reorient" for t in proof.required_transforms)
