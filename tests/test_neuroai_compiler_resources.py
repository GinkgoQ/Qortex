from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, InputContract
from qortex.neuroai.compiler.resources import estimate_resource_plan


def test_unresolved_spatial_shape_never_produces_negative_or_zero_estimate():
    # Reproduces the exact shape MONAIBundleAdapter used to construct before
    # the fix: channel count folded into spatial_shape alongside unresolved
    # -1 placeholders for the true spatial extent.
    contract = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        n_channels=4,
        spatial_shape=(4, -1, -1, -1),
    )

    plan = estimate_resource_plan(
        device="cpu", input_contract=contract, source_size_bytes=10_000_000, max_vram_gb=None
    )

    assert plan.estimated_vram_gb is not None
    assert plan.estimated_vram_gb > 0
    # Falls back to the real file size, not a fabricated negative tensor size.
    assert plan.estimated_input_tensor_gb == 10_000_000 / 1e9


def test_resolved_positive_spatial_shape_does_not_double_count_channels():
    contract = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        n_channels=4,
        spatial_shape=(96, 96, 96),
    )

    plan = estimate_resource_plan(
        device="cpu", input_contract=contract, source_size_bytes=None, max_vram_gb=None
    )

    expected_elements = 96 * 96 * 96
    expected_input_gb = expected_elements * 4 * 4 / 1e9  # 4 channels, float32
    assert plan.estimated_input_tensor_gb == expected_input_gb
    assert plan.evidence_status.value == "confirmed"


def test_zero_dimension_is_rejected_as_unresolved():
    contract = InputContract(
        modality="mri",
        axis_convention=AxisConvention.channels_first,
        n_channels=1,
        spatial_shape=(0, 128, 128),
    )

    plan = estimate_resource_plan(
        device="cpu", input_contract=contract, source_size_bytes=None, max_vram_gb=None
    )

    assert plan.estimated_input_tensor_gb is None
    assert any("unavailable" in n.lower() for n in plan.notes)


def test_roi_size_bounds_working_set_to_patch_not_full_volume():
    # A model that declares a huge spatial_shape but is actually run with a
    # small sliding-window ROI must not blow up its VRAM estimate to the
    # full-volume tensor size.
    contract = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        n_channels=1,
        spatial_shape=(512, 512, 768),
        dtype="float32",
    )

    plan = estimate_resource_plan(
        device="cpu",
        input_contract=contract,
        source_size_bytes=None,
        max_vram_gb=None,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
    )

    full_volume_bytes = 512 * 512 * 768 * 4
    assert plan.estimated_input_tensor_gb is not None
    assert plan.estimated_input_tensor_gb * 1e9 < full_volume_bytes / 100
    expected_patch_gb = 96 * 96 * 96 * 1 * 4 / 1e9
    assert plan.estimated_input_tensor_gb == expected_patch_gb
    assert plan.evidence_status.value == "confirmed"
    assert plan.estimated_vram_gb is not None
    assert plan.estimated_vram_gb < 5.0  # sane patch-based estimate, not volume-sized


def test_file_size_fallback_attaches_compressed_size_caveat():
    contract = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        n_channels=4,
        spatial_shape=(4, -1, -1, -1),
    )

    plan = estimate_resource_plan(
        device="cpu", input_contract=contract, source_size_bytes=10_000_000, max_vram_gb=None
    )

    assert any("poor proxy" in n.lower() for n in plan.notes)
    assert plan.evidence_status.value == "inferred"


def test_estimated_memory_mb_hint_increases_estimate():
    contract = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        n_channels=1,
        spatial_shape=(96, 96, 96),
        dtype="float32",
    )

    plan_without_hint = estimate_resource_plan(
        device="cpu", input_contract=contract, source_size_bytes=None, max_vram_gb=None
    )
    plan_with_hint = estimate_resource_plan(
        device="cpu",
        input_contract=contract,
        source_size_bytes=None,
        max_vram_gb=None,
        estimated_memory_mb=4096.0,
    )

    assert plan_with_hint.estimated_vram_gb > plan_without_hint.estimated_vram_gb
    assert any("weights/parameter memory hint" in n for n in plan_with_hint.notes)


def test_evidence_status_inferred_when_heuristic_fallback_used():
    plan = estimate_resource_plan(
        device="cpu", input_contract=None, source_size_bytes=5_000_000, max_vram_gb=None
    )
    assert plan.evidence_status.value == "inferred"
