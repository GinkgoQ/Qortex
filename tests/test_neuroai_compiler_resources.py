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
