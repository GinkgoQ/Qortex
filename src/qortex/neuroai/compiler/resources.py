"""Resource estimates and limits for NeuroAI compilation."""

from __future__ import annotations

from qortex.neuroai.contracts import BaseModel, EvidenceStatus, Field, InputContract

_DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "uint16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}


def _resolved_positive_dims(spatial_shape: tuple[int, ...] | list[int] | None) -> list[int] | None:
    """Return spatial_shape as a list of ints only if every dimension is a
    real, resolved, positive size. Rejects unresolved placeholder values
    (0, negative, or any sentinel like -1 used to mean "unknown") instead
    of silently multiplying them into a nonsensical (often negative)
    element count.
    """
    if not spatial_shape:
        return None
    dims = [int(d) for d in spatial_shape]
    if any(d <= 0 for d in dims):
        return None
    return dims


# Activation working-set multiplier: an INFERRED upper-bound heuristic for a
# typical encoder-decoder (U-Net-family) segmentation CNN. Encoder/decoder
# skip connections keep several downsampled copies of the feature maps alive
# simultaneously; 7x the raw input+output patch tensor bytes is a defensible
# ceiling for that class of architecture at common depths/widths, not a
# measured profiling number for any specific model.
_ACTIVATION_MULTIPLIER = 7.0

# Framework/CUDA-context overhead: the fixed cost of a CUDA context plus
# framework (PyTorch/MONAI) bookkeeping before any tensor is allocated.
# Typical observed baseline is ~0.3-0.5 GB; documented estimate, not measured
# per-model.
_FRAMEWORK_OVERHEAD_GB = 0.4


class ResourcePlan(BaseModel):
    device: str
    estimated_vram_gb: float | None = None
    estimated_input_tensor_gb: float | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    blockers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def estimate_resource_plan(
    *,
    device: str,
    input_contract: InputContract | None,
    source_size_bytes: int | None,
    max_vram_gb: float | None,
    roi_size: tuple[int, ...] | None = None,
    sw_batch_size: int = 1,
    n_classes: int | None = None,
    estimated_memory_mb: float | None = None,
) -> ResourcePlan:
    input_gb: float | None = None
    evidence = EvidenceStatus.unknown
    notes: list[str] = []
    heuristic_used = False

    # Prefer the sliding-window ROI (the patch actually pushed through the
    # network) over the full declared spatial_shape. For volumetric models
    # scanned with a sliding window, the working set is the ROI, not the
    # whole volume -- a 512^3 CT with a 96^3 ROI never materializes 512^3
    # activations.
    resolved_roi = _resolved_positive_dims(roi_size)
    resolved_spatial_shape = _resolved_positive_dims(
        input_contract.spatial_shape if input_contract is not None else None
    )
    patch_dims = resolved_roi if resolved_roi is not None else resolved_spatial_shape

    channels = int((input_contract.n_channels if input_contract is not None else None) or 1)
    dtype = str((input_contract.dtype if input_contract is not None else None) or "float32").lower()
    dtype_bytes = _DTYPE_BYTES.get(dtype, 4)
    batch = max(1, int(sw_batch_size))

    if patch_dims is not None:
        elements = 1
        for dim in patch_dims:
            elements *= dim
        # spatial_shape/roi_size is purely spatial dims (Z,Y,X / H,W);
        # n_channels is the separate channel count. Multiplying both in is
        # correct exactly once -- do not fold channels into spatial_shape
        # upstream (see models/monai.py's required_input(), which used to do
        # this and produced a negative element count from unresolved -1 dims).
        input_gb = elements * channels * dtype_bytes * batch / 1e9
        evidence = EvidenceStatus.confirmed
        if resolved_roi is not None:
            notes.append(
                "VRAM estimate uses the sliding-window ROI patch size, not the "
                "full declared spatial_shape, since the model processes one "
                "patch at a time."
            )

        output_gb = elements * int(n_classes or 1) * dtype_bytes * batch / 1e9
        activation_gb = (input_gb + output_gb) * _ACTIVATION_MULTIPLIER
    elif input_contract is not None and input_contract.spatial_shape and source_size_bytes is not None:
        input_gb = source_size_bytes / 1e9
        evidence = EvidenceStatus.inferred
        heuristic_used = True
        notes.append(
            "VRAM estimate uses local file size because the model's declared "
            "spatial_shape contains unresolved (non-positive) dimensions and no "
            "roi_size was provided. CAVEAT: a compressed on-disk file size is a "
            "poor proxy for in-memory tensor/activation size (compression ratio, "
            "container overhead, and dtype differences are not accounted for) -- "
            "treat this estimate as a rough upper/lower bound, not a measurement."
        )
        activation_gb = input_gb * _ACTIVATION_MULTIPLIER
    elif source_size_bytes is not None:
        input_gb = source_size_bytes / 1e9
        evidence = EvidenceStatus.inferred
        heuristic_used = True
        notes.append(
            "VRAM estimate uses local file size because the model input tensor "
            "shape is not confirmed. CAVEAT: a compressed on-disk file size is a "
            "poor proxy for in-memory tensor/activation size -- treat this "
            "estimate as a rough bound, not a measurement."
        )
        activation_gb = input_gb * _ACTIVATION_MULTIPLIER
    else:
        notes.append("VRAM estimate unavailable: no confirmed tensor shape or local file size.")
        activation_gb = None

    # Parameter/weights + optimizer-state contribution: only added when the
    # model registry supplies an explicit memory hint. Otherwise this term is
    # zero and we say so explicitly rather than silently under-counting.
    if estimated_memory_mb is not None:
        weights_gb = estimated_memory_mb / 1024.0
        notes.append(
            f"Includes a model-provided weights/parameter memory hint of "
            f"{estimated_memory_mb:.1f} MB."
        )
    else:
        weights_gb = 0.0
        if activation_gb is not None:
            notes.append(
                "Parameter/weights memory is unaccounted: no estimated_memory_mb "
                "hint was available from the model registry."
            )

    estimated_vram_gb: float | None
    if activation_gb is not None:
        estimated_vram_gb = activation_gb + weights_gb + _FRAMEWORK_OVERHEAD_GB
        notes.append(
            f"Adds a {_FRAMEWORK_OVERHEAD_GB:.1f} GB fixed allowance for CUDA "
            "context and framework baseline overhead."
        )
    else:
        estimated_vram_gb = None

    if heuristic_used and evidence != EvidenceStatus.unknown:
        evidence = EvidenceStatus.inferred

    plan = ResourcePlan(
        device=device,
        estimated_vram_gb=estimated_vram_gb,
        estimated_input_tensor_gb=input_gb,
        evidence_status=evidence,
        notes=notes,
    )
    if (
        max_vram_gb is not None
        and estimated_vram_gb is not None
        and estimated_vram_gb > max_vram_gb
    ):
        plan.blockers.append(
            f"Estimated VRAM {estimated_vram_gb:.3f} GB exceeds limit {max_vram_gb:.3f} GB."
        )
    return plan


__all__ = ["ResourcePlan", "estimate_resource_plan"]
