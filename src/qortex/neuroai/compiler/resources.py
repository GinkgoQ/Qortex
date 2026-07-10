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
) -> ResourcePlan:
    input_gb: float | None = None
    evidence = EvidenceStatus.unknown
    notes: list[str] = []

    resolved_spatial_shape = _resolved_positive_dims(
        input_contract.spatial_shape if input_contract is not None else None
    )
    if input_contract is not None and resolved_spatial_shape is not None:
        elements = 1
        for dim in resolved_spatial_shape:
            elements *= dim
        # spatial_shape is purely spatial dims (Z,Y,X / H,W); n_channels is
        # the separate channel count. Multiplying both in is correct exactly
        # once -- do not fold channels into spatial_shape upstream (see
        # models/monai.py's required_input(), which used to do this and
        # produced a negative element count from unresolved -1 dims).
        channels = int(input_contract.n_channels or 1)
        dtype = str(input_contract.dtype or "float32").lower()
        input_gb = elements * channels * _DTYPE_BYTES.get(dtype, 4) / 1e9
        evidence = EvidenceStatus.confirmed
    elif input_contract is not None and input_contract.spatial_shape and source_size_bytes is not None:
        input_gb = source_size_bytes / 1e9
        evidence = EvidenceStatus.inferred
        notes.append(
            "VRAM estimate uses local file size because the model's declared "
            "spatial_shape contains unresolved (non-positive) dimensions."
        )
    elif source_size_bytes is not None:
        input_gb = source_size_bytes / 1e9
        evidence = EvidenceStatus.inferred
        notes.append("VRAM estimate uses local file size because the model input tensor shape is not confirmed.")
    else:
        notes.append("VRAM estimate unavailable: no confirmed tensor shape or local file size.")

    estimated_vram_gb = (input_gb * 3.5) if input_gb is not None else None
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
