"""Typed registry of validated public-model execution profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ExecutionParameter:
    name: str
    kind: str
    default: Any
    label: str
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class ModelExecutionProfile:
    id: str
    model_id: str
    display_name: str
    task: str
    dataset: dict[str, Any]
    parameters: tuple[ExecutionParameter, ...]
    result_contract: str
    artifact_kinds: tuple[str, ...]
    runner: Callable[..., dict[str, Any]] = field(repr=False)

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("runner", None)
        payload["parameters"] = [asdict(parameter) for parameter in self.parameters]
        return payload


def _run_brats(**kwargs: Any) -> dict[str, Any]:
    from qortex.neuroai.public_validation import run_public_brats_validation

    return run_public_brats_validation(**kwargs)


def _run_detection(**kwargs: Any) -> dict[str, Any]:
    from qortex.neuroai.public_detection import run_public_detection_validation

    return run_public_detection_validation(**kwargs)


_PROFILES = (
    ModelExecutionProfile(
        id="public-brats-segmentation-v1",
        model_id="monai.brats_mri_segmentation",
        display_name="Pinned public BraTS segmentation validation",
        task="brain_tumor_segmentation",
        dataset={
            "id": "MedOtter/brats2023-gli-dataset",
            "revision": "b032d353a3e80911a5f850bc54e6fb575298a354",
            "case_id": "BraTS-GLI-00000-000",
        },
        parameters=(
            ExecutionParameter("device", "enum", "auto", "Inference device", ("auto", "cpu", "cuda")),
            ExecutionParameter("case_id", "string", "BraTS-GLI-00000-000", "Pinned public case"),
        ),
        result_contract="qortex.public_validation.brats.v1",
        artifact_kinds=("input", "ground_truth", "prediction", "prediction_regions", "provenance"),
        runner=_run_brats,
    ),
    ModelExecutionProfile(
        id="public-coco-detection-v1",
        model_id="torchvision/fasterrcnn_resnet50_fpn_v2",
        display_name="Pinned public COCO object-detection validation",
        task="object_detection",
        dataset={
            "id": "coco-2017-val",
            "split": "val2017",
            "image_id": 397133,
            "annotation_sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
        },
        parameters=(
            ExecutionParameter("device", "enum", "auto", "Inference device", ("auto", "cpu", "cuda")),
            ExecutionParameter("image_id", "integer", 397133, "Pinned COCO image", minimum=397133, maximum=397133),
            ExecutionParameter("score_threshold", "number", 0.5, "Score threshold", minimum=0.0, maximum=1.0),
            ExecutionParameter("iou_threshold", "number", 0.5, "Match IoU threshold", minimum=0.0, maximum=1.0),
        ),
        result_contract="qortex.public_validation.detection.v1",
        artifact_kinds=("input", "board", "detections", "ground_truth", "showcase_manifest", "provenance"),
        runner=_run_detection,
    ),
)
_BY_ID = {profile.id: profile for profile in _PROFILES}


def list_model_execution_profiles() -> list[dict[str, Any]]:
    return [profile.public() for profile in _PROFILES]


def get_model_execution_profile(profile_id: str) -> dict[str, Any]:
    profile = _BY_ID.get(profile_id)
    if profile is None:
        raise KeyError(f"No model execution profile {profile_id!r}")
    return profile.public()


def _validate_parameter(spec: ExecutionParameter, value: Any) -> Any:
    if spec.kind == "enum":
        if value not in spec.choices:
            raise ValueError(f"{spec.name} must be one of {list(spec.choices)}")
        return value
    if spec.kind == "string":
        if not isinstance(value, str) or not value:
            raise ValueError(f"{spec.name} must be a non-empty string")
        return value
    if spec.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{spec.name} must be an integer")
    elif spec.kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{spec.name} must be a number")
        value = float(value)
    else:
        raise ValueError(f"Unsupported execution parameter kind {spec.kind!r}")
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"{spec.name} must be at least {spec.minimum}")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"{spec.name} must be at most {spec.maximum}")
    return value


def run_model_execution_profile(
    profile_id: str,
    *,
    parameters: dict[str, Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    profile = _BY_ID.get(profile_id)
    if profile is None:
        raise KeyError(f"No model execution profile {profile_id!r}")
    supplied = parameters or {}
    declared = {parameter.name for parameter in profile.parameters}
    unknown = sorted(set(supplied) - declared)
    if unknown:
        raise ValueError(f"Unknown parameters for {profile_id}: {unknown}")
    validated = {
        spec.name: _validate_parameter(spec, supplied.get(spec.name, spec.default))
        for spec in profile.parameters
    }
    execution_profile = {
        "id": profile.id,
        "result_contract": profile.result_contract,
        "parameters": validated,
    }
    return profile.runner(
        **validated,
        on_progress=on_progress,
        execution_profile=execution_profile,
    )


__all__ = ["get_model_execution_profile", "list_model_execution_profiles", "run_model_execution_profile"]
