"""Public Torchvision object-detection models with executable Qortex evidence."""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, OutputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, SecurityPolicy, ZooEntry, ZooEntryType


def register_all() -> None:
    register(ZooEntry(
        id="torchvision/fasterrcnn_resnet50_fpn_v2",
        display_name="Faster R-CNN ResNet-50 FPN v2",
        entry_type=ZooEntryType.model,
        provider="torchvision",
        execution_mode=ExecutionMode.in_process,
        source_url=(
            "https://pytorch.org/vision/stable/models/generated/"
            "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html"
        ),
        model_url="https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
        paper_url="https://arxiv.org/abs/1612.03144",
        docs_url="https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection",
        maintainer="Torchvision maintainers",
        modality=["rgb_image"],
        task=["object_detection"],
        input_contract=InputContract(
            modality="image",
            axis_convention=AxisConvention.channels_first,
            n_channels=3,
            dtype="float32",
            intensity_range=(0.0, 1.0),
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="detection",
            n_classes=91,
            produces_probabilities=True,
        ),
        license=LicenseInfo(
            name="BSD-3-Clause",
            url="https://github.com/pytorch/vision/blob/main/LICENSE",
            commercial_use=True,
            redistribution_allowed=True,
            evidence_status=EvidenceStatus.inferred,
            notes=[
                "The license evidence covers Torchvision source code. The weights metadata does not declare a separate weights license.",
                "COCO source images retain per-image licenses recorded in the dataset annotations.",
            ],
        ),
        security=SecurityPolicy(
            network_required_for_download=True,
            network_required_at_runtime=False,
        ),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))


__all__ = ["register_all"]
