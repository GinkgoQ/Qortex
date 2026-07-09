"""Runtime prompt value object for promptable segmentation models.

Kept separate from InteractionContract (zoo/schema.py) per the design
spec's explicit correction (section 8.1): a prompt is the interaction
data passed at inference time, not the declared capability. A model
declares what it supports via InteractionContract; a caller supplies a
Prompt and validates it against that contract before inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType


@dataclass
class Prompt:
    points: list[tuple[float, ...]] | None = None
    point_labels: list[int] | None = None
    boxes: list[tuple[float, ...]] | None = None
    text: str | None = None

    def validate_against(self, contract: InteractionContract) -> list[str]:
        violations: list[str] = []
        supported = set(contract.supported_prompt_types)

        if self.points is not None:
            if PromptType.point not in supported:
                violations.append("prompt has points but model does not support point prompts")
            if self.point_labels is not None and len(self.point_labels) != len(self.points):
                violations.append(
                    f"point_labels length ({len(self.point_labels)}) does not match "
                    f"points length ({len(self.points)})"
                )
            if contract.max_points is not None and len(self.points) > contract.max_points:
                violations.append(
                    f"prompt has {len(self.points)} points, exceeds model's max_points={contract.max_points}"
                )

        if self.boxes is not None:
            if PromptType.box not in supported:
                violations.append("prompt has boxes but model does not support box prompts")
            if contract.max_boxes is not None and len(self.boxes) > contract.max_boxes:
                violations.append(
                    f"prompt has {len(self.boxes)} boxes, exceeds model's max_boxes={contract.max_boxes}"
                )

        if self.text is not None and PromptType.text not in supported:
            violations.append("prompt has text but model does not support text prompts")

        return violations


__all__ = ["Prompt"]
