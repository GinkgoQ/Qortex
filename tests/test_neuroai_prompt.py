from __future__ import annotations

from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType


def test_prompt_with_no_fields_set_is_empty():
    prompt = Prompt()
    assert prompt.points is None
    assert prompt.boxes is None
    assert prompt.text is None


def test_validate_against_rejects_unsupported_prompt_type():
    contract = InteractionContract(supported_prompt_types=[PromptType.point])
    prompt = Prompt(boxes=[(0.0, 0.0, 10.0, 10.0)])

    violations = prompt.validate_against(contract)

    assert any("box" in v.lower() for v in violations)


def test_validate_against_accepts_supported_prompt_type():
    contract = InteractionContract(supported_prompt_types=[PromptType.point, PromptType.box])
    prompt = Prompt(points=[(5.0, 5.0)], point_labels=[1], boxes=[(0.0, 0.0, 10.0, 10.0)])

    assert prompt.validate_against(contract) == []


def test_validate_against_rejects_mismatched_point_labels_length():
    contract = InteractionContract(supported_prompt_types=[PromptType.point])
    prompt = Prompt(points=[(1.0, 1.0), (2.0, 2.0)], point_labels=[1])

    violations = prompt.validate_against(contract)

    assert any("point_labels" in v for v in violations)


def test_validate_against_rejects_too_many_points():
    contract = InteractionContract(supported_prompt_types=[PromptType.point], max_points=1)
    prompt = Prompt(points=[(1.0, 1.0), (2.0, 2.0)], point_labels=[1, 0])

    violations = prompt.validate_against(contract)

    assert any("max_points" in v for v in violations)


def test_validate_against_rejects_text_when_unsupported():
    contract = InteractionContract(supported_prompt_types=[PromptType.point])
    prompt = Prompt(text="liver")

    violations = prompt.validate_against(contract)

    assert any("text" in v.lower() for v in violations)
