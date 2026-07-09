# tests/test_neuroai_promptable.py
from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import ModelProfile, InputContract, OutputContract, AxisConvention
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType


class _FakePromptOnlyAdapter(PromptableModelAdapter):
    """Minimal concrete subclass for testing the ABC's default behavior."""

    def inspect(self) -> ModelProfile:
        return ModelProfile(model_id="fake", provider="fake")

    def required_input(self) -> InputContract:
        return InputContract(modality="ct", axis_convention=AxisConvention.channels_first)

    def output_schema(self) -> OutputContract:
        return OutputContract(output_type="segmentation")

    def load(self, runtime) -> None:
        self._loaded = True

    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(supported_prompt_types=[PromptType.point, PromptType.box])

    def predict_with_prompt(self, batch, prompt: Prompt) -> ModelOutput:
        return ModelOutput(output_type="segmentation", raw=batch, metadata={"prompt_used": True})


class _FakeAutomaticCapableAdapter(_FakePromptOnlyAdapter):
    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(
            supported_prompt_types=[PromptType.point],
            supports_automatic_mode=True,
        )

    def predict_automatic(self, batch) -> ModelOutput:
        return ModelOutput(output_type="segmentation", raw=batch, metadata={"automatic": True})


def test_predict_without_prompt_raises_when_automatic_mode_unsupported():
    adapter = _FakePromptOnlyAdapter()

    with pytest.raises(ModelAdapterError, match="predict_with_prompt"):
        adapter.predict(batch="fake_batch")


def test_predict_with_prompt_returns_output():
    adapter = _FakePromptOnlyAdapter()

    output = adapter.predict_with_prompt("fake_batch", Prompt(points=[(1.0, 2.0)], point_labels=[1]))

    assert output.metadata["prompt_used"] is True


def test_predict_falls_back_to_automatic_when_supported():
    adapter = _FakeAutomaticCapableAdapter()

    output = adapter.predict(batch="fake_batch")

    assert output.metadata["automatic"] is True


def test_predict_automatic_default_raises_not_implemented():
    adapter = _FakePromptOnlyAdapter()

    with pytest.raises(NotImplementedError):
        adapter.predict_automatic("fake_batch")
