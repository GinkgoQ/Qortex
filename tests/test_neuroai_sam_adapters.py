from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.sam_adapters import MedSAMAdapter, SAMMed3DAdapter
from qortex.neuroai.models.zoo.schema import PromptType
from qortex.neuroai.spec import ModelSpec


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_interaction_contract_is_point_and_box_only(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))
    contract = adapter.interaction_contract()

    assert set(contract.supported_prompt_types) == {PromptType.point, PromptType.box}
    assert contract.supports_automatic_mode is False


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_inspect_works_offline_without_loading_weights(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))

    profile = adapter.inspect()

    assert profile.provider in ("medsam", "sam_med3d")


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_predict_with_prompt_before_load_raises_clear_error(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))

    with pytest.raises(ModelAdapterError, match="load"):
        adapter.predict_with_prompt("fake_batch", Prompt(points=[(1.0, 2.0)], point_labels=[1]))


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_predict_with_text_prompt_rejected(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))

    with pytest.raises(ModelAdapterError):
        adapter.predict_with_prompt("fake_batch", Prompt(text="liver"))


def test_medsam_provider_dispatches_correctly():
    from qortex.neuroai.models._registry import make_model_adapter

    adapter = make_model_adapter(ModelSpec(provider="medsam", id="medsam"))
    assert isinstance(adapter, MedSAMAdapter)


def test_sam_med3d_provider_dispatches_correctly():
    from qortex.neuroai.models._registry import make_model_adapter

    adapter = make_model_adapter(ModelSpec(provider="sam_med3d", id="sam_med3d"))
    assert isinstance(adapter, SAMMed3DAdapter)
