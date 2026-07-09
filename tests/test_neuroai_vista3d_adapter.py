from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import PromptType


def test_vista3d_interaction_contract_declares_point_box_and_automatic():
    from qortex.neuroai.models.monai import VISTA3DAdapter
    from qortex.neuroai.spec import ModelSpec

    adapter = VISTA3DAdapter(ModelSpec(provider="vista3d", id="vista3d"))
    contract = adapter.interaction_contract()

    assert set(contract.supported_prompt_types) == {PromptType.point, PromptType.box}
    assert contract.supports_automatic_mode is True


def test_vista3d_rejects_text_prompt_before_touching_the_model():
    from qortex.neuroai.models.monai import VISTA3DAdapter
    from qortex.neuroai.spec import ModelSpec

    adapter = VISTA3DAdapter(ModelSpec(provider="vista3d", id="vista3d"))
    bad_prompt = Prompt(text="liver")

    with pytest.raises(ModelAdapterError):
        adapter.predict_with_prompt(batch="fake_batch", prompt=bad_prompt)


def test_vista3d_provider_dispatches_to_vista3d_adapter():
    from qortex.neuroai.models._registry import make_model_adapter
    from qortex.neuroai.models.monai import VISTA3DAdapter
    from qortex.neuroai.spec import ModelSpec

    adapter = make_model_adapter(ModelSpec(provider="vista3d", id="vista3d"))

    assert isinstance(adapter, VISTA3DAdapter)


def test_zoo_vista3d_entry_is_promptable_with_confirmed_contract():
    from qortex.neuroai.models.zoo.registry import lookup

    entry = lookup("monai.vista3d")

    assert entry.entry_type.value == "promptable_model"
    assert entry.interaction_contract is not None
    assert set(entry.interaction_contract.supported_prompt_types) == {"point", "box"}
