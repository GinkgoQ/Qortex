from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models import zoo as _zoo  # noqa: F401  (triggers zoo registration)
from qortex.neuroai.models._registry import make_model_adapter
from qortex.neuroai.spec import ModelSpec


def test_make_model_adapter_blocks_unknown_license_by_default():
    # monai.vista3d's LicenseInfo.evidence_status is unknown (never verified).
    # Calling make_model_adapter directly (bypassing any CLI-layer check)
    # must still be blocked -- this is the actual bug the audit identified:
    # only CLI paths enforced the gate before this fix.
    with pytest.raises(ModelAdapterError, match="license"):
        make_model_adapter(ModelSpec(provider="vista3d", id="monai.vista3d"))


def test_make_model_adapter_allows_with_explicit_opt_in():
    adapter = make_model_adapter(ModelSpec(
        provider="vista3d",
        id="monai.vista3d",
        extra={"accept_unknown_license_risk": True},
    ))
    assert adapter is not None


def test_gate_enforcement_does_not_mask_unknown_provider_in_offline_validation():
    # Regression: adding gate enforcement to make_model_adapter() initially
    # broke zoo/validate.py's offline provider-dispatch check, because an
    # entry with both an unknown license AND a bad provider string would
    # raise ModelAdapterError (license) before ever reaching the ValueError
    # (bad provider) -- silently hiding the real registry defect. The
    # validator must use the gate-free resolve_provider_dispatch(), not
    # make_model_adapter(), for exactly this reason.
    from qortex.neuroai.models._registry import resolve_provider_dispatch

    with pytest.raises(ValueError, match="Unknown model provider"):
        resolve_provider_dispatch(ModelSpec(provider="not_a_real_provider", id="whatever"))


def test_make_model_adapter_is_a_noop_gate_for_unregistered_ids():
    # A raw local path/id with no corresponding zoo entry has no declared
    # policy to enforce -- this must not raise, since a torch checkpoint
    # loaded by direct file path was never opted into the zoo's gates.
    from qortex.neuroai.models.torch import TorchModelAdapter

    adapter = make_model_adapter(ModelSpec(
        provider="torch",
        id="/nonexistent/local/checkpoint.pt",
        input_contract={"modality": "eeg", "axis_convention": "batch_channels_time"},
        output_contract={"output_type": "classification"},
    ))
    assert isinstance(adapter, TorchModelAdapter)
