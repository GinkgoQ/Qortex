"""A small, curated catalog of real ``ModelProfile`` contracts for the Atlas
Compatibility screen.

Qortex's ``qortex.neuroai`` subsystem ships a *compatibility engine*
(``CompatibilityEngine.check``) and a full contract type system
(``SourceProfile`` / ``ModelProfile`` / ``InputContract``), but — correctly —
no opinion about which specific published models exist. That catalog is a
product decision, not a library concern, so it lives here in the console
layer: real ``ModelProfile`` objects, each a faithful contract for a
well-known EEG/MEG architecture, evaluated by the real, unmodified
``CompatibilityEngine``.
"""

from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus, InputContract, ModelProfile, OutputContract

MODEL_CATALOG: dict[str, ModelProfile] = {
    "braindecode/eegnet": ModelProfile(
        model_id="braindecode/eegnet",
        provider="braindecode",
        task="eeg_classification",
        license="BSD-3-Clause",
        input_contract=InputContract(
            modality="eeg",
            axis_convention="channels_first",
            n_channels=None,  # architecture is channel-count agnostic (depthwise conv)
            sampling_rate_hz=128.0,
            window_duration_s=2.0,
            dtype="float32",
            required_metadata=["sampling_rate_hz"],
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(output_type="classification", produces_probabilities=True),
        estimated_params=2_000,
    ),
    "braindecode/deep4": ModelProfile(
        model_id="braindecode/deep4",
        provider="braindecode",
        task="eeg_classification",
        license="BSD-3-Clause",
        input_contract=InputContract(
            modality="eeg",
            axis_convention="channels_first",
            n_channels=22,
            sampling_rate_hz=250.0,
            window_duration_s=4.0,
            dtype="float32",
            required_metadata=["sampling_rate_hz", "channel_names"],
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(output_type="classification", produces_probabilities=True),
        estimated_params=280_000,
    ),
    "custom/self-supervised-meg": ModelProfile(
        model_id="custom/self-supervised-meg",
        provider="custom",
        task="representation_learning",
        input_contract=InputContract(
            modality="meg",
            axis_convention="channels_first",
            sampling_rate_hz=1000.0,
            window_duration_s=10.0,
            dtype="float32",
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(output_type="embedding", produces_probabilities=False),
    ),
}


def list_models() -> list[dict]:
    return [
        {
            "id": mp.model_id, "provider": mp.provider, "task": mp.task,
            "modality": mp.input_contract.modality if mp.input_contract else None,
            "sampling_rate_hz": mp.input_contract.sampling_rate_hz if mp.input_contract else None,
            "n_channels": mp.input_contract.n_channels if mp.input_contract else None,
            "window_duration_s": mp.input_contract.window_duration_s if mp.input_contract else None,
        }
        for mp in MODEL_CATALOG.values()
    ]
