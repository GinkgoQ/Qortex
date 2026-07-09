from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.extractors.braindecode_model import (
    ExtractedBraindecodeContract,
    extract_braindecode_contract,
)


def test_extract_full_config_populates_contracts():
    config = {
        "n_chans": 22,
        "n_times": 1000,
        "sfreq": 250.0,
        "n_outputs": 4,
        "id2label": {"0": "left_hand", "1": "right_hand", "2": "feet", "3": "tongue"},
    }

    extracted = extract_braindecode_contract("test/model", config)

    assert extracted.model_id == "test/model"
    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels == 22
    assert extracted.input_contract.sampling_rate_hz == 250.0
    assert extracted.input_contract.window_duration_s == 4.0
    assert extracted.input_contract.evidence_status == EvidenceStatus.confirmed
    assert extracted.output_contract is not None
    assert extracted.output_contract.n_classes == 4
    assert extracted.output_contract.classes == ["left_hand", "right_hand", "feet", "tongue"]


def test_extract_accepts_legacy_n_channels_alias():
    config = {"n_channels": 64, "n_times": 500, "sfreq": 125.0}

    extracted = extract_braindecode_contract("legacy/model", config)

    assert extracted.input_contract.n_channels == 64


def test_extract_empty_config_returns_none_contracts():
    extracted = extract_braindecode_contract("bare/model", {})

    assert extracted.input_contract is None
    assert extracted.output_contract is None


def test_extract_partial_config_does_not_guess_missing_fields():
    config = {"n_chans": 22}

    extracted = extract_braindecode_contract("partial/model", config)

    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels == 22
    assert extracted.input_contract.sampling_rate_hz is None
    assert extracted.input_contract.window_duration_s is None
    assert extracted.input_contract.evidence_status == EvidenceStatus.inferred
    assert extracted.output_contract is None


def test_extract_without_id2label_still_populates_n_classes():
    config = {"n_chans": 22, "n_times": 1000, "sfreq": 250.0, "n_outputs": 4}

    extracted = extract_braindecode_contract("no_labels/model", config)

    assert extracted.output_contract.n_classes == 4
    assert extracted.output_contract.classes == []


def test_extract_uses_input_window_seconds_when_n_times_absent():
    config = {"n_chans": 22, "sfreq": 250.0, "input_window_seconds": 4.0}

    extracted = extract_braindecode_contract("window_seconds/model", config)

    assert extracted.input_contract.window_duration_s == 4.0
    # n_times was never given, so this cannot be "confirmed" even though
    # window_duration_s is populated via input_window_seconds.
    assert extracted.input_contract.evidence_status == EvidenceStatus.inferred


def test_extract_sfreq_zero_is_never_confirmed_and_never_divides():
    config = {"n_chans": 22, "n_times": 1000, "sfreq": 0}

    extracted = extract_braindecode_contract("zero_sfreq/model", config)

    assert extracted.input_contract.sampling_rate_hz == 0
    assert extracted.input_contract.window_duration_s is None
    assert extracted.input_contract.evidence_status == EvidenceStatus.inferred
