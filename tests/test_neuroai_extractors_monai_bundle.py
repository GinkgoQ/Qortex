from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.extractors.monai_bundle import (
    ExtractedMONAIContract,
    extract_monai_contract,
)


def test_extract_full_metadata_populates_contracts():
    metadata = {
        "network_data_format": {
            "inputs": {
                "image": {
                    "type": "image",
                    "format": "magnitude",
                    "num_channels": 4,
                    "spatial_shape": [240, 240, 155],
                    "dtype": "float32",
                }
            },
            "outputs": {
                "pred": {
                    "type": "image",
                    "format": "segmentation",
                    "num_channels": 4,
                    "dtype": "float32",
                }
            },
        }
    }

    extracted = extract_monai_contract("test.bundle", metadata)

    assert extracted.model_id == "test.bundle"
    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels == 4
    assert extracted.input_contract.spatial_shape == (240, 240, 155)
    assert extracted.input_contract.evidence_status == EvidenceStatus.confirmed
    assert extracted.output_contract is not None
    assert extracted.output_contract.n_classes == 4
    assert extracted.unresolved_transforms == []


def test_extract_missing_network_data_format_returns_unknown_contracts():
    extracted = extract_monai_contract("bare.bundle", {})

    assert extracted.input_contract is None
    assert extracted.output_contract is None


def test_extract_partial_metadata_does_not_guess_missing_fields():
    metadata = {
        "network_data_format": {
            "inputs": {"image": {"type": "image", "format": "magnitude"}},
            "outputs": {},
        }
    }

    extracted = extract_monai_contract("partial.bundle", metadata)

    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels is None
    assert extracted.input_contract.spatial_shape is None
    assert extracted.input_contract.evidence_status == EvidenceStatus.inferred
    assert extracted.output_contract is None


def test_extract_flags_unresolved_custom_transforms():
    metadata = {
        "network_data_format": {
            "inputs": {"image": {"type": "image", "format": "magnitude", "num_channels": 1}},
            "outputs": {},
        }
    }
    inference = {
        "preprocessing": [
            {"_target_": "LoadImaged"},
            {"_target_": "my_custom_module.WeirdTransform"},
        ]
    }

    extracted = extract_monai_contract("custom.bundle", metadata, inference)

    assert "my_custom_module.WeirdTransform" in extracted.unresolved_transforms
    assert "LoadImaged" not in extracted.unresolved_transforms
