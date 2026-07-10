from __future__ import annotations

import pytest

from qortex.neuroai.models.zoo.registry import clear_registry
from qortex.neuroai.models.zoo.seed_examples import register_all as seed_examples_register_all
from qortex.neuroai.models.zoo.monai_imaging import register_all as monai_imaging_register_all
from qortex.neuroai.models.zoo.monai_generative import register_all as monai_generative_register_all
from qortex.neuroai.models.zoo.braindecode_eeg import register_all as braindecode_eeg_register_all
from qortex.neuroai.models.zoo.external_engines import register_all as external_engines_register_all
from qortex.neuroai.models.zoo.foundation_segmentation import register_all as foundation_segmentation_register_all


@pytest.fixture(autouse=True)
def _seeded_zoo_registry():
    # ponytail: zoo registry is a tiny in-memory dict unrelated to other
    # subsystems, so clearing+reseeding it before every test in the suite
    # is harmless for tests that never touch it, and it's the single
    # source of truth for zoo test isolation (was duplicated per-file).
    clear_registry()
    seed_examples_register_all()
    monai_imaging_register_all()
    monai_generative_register_all()
    braindecode_eeg_register_all()
    external_engines_register_all()
    foundation_segmentation_register_all()
    yield
    clear_registry()
