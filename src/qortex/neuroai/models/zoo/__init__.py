"""Qortex NeuroAI model zoo — contract-validated capability registry.

Importing this package registers all curated ZooEntry instances (seed
examples now; MONAI/Braindecode/vision/external-engine domain modules as
each phase lands — see docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md).
"""

from __future__ import annotations

from ._backend import backend_availability
from qortex.neuroai.models.zoo import seed_examples as _seed_examples
from qortex.neuroai.models.zoo import monai_imaging as _monai_imaging
from qortex.neuroai.models.zoo import monai_generative as _monai_generative

_seed_examples.register_all()
_monai_imaging.register_all()
_monai_generative.register_all()

__all__ = ["backend_availability"]
