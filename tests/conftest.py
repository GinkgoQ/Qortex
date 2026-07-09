from __future__ import annotations

import pytest

from qortex.neuroai.models.zoo.registry import clear_registry
from qortex.neuroai.models.zoo.seed_examples import register_all


@pytest.fixture(autouse=True)
def _seeded_zoo_registry():
    # ponytail: zoo registry is a tiny in-memory dict unrelated to other
    # subsystems, so clearing+reseeding it before every test in the suite
    # is harmless for tests that never touch it, and it's the single
    # source of truth for zoo test isolation (was duplicated per-file).
    clear_registry()
    register_all()
    yield
    clear_registry()
