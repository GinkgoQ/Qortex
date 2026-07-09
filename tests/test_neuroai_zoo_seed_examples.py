from __future__ import annotations

import pytest

from qortex.neuroai.models.zoo.registry import clear_registry, lookup
from qortex.neuroai.models.zoo.validate import validate_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_importing_zoo_package_registers_seed_examples():
    import importlib
    import qortex.neuroai.models.zoo as zoo_pkg
    importlib.reload(zoo_pkg)

    assert lookup("monai.brats_mri_segmentation") is not None
    assert lookup("braindecode.EEGNet") is not None
    assert lookup("external.totalsegmentator") is not None


def test_seed_examples_pass_offline_validation():
    import importlib
    import qortex.neuroai.models.zoo as zoo_pkg
    importlib.reload(zoo_pkg)

    issues = validate_registry()
    assert issues == [], f"seed examples must be fully valid, got: {issues}"
