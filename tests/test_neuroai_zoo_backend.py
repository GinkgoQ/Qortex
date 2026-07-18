from __future__ import annotations

from qortex.neuroai.models.zoo import _backend


def test_backend_probe_reports_broken_installed_runtime(monkeypatch):
    def fail_import(name: str):
        if name == "tensorflow":
            raise AttributeError("incompatible numpy")
        return object()

    monkeypatch.setattr(_backend.importlib, "import_module", fail_import)

    diagnostics = _backend.backend_diagnostics()

    assert diagnostics["keras"]["available"] is False
    assert diagnostics["keras"]["error"] == "AttributeError: incompatible numpy"
    assert _backend.backend_availability()["keras"] is False
