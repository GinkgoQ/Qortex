from __future__ import annotations

from typing import Any

import pytest

from qortex.console import model_execution


def test_public_profiles_are_typed_and_linked_to_registered_models() -> None:
    profiles = model_execution.list_model_execution_profiles()

    assert {profile["id"] for profile in profiles} == {
        "public-brats-segmentation-v1", "public-coco-detection-v1",
    }
    assert all(profile["model_id"] for profile in profiles)
    assert all(profile["result_contract"].startswith("qortex.public_validation.") for profile in profiles)
    assert all(profile["parameters"] for profile in profiles)


def test_execution_rejects_undeclared_parameters_before_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return kwargs

    profile = model_execution.ModelExecutionProfile(
        id="test-profile",
        model_id="test/model",
        display_name="Test",
        task="test",
        dataset={"id": "real-dataset"},
        parameters=(model_execution.ExecutionParameter("device", "enum", "cpu", "Device", ("cpu",)),),
        result_contract="test.v1",
        artifact_kinds=("result",),
        runner=runner,
    )
    monkeypatch.setitem(model_execution._BY_ID, profile.id, profile)

    with pytest.raises(ValueError, match="Unknown parameters"):
        model_execution.run_model_execution_profile(profile.id, parameters={"fabricated": True})

    assert called is False


def test_execution_passes_validated_parameters_and_profile_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, Any] = {}

    def runner(**kwargs: Any) -> dict[str, Any]:
        received.update(kwargs)
        return {"status": "completed", "execution_profile": kwargs["execution_profile"]}

    profile = model_execution.ModelExecutionProfile(
        id="test-profile",
        model_id="test/model",
        display_name="Test",
        task="test",
        dataset={"id": "real-dataset"},
        parameters=(model_execution.ExecutionParameter("threshold", "number", 0.5, "Threshold", minimum=0.0, maximum=1.0),),
        result_contract="test.v1",
        artifact_kinds=("result",),
        runner=runner,
    )
    monkeypatch.setitem(model_execution._BY_ID, profile.id, profile)
    progress = lambda done, total: None

    result = model_execution.run_model_execution_profile(
        profile.id, parameters={"threshold": 0.75}, on_progress=progress,
    )

    assert received["threshold"] == 0.75
    assert received["on_progress"] is progress
    assert received["execution_profile"] == {
        "id": "test-profile",
        "result_contract": "test.v1",
        "parameters": {"threshold": 0.75},
    }
    assert result["execution_profile"] == received["execution_profile"]


def test_execution_rejects_out_of_range_numeric_parameter(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = model_execution.ModelExecutionProfile(
        id="test-profile",
        model_id="test/model",
        display_name="Test",
        task="test",
        dataset={"id": "real-dataset"},
        parameters=(model_execution.ExecutionParameter("threshold", "number", 0.5, "Threshold", minimum=0.0, maximum=1.0),),
        result_contract="test.v1",
        artifact_kinds=("result",),
        runner=lambda **kwargs: kwargs,
    )
    monkeypatch.setitem(model_execution._BY_ID, profile.id, profile)

    with pytest.raises(ValueError, match="at most"):
        model_execution.run_model_execution_profile(profile.id, parameters={"threshold": 1.1})
