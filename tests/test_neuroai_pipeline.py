"""Regression tests for NeuroAI pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import qortex.neuroai.pipeline as pipeline_mod
from qortex.neuroai.contracts import (
    AxisConvention,
    InputContract,
    Modality,
    ModelProfile,
    OutputContract,
    QortexTimeSeries,
    SourceProfile,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.pipeline import Pipeline
from qortex.neuroai.sources._base import SourceAdapter


class _ReplaySource(SourceAdapter):
    def __init__(self, spec: Any):
        self.spec = spec
        self.replay_speeds: list[float] = []

    def probe(self) -> SourceProfile:
        return SourceProfile(
            source_id="test-source",
            source_type="local_file",
            path=str(self.spec.path),
            modality=Modality.tabular,
            abstraction="timeseries",
            n_channels=2,
            sampling_rate_hz=2.0,
            channel_names=["a", "b"],
            duration_s=2.0,
            dtype="float32",
            axis_convention=AxisConvention.channels_time,
        )

    def read_batch(self) -> list[Any]:
        return list(self.stream())

    def stream(self):
        raise AssertionError("Pipeline.replay() must use SourceAdapter.replay()")

    def replay(self, speed: float = 1.0):
        self.replay_speeds.append(speed)
        yield QortexTimeSeries(
            shape=(2, 4),
            axes=["channels", "time"],
            dtype="float32",
            sampling_frequency_hz=2.0,
            channel_names=["a", "b"],
            data=np.ones((2, 4), dtype=np.float32),
        )


class _Model(ModelAdapter):
    def __init__(self, spec: Any):
        self.spec = spec
        self.loaded = False

    def inspect(self) -> ModelProfile:
        return ModelProfile(
            model_id=str(self.spec.id),
            provider="custom",
            task="classification",
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality=Modality.tabular,
            axis_convention=AxisConvention.channels_time,
            n_channels=2,
            sampling_rate_hz=2.0,
            dtype="float32",
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(
            output_type="classification",
            classes=["ok"],
            n_classes=1,
            produces_probabilities=True,
        )

    def load(self, runtime) -> None:
        self.loaded = True

    def predict(self, batch: Any) -> ModelOutput:
        assert self.loaded
        return ModelOutput(
            output_type="classification",
            raw=np.asarray([1.0], dtype=np.float32),
            class_name="ok",
            class_index=0,
            probabilities={"ok": 1.0},
        )

    def unload(self) -> None:
        self.loaded = False


class _Output(OutputAdapter):
    def __init__(self, spec: Any):
        self.spec = spec
        self.records: list[tuple[ModelOutput, dict[str, Any] | None]] = []
        self._is_open = False
        self._n_written = 0

    def open(self) -> None:
        self._is_open = True

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        assert self._is_open
        self.records.append((output, metadata))
        self._n_written += 1

    def close(self) -> None:
        self._is_open = False


def _spec(path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "name": "replay-test",
        "source": {"type": "local_file", "path": str(path), "modality": "tabular"},
        "model": {
            "provider": "custom",
            "id": "fake-model",
            "task": "classification",
            "trust_remote_code": True,
        },
        "runtime": {"device": "cpu", "max_windows": 1},
        "outputs": [{"type": "jsonl", "path": str(output_path)}],
    }


def test_replay_uses_source_replay_speed(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "source.csv"
    replay_path = tmp_path / "replay.csv"
    source_path.write_text("a,b\n1,1\n", encoding="utf-8")
    replay_path.write_text("a,b\n1,1\n", encoding="utf-8")
    sources: list[_ReplaySource] = []
    outputs: list[_Output] = []

    def make_source(spec, **kwargs):
        source = _ReplaySource(spec)
        sources.append(source)
        return source

    def make_output(spec, **kwargs):
        output = _Output(spec)
        outputs.append(output)
        return output

    monkeypatch.setattr(pipeline_mod, "make_source_adapter", make_source)
    monkeypatch.setattr(pipeline_mod, "make_model_adapter", lambda spec: _Model(spec))
    monkeypatch.setattr(pipeline_mod, "make_output_adapter", make_output)

    pipe = Pipeline.from_dict(_spec(source_path, tmp_path / "predictions.jsonl"))
    report = pipe.replay(replay_path, speed=2.5, output_dir=tmp_path / "replay_out")

    assert report.success, report.errors
    assert report.n_windows_processed == 1
    assert sources[-1].replay_speeds == [2.5]
    assert outputs[-1].n_written == 1
    assert report.source_profile is not None
    assert report.source_profile.path == str(replay_path)


def test_replay_rejects_nonpositive_speed(tmp_path: Path):
    source_path = tmp_path / "source.csv"
    source_path.write_text("a,b\n1,1\n", encoding="utf-8")
    pipe = Pipeline.from_dict(_spec(source_path, tmp_path / "predictions.jsonl"))

    with pytest.raises(ValueError, match="Replay speed"):
        pipe.replay(source_path, speed=0)
