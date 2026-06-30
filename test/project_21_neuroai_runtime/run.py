from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from qortex.neuroai import Pipeline, validate_artifact
from qortex.neuroai.compatibility import CompatibilityEngine
from qortex.neuroai.contracts import (
    AxisConvention,
    CompatibilityReport,
    CompatibilityStatus,
    InputContract,
    ModelProfile,
    SourceProfile,
    TransformDescriptor,
    TransformKind,
)
from qortex.neuroai.preprocess import PreprocessPlanner, TransformExecutor
from qortex.neuroai.spec import PipelineSpec, PreprocessSpec


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    data_path = ROOT / "input_features.csv"
    plugin_path = ROOT / "tabular_risk_plugin.py"
    jsonl_path = OUT / "predictions.jsonl"
    csv_path = OUT / "predictions.csv"

    alias_spec = PipelineSpec.from_dict({
        "name": "project_21_schema_aliases",
        "source": {
            "type": "local_file",
            "path": str(data_path),
            "subject": "01",
            "session": "baseline",
        },
        "window": {
            "duration_s": "2s",
            "step_s": "500ms",
            "overlap_frac": 0.25,
            "drop_short": "false",
        },
        "model": {
            "provider": "plugin",
            "id": str(plugin_path),
            "trust_remote_code": "true",
        },
        "preprocessing": {
            "mode": "auto",
            "normalize": "false",
            "resample": "false",
            "channel_select": "false",
        },
        "runtime": {"device": "cpu", "cache_model": "false"},
        "outputs": {"type": "jsonl", "path": str(OUT / "alias.jsonl"), "append": "true"},
    })
    alias_dict = alias_spec.to_dict()
    print("ALIAS_SOURCE_SUBJECTS", alias_spec.source.subjects)
    print("ALIAS_SOURCE_SESSIONS", alias_spec.source.sessions)
    print("ALIAS_WINDOW", alias_dict["window"])
    print("ALIAS_PREPROCESSING", alias_dict["preprocessing"])
    print("ALIAS_RUNTIME", alias_dict["runtime"])
    assert alias_spec.source.subjects == ["01"]
    assert alias_spec.source.sessions == ["baseline"]
    assert alias_spec.window.duration_s == 2.0
    assert alias_spec.window.step_s == 0.5
    assert alias_spec.window.overlap_frac == 0.25
    assert alias_spec.window.drop_short is False
    assert alias_spec.preprocessing.allows("normalize") is False
    assert alias_spec.preprocessing.allows("resample") is False
    assert alias_spec.preprocessing.allows("resample_spatial") is False
    assert alias_spec.preprocessing.allows("channel_select") is False
    assert alias_spec.runtime.cache_model is False
    assert alias_spec.outputs[0].append is True

    contradictory = PipelineSpec.from_dict({
        "name": "project_21_policy_contradiction",
        "source": {"type": "local_file", "path": str(data_path)},
        "model": {"provider": "plugin", "id": str(plugin_path), "trust_remote_code": True},
        "preprocessing": {"mode": "auto", "allow": ["normalize"], "normalize": False},
        "outputs": {"type": "jsonl", "path": str(OUT / "contradiction.jsonl")},
    })
    contradiction_errors = contradictory.validate()
    print("POLICY_CONTRADICTION_ERRORS", contradiction_errors)
    assert any("normalize is False" in e for e in contradiction_errors)

    deny_cast_report = CompatibilityEngine().check(
        SourceProfile(
            source_id="project_21_contract_source",
            source_type="local_file",
            modality="tabular",
            n_channels=4,
            dtype="float64",
            axis_convention=AxisConvention.channels_time,
        ),
        ModelProfile(
            model_id="project_21_contract_model",
            provider="plugin",
            input_contract=InputContract(
                modality="tabular",
                axis_convention=AxisConvention.channels_time,
                n_channels=4,
                dtype="float32",
            ),
        ),
        PreprocessSpec(mode="auto", deny=["cast_dtype"]),
    )
    print("DENY_CAST_COMPATIBILITY_STATUS", deny_cast_report.status.value)
    print("DENY_CAST_BLOCKERS", [b.code for b in deny_cast_report.blockers])
    assert deny_cast_report.status.value == "incompatible"
    assert any(b.code == "DTYPE_MISMATCH" for b in deny_cast_report.blockers)

    channel_plan = PreprocessPlanner().build_plan(
        CompatibilityReport(
            status=CompatibilityStatus.compatible_with_transforms,
            source_id="manual_channel_source",
            model_id="manual_channel_model",
            required_transforms=[
                TransformDescriptor(
                    kind=TransformKind.channel_select,
                    required_by="input_contract.required_channels",
                    params={
                        "mode": "names",
                        "names": ["C3", "C4"],
                        "source_names": ["Fp1", "C3", "C4", "Oz"],
                        "missing_policy": "error",
                    },
                    reversible=True,
                )
            ],
        ),
        window_duration_s=2.0,
        model_provider="onnx",
    )
    channel_transform_names = [
        t.kind.value if hasattr(t.kind, "value") else str(t.kind)
        for t in channel_plan.transforms
    ]
    selected = TransformExecutor(channel_plan).apply(
        np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ], dtype=np.float32)
    )
    print("CHANNEL_PLAN_TRANSFORMS", channel_transform_names)
    print("CHANNEL_SELECT_RESULT", selected.tolist())
    assert "window" not in channel_transform_names
    assert selected.shape == (2, 3)
    assert selected.tolist() == [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]

    spec_dict = {
        "name": "project_21_neuroai_runtime",
        "source": {
            "type": "local_file",
            "path": str(data_path),
        },
        "model": {
            "provider": "plugin",
            "id": str(plugin_path),
            "task": "tabular_classification",
            "trust_remote_code": True,
        },
        "preprocessing": {
            "mode": "auto",
            "allow": ["normalize", "cast_dtype", "to_tensor"],
        },
        "runtime": {
            "device": "cpu",
            "latency_budget_ms": 500,
            "optimize": "safe",
            "batch_size": 1,
        },
        "outputs": [
            {"type": "jsonl", "path": str(jsonl_path)},
            {"type": "csv", "path": str(csv_path)},
        ],
        "trigger": {
            "when": {"class": "alert", "probability_gte": 0.70, "stable_for": 1},
            "emit": {"label": "tabular_alert_marker"},
        },
    }

    spec = PipelineSpec.from_dict(spec_dict)
    errors = spec.validate()
    print("SPEC_VALIDATION_ERRORS", errors)
    assert errors == [], errors

    pipe = Pipeline.from_dict(spec_dict)
    compat = pipe.check()
    print("COMPATIBILITY_STATUS", compat.status.value)
    print("COMPATIBILITY_IS_RUNNABLE", compat.is_runnable)
    print("REQUIRED_TRANSFORMS", [t.kind.value if hasattr(t.kind, "value") else str(t.kind) for t in compat.required_transforms])
    print("EVIDENCE_CHECKS", compat.evidence)
    assert compat.is_runnable, compat.summary()
    assert any(item.get("check") == "memory_estimate" for item in compat.evidence)

    plan = pipe.plan_preprocessing()
    print("PREPROCESS_PLAN")
    print(plan.summary())

    report = pipe.run(artifact_dir=OUT / "artifact")
    artifact_jsonl_path = OUT / "artifact" / "outputs" / "predictions.jsonl"
    artifact_csv_path = OUT / "artifact" / "outputs" / "predictions.csv"
    print("RUN_SUCCESS", report.success)
    print("WINDOWS_PROCESSED", report.n_windows_processed)
    print("OUTPUTS_WRITTEN", report.n_outputs_written)
    print("OUTPUT_RECORD_COUNTS", report.outputs)
    print("LATENCY_STATUS", report.latency_report.status if report.latency_report else None)
    print("RUN_ERRORS", report.errors)
    assert report.success, report.errors
    assert report.n_outputs_written == 2, report.outputs
    assert artifact_jsonl_path.exists()
    assert artifact_csv_path.exists()
    assert any(item.get("n_marker_records") == 1 for item in report.outputs)

    jsonl_records = [
        json.loads(line)
        for line in artifact_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print("JSONL_RECORDS", json.dumps(jsonl_records, indent=2, sort_keys=True))
    prediction_records = [r for r in jsonl_records if r.get("output_type") == "classification"]
    marker_records = [r for r in jsonl_records if r.get("record_type") == "event_marker"]
    assert len(prediction_records) == 1
    assert len(marker_records) == 1
    prediction = prediction_records[0]
    marker = marker_records[0]
    assert "probabilities" in prediction
    assert prediction["class"] == "alert"
    assert prediction["trigger_fired"] is True
    assert prediction["input_shape"]
    assert prediction["preprocessed_shape"]
    assert marker["label"] == "alert"

    with artifact_csv_path.open(newline="", encoding="utf-8") as f:
        csv_records = list(csv.DictReader(f))
    print("CSV_RECORDS", json.dumps(csv_records, indent=2, sort_keys=True))
    assert len(csv_records) == 1
    assert csv_records[0]["probabilities_json"]
    assert csv_records[0]["raw_summary_json"]
    assert csv_records[0]["trigger_fired"] == "True"

    provenance_files = sorted((OUT / "artifact").glob("*.json"))
    print("ARTIFACT_FILES", [p.name for p in provenance_files])
    assert (OUT / "artifact" / "artifact_manifest.json").exists()
    assert (OUT / "artifact" / "provenance.json").exists()
    manifest = json.loads((OUT / "artifact" / "artifact_manifest.json").read_text(encoding="utf-8"))
    print("ARTIFACT_MANIFEST_FILES", sorted(manifest["files"]))
    assert "outputs/predictions.jsonl" in manifest["files"]
    assert "outputs/predictions.csv" in manifest["files"]

    validation = validate_artifact(OUT / "artifact")
    print("ARTIFACT_VALIDATION_STATUS", validation.status)
    print("ARTIFACT_VALIDATION_SUMMARY", validation.summary())
    print("ARTIFACT_VALIDATION_OUTPUTS", json.dumps(validation.output_files, indent=2, sort_keys=True))
    assert validation.status == "PASS", validation.to_json()
    assert validation.n_prediction_records == 2
    assert validation.n_marker_records == 1

    print("project_21_neuroai_runtime complete")


if __name__ == "__main__":
    main()
