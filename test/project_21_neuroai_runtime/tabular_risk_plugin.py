from __future__ import annotations

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    Modality,
    ModelProfile,
    OutputContract,
)
from qortex.neuroai.models._base import ModelOutput


class QortexPlugin:
    """Local clinical-feature classifier used by the runtime smoke project.

    The plugin follows the same contract as a production local model adapter:
    it declares an inspectable input/output schema, loads explicitly, and
    returns structured ModelOutput records.
    """

    def __init__(self) -> None:
        self._loaded = False

    def inspect(self) -> ModelProfile:
        return ModelProfile(
            model_id="local/tabular-neuro-risk-v1",
            provider="plugin",
            task="tabular_classification",
            trusted=True,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
            estimated_params=8,
            estimated_memory_mb=1.0,
            supported_devices=["cpu"],
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality=Modality.tabular,
            axis_convention=AxisConvention.channels_last,
            n_channels=4,
            dtype="float32",
            evidence_status=EvidenceStatus.confirmed,
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(
            output_type="classification",
            classes=["restable", "alert"],
            n_classes=2,
            produces_probabilities=True,
        )

    def load(self, runtime) -> None:
        if runtime.device not in {"auto", "cpu"}:
            raise RuntimeError(f"Plugin supports CPU execution, got device={runtime.device!r}")
        self._loaded = True

    def predict(self, batch) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Plugin was not loaded")
        if hasattr(batch, "detach"):
            arr = batch.detach().cpu().numpy()
        else:
            arr = np.asarray(batch, dtype=np.float32)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 4:
            raise ValueError(f"Expected a 2D table with >=4 numeric features, got shape={arr.shape}")

        alpha = float(np.mean(arr[:, 0]))
        beta = float(np.mean(arr[:, 1]))
        theta = float(np.mean(arr[:, 2]))
        artifact = float(np.mean(arr[:, 3]))
        risk_logit = (1.55 * beta) + (1.10 * theta) + (1.75 * artifact) - (0.95 * alpha)
        alert_probability = float(1.0 / (1.0 + np.exp(-risk_logit)))
        rest_probability = float(1.0 - alert_probability)
        label = "alert" if alert_probability >= 0.55 else "restable"

        return ModelOutput(
            output_type="classification",
            raw=np.array([rest_probability, alert_probability], dtype=np.float32),
            class_name=label,
            class_index=1 if label == "alert" else 0,
            probabilities={
                "restable": rest_probability,
                "alert": alert_probability,
            },
            regression_value=alert_probability,
            metadata={
                "mean_alpha_power": alpha,
                "mean_beta_power": beta,
                "mean_theta_power": theta,
                "mean_artifact_ratio": artifact,
            },
        )

    def unload(self) -> None:
        self._loaded = False
