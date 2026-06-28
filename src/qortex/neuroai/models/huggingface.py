"""HuggingFace model adapter.

Loads any HuggingFace model and exposes it through the ModelAdapter interface.
Supports AutoModel, AutoModelForSequenceClassification, AutoModelForImageClassification,
and custom pipelines.

Security policy:
  ``trust_remote_code=True`` must be explicitly set in the model spec and surfaced as
  a warning.  We never silently execute remote code.

Input contract is inferred from the model config when possible; when not deterministic,
the evidence_status is set to EvidenceStatus.inferred or EvidenceStatus.unknown.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    Modality,
    ModelProfile,
    OutputContract,
    WarningItem,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

log = logging.getLogger(__name__)

_TASK_TO_MODALITY: dict[str, str] = {
    "eeg_classification": "eeg",
    "eeg_regression": "eeg",
    "image-classification": "image",
    "image-segmentation": "image",
    "object-detection": "image",
    "audio-classification": "audio",
    "text-classification": "tabular",
}


class HuggingFaceAdapter(ModelAdapter):
    """Load and run a HuggingFace model via the ``transformers`` library.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="huggingface"`` and ``id="org/model"``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._processor = None
        self._config = None
        self._profile: ModelProfile | None = None
        self._loaded = False

    def inspect(self) -> ModelProfile:
        if self._profile is not None:
            return self._profile

        try:
            from transformers import AutoConfig
        except ImportError:
            raise ImportError(
                "HuggingFace model inspection requires transformers. "
                "Install with: pip install 'qortex[huggingface]'"
            )

        warnings: list[WarningItem] = []

        if self._spec.trust_remote_code:
            warnings.append(WarningItem(
                code="TRUST_REMOTE_CODE",
                message="trust_remote_code=True: this model may execute arbitrary code. "
                        "Only use with models from trusted sources.",
                severity="warning",
                suggestion="Verify the model repository before enabling trust_remote_code.",
            ))

        try:
            cfg = AutoConfig.from_pretrained(
                self._spec.id,
                revision=self._spec.revision,
                trust_remote_code=self._spec.trust_remote_code,
            )
            self._config = cfg
        except Exception as exc:
            warnings.append(WarningItem(
                code="CONFIG_LOAD_FAILED",
                message=f"Cannot load model config: {exc}",
                severity="error",
            ))
            profile = ModelProfile(
                model_id=self._spec.id,
                provider="huggingface",
                revision=self._spec.revision,
                task=self._spec.task,
                warnings=warnings,
            )
            self._profile = profile
            return profile

        n_classes = getattr(cfg, "num_labels", None)
        classes = list(getattr(cfg, "id2label", {}).values()) if hasattr(cfg, "id2label") else []

        input_contract = self._infer_input_contract(cfg, warnings)
        output_contract = OutputContract(
            output_type=self._spec.task or "classification",
            classes=classes,
            n_classes=n_classes or len(classes) or None,
            produces_probabilities=True,
        )

        self._profile = ModelProfile(
            model_id=self._spec.id,
            provider="huggingface",
            revision=self._spec.revision,
            task=self._spec.task,
            license=getattr(cfg, "license", None),
            trusted=self._spec.trust_remote_code,
            input_contract=input_contract,
            output_contract=output_contract,
            warnings=warnings,
        )
        return self._profile

    def required_input(self) -> InputContract:
        return self.inspect().input_contract or InputContract(
            modality="unknown",
            axis_convention=AxisConvention.channels_first,
            evidence_status=EvidenceStatus.unknown,
        )

    def output_schema(self) -> OutputContract:
        return self.inspect().output_contract or OutputContract(
            output_type=self._spec.task or "unknown",
        )

    def load(self, runtime: RuntimeSpec) -> None:
        try:
            import torch
            from transformers import AutoModel, pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "HuggingFace inference requires transformers + torch. "
                "Install with: pip install 'qortex[huggingface]'"
            )

        device = self._resolve_device(runtime.device)
        log.info("Loading HuggingFace model %s on %s", self._spec.id, device)

        task = self._spec.task
        try:
            if task:
                self._model = hf_pipeline(
                    task,
                    model=self._spec.id,
                    revision=self._spec.revision,
                    device=device,
                    trust_remote_code=self._spec.trust_remote_code,
                )
            else:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self._spec.id,
                    revision=self._spec.revision,
                    trust_remote_code=self._spec.trust_remote_code,
                ).to(device)
        except Exception as exc:
            from qortex.core.exceptions import ModelAdapterError
            raise ModelAdapterError(
                f"Failed to load model {self._spec.id!r}: {exc}",
                model_id=self._spec.id,
                provider="huggingface",
            ) from exc

        self._loaded = True
        log.info("Model %s loaded.", self._spec.id)

    def predict(self, batch: Any) -> ModelOutput:
        if not self._loaded or self._model is None:
            raise RuntimeError("Call load() before predict().")

        profile = self.inspect()
        out_contract = profile.output_contract

        raw_out = self._model(batch) if callable(self._model) else None

        if isinstance(raw_out, list) and raw_out and isinstance(raw_out[0], dict):
            # HF pipeline output: [{"label": "...", "score": 0.9}]
            top = max(raw_out, key=lambda x: x.get("score", 0))
            classes = out_contract.classes if out_contract else []
            probs = {item["label"]: float(item["score"]) for item in raw_out}
            return ModelOutput(
                output_type="classification",
                raw=raw_out,
                class_name=top.get("label"),
                class_index=classes.index(top["label"]) if top.get("label") in classes else None,
                probabilities=probs,
            )

        if hasattr(raw_out, "logits"):
            import torch
            probs_tensor = torch.softmax(raw_out.logits, dim=-1).squeeze()
            probs_np = probs_tensor.detach().cpu().numpy()
            classes = out_contract.classes if out_contract else [str(i) for i in range(len(probs_np))]
            pred_idx = int(probs_np.argmax())
            return ModelOutput(
                output_type="classification",
                raw=raw_out,
                class_index=pred_idx,
                class_name=classes[pred_idx] if pred_idx < len(classes) else str(pred_idx),
                probabilities={c: float(probs_np[i]) for i, c in enumerate(classes)},
            )

        return ModelOutput(output_type="unknown", raw=raw_out)

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_input_contract(self, cfg, warnings: list[WarningItem]) -> InputContract:
        """Infer InputContract from model config with explicit evidence tracking."""

        modality_str = _TASK_TO_MODALITY.get(self._spec.task or "", "unknown")
        evidence = EvidenceStatus.confirmed if modality_str != "unknown" else EvidenceStatus.inferred

        # Look for hidden_size / sequence_length clues
        n_ch: int | None = getattr(cfg, "num_channels", None) or getattr(cfg, "in_channels", None)
        max_len: int | None = getattr(cfg, "max_position_embeddings", None)

        if n_ch is None:
            evidence = EvidenceStatus.inferred
            warnings.append(WarningItem(
                code="INPUT_CONTRACT_INFERRED",
                message="Cannot determine required channel count from model config. "
                        "Input contract is inferred, not confirmed.",
                severity="info",
                suggestion="Check the model card for the exact expected input shape.",
            ))

        return InputContract(
            modality=modality_str,
            axis_convention=AxisConvention.channels_time,
            n_channels=n_ch,
            window_duration_s=max_len / 256.0 if max_len else None,  # rough estimate
            dtype="float32",
            evidence_status=evidence,
        )

    def _resolve_device(self, device_str: str) -> str | int:
        if device_str == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return 0   # CUDA device 0
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                return "cpu"
            except ImportError:
                return "cpu"
        return device_str
