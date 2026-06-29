"""Braindecode model adapter.

Braindecode is a deep learning library for EEG/BCI built on PyTorch.
Supports: ShallowFBCSPNet, EEGNet, Deep4Net, EEGInception, EEGConformer, etc.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    ModelProfile,
    OutputContract,
    WarningItem,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

log = logging.getLogger(__name__)

_KNOWN_BRAINDECODE_MODELS = {
    "shallowfbcspnet": "ShallowFBCSPNet",
    "eegnet": "EEGNetv4",
    "deep4net": "Deep4Net",
    "eeginception": "EEGInception",
    "eegconformer": "EEGConformer",
    "tcn": "TCN",
}


class BrainDecodeAdapter(ModelAdapter):
    """Adapter for Braindecode EEG/BCI models.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="braindecode"`` and
        ``id=<model_name or hf://org/model_name>``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._device = "cpu"
        self._n_channels: int | None = None
        self._n_times: int | None = None
        self._n_classes: int | None = None
        self._class_names: list[str] = []

    # ── ModelAdapter interface ────────────────────────────────────────────────

    def inspect(self) -> ModelProfile:
        # Fast path: curated registry has confirmed contracts for known BD models.
        from qortex.neuroai.models._contracts import lookup as _registry_lookup
        entry = _registry_lookup(self._spec.id)
        if entry is not None:
            # Apply registry values so required_input() / output_schema() are consistent.
            ic = entry.input_contract
            self._n_channels = ic.n_channels
            self._n_classes = entry.output_contract.n_classes
            self._class_names = list(entry.output_contract.classes or [])
            return ModelProfile(
                model_id=self._spec.id,
                provider="braindecode",
                task=self._spec.task or "eeg_classification",
                revision=self._spec.revision,
                model_hash=None,
                estimated_memory_mb=entry.estimated_memory_mb,
                input_contract=ic,
                output_contract=entry.output_contract,
                warnings=[WarningItem(
                    code="CONTRACT_FROM_REGISTRY",
                    message=f"Input contract loaded from curated registry. {entry.notes}",
                    severity="info",
                )],
            )

        # Fallback: try to read config from HuggingFace Hub.
        _require_braindecode()
        self._load_model_config()

        return ModelProfile(
            model_id=self._spec.id,
            provider="braindecode",
            task=self._spec.task or "eeg_classification",
            revision=self._spec.revision,
            model_hash=None,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality="eeg",
            n_channels=self._n_channels,
            sampling_rate_hz=None,
            spatial_shape=None,
            dtype="float32",
            axis_convention=AxisConvention.batch_channels_time,
            evidence_status=(
                EvidenceStatus.confirmed if self._n_channels else EvidenceStatus.unknown
            ),
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(
            output_type="classification",
            n_classes=self._n_classes,
            classes=self._class_names,
        )

    def load(self, runtime: RuntimeSpec) -> None:
        import torch
        braindecode = _require_braindecode()
        self._device = _resolve_device(runtime.device)

        model_id = self._spec.id
        model_key = model_id.lower().split("/")[-1]

        # Try braindecode built-in models
        for key, class_name in _KNOWN_BRAINDECODE_MODELS.items():
            if key in model_key:
                try:
                    n_ch = self._n_channels or 64
                    n_t = self._n_times or 512
                    n_cl = self._n_classes or 2
                    cls = getattr(braindecode.models, class_name)
                    self._model = cls(n_chans=n_ch, n_times=n_t, n_outputs=n_cl)
                    log.info("Loaded braindecode model: %s", class_name)
                    break
                except Exception as exc:
                    log.warning("Failed to init %s: %s", class_name, exc)

        # Fallback: try loading from HuggingFace via transformers
        if self._model is None:
            try:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=self._spec.trust_remote_code,
                )
                log.info("Loaded braindecode/HF model: %s", model_id)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load braindecode model {model_id!r}: {exc}"
                ) from exc

        self._model.eval()
        self._model.to(self._device)
        if runtime.fp16 and "cuda" in self._device:
            self._model = self._model.half()
        self._loaded = True

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        import torch

        if isinstance(batch, np.ndarray):
            x = torch.from_numpy(batch.astype(np.float32)).to(self._device)
        elif hasattr(batch, "data"):
            x = torch.from_numpy(np.array(batch.data, dtype=np.float32)).to(self._device)
        else:
            x = batch

        if x.ndim == 2:
            x = x.unsqueeze(0)  # [Ch, T] → [1, Ch, T]

        with torch.no_grad():
            out = self._model(x)

        if hasattr(out, "logits"):
            logits = out.logits.cpu().numpy()
        else:
            logits = out.cpu().numpy() if hasattr(out, "numpy") else np.array(out)

        if logits.ndim > 1:
            logits = logits[0]

        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        class_idx = int(np.argmax(probs))
        class_name = (
            self._class_names[class_idx]
            if class_idx < len(self._class_names)
            else f"class_{class_idx}"
        )

        return ModelOutput(
            output_type="classification",
            raw=logits,
            class_index=class_idx,
            class_name=class_name,
            probabilities={
                (self._class_names[i] if i < len(self._class_names) else f"class_{i}"): float(p)
                for i, p in enumerate(probs)
            },
        )

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model_config(self) -> None:
        """Try to load model config from HuggingFace Hub to get n_channels etc."""
        try:
            from huggingface_hub import hf_hub_download
            import json
            config_path = hf_hub_download(
                repo_id=self._spec.id,
                filename="config.json",
                revision=self._spec.revision,
            )
            with open(config_path) as f:
                cfg = json.load(f)
            self._n_channels = cfg.get("n_chans") or cfg.get("in_channels")
            self._n_times = cfg.get("n_times") or cfg.get("input_size")
            self._n_classes = cfg.get("n_outputs") or cfg.get("num_labels")
            id2label = cfg.get("id2label", {})
            self._class_names = [id2label.get(str(i), f"class_{i}") for i in range(self._n_classes or 0)]
        except Exception as exc:
            log.debug("Could not load config for %s: %s", self._spec.id, exc)


def _require_braindecode():
    try:
        import braindecode
        return braindecode
    except ImportError:
        raise ImportError(
            "Braindecode model adapter requires braindecode. "
            "Install with: pip install 'qortex[eeg]' or pip install braindecode"
        )


def _resolve_device(device: str) -> str:
    try:
        import torch
        if device in ("auto", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    except ImportError:
        return "cpu"
