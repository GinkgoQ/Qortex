"""PyTorch and TorchScript model adapter.

Handles:
  - TorchScript models (.pt / .ts files) loaded with ``torch.jit.load``
  - Raw PyTorch nn.Module checkpoints loaded with ``torch.load``
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

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


class TorchModelAdapter(ModelAdapter):
    """Adapter for PyTorch and TorchScript models.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="torch"`` or ``"torchscript"`` and
        ``id=<path to .pt or .ts file>``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model_path = Path(spec.id).expanduser().resolve()
        self._model = None
        self._is_torchscript = False
        self._device = "cpu"

    # ── ModelAdapter interface ────────────────────────────────────────────────

    def inspect(self) -> ModelProfile:
        torch = _require_torch()
        model_hash = _file_sha256(self._model_path)
        n_params = None
        input_shape = None

        is_ts = self._spec.provider in ("torchscript", "ts") or self._model_path.suffix in (
            ".ts", ".torchscript"
        )

        try:
            if is_ts:
                m = torch.jit.load(str(self._model_path), map_location="cpu")
                m.eval()
                # Try to get input shape from graph
                try:
                    graph = m.graph
                    for node in graph.inputs():
                        t = node.type()
                        if hasattr(t, "sizes"):
                            sizes = t.sizes()
                            if sizes:
                                input_shape = list(sizes)
                                break
                except Exception:
                    pass
                n_params = sum(p.numel() for p in m.parameters())
            else:
                obj = torch.load(str(self._model_path), map_location="cpu", weights_only=False)
                m = obj if hasattr(obj, "parameters") else obj.get("model", obj)
                if hasattr(m, "parameters"):
                    n_params = sum(p.numel() for p in m.parameters())
                    # Infer input shape from first layer
                    for module in m.modules():
                        if hasattr(module, "in_features"):
                            input_shape = [1, module.in_features]
                            break
                        if hasattr(module, "in_channels"):
                            input_shape = [1, module.in_channels, -1, -1]
                            break
        except Exception as exc:
            log.warning("TorchModelAdapter.inspect(): %s", exc)

        task = self._spec.task or "unknown"
        contract = self.required_input()
        # Embed shape hint into contract evidence so it survives serialisation
        if input_shape:
            contract.evidence_status = EvidenceStatus.inferred
        return ModelProfile(
            model_id=str(self._model_path),
            provider="torch",
            task=task,
            revision=None,
            model_hash=model_hash,
            estimated_params=n_params,
            input_contract=contract,
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality=self._spec.task.split("_")[0] if self._spec.task and "_" in self._spec.task else "unknown",
            n_channels=None,
            sampling_rate_hz=None,
            spatial_shape=None,
            dtype="float32",
            axis_convention=AxisConvention.batch_channels_time,
            evidence_status=EvidenceStatus.inferred,
        )

    def output_schema(self) -> OutputContract:
        task = self._spec.task or "unknown"
        return OutputContract(
            output_type=_task_to_output_type(task),
            n_classes=None,
        )

    def load(self, runtime: RuntimeSpec) -> None:
        torch = _require_torch()
        self._device = _resolve_device(runtime.device)
        is_ts = self._spec.provider in ("torchscript", "ts") or self._model_path.suffix in (
            ".ts", ".torchscript"
        )
        try:
            if is_ts:
                self._model = torch.jit.load(str(self._model_path), map_location=self._device)
                self._is_torchscript = True
            else:
                obj = torch.load(str(self._model_path), map_location=self._device, weights_only=False)
                self._model = obj if hasattr(obj, "parameters") else obj.get("model", obj)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load PyTorch model from {self._model_path}: {exc}"
            ) from exc

        self._model.eval()
        if runtime.fp16 and "cuda" in self._device:
            self._model = self._model.half()
        self._loaded = True
        log.info("Loaded PyTorch model: %s on %s", self._model_path.name, self._device)

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        torch = _require_torch()
        import numpy as np

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

        raw = out.cpu().numpy() if hasattr(out, "numpy") else out
        return _parse_output(raw, self._spec.task)

    def unload(self) -> None:
        self._model = None
        self._loaded = False
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch model adapter requires torch. "
            "Install with: pip install 'qortex[torch]' or pip install torch"
        )


def _resolve_device(device: str) -> str:
    try:
        import torch
        if device in ("auto", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    except ImportError:
        return "cpu"


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _task_to_output_type(task: str) -> str:
    task_l = task.lower()
    if "classif" in task_l or "eeg" in task_l:
        return "classification"
    if "segment" in task_l:
        return "segmentation"
    if "detect" in task_l:
        return "detection"
    if "regress" in task_l:
        return "regression"
    if "embed" in task_l:
        return "embedding"
    return "unknown"


def _parse_output(raw, task: str | None) -> ModelOutput:
    import numpy as np

    raw_arr = np.array(raw)
    task_l = (task or "").lower()

    if raw_arr.ndim == 1 or (raw_arr.ndim == 2 and raw_arr.shape[0] == 1):
        arr = raw_arr.flatten()
        # Classification: softmax
        exp = np.exp(arr - arr.max())
        probs = exp / exp.sum()
        class_idx = int(np.argmax(probs))
        return ModelOutput(
            output_type="classification",
            raw=raw,
            class_index=class_idx,
            class_name=f"class_{class_idx}",
            probabilities={f"class_{i}": float(p) for i, p in enumerate(probs)},
        )

    if "segment" in task_l and raw_arr.ndim >= 3:
        mask = np.argmax(raw_arr, axis=0) if raw_arr.shape[0] > 1 else raw_arr[0]
        return ModelOutput(output_type="segmentation", raw=raw, mask=mask)

    return ModelOutput(output_type="unknown", raw=raw)
