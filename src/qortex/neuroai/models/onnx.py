"""ONNX model adapter.

Loads an ONNX model via onnxruntime and exposes it through the ModelAdapter
interface.  The input contract is read directly from the ONNX graph's input
nodes, giving fully confirmed (not inferred) shape information.
"""

from __future__ import annotations

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

_ONNX_DTYPE_MAP = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 10: "float16", 11: "float64", 12: "uint32",
    13: "uint64",
}


class ONNXModelAdapter(ModelAdapter):
    """Run ONNX models via onnxruntime.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="onnx"`` and ``id=<path_to_model.onnx>``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._session = None
        self._input_nodes: list = []
        self._output_nodes: list = []
        self._loaded = False
        self._profile: ModelProfile | None = None

    def inspect(self) -> ModelProfile:
        if self._profile is not None:
            return self._profile

        # Registry lookup: ONNX models are often exported versions of known models.
        from qortex.neuroai.models._contracts import lookup as _registry_lookup
        entry = _registry_lookup(self._spec.id)
        if entry is not None:
            self._profile = ModelProfile(
                model_id=self._spec.id,
                provider="onnx",
                task=self._spec.task or entry.output_contract.output_type,
                input_contract=entry.input_contract,
                output_contract=entry.output_contract,
                estimated_memory_mb=entry.estimated_memory_mb,
                warnings=[WarningItem(
                    code="CONTRACT_FROM_REGISTRY",
                    message=f"Input contract loaded from curated registry. {entry.notes}",
                    severity="info",
                )],
            )
            return self._profile

        try:
            import onnxruntime as ort
            import onnx
        except ImportError:
            raise ImportError(
                "ONNX inspection requires onnxruntime + onnx. "
                "Install with: pip install 'qortex[onnx]'"
            )

        model_path = Path(self._spec.id).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        model_proto = onnx.load(str(model_path))
        graph = model_proto.graph
        warnings: list[WarningItem] = []

        # Read input shapes from the ONNX graph
        inputs = list(graph.input)
        outputs = list(graph.output)
        self._input_nodes = inputs
        self._output_nodes = outputs

        input_contract = self._parse_input_contract(inputs, warnings)
        output_contract = self._parse_output_contract(outputs)

        # Model hash from file
        try:
            import hashlib
            file_bytes = model_path.read_bytes()
            model_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        except Exception:
            model_hash = None

        self._profile = ModelProfile(
            model_id=str(model_path),
            provider="onnx",
            model_hash=model_hash,
            task=self._spec.task,
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
        return self.inspect().output_contract or OutputContract(output_type="unknown")

    def load(self, runtime: RuntimeSpec) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install 'qortex[onnx]'"
            )

        providers = self._select_providers(runtime.device)
        log.info("Loading ONNX model %s with providers=%s", self._spec.id, providers)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if runtime.optimize in ("speed",)
            else ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self._session = ort.InferenceSession(
            self._spec.id,
            sess_options=opts,
            providers=providers,
        )
        self._input_nodes = self._session.get_inputs()
        self._output_nodes = self._session.get_outputs()
        self._loaded = True
        log.info("ONNX model loaded.")

    def predict(self, batch: Any) -> ModelOutput:
        if not self._loaded or self._session is None:
            raise RuntimeError("Call load() before predict().")

        inp_name = self._input_nodes[0].name
        arr = self._coerce_to_numpy(batch)
        feeds = {inp_name: arr}

        raw_outputs = self._session.run(None, feeds)
        out_node = self._output_nodes[0]

        # Classification: assume (batch, n_classes) logits or probabilities
        raw = raw_outputs[0]
        if raw.ndim >= 2:
            probs = _softmax(raw[0])
        else:
            probs = _softmax(raw)

        profile = self.inspect()
        out_contract = profile.output_contract
        classes = out_contract.classes if out_contract else [str(i) for i in range(len(probs))]

        pred_idx = int(np.argmax(probs))
        return ModelOutput(
            output_type=self._spec.task or "classification",
            raw=raw_outputs,
            class_index=pred_idx,
            class_name=classes[pred_idx] if pred_idx < len(classes) else str(pred_idx),
            probabilities={c: float(probs[i]) for i, c in enumerate(classes[:len(probs)])},
        )

    def unload(self) -> None:
        self._session = None
        self._loaded = False

    # ── ONNX graph parsing ────────────────────────────────────────────────────

    def _parse_input_contract(self, inputs: list, warnings: list) -> InputContract:
        if not inputs:
            return InputContract(
                modality="unknown",
                axis_convention=AxisConvention.channels_first,
                evidence_status=EvidenceStatus.unknown,
            )

        first = inputs[0]
        shape = self._extract_shape(first)
        dtype = self._extract_dtype(first)

        # Guess modality from task or shape
        task = self._spec.task or ""
        modality = "eeg" if "eeg" in task else ("mri" if len(shape) in (4, 5) else "unknown")

        # (batch, channels, time) → channels_time convention
        axis_conv = AxisConvention.channels_first if len(shape) >= 3 else AxisConvention.channels_time

        n_channels: int | None = None
        if len(shape) >= 3 and isinstance(shape[1], int):
            n_channels = shape[1]

        evidence = EvidenceStatus.confirmed if shape else EvidenceStatus.missing
        if None in (shape or []):
            evidence = EvidenceStatus.inferred
            warnings.append(WarningItem(
                code="DYNAMIC_SHAPE",
                message=f"ONNX model has dynamic input dimensions: shape={shape}. "
                        "Channel count and window size may not be fully determined.",
                severity="info",
            ))

        return InputContract(
            modality=modality,
            axis_convention=axis_conv,
            n_channels=n_channels,
            dtype=dtype,
            evidence_status=evidence,
        )

    def _parse_output_contract(self, outputs: list) -> OutputContract:
        if not outputs:
            return OutputContract(output_type="unknown")
        out = outputs[0]
        shape = self._extract_shape(out)
        n_classes = shape[-1] if shape and isinstance(shape[-1], int) else None
        return OutputContract(
            output_type=self._spec.task or "classification",
            n_classes=n_classes,
            output_dtype=self._extract_dtype(out),
            produces_probabilities=True,
        )

    @staticmethod
    def _extract_shape(node) -> list:
        try:
            # ort.NodeArg shape
            if hasattr(node, "shape"):
                return list(node.shape)
            # onnx.ValueInfoProto
            t = node.type.tensor_type
            if t.HasField("shape"):
                dims = []
                for d in t.shape.dim:
                    dims.append(d.dim_value if d.dim_value > 0 else None)
                return dims
        except Exception:
            pass
        return []

    @staticmethod
    def _extract_dtype(node) -> str:
        try:
            if hasattr(node, "type") and isinstance(node.type, str):
                return node.type
            t = node.type.tensor_type
            return _ONNX_DTYPE_MAP.get(t.elem_type, "float32")
        except Exception:
            return "float32"

    @staticmethod
    def _select_providers(device: str) -> list[str]:
        providers = ["CPUExecutionProvider"]
        if device in ("cuda", "auto"):
            providers = ["CUDAExecutionProvider"] + providers
        elif device == "cpu":
            pass
        return providers

    @staticmethod
    def _coerce_to_numpy(batch: Any) -> np.ndarray:
        if isinstance(batch, np.ndarray):
            arr = batch.astype(np.float32)
        elif hasattr(batch, "numpy"):
            arr = batch.numpy().astype(np.float32)
        elif hasattr(batch, "shape"):
            arr = np.array(batch, dtype=np.float32)
        else:
            arr = np.array(batch, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis]  # (1, channels, time)
        return arr


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
