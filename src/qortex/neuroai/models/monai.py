"""MONAI Bundle model adapter.

MONAI bundles are self-contained model packages (ZIP or directory) with:
  - ``configs/metadata.json``  — model metadata
  - ``configs/inference.json`` — inference configuration
  - ``models/model.pt``        — model weights
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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


class MONAIBundleAdapter(ModelAdapter):
    """Adapter for MONAI bundle models.

    Parameters
    ----------
    spec:
        ``ModelSpec`` with ``provider="monai"`` and:
        - ``id=<local path to bundle ZIP or directory>``
        - or ``id=<org/bundle_name>`` for download from MONAI Hub
    """

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._bundle_dir: Path | None = None
        self._metadata: dict = {}
        self._infer_config: dict = {}
        self._device = "cpu"

    # ── ModelAdapter interface ────────────────────────────────────────────────

    def inspect(self) -> ModelProfile:
        _require_monai()
        self._resolve_bundle()
        task = (
            self._metadata.get("task", {}).get("name", "")
            or self._spec.task
            or "segmentation"
        )
        modality = _detect_modality(self._metadata)
        in_channels, spatial_dims, output_classes = self._parse_network_def()

        return ModelProfile(
            model_id=self._spec.id,
            provider="monai",
            task=task,
            revision=self._metadata.get("version"),
            model_hash=None,
            n_parameters=None,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
            extra={
                "spatial_dims": spatial_dims,
                "bundle_dir": str(self._bundle_dir),
                "monai_version": self._metadata.get("monai_version", "unknown"),
            },
        )

    def required_input(self) -> InputContract:
        in_channels, spatial_dims, _ = self._parse_network_def()
        shape = (
            [in_channels, -1, -1, -1] if spatial_dims == 3
            else [in_channels, -1, -1] if spatial_dims == 2
            else None
        )
        return InputContract(
            modality=_detect_modality(self._metadata),
            n_channels=in_channels,
            sampling_rate_hz=None,
            spatial_shape=shape,
            dtype="float32",
            axis_convention=AxisConvention.batch_channels_spatial,
            evidence={
                "n_channels": EvidenceStatus.confirmed if in_channels else EvidenceStatus.inferred,
                "spatial_shape": EvidenceStatus.inferred,
            },
        )

    def output_schema(self) -> OutputContract:
        _, _, n_classes = self._parse_network_def()
        return OutputContract(
            output_type="segmentation",
            n_classes=n_classes,
            class_labels={},
            evidence={
                "output_type": EvidenceStatus.confirmed,
                "n_classes": EvidenceStatus.confirmed if n_classes else EvidenceStatus.inferred,
            },
        )

    def load(self, runtime: RuntimeSpec) -> None:
        import torch
        monai = _require_monai()
        self._device = _resolve_device(runtime.device)
        self._resolve_bundle()

        config_path = self._bundle_dir / "configs" / "inference.json"
        if not config_path.exists():
            config_path = self._bundle_dir / "configs" / "train.json"

        try:
            parser = monai.bundle.ConfigParser()
            parser.read_config(str(config_path))
            parser["device"] = self._device
            self._model = parser.get_parsed_content("network_def")
            # Load weights
            model_pt = self._bundle_dir / "models" / "model.pt"
            if model_pt.exists():
                state = torch.load(str(model_pt), map_location=self._device, weights_only=True)
                if "state_dict" in state:
                    state = state["state_dict"]
                self._model.load_state_dict(state, strict=False)
            self._model.eval()
            self._model.to(self._device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MONAI bundle from {self._bundle_dir}: {exc}"
            ) from exc

        if runtime.fp16 and "cuda" in self._device:
            self._model = self._model.half()
        self._loaded = True
        log.info("Loaded MONAI bundle: %s on %s", self._spec.id, self._device)

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        import torch
        monai = _require_monai()

        if isinstance(batch, np.ndarray):
            x = torch.from_numpy(batch.astype(np.float32)).to(self._device)
        elif hasattr(batch, "data"):
            x = torch.from_numpy(np.array(batch.data, dtype=np.float32)).to(self._device)
        else:
            x = batch

        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)  # [Z,Y,X] → [1,1,Z,Y,X]
        elif x.ndim == 4:
            x = x.unsqueeze(0)  # [C,Z,Y,X] → [1,C,Z,Y,X]

        with torch.no_grad():
            try:
                out = monai.inferers.sliding_window_inference(
                    x, roi_size=(96, 96, 96), sw_batch_size=1,
                    predictor=self._model, overlap=0.5,
                )
            except Exception:
                out = self._model(x)

        raw = out.cpu().numpy()
        mask = np.argmax(raw[0], axis=0) if raw.shape[1] > 1 else raw[0, 0]
        return ModelOutput(output_type="segmentation", raw=raw, mask=mask)

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_bundle(self) -> None:
        if self._bundle_dir is not None:
            return
        monai = _require_monai()
        candidate = Path(self._spec.id)

        if candidate.exists() and candidate.is_dir():
            self._bundle_dir = candidate
        elif candidate.exists() and candidate.suffix == ".zip":
            import tempfile, zipfile
            tmp = Path(tempfile.mkdtemp(prefix="qortex_monai_"))
            with zipfile.ZipFile(candidate) as zf:
                zf.extractall(tmp)
            inner = [d for d in tmp.iterdir() if d.is_dir()]
            self._bundle_dir = inner[0] if inner else tmp
        else:
            # Try MONAI Hub download
            try:
                bundle_dir = monai.bundle.load(
                    name=self._spec.id.split("/")[-1],
                    version=self._spec.revision,
                )
                self._bundle_dir = Path(bundle_dir)
            except Exception as exc:
                raise ValueError(
                    f"Cannot resolve MONAI bundle {self._spec.id!r}: {exc}"
                ) from exc

        self._metadata = _load_json(self._bundle_dir / "configs" / "metadata.json")
        self._infer_config = _load_json(self._bundle_dir / "configs" / "inference.json")

    def _parse_network_def(self) -> tuple[int | None, int, int | None]:
        cfg = self._infer_config
        net = cfg.get("network_def", {})
        in_channels = net.get("in_channels")
        spatial_dims = int(net.get("spatial_dims", 3))
        out_channels = net.get("out_channels")
        return in_channels, spatial_dims, out_channels


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_monai():
    try:
        import monai
        return monai
    except ImportError:
        raise ImportError(
            "MONAI model adapter requires monai. "
            "Install with: pip install 'qortex[monai]' or pip install monai"
        )


def _resolve_device(device: str) -> str:
    try:
        import torch
        if device in ("auto", "gpu"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    except ImportError:
        return "cpu"


def _detect_modality(metadata: dict) -> str:
    desc = str(metadata.get("description", "")).lower()
    for m in ("ct", "mri", "pet", "x-ray", "pathology", "ultrasound"):
        if m in desc:
            return m
    return "mri"


def _load_json(path: Path) -> dict:
    try:
        if path.exists():
            with path.open() as f:
                return json.load(f)
    except Exception:
        pass
    return {}
