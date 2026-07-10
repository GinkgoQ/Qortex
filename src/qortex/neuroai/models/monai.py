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
import tempfile
from typing import Any
import zipfile

import numpy as np

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    ModelProfile,
    OutputContract,
    WarningItem,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType
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
        self._inference_settings: dict[str, Any] = {}
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
        in_channels, spatial_dims, output_classes = self._parse_network_def()

        return ModelProfile(
            model_id=self._spec.id,
            provider="monai",
            task=task,
            revision=self._metadata.get("version"),
            model_hash=None,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
            warnings=_monai_transform_warnings(self._infer_config, self._spec.extra),
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
            axis_convention=AxisConvention.batch_channels_xyz,
            required_transforms=_parse_required_transforms(self._infer_config, self._spec.extra),
            evidence_status=(
                EvidenceStatus.confirmed if in_channels else EvidenceStatus.inferred
            ),
        )

    def output_schema(self) -> OutputContract:
        _, _, n_classes = self._parse_network_def()
        return OutputContract(
            output_type="segmentation",
            n_classes=n_classes,
            produces_probabilities=False,
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
            self._inference_settings = _parse_inference_settings(
                self._infer_config,
                self._spec.extra,
                spatial_dims=self._parse_network_def()[1],
            )
            # Load weights
            model_pt = self._bundle_dir / "models" / "model.pt"
            if model_pt.exists():
                state = torch.load(str(model_pt), map_location=self._device, weights_only=True)
                if "state_dict" in state:
                    state = state["state_dict"]
                result = self._model.load_state_dict(state, strict=False)
                if result.missing_keys or result.unexpected_keys:
                    raise ModelAdapterError(
                        "MONAI bundle state_dict mismatch: "
                        f"missing={list(result.missing_keys)!r}, "
                        f"unexpected={list(result.unexpected_keys)!r}"
                    )
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

        if isinstance(batch, torch.Tensor):
            x = batch.to(self._device)
        elif isinstance(batch, np.ndarray):
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
            out = monai.inferers.sliding_window_inference(
                x,
                roi_size=self._inference_settings.get("roi_size"),
                sw_batch_size=int(self._inference_settings.get("sw_batch_size", 1)),
                predictor=self._model,
                overlap=float(self._inference_settings.get("overlap", 0.25)),
            )

        out = _apply_monai_postprocess(out, self._inference_settings)
        raw = out.cpu().numpy()
        argmax_axis = self._inference_settings.get("argmax_axis", 1)
        if raw.ndim >= 2 and raw.shape[int(argmax_axis)] > 1:
            mask = np.argmax(raw, axis=int(argmax_axis))[0]
        else:
            threshold = self._inference_settings.get("threshold")
            if threshold is not None:
                mask = (raw[0, 0] >= float(threshold)).astype(np.uint8)
            else:
                mask = raw[0, 0]
        return ModelOutput(
            output_type="segmentation",
            raw=raw,
            mask=mask,
            metadata={
                "monai_inference": self._inference_settings,
                "label_map": self._inference_settings.get("label_map", {}),
            },
        )

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
            tmp = Path(tempfile.mkdtemp(prefix="qortex_monai_"))
            _safe_extract_zip(candidate, tmp)
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


class VISTA3DAdapter(MONAIBundleAdapter, PromptableModelAdapter):
    """VISTA3D: a MONAI bundle with both automatic and point/box-prompted
    3D CT segmentation. Reuses MONAIBundleAdapter's real bundle loading and
    sliding-window inference entirely -- this class only adds the prompt
    path on top, per docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
    section 12.4 ("use one canonical entry ID with two capabilities instead
    of duplicate entries").

    VISTA3D's paper (arXiv:2406.05285) documents both automatic
    whole-organ segmentation and interactive point/box-prompted
    segmentation; text prompts are not part of its documented interface
    and are deliberately not declared here.
    """

    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            supports_automatic_mode=True,
            evidence_status=EvidenceStatus.confirmed,
        )

    def predict(self, batch: Any) -> ModelOutput:
        # MRO would otherwise resolve predict() to MONAIBundleAdapter's
        # implementation (it comes first in the base list), silently
        # bypassing PromptableModelAdapter's automatic/prompt-required
        # dispatch. Routing through it explicitly means a future change to
        # supports_automatic_mode is respected rather than silently
        # ignored.
        return PromptableModelAdapter.predict(self, batch)

    def predict_automatic(self, batch: Any) -> ModelOutput:
        # VISTA3D's already-proven whole-organ automatic segmentation path
        # -- identical to MONAIBundleAdapter.predict() for every other
        # MONAI segmentation bundle in the zoo.
        return MONAIBundleAdapter.predict(self, batch)

    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        violations = prompt.validate_against(self.interaction_contract())
        if violations:
            raise ModelAdapterError(
                "VISTA3D prompt is invalid: " + "; ".join(violations)
            )
        raise ModelAdapterError(
            "VISTA3D prompted inference is not executable in Qortex yet: "
            "the MONAI bundle-specific prompt transforms and coordinate "
            "restoration path are not wired. Use automatic mode only after "
            "the bundle has passed compatibility validation."
        )


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
    if path.exists():
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in MONAI bundle config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"MONAI bundle config {path} must contain a JSON object")
        return data
    return {}


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"MONAI bundle ZIP contains unsafe path: {member.filename!r}")
        zf.extractall(destination)


def _parse_inference_settings(
    config: dict,
    extra: dict[str, Any],
    *,
    spatial_dims: int,
) -> dict[str, Any]:
    roi_size = (
        extra.get("roi_size")
        or _find_config_value(config, ("roi_size", "patch_size", "spatial_size"))
        or ((96, 96, 96) if spatial_dims == 3 else (256, 256))
    )
    settings = {
        "roi_size": _as_tuple(roi_size, spatial_dims),
        "sw_batch_size": int(
            extra.get("sw_batch_size")
            or _find_config_value(config, ("sw_batch_size", "sliding_window_batch_size"))
            or 1
        ),
        "overlap": float(
            extra.get("overlap")
            or _find_config_value(config, ("overlap", "sliding_window_overlap"))
            or 0.25
        ),
        "activation": extra.get("activation") or _find_config_value(config, ("activation", "post_activation")),
        "argmax_axis": int(extra.get("argmax_axis", _find_config_value(config, ("argmax_axis",)) or 1)),
        "threshold": extra.get("threshold", _find_config_value(config, ("threshold",))),
        "label_map": extra.get("label_map") or _find_config_value(config, ("label_map", "labels")) or {},
    }
    if settings["overlap"] < 0 or settings["overlap"] >= 1:
        raise ValueError(f"MONAI sliding-window overlap must be in [0, 1), got {settings['overlap']}")
    return settings


def _parse_required_transforms(config: dict, extra: dict[str, Any]) -> list[dict[str, Any]]:
    required = extra.get("required_transforms")
    if required:
        return list(required)
    # MONAI configs frequently declare spacing/orientation as transform objects,
    # but converting those to executable Qortex transforms requires source
    # affine/orientation provenance and inverse tracking. Do not guess here.
    return []


def _monai_transform_warnings(config: dict, extra: dict[str, Any]) -> list[WarningItem]:
    if extra.get("required_transforms"):
        return []
    transform_names = _collect_monai_transform_names(config)
    unsupported = sorted(
        name for name in transform_names
        if _is_preprocessing_transform_requiring_mapping(name)
    )
    if not unsupported:
        return []
    return [WarningItem(
        code="MONAI_REQUIRED_PREPROCESSING_UNMAPPED",
        message=(
            "MONAI bundle config declares preprocessing transforms that Qortex "
            "will not translate implicitly: "
            f"{', '.join(unsupported[:12])}. Provide model.required_transforms "
            "with explicit Qortex transforms before running this bundle."
        ),
        severity="error",
        evidence={"monai_transforms": unsupported[:32]},
        suggestion=(
            "Declare explicit Qortex required_transforms for spacing/orientation/"
            "intensity/crop/pad steps, or use a source already preprocessed exactly "
            "as the MONAI bundle expects."
        ),
    )]


def _collect_monai_transform_names(obj: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"_target_", "target", "name", "type"} and isinstance(value, str):
                names.add(value.rsplit(".", 1)[-1])
            if isinstance(key, str) and key.endswith("d") and key[:1].isupper():
                names.add(key)
            names.update(_collect_monai_transform_names(value))
    elif isinstance(obj, list):
        for value in obj:
            names.update(_collect_monai_transform_names(value))
    elif isinstance(obj, str) and obj.endswith("d") and obj[:1].isupper():
        names.add(obj.rsplit(".", 1)[-1])
    return names


def _is_preprocessing_transform_requiring_mapping(name: str) -> bool:
    low = name.lower()
    return any(
        token in low
        for token in (
            "spacing", "orientation", "scaleintensity", "normalizeintensity",
            "cropforeground", "resize", "resized", "spatialpad", "borderpad",
            "divisiblepad", "reorient",
        )
    )


def _apply_monai_postprocess(out: Any, settings: dict[str, Any]) -> Any:
    activation = str(settings.get("activation") or "").lower()
    if not activation:
        return out
    import torch
    if activation == "softmax":
        return torch.softmax(out, dim=int(settings.get("argmax_axis", 1)))
    if activation == "sigmoid":
        return torch.sigmoid(out)
    return out


def _find_config_value(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key) in keys:
                return value
            found = _find_config_value(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_config_value(value, keys)
            if found is not None:
                return found
    return None


def _as_tuple(value: Any, n: int) -> tuple[int | float, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("x", ",").split(",") if p.strip()]
        values = [float(p) for p in parts]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value] * n
    if len(values) == 1:
        values = values * n
    return tuple(int(v) if float(v).is_integer() else float(v) for v in values[:n])
