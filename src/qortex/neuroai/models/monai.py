"""MONAI Bundle model adapter.

MONAI bundles are self-contained model packages (ZIP or directory) with:
  - ``configs/metadata.json``  — model metadata
  - ``configs/inference.json`` — inference configuration
  - ``models/model.pt``        — model weights
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import sys
import tempfile
import zipfile
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any

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
from qortex.neuroai.models.artifacts import (
    download_vista3d_bundle,
    resolve_vista3d_bundle,
    resolve_vista3d_weights,
)
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
        self._use_autocast = False

    # ── ModelAdapter interface ────────────────────────────────────────────────

    def inspect(self) -> ModelProfile:
        _require_monai()
        self._resolve_bundle()
        metadata_task = self._metadata.get("task")
        if isinstance(metadata_task, dict):
            metadata_task = metadata_task.get("name")
        task = str(metadata_task or self._spec.task or "segmentation")
        in_channels, spatial_dims, output_classes = self._parse_network_def()

        return ModelProfile(
            model_id=self._spec.id,
            provider="monai",
            task=task,
            revision=self._metadata.get("version"),
            model_hash=self._compute_model_hash(),
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
            warnings=(
                _monai_transform_warnings(self._infer_config, self._spec.extra)
                + _monai_postprocess_warnings(self._infer_config)
            ),
        )

    def _compute_model_hash(self) -> str | None:
        # Provenance: record the exact checkpoint bytes that will run, not
        # just the bundle id/version, so a run can be traced back to the
        # weights that actually produced it. Only possible once the bundle
        # is resolved to a real local path with a models/model.pt on disk --
        # a bundle id that isn't resolvable offline (e.g. hub download not
        # yet fetched) has nothing to hash, so this honestly stays None
        # rather than fabricating a value.
        if self._bundle_dir is None:
            return None
        model_pt = self._resolve_weight_path()
        if model_pt is None:
            return None
        return _sha256_file(model_pt)

    def required_input(self) -> InputContract:
        in_channels, _, _ = self._parse_network_def()
        # spatial_shape carries ONLY spatial dimensions (Z,Y,X / H,W), never
        # the channel count -- n_channels is the separate, dedicated field
        # every other adapter in this codebase already uses this way (e.g.
        # zoo/monai_imaging.py's wholeBody_ct_segmentation entry). Bundle
        # configs never confirm the actual spatial extent (MONAI bundles are
        # sliding-window and accept arbitrary spatial size), so there is
        # nothing real to put here -- leaving it None is the honest value,
        # not a placeholder tuple of -1s that reads as a resolved shape to
        # any downstream consumer (e.g. resource estimation) that doesn't
        # know to special-case -1.
        return InputContract(
            modality=_detect_modality(self._metadata),
            n_channels=in_channels,
            sampling_rate_hz=None,
            spatial_shape=None,
            dtype="float32",
            axis_convention=AxisConvention.batch_channels_xyz,
            required_transforms=_parse_required_transforms(self._infer_config, self._spec.extra),
            evidence_status=(
                EvidenceStatus.confirmed if in_channels else EvidenceStatus.inferred
            ),
        )

    def _zoo_entry(self):
        """Look up this bundle's own zoo registry entry, if it has one.

        Used to detect generative bundles (entry_type=generative_model) so
        this adapter never mislabels their output as a segmentation mask --
        the same registry entry that declares output_type="image_generation"
        must be consulted here rather than this adapter blindly assuming
        every MONAI bundle is a segmentation model.
        """
        try:
            from qortex.neuroai.models.zoo.registry import lookup
            return lookup(self._spec.id)
        except Exception:
            return None

    def _is_generative_bundle(self) -> bool:
        entry = self._zoo_entry()
        return entry is not None and entry.entry_type.value == "generative_model"

    def output_schema(self) -> OutputContract:
        entry = self._zoo_entry()
        if entry is not None and entry.output_contract is not None:
            # Trust the curated registry's confirmed output contract over a
            # blind "segmentation" guess -- this is what actually fixes the
            # generative-bundle mislabeling bug: entry.output_contract.output_type
            # is "image_generation" for every zoo/monai_generative.py entry.
            return entry.output_contract
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
            # Load weights. Fail closed if no checkpoint is present -- silently
            # continuing with the freshly-constructed network's random-init
            # parameters would let a caller believe they ran a real trained
            # segmentation model when they actually ran untrained noise.
            # Bypass only via an explicit opt-in (mirrors the trust_remote_code
            # explicit-opt-in pattern in models/plugin.py), for genuine
            # architecture-only smoke testing.
            model_pt = self._resolve_weight_path()
            allow_missing_weights = bool(self._spec.extra.get("allow_missing_weights", False))
            if model_pt is None:
                if not allow_missing_weights:
                    raise ModelAdapterError(
                        f"MONAI bundle {self._spec.id!r} has no valid checkpoint. "
                        f"Bundle path: {self._bundle_dir}. Refusing to run with randomly-initialized weights. "
                        "Pass model.allow_missing_weights: true only for deliberate "
                        "architecture-only testing."
                    )
                log.warning(
                    "MONAI bundle %s has no checkpoint; running with random-init "
                    "weights because allow_missing_weights=True was explicitly set.",
                    self._spec.id,
                )
            else:
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
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MONAI bundle from {self._bundle_dir}: {exc}"
            ) from exc

        # ponytail: full-model .half() is dropped in favor of torch.autocast
        # in predict() — autocast keeps numerically-sensitive ops (e.g. norm
        # layers) in fp32 automatically instead of forcing every parameter
        # to fp16, which is the standard mature mixed-precision pattern.
        self._use_autocast = bool(runtime.fp16 and "cuda" in self._device)
        self._loaded = True
        log.info("Loaded MONAI bundle: %s on %s", self._spec.id, self._device)

    def predict(self, batch: Any) -> ModelOutput:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        if self._is_generative_bundle():
            # There is no real generative execution path yet -- no sampler,
            # no conditioning contract, no seed handling, no synthetic-output
            # writer. Running this bundle through the segmentation-style
            # sliding-window inference below would silently mislabel a
            # generative network's raw output as a segmentation mask. Refuse
            # rather than fabricate a result this adapter cannot honestly
            # produce.
            raise ModelAdapterError(
                f"{self._spec.id!r} is a generative model entry; MONAIBundleAdapter "
                "has no generative execution path (sampler/conditioning/seed handling) "
                "implemented yet. Refusing to run segmentation-style inference against it."
            )
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

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_autocast
            else _nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
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
        threshold = self._inference_settings.get("threshold")
        if threshold is not None:
            # A threshold declares independent output channels (for example
            # BraTS TC/WT/ET nested regions). Argmax would incorrectly force
            # those overlapping regions into mutually exclusive classes.
            mask = (raw[0] >= float(threshold)).astype(np.uint8)
        elif raw.ndim >= 2 and raw.shape[int(argmax_axis)] > 1:
            mask = np.argmax(raw, axis=int(argmax_axis))[0]
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
        if self._is_vista3d_spec():
            explicit_bundle = self._spec.extra.get("bundle") or self._spec.extra.get("bundle_dir")
            if self._spec.extra.get("download_artifacts") or self._spec.extra.get("download_bundle"):
                self._bundle_dir = download_vista3d_bundle(
                    revision=self._spec.revision,
                    cache_dir=explicit_bundle,
                    token=self._spec.extra.get("hf_token"),
                )
            else:
                self._bundle_dir = resolve_vista3d_bundle(explicit_bundle)
            self._metadata = _load_json(self._bundle_dir / "configs" / "metadata.json")
            if not self._metadata:
                self._metadata = _load_json(self._bundle_dir / "metadata.json")
            self._infer_config = _load_json(self._bundle_dir / "configs" / "inference.json")
            return
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
                bundle_name = str(self._spec.extra.get("bundle_name") or self._spec.id.split("/")[-1])
                if bundle_name.startswith("monai."):
                    bundle_name = bundle_name.split(".", 1)[1]
                bundle_dir = monai.bundle.load(
                    name=bundle_name,
                    version=self._spec.revision,
                )
                self._bundle_dir = Path(bundle_dir)
            except Exception as exc:
                raise ValueError(
                    f"Cannot resolve MONAI bundle {self._spec.id!r}: {exc}"
                ) from exc

        self._metadata = _load_json(self._bundle_dir / "configs" / "metadata.json")
        self._infer_config = _load_json(self._bundle_dir / "configs" / "inference.json")

    def _is_vista3d_spec(self) -> bool:
        provider = str(getattr(self._spec, "provider", "") or "").lower()
        model_id = str(getattr(self._spec, "id", "") or "").lower()
        return provider == "vista3d" or model_id in {"monai.vista3d", "vista3d"}

    def _resolve_weight_path(self) -> Path | None:
        if self._bundle_dir is None:
            return None
        if self._is_vista3d_spec():
            return resolve_vista3d_weights(self._bundle_dir)
        model_pt = self._bundle_dir / "models" / "model.pt"
        return model_pt if model_pt.exists() else None

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

    def load(self, runtime: RuntimeSpec) -> None:
        _require_monai()
        self._device = _resolve_device(runtime.device)
        self._resolve_bundle()
        if self._is_hf_bundle():
            try:
                import torch
                bundle_dir = str(self._bundle_dir)
                if bundle_dir not in sys.path:
                    sys.path.insert(0, bundle_dir)
                vista_model_mod = importlib.import_module("vista3d_model")
                if not hasattr(vista_model_mod.VISTA3DModel, "all_tied_weights_keys"):
                    vista_model_mod.VISTA3DModel.all_tied_weights_keys = {}
                helper_mod = importlib.import_module("hugging_face_pipeline")
                helper = helper_mod.HuggingFacePipelineHelper("vista3d")
                model_dir = self._bundle_dir / "vista3d_pretrained_model"
                device = torch.device(self._device)
                self._model = helper.init_pipeline(str(model_dir), device=device)
                self._loaded = True
                return
            except Exception as exc:
                raise ModelAdapterError(
                    f"Failed to load VISTA3D-HF pipeline from {self._bundle_dir}: {exc}"
                ) from exc
        super().load(runtime)

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
        if self._is_hf_bundle():
            return self._predict_hf_with_prompt(batch, prompt)
        raise ModelAdapterError(
            "VISTA3D prompted inference requires the MONAI/VISTA3D-HF artifact. "
            "Run `qortex neuroai zoo download-artifact monai.vista3d` and pass "
            "--bundle if using a custom path."
        )

    def _is_hf_bundle(self) -> bool:
        return (
            self._bundle_dir is not None
            and (self._bundle_dir / "hugging_face_pipeline.py").is_file()
            and (self._bundle_dir / "vista3d_pretrained_model" / "config.json").is_file()
        )

    def _predict_hf_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        if self._model is None:
            raise ModelAdapterError("VISTA3D-HF pipeline is not loaded.")
        image_path = _coerce_image_path(batch)
        inputs: dict[str, Any] = {"image": image_path}
        if prompt.points is not None:
            inputs["points"] = [list(point[:3]) for point in prompt.points]
            inputs["point_labels"] = list(prompt.point_labels or [1] * len(prompt.points))
        output_dir = Path(tempfile.mkdtemp(prefix="qortex_vista3d_hf_"))
        try:
            result = self._model(
                inputs,
                output_dir=str(output_dir),
                amp=False,
                save_output=True,
                separate_folder=False,
            )
            mask_path = _find_first_nifti(output_dir)
            if mask_path is None:
                raise ModelAdapterError(
                    f"VISTA3D-HF completed without writing a NIfTI mask under {output_dir}."
                )
            import nibabel as nib
            mask = np.asarray(nib.load(str(mask_path)).get_fdata()).astype(np.uint8)
            return ModelOutput(
                output_type="segmentation",
                raw=result,
                mask=mask,
                metadata={
                    "provider": "vista3d",
                    "artifact": "MONAI/VISTA3D-HF",
                    "mask_path": str(mask_path),
                    "output_dir": str(output_dir),
                },
            )
        except ModelAdapterError:
            raise
        except Exception as exc:
            raise ModelAdapterError(f"VISTA3D-HF prompted inference failed: {exc}") from exc


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


def _coerce_image_path(batch: Any) -> str:
    if isinstance(batch, (str, Path)):
        path = Path(batch).expanduser().resolve()
        if not path.is_file():
            raise ModelAdapterError(f"VISTA3D-HF image path does not exist: {path}")
        return str(path)
    source_path = getattr(batch, "path", None) or getattr(batch, "source_path", None)
    if source_path:
        path = Path(source_path).expanduser().resolve()
        if path.is_file():
            return str(path)
    raise ModelAdapterError(
        "VISTA3D-HF requires a real NIfTI file path so its official preprocessing "
        "pipeline can preserve affine/original image metadata."
    )


def _find_first_nifti(root: Path) -> Path | None:
    for path in sorted(root.rglob("*")):
        name = path.name.lower()
        if path.is_file() and (name.endswith(".nii") or name.endswith(".nii.gz")):
            return path
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "activation": (
            extra.get("activation")
            or _find_config_value(config, ("activation", "post_activation"))
            or _configured_activation(config)
        ),
        "argmax_axis": int(extra.get("argmax_axis", _find_config_value(config, ("argmax_axis",)) or 1)),
        "threshold": extra.get("threshold", _find_config_value(config, ("threshold",))),
        "label_map": extra.get("label_map") or _find_config_value(config, ("label_map", "labels")) or {},
    }
    if settings["overlap"] < 0 or settings["overlap"] >= 1:
        raise ValueError(f"MONAI sliding-window overlap must be in [0, 1), got {settings['overlap']}")
    return settings


def _configured_activation(config: dict[str, Any]) -> str | None:
    """Resolve a declarative MONAI ``Activationsd`` configuration."""
    postprocessing = config.get("postprocessing")
    stack = [postprocessing]
    while stack:
        value = stack.pop()
        if isinstance(value, list):
            stack.extend(value)
            continue
        if not isinstance(value, dict):
            continue
        target = str(value.get("_target_") or value.get("target") or "")
        if target.rsplit(".", 1)[-1].lower() in {"activations", "activationsd"}:
            if value.get("sigmoid") is True:
                return "sigmoid"
            if value.get("softmax") is True:
                return "softmax"
        stack.extend(value.values())
    return None


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


# Postprocessing transforms Qortex's predict() actually executes today
# (via activation/argmax/threshold parsed from _parse_inference_settings).
# Anything else declared in the bundle's postprocessing config runs
# unexecuted -- e.g. KeepLargestConnectedComponentd, Invertd (undoes
# resize/orientation back to source space), SaveImaged.
_HANDLED_POSTPROCESS_TRANSFORMS = {"activationsd", "asdiscreted"}


def _monai_postprocess_warnings(config: dict) -> list[WarningItem]:
    postproc = config.get("postprocessing")
    if not postproc:
        return []
    names = _collect_monai_transform_names(postproc)
    unexecuted = sorted(
        name for name in names
        if name.lower() not in _HANDLED_POSTPROCESS_TRANSFORMS
    )
    if not unexecuted:
        return []
    return [WarningItem(
        code="MONAI_POSTPROCESSING_NOT_EXECUTED",
        message=(
            "MONAI bundle declares postprocessing transforms Qortex does not "
            f"execute: {', '.join(unexecuted[:12])}. Only activation "
            "(softmax/sigmoid) and argmax/threshold from the bundle's "
            "inference settings are applied -- the output is not guaranteed "
            "bundle-faithful for steps like connected-component filtering or "
            "inverse resampling to source space."
        ),
        severity="warning",
        evidence={"monai_postprocess_transforms": unexecuted[:32]},
        suggestion=(
            "Apply any remaining bundle postprocessing steps downstream "
            "yourself, or treat the raw/mask output as pre-final."
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
