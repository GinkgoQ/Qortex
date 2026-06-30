"""Curated model contract registry.

Maps known model IDs to their verified ``InputContract`` and ``OutputContract``.
The registry is consulted by every model adapter *before* falling back to
config-file inference, so compatibility checking returns ``confirmed`` evidence
rather than ``inferred`` or ``unknown`` for models that appear here.

Adding a new entry
------------------
Add a ``ModelContractEntry`` to ``_REGISTRY`` and validate it against a real
inference run.  Only include fields you can confirm from the paper or official
model card — leave unknown fields as ``None`` rather than guessing.

Current coverage (13 models):
  EEG / Braindecode:  EEGNet, ShallowFBCSPNet, Deep4Net, EEGConformer
  Image / Vision:     ViT-B/16, ResNet-50, DeiT-B
  Audio:              Whisper-base
  Medical Imaging:    MONAI wholeBody_ct_segmentation, nnU-Net
  ONNX:               wholeBodySeg
"""

from __future__ import annotations

from dataclasses import dataclass

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    OutputContract,
)


@dataclass(frozen=True)
class ModelContractEntry:
    """One curated registry entry.

    Attributes
    ----------
    model_id:
        Canonical model identifier (HuggingFace ``org/name``, MONAI bundle id,
        or any string the adapter normalises to before lookup).
    aliases:
        Alternative identifiers that should resolve to the same entry (e.g.
        ``hf://org/name``, short names, versioned ids).
    provider:
        Which adapter owns this model.
    input_contract:
        Verified input requirements — only populate confirmed fields.
    output_contract:
        Verified output schema.
    estimated_memory_mb:
        Peak VRAM/RAM in float32 mode.
    notes:
        Human-readable note visible in suggest-models output.
    """
    model_id: str
    provider: str
    input_contract: InputContract
    output_contract: OutputContract
    aliases: tuple[str, ...] = ()
    estimated_memory_mb: float | None = None
    notes: str = ""


def _eeg_contract(
    n_channels: int | None,
    sampling_rate_hz: float | None,
    window_s: float | None = None,
    n_classes: int | None = None,
    classes: list[str] | None = None,
    evidence: EvidenceStatus = EvidenceStatus.confirmed,
) -> tuple[InputContract, OutputContract]:
    ic = InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        n_channels=n_channels,
        sampling_rate_hz=sampling_rate_hz,
        window_duration_s=window_s,
        dtype="float32",
        evidence_status=evidence,
    )
    oc = OutputContract(
        output_type="eeg_classification",
        n_classes=n_classes,
        classes=classes or [],
        produces_probabilities=True,
    )
    return ic, oc


def _image_contract(
    spatial_shape: tuple[int, ...],
    n_channels: int = 3,
    n_classes: int | None = None,
    classes: list[str] | None = None,
    dtype: str = "float32",
    intensity_range: tuple[float, float] | None = (0.0, 1.0),
) -> tuple[InputContract, OutputContract]:
    ic = InputContract(
        modality="image",
        axis_convention=AxisConvention.channels_first,
        n_channels=n_channels,
        spatial_shape=spatial_shape,
        dtype=dtype,
        intensity_range=intensity_range,
        evidence_status=EvidenceStatus.confirmed,
    )
    oc = OutputContract(
        output_type="image_classification",
        n_classes=n_classes,
        classes=classes or [],
        produces_probabilities=True,
    )
    return ic, oc


# ── Curated entries ────────────────────────────────────────────────────────────

_REGISTRY: list[ModelContractEntry] = [

    # ── EEG / Braindecode ──────────────────────────────────────────────────────

    ModelContractEntry(
        model_id="braindecode/EEGNet_8_2",
        provider="braindecode",
        aliases=("eegnet", "eegnet_8_2", "EEGNet_8_2"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_eeg_contract(n_channels=64, sampling_rate_hz=250.0, window_s=4.0, n_classes=4,
                           classes=["left_hand", "right_hand", "feet", "tongue"])
        ),
        estimated_memory_mb=120.0,
        notes="BCI Competition IV-2a 4-class motor imagery. Default: 64-ch, 250 Hz, 4-s windows.",
    ),

    ModelContractEntry(
        model_id="braindecode/ShallowFBCSPNet",
        provider="braindecode",
        aliases=("shallowfbcspnet", "shallow_fbcsp", "ShallowFBCSPNet"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_eeg_contract(n_channels=22, sampling_rate_hz=250.0, window_s=4.0, n_classes=4,
                           classes=["left_hand", "right_hand", "feet", "tongue"])
        ),
        estimated_memory_mb=80.0,
        notes="FBCSP-based shallow ConvNet. Best for BCI motor imagery.",
    ),

    ModelContractEntry(
        model_id="braindecode/Deep4Net",
        provider="braindecode",
        aliases=("deep4net", "Deep4Net"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_eeg_contract(n_channels=22, sampling_rate_hz=250.0, window_s=4.0, n_classes=4)
        ),
        estimated_memory_mb=200.0,
        notes="4-layer deep ConvNet for EEG. Larger than ShallowFBCSPNet.",
    ),

    ModelContractEntry(
        model_id="braindecode/EEGConformer",
        provider="braindecode",
        aliases=("eegconformer", "EEGConformer"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_eeg_contract(n_channels=22, sampling_rate_hz=250.0, window_s=4.0, n_classes=4,
                           evidence=EvidenceStatus.inferred)
        ),
        estimated_memory_mb=350.0,
        notes="Transformer-based EEG model. Channel and window counts are architecture defaults.",
    ),

    # ── Image classification (HuggingFace / ImageNet) ─────────────────────────

    ModelContractEntry(
        model_id="google/vit-base-patch16-224",
        provider="huggingface",
        aliases=("vit-base-patch16-224", "vit_base"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_image_contract(spatial_shape=(224, 224), n_channels=3, n_classes=1000,
                             intensity_range=(0.0, 1.0))
        ),
        estimated_memory_mb=340.0,
        notes="ViT-B/16 trained on ImageNet-21k, fine-tuned on ImageNet-1k.",
    ),

    ModelContractEntry(
        model_id="microsoft/resnet-50",
        provider="huggingface",
        aliases=("resnet-50", "resnet50"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_image_contract(spatial_shape=(224, 224), n_channels=3, n_classes=1000,
                             intensity_range=(0.0, 1.0))
        ),
        estimated_memory_mb=100.0,
        notes="ResNet-50 pretrained on ImageNet.",
    ),

    ModelContractEntry(
        model_id="facebook/deit-base-patch16-224",
        provider="huggingface",
        aliases=("deit-base-patch16-224", "deit_base"),
        **(lambda ic, oc: {"input_contract": ic, "output_contract": oc})(
            *_image_contract(spatial_shape=(224, 224), n_channels=3, n_classes=1000,
                             intensity_range=(0.0, 1.0))
        ),
        estimated_memory_mb=340.0,
        notes="Data-efficient Image Transformer (DeiT-B) for natural images.",
    ),

    # ── Audio (HuggingFace) ───────────────────────────────────────────────────

    ModelContractEntry(
        model_id="openai/whisper-base",
        provider="huggingface",
        aliases=("whisper-base", "whisper_base"),
        input_contract=InputContract(
            modality="audio",
            axis_convention=AxisConvention.channels_time,
            n_channels=1,
            sampling_rate_hz=16000.0,
            dtype="float32",
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="audio_transcription",
            produces_probabilities=False,
        ),
        estimated_memory_mb=290.0,
        notes="Whisper base model. 16 kHz mono audio only.",
    ),

    # ── Medical imaging (MONAI bundles) ───────────────────────────────────────

    ModelContractEntry(
        model_id="wholeBody_ct_segmentation",
        provider="monai",
        aliases=("monai/wholeBody_ct_segmentation", "wholebody_ct"),
        input_contract=InputContract(
            modality="ct",
            axis_convention=AxisConvention.channels_first,
            n_channels=1,
            spatial_shape=None,           # accepts arbitrary spatial size
            intensity_range=(-1024.0, 3071.0),  # Hounsfield Units
            dtype="float32",
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="segmentation",
            n_classes=105,
            produces_probabilities=False,
        ),
        estimated_memory_mb=4096.0,
        notes=(
            "MONAI whole-body CT segmentation (104 anatomical structures + background). "
            "Input must be in HU range. Arbitrary spatial size — model uses sliding window."
        ),
    ),

    ModelContractEntry(
        model_id="msd_brain_tumor",
        provider="monai",
        aliases=("monai/msd_brain_tumor", "msd_brain"),
        input_contract=InputContract(
            modality="mri",
            axis_convention=AxisConvention.channels_first,
            n_channels=4,            # FLAIR, T1w, T1gd, T2w
            spatial_shape=(240, 240, 155),
            dtype="float32",
            intensity_range=None,    # per-channel z-score expected (specified in InputContract)
            required_transforms=[
                {
                    "kind": "normalize",
                    "required_by": "model_contract.preprocessing",
                    "params": {"method": "channel_zscore", "eps": 1e-8},
                    "reversible": False,
                    "irreversible_reason": "MRI channel z-score normalization changes intensity scale.",
                }
            ],
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="segmentation",
            n_classes=4,
            classes=["background", "NCR_NET", "edema", "enhancing_tumour"],
            produces_probabilities=False,
        ),
        estimated_memory_mb=8000.0,
        notes=(
            "MSD BraTS tumour segmentation. Requires 4-channel MRI: FLAIR, T1w, T1gd, T2w. "
            "Each channel must be z-score normalised independently before inference."
        ),
    ),

    # ── Object detection (Ultralytics) ────────────────────────────────────────

    ModelContractEntry(
        model_id="ultralytics/yolov8n",
        provider="ultralytics",
        aliases=("yolov8n", "yolo_v8n"),
        input_contract=(lambda ic, oc: ic)(
            *_image_contract(spatial_shape=(640, 640), n_channels=3, intensity_range=(0.0, 1.0))
        ),
        output_contract=OutputContract(
            output_type="detection",
            n_classes=80,
            produces_probabilities=True,
        ),
        estimated_memory_mb=14.0,
        notes="YOLOv8 nano (3.2M params). 640×640 RGB. COCO 80 classes.",
    ),
]


# ── Lookup helpers ─────────────────────────────────────────────────────────────

def _normalise_id(model_id: str) -> str:
    """Strip provider prefixes and lower-case for alias matching."""
    s = model_id.strip()
    for prefix in ("hf://", "huggingface://", "braindecode://", "monai://"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.lower()


def lookup(model_id: str) -> ModelContractEntry | None:
    """Return the registry entry for *model_id*, or ``None`` if not found.

    Matching is case-insensitive and strips common provider prefixes.

    Parameters
    ----------
    model_id:
        Any form of the model identifier: ``"braindecode/EEGNet_8_2"``,
        ``"hf://google/vit-base-patch16-224"``, ``"eegnet"``, etc.
    """
    key = _normalise_id(model_id)
    for entry in _REGISTRY:
        if _normalise_id(entry.model_id) == key:
            return entry
        if any(_normalise_id(a) == key for a in entry.aliases):
            return entry
    return None


def list_entries(
    *,
    provider: str | None = None,
    modality: str | None = None,
) -> list[ModelContractEntry]:
    """Return all registry entries, optionally filtered.

    Parameters
    ----------
    provider:
        Filter by provider string (case-insensitive): ``"braindecode"``,
        ``"huggingface"``, ``"monai"``, ``"ultralytics"``, ``"onnx"``.
    modality:
        Filter by ``input_contract.modality`` string.
    """
    results = list(_REGISTRY)
    if provider:
        p = provider.lower()
        results = [e for e in results if e.provider.lower() == p]
    if modality:
        m = modality.lower()
        results = [
            e for e in results
            if str(getattr(e.input_contract, "modality", "") or "").lower() == m
        ]
    return results
