"""qortex.datasets.msd_brain — Medical Segmentation Decathlon Brain Tumour.

Usage::

    from qortex.datasets import msd_brain

    card = msd_brain.describe()
    bundle = msd_brain.load_data(split="train", max_cases=20)
    image, mask = bundle.load_pair(0)
    # image: [4, x, y, z] (FLAIR, T1w, T1gd, T2w)
    # mask:  [x, y, z] with values {0, 1, 2, 3}

Dataset facts
-------------
- Task 01 from the Medical Segmentation Decathlon (2018).
- 750 volumes: 484 training + 266 test (no test masks released).
- Multimodal: FLAIR, T1w, T1-contrast enhanced (T1ce/T1gd), T2w.
- Source: BraTS 2016 / BraTS 2017 challenge data.
- Mask labels: 0=background, 1=NCR/NET (necrotic core), 2=ED (edema), 3=ET (enhancing tumour).
- MONAI DecathlonDataset handles download and caching.
- License: CC BY-SA 4.0 (attribution + share-alike).
"""

from __future__ import annotations

from pathlib import Path

from qortex.datasets._base import DatasetCard, SegmentationBundle, _REGISTRY
from qortex.datasets._cache import dataset_cache_dir

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="msd_brain",
    full_name="Medical Segmentation Decathlon — Task01: Brain Tumour",
    version="1.0",
    source_url="http://medicaldecathlon.com/",
    license="Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
    citation=(
        "Antonelli M, et al. The Medical Segmentation Decathlon. "
        "Nat Commun. 2022;13(1):4128. doi:10.1038/s41467-022-30695-9"
    ),
    modality="mri",
    n_subjects=750,
    n_channels=4,
    sampling_hz=None,
    image_shape=(240, 240, 155),
    n_classes=4,
    description=(
        "750 multimodal brain MRI volumes for tumour segmentation.\n"
        "484 training (with masks) + 266 test (no masks released).\n"
        "Modalities: FLAIR, T1w, T1-contrast enhanced, T2w.\n"
        "Mask labels: 0=background, 1=NCR/NET, 2=edema, 3=enhancing tumour.\n"
        "Downloaded via MONAI's DecathlonDataset with automatic caching."
    ),
    tasks=["brain_tumour_segmentation"],
    tutorial_ids=["T08"],
    size_gb_approx=16.0,
    requires_registration=False,
    access_instructions=(
        "MONAI will automatically download from the official MSD mirrors.\n"
        "Set QORTEX_DATA_DIR or pass local_root= to control the cache location.\n"
        "Requires: pip install 'qortex[monai]'"
    ),
)
_REGISTRY.register(_CARD)


# ── Label map ─────────────────────────────────────────────────────────────────

LABEL_MAP = {
    0: "background",
    1: "NCR_NET",   # Necrotic core / non-enhancing tumour
    2: "edema",
    3: "enhancing_tumour",
}

MODALITIES = ["FLAIR", "T1w", "T1gd", "T2w"]


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    local_root: Path | str | None = None,
    split: str = "train",
    max_cases: int | None = None,
    download: bool = True,
    seed: int = 42,
) -> SegmentationBundle:
    """Load MSD Brain Tumour data via MONAI.

    Parameters
    ----------
    local_root : Directory for dataset storage.
                 Falls back to QORTEX_DATA_DIR/msd_brain.
    split      : "train" (484 cases with masks) or "test" (266 cases, no masks).
    max_cases  : Limit for fast experiments.
    download   : Download if not already cached.
    seed       : Random seed for reproducible split (if MONAI applies one).

    Returns
    -------
    SegmentationBundle with:
      - image_paths: list of lists shaped (case, modality)
      - mask_paths: list of mask paths (empty for test split)
      - label_map: {0: background, 1: NCR_NET, 2: edema, 3: enhancing_tumour}
      - modalities: ['FLAIR', 'T1w', 'T1gd', 'T2w']

    Examples
    --------
    >>> bundle = msd_brain.load_data(split="train", max_cases=20)
    >>> image, mask = bundle.load_pair(0)
    >>> print(image.shape, mask.shape)  # (4, 240, 240, 155), (240, 240, 155)
    """
    if local_root is None:
        local_root = dataset_cache_dir("msd_brain")
    local_root = Path(local_root)

    # Try MONAI first (preferred path)
    try:
        return _load_via_monai(local_root, split, max_cases, download, seed)
    except ImportError:
        pass

    # Fallback: direct filesystem scan (user unpacked archive manually)
    return _load_from_filesystem(local_root, split, max_cases)


def _load_via_monai(
    local_root: Path,
    split: str,
    max_cases: int | None,
    download: bool,
    seed: int,
) -> SegmentationBundle:
    """Load via MONAI DecathlonDataset."""
    try:
        from monai.data import DecathlonDataset  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "msd_brain.load_data() via MONAI requires:\n"
            "  pip install 'qortex[monai]'\n"
            "or: pip install monai"
        ) from None

    monai_split = "training" if split == "train" else "test"
    dataset = DecathlonDataset(
        root_dir=str(local_root),
        task="Task01_BrainTumour",
        section=monai_split,
        download=download,
        seed=seed,
    )

    n = len(dataset)
    if max_cases is not None:
        n = min(n, max_cases)

    case_ids: list[str] = []
    image_paths: list[list[Path]] = []
    mask_paths: list[Path] = []

    for i in range(n):
        item = dataset[i]
        img_path = Path(item["image"] if isinstance(item["image"], str) else str(item["image"]))
        case_ids.append(f"case_{i:04d}")
        image_paths.append([img_path])  # MONAI stacks modalities in one 4D file
        if "label" in item:
            mask_paths.append(Path(item["label"] if isinstance(item["label"], str) else str(item["label"])))

    return SegmentationBundle(
        card=_CARD,
        case_ids=case_ids,
        image_paths=image_paths,
        mask_paths=mask_paths,
        label_map=LABEL_MAP,
        modalities=MODALITIES,
        split=split,
    )


def _load_from_filesystem(
    local_root: Path,
    split: str,
    max_cases: int | None,
) -> SegmentationBundle:
    """Fallback: scan unpacked MSD Task01_BrainTumour directory.

    Expected layout after unpacking:
      Task01_BrainTumour/
        imagesTr/  (BraTS_XXX_flair.nii.gz, _t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz)
        labelsTr/  (BraTS_XXX.nii.gz)
        imagesTs/
    """
    task_dir = local_root / "Task01_BrainTumour"
    if not task_dir.exists():
        task_dir = local_root
        if not (task_dir / "imagesTr").exists():
            raise FileNotFoundError(
                f"MSD Brain Tumour data not found at {local_root}.\n"
                f"Install MONAI (pip install monai) for automatic download, or\n"
                f"unpack the dataset manually to {local_root}/Task01_BrainTumour/"
            )

    if split == "train":
        img_dir = task_dir / "imagesTr"
        msk_dir = task_dir / "labelsTr"
    else:
        img_dir = task_dir / "imagesTs"
        msk_dir = None

    case_ids: list[str] = []
    image_paths: list[list[Path]] = []
    mask_paths: list[Path] = []

    # Each training case has 4 modality files: *_flair, *_t1, *_t1ce, *_t2
    # or a stacked 4D volume in newer MSD format
    all_imgs = sorted(img_dir.glob("*.nii.gz"))
    if not all_imgs:
        all_imgs = sorted(img_dir.glob("*.nii"))

    # Group by case ID
    case_map: dict[str, list[Path]] = {}
    for p in all_imgs:
        # Filename pattern: BraTS_001_0000.nii.gz (channel 0–3 = FLAIR, T1w, T1gd, T2w)
        parts = p.stem.replace(".nii", "").rsplit("_", 1)
        case_key = parts[0] if len(parts) == 2 else p.stem
        case_map.setdefault(case_key, []).append(p)

    for i, (case_key, paths) in enumerate(sorted(case_map.items())):
        if max_cases is not None and i >= max_cases:
            break
        sorted_paths = sorted(paths)
        case_ids.append(case_key)
        image_paths.append(sorted_paths)

        if msk_dir:
            # mask: BraTS_001.nii.gz
            mask_candidate = msk_dir / (case_key + ".nii.gz")
            if not mask_candidate.exists():
                mask_candidate = msk_dir / (case_key + ".nii")
            mask_paths.append(mask_candidate)

    return SegmentationBundle(
        card=_CARD,
        case_ids=case_ids,
        image_paths=image_paths,
        mask_paths=mask_paths,
        label_map=LABEL_MAP,
        modalities=MODALITIES,
        split=split,
    )
