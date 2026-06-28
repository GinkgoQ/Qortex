"""qortex.datasets.ixi — IXI Multimodal Brain MRI Dataset.

Usage::

    from qortex.datasets import ixi

    card = ixi.describe()
    bundle = ixi.load_data(local_root="/data/ixi", modalities=["T1"],
                           task="age_regression", max_subjects=100)
    # bundle.label_col = "age"
    # bundle.labels: float array of ages

    # Sex classification
    bundle2 = ixi.load_data(local_root="/data/ixi", task="sex_classification")
    # bundle2.label_map = {0: "male", 1: "female"}

Dataset facts
-------------
- ~600 healthy subjects from 3 UK hospitals (Hammersmith, Guy's, IOP).
- Modalities per subject: T1, T2, PD (proton density), MRA, DWI.
- Demographics: age, sex, acquisition site.
- License: CC BY-SA 3.0 (attribution + share-alike required).
- Source: https://brain-development.org/ixi-dataset/
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from qortex.datasets._base import DatasetCard, MRIBundle, _REGISTRY
from qortex.datasets._cache import dataset_cache_dir

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="ixi",
    full_name="IXI Dataset — Information eXtraction from Images",
    version="1.0",
    source_url="https://brain-development.org/ixi-dataset/",
    license="Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
    citation=(
        "IXI Dataset. Brain Development, Imperial College London. "
        "https://brain-development.org/ixi-dataset/"
    ),
    modality="mri",
    n_subjects=600,
    n_channels=None,
    sampling_hz=None,
    image_shape=(256, 256, 150),
    n_classes=2,
    description=(
        "~600 healthy subjects scanned across 3 UK hospitals.\n"
        "Modalities: T1, T2, PD, MRA, DWI; demographics include age and sex.\n"
        "Tasks: age regression (continuous), sex classification (binary),\n"
        "       scanner/site QC.\n"
        "License: CC BY-SA 3.0 — attribution and share-alike required."
    ),
    tasks=["age_regression", "sex_classification", "scanner_site_qc"],
    tutorial_ids=["T06"],
    size_gb_approx=14.5,
    requires_registration=False,
    access_instructions=(
        "Download from https://brain-development.org/ixi-dataset/ (no registration).\n"
        "Expected layout:\n"
        "  <local_root>/\n"
        "    IXI-T1/  (T1 NIfTI files: IXI001-Guys-0828-T1.nii.gz, ...)\n"
        "    IXI-T2/\n"
        "    IXI-PD/\n"
        "    IXI-MRA/\n"
        "    IXI-DTI/\n"
        "    IXI.xls  (demographic spreadsheet; also available as CSV)"
    ),
)
_REGISTRY.register(_CARD)


# ── Label maps ────────────────────────────────────────────────────────────────

LABEL_MAP_SEX = {0: "male", 1: "female"}

_MODALITY_DIR = {
    "T1": "IXI-T1",
    "T2": "IXI-T2",
    "PD": "IXI-PD",
    "MRA": "IXI-MRA",
    "DWI": "IXI-DTI",
}

# Site codes in filenames
_SITE_NAMES = {"Guys": "Guys_Hospital", "HH": "Hammersmith_Hospital", "IOP": "IOP"}


# ── Demographic table parser ──────────────────────────────────────────────────

def load_demographics(local_root: Path) -> dict[str, dict[str, Any]]:
    """Parse IXI demographics from IXI.csv or IXI.xls.

    Returns {ixi_id: {age, sex, site, ethnic_group, ...}}.

    IXI ID is an integer (1–628); the NIfTI filename encodes it as IXI%03d.
    sex codes: 1=Male, 2=Female.
    """
    table: dict[str, dict[str, Any]] = {}

    # Try CSV first (user-converted), then xls
    csv_path = local_root / "IXI.csv"
    xls_path = local_root / "IXI.xls"
    xlsx_path = local_root / "IXI.xlsx"

    if csv_path.exists():
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                ixi_id = row.get("IXI_ID", "").strip()
                if not ixi_id:
                    continue
                parsed: dict[str, Any] = {}
                for k, v in row.items():
                    v = v.strip() if isinstance(v, str) else v
                    try:
                        parsed[k] = float(v) if "." in str(v) else int(v)
                    except (ValueError, TypeError):
                        parsed[k] = v
                # Normalize sex to 0=male, 1=female
                sex_raw = parsed.get("SEX_ID", None)
                if sex_raw is not None:
                    try:
                        parsed["sex_label"] = 0 if int(sex_raw) == 1 else 1
                    except (ValueError, TypeError):
                        parsed["sex_label"] = None
                table[str(ixi_id)] = parsed
    elif xls_path.exists() or xlsx_path.exists():
        try:
            import openpyxl  # type: ignore[import]
            wb_path = xlsx_path if xlsx_path.exists() else xls_path
            wb = openpyxl.load_workbook(str(wb_path), read_only=True, data_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            if rows:
                header = [str(h) for h in rows[0]]
                for row in rows[1:]:
                    row_dict = dict(zip(header, row))
                    ixi_id = str(row_dict.get("IXI_ID", "")).strip()
                    if not ixi_id or ixi_id in ("None", ""):
                        continue
                    sex_raw = row_dict.get("SEX_ID")
                    row_dict["sex_label"] = (0 if int(sex_raw) == 1 else 1) if sex_raw else None
                    table[ixi_id] = row_dict
        except ImportError:
            import warnings
            warnings.warn(
                "openpyxl not installed; cannot read IXI.xls. "
                "Convert IXI.xls to IXI.csv manually or install openpyxl.",
                RuntimeWarning,
                stacklevel=3,
            )

    return table


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    local_root: Path | str | None = None,
    modalities: list[str] | None = None,
    task: str = "age_regression",
    exclude_missing_labels: bool = True,
    max_subjects: int | None = None,
) -> MRIBundle:
    """Load IXI data from a local directory.

    Parameters
    ----------
    local_root            : Root of the downloaded IXI dataset.
                            Falls back to QORTEX_DATA_DIR/ixi.
    modalities            : List of modalities to load. Defaults to ["T1"].
                            Options: "T1", "T2", "PD", "MRA", "DWI".
                            Only the first modality's paths are stored as primary.
    task                  : "age_regression" or "sex_classification".
    exclude_missing_labels : Skip subjects with no label for the chosen task.
    max_subjects          : Limit for quick experiments.

    Returns
    -------
    MRIBundle with:
      - For age_regression: label_col="AGE", labels=float array, label_map=None.
      - For sex_classification: label_col="sex_label", labels=int array,
        label_map={0: "male", 1: "female"}.

    Examples
    --------
    >>> bundle = ixi.load_data(local_root="/data/ixi", task="age_regression",
    ...                        max_subjects=50)
    >>> bundle.load_images(max_subjects=10)
    """
    import numpy as np

    if local_root is None:
        local_root = dataset_cache_dir("ixi")
    local_root = Path(local_root)

    if modalities is None:
        modalities = ["T1"]

    demographics = load_demographics(local_root)

    # Determine primary modality directory
    primary_mod = modalities[0]
    mod_dir = local_root / _MODALITY_DIR.get(primary_mod, f"IXI-{primary_mod}")

    if not mod_dir.exists():
        raise FileNotFoundError(
            f"Modality directory not found: {mod_dir}.\n"
            f"Download IXI-{primary_mod} from https://brain-development.org/ixi-dataset/"
        )

    nifti_files = sorted(mod_dir.glob("*.nii.gz")) + sorted(mod_dir.glob("*.nii"))

    subjects: list[str] = []
    local_paths: list[Path] = []
    labels_list: list[float] = []
    metadata: dict[str, Any] = {}

    label_col = "AGE" if task == "age_regression" else "sex_label"

    for nifti_path in nifti_files:
        # Extract IXI ID from filename: IXI001-Guys-0828-T1.nii.gz → "1"
        stem = nifti_path.name.split("-")[0]  # "IXI001"
        ixi_id = stem.replace("IXI", "").lstrip("0") or "0"

        row = demographics.get(ixi_id)
        if row is None:
            continue

        label_val = row.get(label_col)
        if exclude_missing_labels and label_val is None:
            continue

        subjects.append(ixi_id)
        local_paths.append(nifti_path)
        labels_list.append(float(label_val) if label_val is not None else float("nan"))

        # Extract site from filename
        parts = nifti_path.stem.split("-")
        site = parts[1] if len(parts) > 1 else "unknown"
        metadata[ixi_id] = {**row, "site": site, "modality": primary_mod}

        if max_subjects is not None and len(subjects) >= max_subjects:
            break

    if task == "age_regression":
        labels_arr = np.array(labels_list, dtype=np.float32) if labels_list else None
        label_map = None
    else:
        labels_arr = np.array(labels_list, dtype=np.int64) if labels_list else None
        label_map = LABEL_MAP_SEX

    return MRIBundle(
        card=_CARD,
        subjects=subjects,
        modality=primary_mod,
        local_paths=local_paths,
        metadata=metadata,
        labels=labels_arr,
        label_col=label_col,
        label_map=label_map,
    )
