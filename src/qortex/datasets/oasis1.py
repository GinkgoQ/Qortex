"""qortex.datasets.oasis1 — OASIS-1 Cross-Sectional Structural MRI Dataset.

Usage::

    from qortex.datasets import oasis1

    card = oasis1.describe()
    bundle = oasis1.load_data(local_root="/data/oasis1")
    # bundle.metadata["table"]: demographic + clinical data per subject
    # bundle.labels: array of CDR-binary labels (0=no_dementia, 1=dementia)

    report = bundle.run_qc(max_subjects=5)

Dataset facts
-------------
- 416 subjects, 434 MR sessions (some subjects scanned twice).
- 3–4 T1-weighted MRI scans per subject (averaged for analysis).
- Clinical measures: CDR, MMSE, SES, eTIV, nWBV, ASF.
- Primary research target: CDR=0 (no dementia) vs CDR>0 (dementia).
- Age range: 18–96 years.
- Registration required: https://sites.wustl.edu/oasisbrains/home/oasis-1/
- For research purposes only — not a diagnostic tool.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from qortex.datasets._base import DatasetCard, MRIBundle, _REGISTRY
from qortex.datasets._cache import dataset_cache_dir

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="oasis1",
    full_name="OASIS-1: Cross-Sectional MRI Dataset in Young, Middle Aged, Nondemented and Demented Older Adults",
    version="1.0.0",
    source_url="https://sites.wustl.edu/oasisbrains/home/oasis-1/",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported",
    citation=(
        "Marcus DS, Wang TH, Parker J, Csernansky JG, Morris JC, Buckner RL. "
        "Open Access Series of Imaging Studies (OASIS): Cross-Sectional MRI "
        "Data in Young, Middle Aged, Nondemented, and Demented Older Adults. "
        "J Cogn Neurosci. 2007;19(9):1498–1507."
    ),
    modality="mri",
    n_subjects=416,
    n_channels=None,
    sampling_hz=None,
    image_shape=(176, 208, 176),
    description=(
        "416 cross-sectional subjects aged 18–96 with T1-weighted MRI.\n"
        "Clinical measures: CDR (Clinical Dementia Rating), MMSE, SES, eTIV, nWBV.\n"
        "Research target: CDR=0 (no dementia) vs CDR>0 (very mild to moderate dementia).\n"
        "Registration required at sites.wustl.edu/oasisbrains — for research only."
    ),
    tasks=["dementia_research_baseline", "confound_analysis", "subject_level_classification"],
    tutorial_ids=["T05"],
    size_gb_approx=1.5,
    requires_registration=True,
    access_instructions=(
        "Register at https://sites.wustl.edu/oasisbrains/home/oasis-1/\n"
        "Download the T1 archives and the demographic/clinical CSV.\n"
        "Extract to a directory and pass local_root= to oasis1.load_data().\n"
        "Expected layout:\n"
        "  <local_root>/\n"
        "    OAS1_XXXX_MR1/   (one dir per session)\n"
        "      RAW/\n"
        "        OAS1_XXXX_MR1_mpr-1_anon.img  (or .nii / .nii.gz)\n"
        "    oasis_cross-sectional.csv  (demographic table)"
    ),
)
_REGISTRY.register(_CARD)


# ── Label map ─────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "no_dementia", 1: "dementia"}

# Clinical variables from the CSV (covariates, not labels)
_COVARIATE_COLUMNS = ["Age", "M/F", "Hand", "Educ", "SES", "MMSE", "eTIV", "nWBV", "ASF", "Delay"]


# ── Table parser ──────────────────────────────────────────────────────────────

def load_clinical_table(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Parse oasis_cross-sectional.csv into {subject_id: {col: val}}.

    CDR column: 0=no dementia, 0.5=very mild, 1=mild, 2=moderate AD.
    CDR binary: CDR=0 → 0 (no_dementia); CDR>0 → 1 (dementia).
    """
    table: dict[str, dict[str, Any]] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = row.get("ID", "").strip()
            if not sid:
                continue
            parsed: dict[str, Any] = {}
            for col, val in row.items():
                val = val.strip() if isinstance(val, str) else val
                # Try numeric coercion
                try:
                    parsed[col] = float(val) if "." in str(val) else int(val)
                except (ValueError, TypeError):
                    parsed[col] = val

            # CDR binary label
            cdr = parsed.get("CDR", None)
            if cdr is None:
                parsed["cdr_binary"] = None
            else:
                try:
                    parsed["cdr_binary"] = 0 if float(cdr) == 0.0 else 1
                except (ValueError, TypeError):
                    parsed["cdr_binary"] = None

            table[sid] = parsed
    return table


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    local_root: Path | str | None = None,
    cdr_binary: bool = True,
    exclude_missing_cdr: bool = True,
    modality: str = "T1",
    max_subjects: int | None = None,
) -> MRIBundle:
    """Load OASIS-1 data from a local directory.

    OASIS-1 requires prior registration and manual download.
    See describe().access_instructions for the expected directory layout.

    Parameters
    ----------
    local_root          : Root of the extracted OASIS-1 dataset.
                          Falls back to QORTEX_DATA_DIR/oasis1.
    cdr_binary          : Use CDR=0 vs CDR>0 as the binary label.
    exclude_missing_cdr : Exclude subjects with no CDR score.
    modality            : "T1" (only T1 images are currently supported).
    max_subjects        : Limit number of subjects (useful for quick tests).

    Returns
    -------
    MRIBundle with:
      - label_col = 'cdr_binary'
      - label_map = {0: 'no_dementia', 1: 'dementia'}
      - metadata = clinical table keyed by subject_id
      - labels = np.ndarray of 0/1 values aligned to subjects list
    """
    import numpy as np

    if local_root is None:
        local_root = dataset_cache_dir("oasis1")
    local_root = Path(local_root)

    # Locate CSV
    csv_candidates = list(local_root.glob("oasis_cross-sectional*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(
            f"Could not find oasis_cross-sectional.csv in {local_root}.\n"
            "Download OASIS-1 from https://sites.wustl.edu/oasisbrains/ "
            "and set local_root accordingly."
        )
    table = load_clinical_table(csv_candidates[0])

    # Discover session directories
    session_dirs = sorted(local_root.glob("OAS1_*_MR*"))
    if not session_dirs:
        raise FileNotFoundError(
            f"No OAS1_* session directories found in {local_root}."
        )

    subjects: list[str] = []
    local_paths: list[Path] = []
    labels_list: list[int] = []
    metadata: dict[str, Any] = {}

    for sess_dir in session_dirs:
        sid = sess_dir.name  # e.g. OAS1_0001_MR1
        subject_key = sid.rsplit("_", 1)[0]  # OAS1_0001

        row = table.get(sid) or table.get(subject_key)
        if row is None:
            continue

        cdr_label = row.get("cdr_binary")
        if exclude_missing_cdr and cdr_label is None:
            continue

        # Find the NIfTI file
        nifti_path = _find_nifti(sess_dir, modality)
        if nifti_path is None:
            continue

        subjects.append(sid)
        local_paths.append(nifti_path)
        labels_list.append(int(cdr_label) if cdr_label is not None else -1)
        metadata[sid] = row

        if max_subjects is not None and len(subjects) >= max_subjects:
            break

    labels_arr = np.array(labels_list, dtype=np.int64) if labels_list else None

    return MRIBundle(
        card=_CARD,
        subjects=subjects,
        modality=modality,
        local_paths=local_paths,
        metadata=metadata,
        labels=labels_arr,
        label_col="cdr_binary",
        label_map=LABEL_MAP if cdr_binary else None,
    )


def _find_nifti(session_dir: Path, modality: str) -> Path | None:
    """Search for a NIfTI file in standard OASIS-1 subdirectories."""
    search_dirs = [
        session_dir / "PROCESSED" / "MPRAGE" / "SUBJ_111",
        session_dir / "RAW",
        session_dir,
    ]
    extensions = [".nii.gz", ".nii", ".img"]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        for ext in extensions:
            candidates = list(sdir.glob(f"*{ext}"))
            if candidates:
                return candidates[0]
    return None
