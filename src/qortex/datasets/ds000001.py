"""qortex.datasets.ds000001 — OpenfMRI / OpenNeuro ds000001 BART fMRI.

Usage::

    from qortex.datasets import ds000001

    card = ds000001.describe()
    bundle = ds000001.load_data(local_root="/data/ds000001", subjects=["01","02"])
    bundle.load_events()
    # bundle.events: list of event-row dicts per subject
    report = bundle.run_preflight(Path("/data/ds000001"))

Dataset facts
-------------
- Balloon Analog Risk-Taking Task (BART) fMRI.
- 16 subjects, Siemens Allegra 3T, TR=2s.
- BIDS-converted; corrected event files in revision 2.0.4.
- License: PDDL (Public Domain Dedication and License).
- Source: https://legacy.openfmri.org/dataset/ds000001/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.datasets._base import DatasetCard, FMRIBundle, _REGISTRY
from qortex.datasets._cache import dataset_cache_dir

# ── Dataset card ──────────────────────────────────────────────────────────────

_CARD = DatasetCard(
    name="ds000001",
    full_name="ds000001 — Balloon Analog Risk-taking Task (BART) fMRI",
    version="2.0.4",
    source_url="https://legacy.openfmri.org/dataset/ds000001/",
    license="Public Domain Dedication and License (PDDL) 1.0",
    citation=(
        "Schonberg T, Fox CR, Mumford JA, et al. Decreasing ventromedial prefrontal "
        "cortex activity during sequential risk-taking: an fMRI investigation of the "
        "balloon analog risk task. Front Neurosci. 2012;6:80."
    ),
    modality="fmri",
    n_subjects=16,
    n_channels=None,
    sampling_hz=None,
    image_shape=(64, 64, 34),
    description=(
        "16 subjects performing the Balloon Analog Risk-taking Task (BART).\n"
        "BIDS-format; TR=2s; Siemens Allegra 3T.\n"
        "Revision 2.0.4 corrects the event timing files.\n"
        "Tutorial T07 uses this dataset for event/design readiness validation.\n"
        "No ML model required; deterministic event/GLM diagnostic workflow."
    ),
    tasks=["fmri_event_design_validation", "glm_diagnostic"],
    tutorial_ids=["T07"],
    size_gb_approx=1.4,
    requires_registration=False,
    access_instructions=(
        "Download from https://legacy.openfmri.org/dataset/ds000001/ or via OpenNeuro.\n"
        "Expected BIDS layout:\n"
        "  <local_root>/\n"
        "    dataset_description.json\n"
        "    sub-01/\n"
        "      func/\n"
        "        sub-01_task-balloonanalogrisktask_bold.nii.gz\n"
        "        sub-01_task-balloonanalogrisktask_events.tsv\n"
        "    sub-02/\n"
        "    ..."
    ),
)
_REGISTRY.register(_CARD)


# ── BIDS path helpers ─────────────────────────────────────────────────────────

_TASK_NAME = "balloonanalogrisktask"
_TR = 2.0


def _bold_path(root: Path, sub: str) -> Path:
    return root / f"sub-{sub}" / "func" / f"sub-{sub}_task-{_TASK_NAME}_bold.nii.gz"


def _events_path(root: Path, sub: str) -> Path:
    return root / f"sub-{sub}" / "func" / f"sub-{sub}_task-{_TASK_NAME}_events.tsv"


def _find_bold_path(root: Path, sub: str) -> Path:
    legacy = _bold_path(root, sub)
    if legacy.exists():
        return legacy
    matches = sorted(
        (root / f"sub-{sub}" / "func").glob(
            f"sub-{sub}_task-{_TASK_NAME}*_bold.nii.gz"
        )
    )
    return matches[0] if matches else legacy


def _find_events_path(root: Path, sub: str) -> Path:
    legacy = _events_path(root, sub)
    if legacy.exists():
        return legacy
    matches = sorted(
        (root / f"sub-{sub}" / "func").glob(
            f"sub-{sub}_task-{_TASK_NAME}*_events.tsv"
        )
    )
    return matches[0] if matches else legacy


def _discover_subjects(root: Path) -> list[str]:
    """Return sorted 2-digit subject IDs from BIDS sub-XX directories."""
    subs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subs.append(d.name[4:])
    return subs


# ── Public API ────────────────────────────────────────────────────────────────

def describe() -> DatasetCard:
    """Return the DatasetCard without downloading anything."""
    return _CARD


def load_data(
    local_root: Path | str | None = None,
    subjects: list[str] | None = None,
    tr: float = _TR,
) -> FMRIBundle:
    """Load ds000001 BIDS dataset from a local directory.

    ds000001 is available for free download — no registration required.

    Parameters
    ----------
    local_root : Root of the BIDS dataset directory.
                 Falls back to QORTEX_DATA_DIR/ds000001.
    subjects   : List of 2-digit subject IDs (e.g. ["01", "02"]).
                 Defaults to all discovered subjects.
    tr         : Repetition time in seconds (default 2.0).

    Returns
    -------
    FMRIBundle with BOLD and event file paths.

    Examples
    --------
    >>> bundle = ds000001.load_data(local_root="/data/ds000001", subjects=["01","02"])
    >>> events = bundle.load_events()
    >>> report = bundle.run_preflight(Path("/data/ds000001"))
    """
    if local_root is None:
        local_root = dataset_cache_dir("ds000001")
    local_root = Path(local_root)

    if not local_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {local_root}.\n"
            f"Download ds000001 from {_CARD.source_url} "
            f"and pass local_root= to ds000001.load_data()."
        )

    if subjects is None:
        subjects = _discover_subjects(local_root)
        if not subjects:
            subjects = [f"{i:02d}" for i in range(1, 17)]

    bold_paths: list[Path] = []
    event_paths: list[Path] = []
    missing: list[str] = []

    for sub in subjects:
        bp = _find_bold_path(local_root, sub)
        ep = _find_events_path(local_root, sub)
        bold_paths.append(bp)
        event_paths.append(ep)
        if not bp.exists() or not ep.exists():
            missing.append(sub)

    metadata: dict[str, Any] = {
        "bids_root": str(local_root),
        "task": _TASK_NAME,
        "tr": tr,
        "subjects_with_missing_files": missing,
        "n_subjects_requested": len(subjects),
        "n_subjects_complete": len(subjects) - len(missing),
    }

    # Try to read dataset_description.json for TR/version validation
    desc_path = local_root / "dataset_description.json"
    if desc_path.exists():
        import json
        try:
            with open(desc_path) as fh:
                desc = json.load(fh)
            metadata["dataset_description"] = desc
        except Exception:
            pass

    if missing:
        import warnings
        warnings.warn(
            f"{len(missing)} subjects have missing BOLD or event files: {missing[:5]}{'...' if len(missing)>5 else ''}",
            RuntimeWarning,
            stacklevel=2,
        )

    return FMRIBundle(
        card=_CARD,
        subjects=subjects,
        task=_TASK_NAME,
        tr=tr,
        bold_paths=bold_paths,
        event_paths=event_paths,
        n_volumes=None,
        preflight=None,
    )
