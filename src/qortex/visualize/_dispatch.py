"""Visual asset detection and rendering dispatch.

inspect_file(path)     → VisualAsset   (fast, no pixel data)
plan_from_asset(asset) → VisualPlan    (what will be rendered)
render_asset(asset)    → VisualResult  (actual rendering)

Detection is intentionally defensive:
- Always falls back gracefully when optional deps are missing.
- Never loads pixel data just to detect the file type.
- Checks companions automatically (bvec/bval, JSON sidecar, events).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from qortex.visualize._asset import (
    VisualAsset, VisualPlan, VisualResult, VisualWarning,
    INTENT_ANATOMICAL, INTENT_BOLD, INTENT_CT, INTENT_DWI,
    INTENT_FIELDMAP, INTENT_LABELMAP, INTENT_MASK, INTENT_PET,
    INTENT_RAW_SIGNAL, INTENT_SERIES_BROWSER, INTENT_STAT_MAP,
    INTENT_SURFACE, INTENT_UNKNOWN,
    MODE_INTERACTIVE, MODE_STATIC, MODE_SUMMARY, MODE_THUMBNAIL,
)

log = logging.getLogger(__name__)

# ── Extension maps ────────────────────────────────────────────────────────────

_NIFTI_EXTS = frozenset({".nii", ".mgz", ".mgh", ".mnc"})
_DICOM_EXTS = frozenset({".dcm", ".dicom", ".ima"})
_EEG_EXTS   = frozenset({".edf", ".bdf", ".fif", ".set", ".cnt", ".vhdr", ".gdf", ".mff", ".egi"})
_GIFTI_EXTS = frozenset({".gii"})
_CIFTI_EXTS = frozenset({".nii"})   # CIFTI uses .nii extension but with special header

# BIDS suffixes → visual intent
_SUFFIX_TO_INTENT: dict[str, str] = {
    "T1w": INTENT_ANATOMICAL, "T2w": INTENT_ANATOMICAL,
    "FLAIR": INTENT_ANATOMICAL, "T2star": INTENT_ANATOMICAL,
    "PD": INTENT_ANATOMICAL, "PDw": INTENT_ANATOMICAL,
    "inplaneT1": INTENT_ANATOMICAL, "inplaneT2": INTENT_ANATOMICAL,
    "bold": INTENT_BOLD, "cbv": INTENT_BOLD, "phase": INTENT_BOLD,
    "dwi": INTENT_DWI, "sbref": INTENT_DWI,
    "pet": INTENT_PET,
    "CT": INTENT_CT, "ct": INTENT_CT,
    "epi": INTENT_BOLD,
    "fmap": INTENT_FIELDMAP, "magnitude": INTENT_FIELDMAP,
    "magnitude1": INTENT_FIELDMAP, "magnitude2": INTENT_FIELDMAP,
    "phasediff": INTENT_FIELDMAP, "fieldmap": INTENT_FIELDMAP,
    "stat": INTENT_STAT_MAP, "zmap": INTENT_STAT_MAP,
    "tmap": INTENT_STAT_MAP, "contrast": INTENT_STAT_MAP,
    "seg": INTENT_LABELMAP, "mask": INTENT_MASK,
    "atlas": INTENT_LABELMAP, "dseg": INTENT_LABELMAP,
    "label": INTENT_LABELMAP,
    "eeg": INTENT_RAW_SIGNAL, "meg": INTENT_RAW_SIGNAL,
    "ieeg": INTENT_RAW_SIGNAL,
}

_INTENT_TO_MODALITY: dict[str, str] = {
    INTENT_ANATOMICAL: "mri",
    INTENT_BOLD: "fmri",
    INTENT_DWI: "dwi",
    INTENT_PET: "pet",
    INTENT_CT: "ct",
    INTENT_FIELDMAP: "mri",
    INTENT_MASK: "mri",
    INTENT_LABELMAP: "mri",
    INTENT_STAT_MAP: "mri",
    INTENT_SURFACE: "surface",
    INTENT_RAW_SIGNAL: "eeg",
    INTENT_SERIES_BROWSER: "ct",
}


# ── Name-based classification ─────────────────────────────────────────────────

def _classify_by_name(name: str, suffix: str) -> tuple[str, str]:
    """Return (intent, modality) from BIDS filename or keywords."""
    name_l = name.lower()
    suffix_l = suffix.lower()

    # BIDS suffix lookup
    intent = _SUFFIX_TO_INTENT.get(suffix) or _SUFFIX_TO_INTENT.get(suffix_l)
    if intent:
        return intent, _INTENT_TO_MODALITY.get(intent, "mri")

    # Keyword scan of the full filename
    kw_map = [
        ("bold", INTENT_BOLD, "fmri"),
        ("func", INTENT_BOLD, "fmri"),
        ("t1w", INTENT_ANATOMICAL, "mri"),
        ("t2w", INTENT_ANATOMICAL, "mri"),
        ("flair", INTENT_ANATOMICAL, "mri"),
        ("dwi", INTENT_DWI, "dwi"),
        ("adc", INTENT_DWI, "dwi"),
        ("fa", INTENT_DWI, "dwi"),
        ("pet", INTENT_PET, "pet"),
        ("fdg", INTENT_PET, "pet"),
        ("suv", INTENT_PET, "pet"),
        ("ct", INTENT_CT, "ct"),
        ("zmap", INTENT_STAT_MAP, "mri"),
        ("tmap", INTENT_STAT_MAP, "mri"),
        ("stat", INTENT_STAT_MAP, "mri"),
        ("mask", INTENT_MASK, "mri"),
        ("seg", INTENT_LABELMAP, "mri"),
        ("label", INTENT_LABELMAP, "mri"),
        ("atlas", INTENT_LABELMAP, "mri"),
        ("eeg", INTENT_RAW_SIGNAL, "eeg"),
        ("meg", INTENT_RAW_SIGNAL, "meg"),
        ("ieeg", INTENT_RAW_SIGNAL, "ieeg"),
    ]
    for kw, intent, modality in kw_map:
        if kw in name_l:
            return intent, modality

    return INTENT_ANATOMICAL, "mri"  # default for NIfTI


# ── NIfTI header inspection ───────────────────────────────────────────────────

def _inspect_nifti(path: Path) -> VisualAsset:
    asset = VisualAsset(path=path, family="nifti")
    bids_suffix = _bids_suffix(path)
    intent, modality = _classify_by_name(path.name, bids_suffix)
    asset.intent = intent
    asset.modality = modality

    # Find BIDS companions
    stem = path.name.replace(".gz", "").replace(".nii", "")
    parent = path.parent
    for ext in (".json", "_events.tsv"):
        companion = parent / f"{stem}{ext}"
        if not companion.exists():
            companion = parent / f"{stem.rsplit('_', 1)[0]}{ext}"
        if companion.exists():
            asset.companion_paths.append(companion)

    if bids_suffix == "dwi":
        for ext in (".bvec", ".bval"):
            companion = parent / f"{stem}{ext}"
            if companion.exists():
                asset.companion_paths.append(companion)

    # NiBabel header read (optional — graceful fallback)
    try:
        import nibabel as nib
        img = nib.load(str(path))
        if "cifti" in img.__class__.__name__.lower():
            asset.family = "cifti"
            asset.intent = INTENT_SURFACE
            asset.modality = "surface"
            asset.shape = tuple(int(s) for s in img.shape)
            asset.ndim = len(asset.shape)
            asset.dtype = str(img.get_data_dtype())
            asset.recommended_view = MODE_STATIC
            return asset
        hdr = img.header
        shape = tuple(int(s) for s in img.shape)
        asset.shape = shape
        asset.ndim = len(shape)
        asset.n_timepoints = shape[3] if len(shape) == 4 else 1
        asset.affine = img.affine
        asset.dtype = str(img.get_data_dtype())
        zooms = hdr.get_zooms()
        asset.spacing = tuple(float(z) for z in zooms[:3])

        # Orientation
        try:
            axcodes = nib.aff2axcodes(img.affine)
            asset.orientation = "".join(axcodes)
        except Exception:
            asset.orientation = "unknown"

        # Refinement from header
        intent_code = int(hdr.get("intent_code", 0)) if hasattr(hdr, "get") else 0
        if intent_code in {1006, 2007, 2008}:  # NEURONAMES_BODMAS, Z_SCORE, T_STAT
            asset.intent = INTENT_STAT_MAP
            asset.is_stat_map = True

        # Mask/labelmap detection: filename/BIDS suffix based only (not dtype)
        suffix_l = bids_suffix.lower()
        name_l = path.name.lower()
        if any(kw in suffix_l or kw in name_l
               for kw in ("mask", "seg", "dseg", "label", "atlas", "aparc", "aseg")):
            asset.is_mask = True
            asset.intent = INTENT_MASK if "mask" in suffix_l or "mask" in name_l else INTENT_LABELMAP

        # Large file flag (~4 GB threshold)
        if asset.estimated_memory_mb > 4000:
            asset.is_large = True
            asset.warn("large_file", f"Full load ~{asset.estimated_memory_mb:.0f} MB", "info")

        # Orientation warnings
        if asset.orientation not in {"RAS", "LAS", "RPS", "LPS",
                                      "RAI", "LAI", "RPI", "LPI",
                                      "RSA", "LSA", "RSP", "LSP",
                                      "AIR", "AIL", "PSR", "PSL"}:
            asset.warn("unusual_orientation", f"Orientation {asset.orientation!r} is non-standard")

        # Very anisotropic voxels
        if asset.spacing and len(asset.spacing) == 3:
            ratio = max(asset.spacing) / (min(asset.spacing) + 1e-6)
            if ratio > 5:
                asset.warn("anisotropic",
                            f"Highly anisotropic voxels: {asset.voxel_size_str} (ratio {ratio:.1f}×)", "info")

        # Empty slab (any dim = 1)
        if any(s == 1 for s in shape[:3]):
            asset.warn("single_slice", "One spatial dimension is 1 — this is a 2D slab, not a full volume")

    except ImportError:
        asset.warn("nibabel_missing",
                   "nibabel not installed; shape/affine unknown. pip install nibabel", "info")
    except Exception as exc:
        asset.warn("nifti_read_error", f"Could not read NIfTI header: {exc}", "error")

    # Recommended view
    if asset.intent == INTENT_BOLD:
        asset.recommended_view = MODE_INTERACTIVE
    elif asset.is_large:
        asset.recommended_view = MODE_THUMBNAIL
    elif asset.intent in {INTENT_ANATOMICAL, INTENT_CT}:
        asset.recommended_view = MODE_STATIC
    else:
        asset.recommended_view = MODE_STATIC

    return asset


def _bids_suffix(path: Path) -> str:
    """Return the BIDS suffix before the imaging extension.

    Handles compound extensions and entity-rich names such as
    ``sub-01_space-MNI_desc-preproc_bold.nii.gz`` without being confused by
    entity values. The suffix is the last underscore-delimited component before
    the extension.
    """
    name = path.name
    for ext in (".nii.gz", ".nii", ".mgz", ".mgh", ".mnc", ".gii"):
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    last = name.rsplit("_", 1)[-1] if "_" in name else name
    return last.split("-", 1)[0] if "-" in last else last


# ── DICOM inspection ──────────────────────────────────────────────────────────

def _inspect_dicom(path: Path) -> VisualAsset:
    asset = VisualAsset(path=path, family="dicom")

    if path.is_dir():
        asset.intent = INTENT_SERIES_BROWSER
        asset.recommended_view = MODE_INTERACTIVE
        # Quick file count
        dcm_files = [f for f in path.iterdir() if f.suffix.lower() in _DICOM_EXTS or f.suffix == ""]
        asset.metadata["n_dicom_files"] = len(dcm_files)
        if not dcm_files:
            asset.warn("no_dicom_files", "No DICOM files found in directory", "error")
            return asset
        # Read first file for metadata
        path_to_read = dcm_files[0]
    else:
        path_to_read = path

    try:
        import pydicom
        ds = pydicom.dcmread(str(path_to_read), stop_before_pixels=True, force=True)

        modality_tag = str(getattr(ds, "Modality", "MR")).upper()
        if modality_tag == "CT":
            asset.intent = INTENT_CT
            asset.modality = "ct"
        elif modality_tag == "PT":
            asset.intent = INTENT_PET
            asset.modality = "pet"
        else:
            asset.intent = INTENT_ANATOMICAL
            asset.modality = "mri"

        ps = getattr(ds, "PixelSpacing", None)
        st = float(getattr(ds, "SliceThickness", 1.0) or 1.0)
        if ps:
            asset.spacing = (float(ps[0]), float(ps[1]), st)

        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)
        if wc and ww:
            asset.metadata["window_center"] = float(wc)
            asset.metadata["window_width"] = float(ww)

        # PHI fields (PatientID, StudyInstanceUID, PatientDOB, PatientSex,
        # InstitutionName) are intentionally omitted from VisualAsset.metadata
        # to prevent accidental exposure in logs, provenance dicts, or reports.
        # Full PHI is available only through the dicom-browser --show-phi path.
        asset.metadata.update({
            "modality_tag": modality_tag,
            "series_description": str(getattr(ds, "SeriesDescription", "")),
            "study_description": str(getattr(ds, "StudyDescription", "")),
            "manufacturer": str(getattr(ds, "Manufacturer", "")),
        })

        rows = int(getattr(ds, "Rows", 0))
        cols = int(getattr(ds, "Columns", 0))
        n_files = asset.metadata.get("n_dicom_files", 1)
        if rows and cols:
            asset.shape = (rows, cols, n_files)
            asset.ndim = 3

    except ImportError:
        asset.warn("pydicom_missing",
                   "pydicom not installed; DICOM metadata not read. pip install pydicom", "info")
    except Exception as exc:
        asset.warn("dicom_read_error", f"Could not read DICOM metadata: {exc}", "warning")

    return asset


# ── EEG/MEG inspection ────────────────────────────────────────────────────────

def _inspect_eeg(path: Path) -> VisualAsset:
    asset = VisualAsset(path=path, family="eeg")
    name_l = path.name.lower()
    if "meg" in name_l:
        asset.intent = INTENT_RAW_SIGNAL
        asset.modality = "meg"
    elif "ieeg" in name_l:
        asset.intent = INTENT_RAW_SIGNAL
        asset.modality = "ieeg"
    else:
        asset.intent = INTENT_RAW_SIGNAL
        asset.modality = "eeg"
    asset.recommended_view = MODE_INTERACTIVE

    try:
        import mne
        raw = mne.io.read_raw(str(path), preload=False, verbose=False)
        asset.n_channels = len(raw.info["ch_names"])
        asset.n_timepoints = int(raw.n_times)
        asset.shape = (asset.n_channels, asset.n_timepoints)
        asset.ndim = 2
        asset.spacing = (1000.0 / raw.info["sfreq"],)  # ms per sample
        asset.metadata["sfreq"] = raw.info["sfreq"]
        asset.metadata["duration_s"] = raw.times[-1]
        asset.metadata["ch_names"] = raw.info["ch_names"][:8]  # first 8
    except ImportError:
        asset.warn("mne_missing",
                   "mne not installed; EEG metadata unknown. pip install mne", "info")
    except Exception as exc:
        asset.warn("eeg_read_error", f"Could not read EEG metadata: {exc}", "warning")

    return asset


# ── Directory inspection ──────────────────────────────────────────────────────

def _inspect_directory(path: Path) -> VisualAsset:
    """Classify a directory by its contents before treating it as DICOM."""
    _dcm_exts = {".dcm", ".dicom", ".ima"}

    # 1. Check for DICOM files
    try:
        dcm_candidates = [
            f for f in path.iterdir()
            if f.is_file() and (
                f.suffix.lower() in _dcm_exts
                or (not f.suffix and f.name.isdigit())
            )
        ]
    except PermissionError:
        dcm_candidates = []

    if dcm_candidates:
        return _inspect_dicom(path)

    # 2. BIDS dataset root
    if (path / "dataset_description.json").exists():
        asset = VisualAsset(path=path, family="bids_dataset")
        asset.intent = "dataset_collection"
        asset.recommended_view = "summary"
        asset.warn("bids_root",
                   "This is a BIDS dataset root. Use qortex.visualize.inspect() on individual files.",
                   "info")
        return asset

    # 3. Qortex artifact
    if (path / "artifact_manifest.json").exists():
        asset = VisualAsset(path=path, family="qortex_artifact")
        asset.intent = INTENT_UNKNOWN
        asset.warn("artifact_dir", "Qortex artifact directory", "info")
        return asset

    # 4. Try subdirectories for DICOM series (multi-series study folder)
    try:
        for subdir in sorted(path.iterdir()):
            if subdir.is_dir():
                try:
                    sub_dcm = [
                        f for f in subdir.iterdir()
                        if f.is_file() and f.suffix.lower() in _dcm_exts
                    ]
                    if sub_dcm:
                        return _inspect_dicom(path)  # multi-series study folder
                except PermissionError:
                    continue
    except PermissionError:
        pass

    asset = VisualAsset(path=path, family="unknown")
    asset.warn("unknown_directory", "Directory type could not be determined", "warning")
    return asset


# ── Main inspection entry point ───────────────────────────────────────────────

def inspect_file(source: Any) -> VisualAsset:
    """Inspect any neuroimaging source and return a VisualAsset.

    Does NOT load pixel data.  Always returns a VisualAsset (with warnings if
    the file cannot be read).

    Parameters
    ----------
    source:
        Path (str or Path), nibabel image, MNE Raw, or numpy array.
    """
    import numpy as np

    if isinstance(source, np.ndarray):
        asset = VisualAsset(path=Path("<array>"), family="array")
        if source.ndim == 4:
            asset.intent = INTENT_BOLD
            asset.modality = "fmri"
            asset.n_timepoints = source.shape[3]
        elif source.ndim == 2 and source.shape[0] < source.shape[1]:
            asset.intent = INTENT_RAW_SIGNAL
            asset.modality = "eeg"
            asset.n_channels = source.shape[0]
        else:
            asset.intent = INTENT_ANATOMICAL
            asset.modality = "mri"
        asset.shape = tuple(source.shape)
        asset.ndim = source.ndim
        asset.dtype = str(source.dtype)
        asset.recommended_view = MODE_STATIC
        return asset

    # nibabel duck-type
    if hasattr(source, "get_fdata") and hasattr(source, "affine"):
        asset = VisualAsset(path=Path("<nibabel>"), family="nifti")
        shape = tuple(int(s) for s in source.shape)
        asset.shape = shape
        asset.ndim = len(shape)
        asset.n_timepoints = shape[3] if len(shape) == 4 else 1
        asset.affine = source.affine
        asset.dtype = str(source.get_data_dtype())
        intent, modality = _classify_by_name("", "")
        asset.intent = INTENT_BOLD if len(shape) == 4 else INTENT_ANATOMICAL
        asset.modality = "fmri" if len(shape) == 4 else "mri"
        asset.recommended_view = MODE_INTERACTIVE if len(shape) == 4 else MODE_STATIC
        return asset

    # MNE Raw duck-type
    if hasattr(source, "get_data") and hasattr(source, "info"):
        asset = VisualAsset(path=Path("<mne_raw>"), family="eeg")
        asset.intent = INTENT_RAW_SIGNAL
        asset.modality = "eeg"
        asset.n_channels = len(source.info["ch_names"])
        asset.n_timepoints = int(source.n_times)
        asset.shape = (asset.n_channels, asset.n_timepoints)
        asset.ndim = 2
        asset.metadata["sfreq"] = source.info["sfreq"]
        asset.recommended_view = MODE_INTERACTIVE
        return asset

    path = Path(source)
    if not path.exists():
        asset = VisualAsset(path=path, family="unknown")
        asset.warn("not_found", f"File not found: {path}", "error")
        return asset

    # Classify by extension
    suffix = path.suffix.lower()
    name_lower = path.name.lower()

    if suffix in _EEG_EXTS:
        return _inspect_eeg(path)

    if suffix in _GIFTI_EXTS:
        asset = VisualAsset(path=path, family="gifti")
        asset.intent = INTENT_SURFACE
        asset.modality = "surface"
        asset.recommended_view = MODE_STATIC
        return asset

    if path.is_dir():
        return _inspect_directory(path)

    if suffix in _DICOM_EXTS:
        # Single DICOM file
        return _inspect_dicom(path)

    is_nifti = (suffix in _NIFTI_EXTS) or name_lower.endswith(".nii.gz")
    if is_nifti:
        return _inspect_nifti(path)

    # Unknown extension — try NIfTI, then give up
    asset = VisualAsset(path=path, family="unknown")
    asset.warn("unknown_format", f"Unrecognised extension: {path.suffix!r}", "warning")
    return asset


# ── Planning ──────────────────────────────────────────────────────────────────

def plan_from_asset(asset: VisualAsset, mode: str = "auto", **kwargs) -> VisualPlan:
    """Derive a VisualPlan from a VisualAsset."""
    resolved_mode = _resolve_mode(asset, mode)
    backend = _choose_backend(asset, resolved_mode)
    views = _choose_views(asset)
    window = _choose_window(asset)
    colormap = _choose_colormap(asset)
    overlay = kwargs.get("overlay")

    companions_needed = []
    if asset.intent == INTENT_DWI:
        companions_needed = ["bvec", "bval"]
    elif asset.intent == INTENT_BOLD:
        companions_needed = ["json_sidecar"]
    elif asset.intent in {INTENT_MASK, INTENT_LABELMAP, INTENT_STAT_MAP}:
        companions_needed = ["base_image"]

    return VisualPlan(
        asset=asset,
        mode=resolved_mode,
        backend=backend,
        views=views,
        window_preset=window,
        colormap=colormap,
        overlay_path=Path(overlay) if overlay else None,
        requires_companions=companions_needed,
    )


_MODE_ALIASES: dict[str, str] = {
    "interactive": MODE_INTERACTIVE,
    "html": MODE_INTERACTIVE,
    "png": MODE_THUMBNAIL,
    "thumb": MODE_THUMBNAIL,
    "static_png": MODE_STATIC,
    "quality": "qc",
    "quality_control": "qc",
}


def _resolve_mode(asset: VisualAsset, mode: str) -> str:
    if mode != "auto":
        return _MODE_ALIASES.get(mode, mode)
    return asset.recommended_view


def _choose_backend(asset: VisualAsset, mode: str) -> str:
    if asset.family == "eeg":
        return "mne+plotly"
    if mode == MODE_INTERACTIVE:
        return "plotly"
    if mode == MODE_THUMBNAIL:
        return "pure_python"
    return "pure_python+plotly"


def _choose_views(asset: VisualAsset) -> list[str]:
    if asset.intent == INTENT_BOLD:
        views = ["mean_epi", "single_timepoint"]
        if asset.n_timepoints > 1:
            views.append("time_slider")
        return views
    if asset.intent in {INTENT_ANATOMICAL, INTENT_MASK, INTENT_LABELMAP}:
        return ["orthogonal"]
    if asset.intent == INTENT_CT:
        return ["orthogonal_windowed"]
    if asset.intent == INTENT_DWI:
        return ["b0_volume", "dwi_volume"]
    if asset.intent == INTENT_PET:
        return ["pet_volume"]
    if asset.intent == INTENT_STAT_MAP:
        return ["stat_overlay", "glass_brain"]
    if asset.intent == INTENT_SERIES_BROWSER:
        return ["series_table", "series_preview"]
    if asset.intent == INTENT_RAW_SIGNAL:
        return ["butterfly", "psd", "spectrogram"]
    return ["orthogonal"]


def _choose_window(asset: VisualAsset) -> str | None:
    if asset.modality == "ct":
        return "brain"
    if asset.intent == INTENT_PET:
        return "fdg"
    return "auto"


def _choose_colormap(asset: VisualAsset) -> str:
    if asset.intent == INTENT_PET:
        return "hot"
    if asset.intent == INTENT_STAT_MAP:
        return "RdBu_r"
    return "gray"


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_asset(asset: VisualAsset, mode: str = "auto", **kwargs) -> VisualResult:
    """Render a VisualAsset to a VisualResult.

    This is the main rendering dispatcher.  It routes to the correct viewer
    class based on the asset's family and intent.
    """
    plan = plan_from_asset(asset, mode=mode, **kwargs)

    # Signal/EEG/MEG
    if asset.family == "eeg" or asset.intent == INTENT_RAW_SIGNAL:
        return _render_signal(asset, plan, **kwargs)

    # DICOM series browser
    if asset.family == "dicom":
        return _render_dicom(asset, plan, **kwargs)

    # Surface
    if asset.intent == INTENT_SURFACE:
        try:
            from qortex.visualize.surface import surface_summary
            fig = surface_summary(asset.path, title=kwargs.get("title", ""))
            return VisualResult(
                asset=asset,
                plan=plan,
                figures=[fig],
                warnings=list(asset.warnings),
                provenance={"renderer": "surface_summary", "path": str(asset.path)},
            )
        except Exception as exc:
            result = _render_summary_only(asset, plan, f"Surface render failed: {exc}")
            result.warnings.append(VisualWarning(
                code="surface_render_failed",
                message=f"Surface render failed: {exc}",
                severity="warning",
            ))
            return result

    # Volumetric (NIfTI / DICOM as volume)
    return _render_volume(asset, plan, **kwargs)


def _render_volume(asset: VisualAsset, plan: VisualPlan, **kwargs) -> VisualResult:
    from qortex.visualize.volume import VolumeViewer
    try:
        viewer = VolumeViewer(
            asset.path,
            modality=asset.modality,
            window=plan.window_preset,
            colormap=plan.colormap,
        )

        if plan.mode in {MODE_SUMMARY, "qc"}:
            if asset.intent == INTENT_BOLD and viewer._lazy is not None and len(viewer._lazy.shape) == 4:
                fig = viewer.fmri_summary(title=kwargs.get("title", ""))
                return VisualResult(
                    asset=asset,
                    plan=plan,
                    figures=[fig],
                    warnings=list(asset.warnings),
                    provenance={"renderer": "VolumeViewer.fmri_summary", "path": str(asset.path)},
                )
            if asset.intent == INTENT_DWI:
                from qortex.visualize.dwi import DWIViewer
                fig = DWIViewer(
                    asset.path,
                    bval_path=kwargs.get("bval_path"),
                    bvec_path=kwargs.get("bvec_path"),
                ).dwi_summary(title=kwargs.get("title", ""))
                return VisualResult(
                    asset=asset,
                    plan=plan,
                    figures=[fig],
                    warnings=list(asset.warnings),
                    provenance={"renderer": "DWIViewer.dwi_summary", "path": str(asset.path)},
                )
            if plan.mode == "qc":
                fig = viewer.ortho(title=kwargs.get("title", f"QC — {asset.path.name}"))
                return VisualResult(
                    asset=asset,
                    plan=plan,
                    figures=[fig],
                    warnings=list(asset.warnings),
                    provenance={"renderer": "VolumeViewer.ortho", "mode": "qc", "path": str(asset.path)},
                )
            return _render_summary_only(asset, plan)

        elif plan.mode == "thumbnail":
            # Per-modality lazy thumbnail — reads the minimum data possible
            from qortex.visualize._html import array_to_b64png
            import base64
            if viewer._lazy is not None:
                lazy = viewer._lazy
                if asset.intent == INTENT_BOLD and len(lazy.shape) == 4:
                    # fMRI: render tSNR map center slice (Welford streaming, ~20 frames)
                    tsnr = lazy.tsnr_volume(max_frames=20)
                    cz = tsnr.shape[2] // 2
                    slc = tsnr[:, :, cz].T[::-1, :]
                    pos_vals = slc[slc > 0]
                    ts_vmin = float(np.percentile(pos_vals, 2)) if pos_vals.size else 0.0
                    ts_vmax = float(np.percentile(pos_vals, 98)) if pos_vals.size else 100.0
                    b64 = array_to_b64png(slc, ts_vmin, ts_vmax, "hot")
                else:
                    # All other volumetric modalities: single center axial slice
                    cz = lazy.shape[2] // 2
                    slc = lazy.slice_along(2, cz).T[::-1, :]
                    b64 = array_to_b64png(slc, viewer._vmin, viewer._vmax, viewer.colormap)
            else:
                vol3d = viewer._vol3d()
                cz = vol3d.shape[2] // 2
                slc = vol3d[:, :, cz].T[::-1, :]
                b64 = array_to_b64png(slc, viewer._vmin, viewer._vmax, viewer.colormap)
            png_bytes = base64.b64decode(b64)
            return VisualResult(asset=asset, plan=plan, png_bytes=png_bytes,
                                warnings=list(asset.warnings))

        elif plan.mode == "static":
            if asset.intent == INTENT_DWI:
                from qortex.visualize.dwi import DWIViewer
                fig = DWIViewer(
                    asset.path,
                    bval_path=kwargs.get("bval_path"),
                    bvec_path=kwargs.get("bvec_path"),
                ).dwi_summary(title=kwargs.get("title", ""))
                return VisualResult(
                    asset=asset,
                    plan=plan,
                    figures=[fig],
                    warnings=list(asset.warnings),
                    provenance={"renderer": "DWIViewer.dwi_summary", "path": str(asset.path)},
                )
            # Ortho view via plotly (returns figures)
            try:
                fig = viewer.ortho(title=kwargs.get("title", ""))
                result = VisualResult(asset=asset, plan=plan, figures=[fig],
                                      warnings=list(asset.warnings))
                return result
            except ImportError:
                pass  # fall through to interactive

        # Default / interactive_html
        html = viewer.interactive_html(
            title=kwargs.get("title", f"{asset.intent.replace('_', ' ').title()} — {asset.path.name}"),
            max_slices_per_axis=80,
        )
        return VisualResult(asset=asset, plan=plan, html=html,
                            warnings=list(asset.warnings),
                            provenance={"renderer": "VolumeViewer", "path": str(asset.path)})
    except Exception as exc:
        result = _render_summary_only(asset, plan, str(exc))
        result.warnings.append(VisualWarning(
            code="render_failed", message=f"Volume render failed: {exc}", severity="error"
        ))
        return result


def _render_signal(asset: VisualAsset, plan: VisualPlan, **kwargs) -> VisualResult:
    from qortex.visualize.timeseries import TimeSeriesViewer
    try:
        viewer = TimeSeriesViewer(asset.path, modality=asset.modality)
        html = viewer.dashboard(title=f"Signal — {asset.path.name}")
        return VisualResult(
            asset=asset, plan=plan, html=html,
            warnings=list(asset.warnings),
            provenance={"renderer": "TimeSeriesViewer", "path": str(asset.path)},
        )
    except Exception as exc:
        return _render_summary_only(asset, plan, f"Signal render failed: {exc}")


def _render_dicom(asset: VisualAsset, plan: VisualPlan, **kwargs) -> VisualResult:
    from qortex.visualize.dicom import DicomSeriesBrowser
    try:
        browser = DicomSeriesBrowser(asset.path)
        html = browser.to_html()
        return VisualResult(
            asset=asset, plan=plan, html=html,
            warnings=list(asset.warnings),
            provenance={"renderer": "DicomSeriesBrowser", "path": str(asset.path)},
        )
    except Exception as exc:
        return _render_summary_only(asset, plan, f"DICOM render failed: {exc}")


def _render_summary_only(asset: VisualAsset, plan: VisualPlan, note: str = "") -> VisualResult:
    summary = asset.summary()
    note_html = f"<p style='color:#f96'>{note}</p>" if note else ""
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
  body{{background:#111;color:#ccc;font-family:monospace;margin:24px;}}
  pre{{background:#1a1a1a;padding:16px;border-radius:6px;color:#6af;}}
  h2{{color:#6af;}}
</style></head><body>
<h2>Visual Summary</h2>
{note_html}
<pre>{summary}</pre>
</body></html>"""
    return VisualResult(asset=asset, plan=plan, html=html, warnings=list(asset.warnings))
