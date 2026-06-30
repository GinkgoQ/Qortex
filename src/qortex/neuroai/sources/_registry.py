"""Auto-detect and construct the correct SourceAdapter from a SourceSpec."""

from __future__ import annotations

from pathlib import Path

from qortex.neuroai.sources._base import SourceAdapter
from qortex.neuroai.spec import SourceSpec, WindowSpec

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def make_source_adapter(
    spec: SourceSpec,
    *,
    window_spec: WindowSpec | None = None,
    channel_names: list[str] | None = None,
) -> SourceAdapter:
    """Factory: return the right SourceAdapter for the given spec.

    Parameters
    ----------
    spec:
        Pipeline source specification.
    window_spec:
        Optional windowing for streaming.
    channel_names:
        Optional channel subset.

    Returns
    -------
    SourceAdapter
        Concrete adapter ready to call ``.probe()`` or ``.stream()``.

    Raises
    ------
    ValueError
        When the source type is unknown or unsupported.
    ImportError
        When the required optional dependency is missing.
    """
    src_type = (spec.type or "").lower().strip()

    if src_type == "bids":
        from qortex.neuroai.sources.bids import BIDSSourceAdapter
        return BIDSSourceAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("local_file", "file", "local"):
        from qortex.neuroai.sources.local import LocalFileAdapter
        return LocalFileAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("dicom", "dicom_folder"):
        from qortex.neuroai.sources.dicom import DICOMFolderAdapter
        return DICOMFolderAdapter(spec, window_spec=window_spec)

    if src_type in ("dicomweb", "pacs", "wado", "wado_rs"):
        from qortex.neuroai.sources.dicomweb import DICOMWebAdapter
        return DICOMWebAdapter(spec, window_spec=window_spec)

    if src_type in ("nwb",):
        from qortex.neuroai.sources.nwb import NWBAdapter
        return NWBAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("xdf",):
        from qortex.neuroai.sources.xdf import XDFAdapter
        return XDFAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("lsl",):
        from qortex.neuroai.sources.lsl import LSLSourceAdapter
        return LSLSourceAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("brainflow", "bf"):
        from qortex.neuroai.sources.brainflow import BrainFlowAdapter
        return BrainFlowAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("image", "video", "img"):
        from qortex.neuroai.sources.image import ImageVideoAdapter
        return ImageVideoAdapter(spec, window_spec=window_spec)

    # Auto-detect from path extension or directory structure
    if spec.path:
        path = Path(spec.path)

        if path.is_dir():
            desc = path / "dataset_description.json"
            if desc.exists():
                from qortex.neuroai.sources.bids import BIDSSourceAdapter
                bids_spec = SourceSpec(
                    type="bids", path=str(path),
                    modality=spec.modality, suffix=spec.suffix,
                    subjects=spec.subjects, sessions=spec.sessions,
                    extra=dict(spec.extra or {}),
                )
                return BIDSSourceAdapter(bids_spec, window_spec=window_spec,
                                         channel_names=channel_names)
            # Check for DICOM
            dcm_files = list(path.glob("*.dcm")) + list(path.glob("**/*.dcm"))
            if dcm_files:
                from qortex.neuroai.sources.dicom import DICOMFolderAdapter
                dicom_spec = SourceSpec(type="dicom", path=str(path))
                return DICOMFolderAdapter(dicom_spec, window_spec=window_spec)
            raise ValueError(
                f"Directory source {path} is not a BIDS dataset or DICOM folder."
            )

        ext = path.suffix.lower()
        double_ext = "".join(path.suffixes[-2:]).lower()  # e.g. ".nii.gz"

        if ext == ".dcm":
            from qortex.neuroai.sources.dicom import DICOMFolderAdapter
            dicom_spec = SourceSpec(type="dicom", path=str(path.parent))
            return DICOMFolderAdapter(dicom_spec, window_spec=window_spec)

        if ext in (".nwb",):
            from qortex.neuroai.sources.nwb import NWBAdapter
            return NWBAdapter(spec, window_spec=window_spec, channel_names=channel_names)

        if ext == ".xdf":
            from qortex.neuroai.sources.xdf import XDFAdapter
            return XDFAdapter(spec, window_spec=window_spec, channel_names=channel_names)

        if ext in _IMAGE_EXTS or ext in _VIDEO_EXTS:
            from qortex.neuroai.sources.image import ImageVideoAdapter
            return ImageVideoAdapter(spec, window_spec=window_spec)

        # Default to local file for EDF/FIF/NIfTI/CSV
        from qortex.neuroai.sources.local import LocalFileAdapter
        return LocalFileAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    raise ValueError(
        f"Cannot determine source adapter for type={src_type!r}, path={spec.path!r}. "
        f"Supported types: 'local_file', 'bids', 'dicom', 'dicomweb', "
        f"'nwb', 'xdf', 'lsl', 'brainflow', 'image', 'video'."
    )
