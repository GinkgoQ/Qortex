"""Auto-detect and construct the correct SourceAdapter from a SourceSpec."""

from __future__ import annotations

from pathlib import Path

from qortex.neuroai.sources._base import SourceAdapter
from qortex.neuroai.spec import SourceSpec, WindowSpec


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
    """
    src_type = (spec.type or "").lower().strip()

    if src_type == "bids":
        from qortex.neuroai.sources.bids import BIDSSourceAdapter
        return BIDSSourceAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type in ("local_file", "file", "local"):
        from qortex.neuroai.sources.local import LocalFileAdapter
        return LocalFileAdapter(spec, window_spec=window_spec, channel_names=channel_names)

    if src_type == "lsl":
        try:
            from qortex.neuroai.sources.lsl import LSLSourceAdapter
            return LSLSourceAdapter(spec, window_spec=window_spec, channel_names=channel_names)
        except ImportError:
            raise ImportError(
                "LSL source requires pylsl. "
                "Install with: pip install 'qortex[lsl]'"
            )

    if src_type in ("xdf", "replay"):
        from qortex.neuroai.sources.local import LocalFileAdapter
        if spec.path and Path(spec.path).suffix.lower() == ".xdf":
            return LocalFileAdapter(spec, window_spec=window_spec, channel_names=channel_names)
        raise ValueError(f"XDF source requires spec.path pointing to an .xdf file")

    # Auto-detect from path extension
    if spec.path:
        path = Path(spec.path)
        if path.is_dir():
            desc = path / "dataset_description.json"
            if desc.exists():
                from qortex.neuroai.sources.bids import BIDSSourceAdapter
                bids_spec = SourceSpec(type="bids", path=str(path),
                                       modality=spec.modality, suffix=spec.suffix,
                                       subjects=spec.subjects)
                return BIDSSourceAdapter(bids_spec, window_spec=window_spec,
                                         channel_names=channel_names)
            raise ValueError(f"Directory source {path} is not a BIDS dataset "
                             f"(no dataset_description.json)")
        else:
            local_spec = SourceSpec(type="local_file", path=str(path),
                                    modality=spec.modality)
            from qortex.neuroai.sources.local import LocalFileAdapter
            return LocalFileAdapter(local_spec, window_spec=window_spec,
                                    channel_names=channel_names)

    raise ValueError(
        f"Cannot determine source adapter for type={src_type!r}. "
        f"Supported types: 'local_file', 'bids', 'lsl', 'xdf'."
    )
