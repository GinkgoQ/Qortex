from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from qortex.convert.formats._base import FormatWriter

__all__ = [
    "FormatWriter",
    "get_writer",
]


@dataclass(frozen=True)
class _WriterSpec:
    module: str
    class_name: str


_REGISTRY: dict[str, _WriterSpec] = {
    "parquet": _WriterSpec("qortex.convert.formats.parquet", "ParquetWriter"),
    "zarr": _WriterSpec("qortex.convert.formats.zarr", "ZarrWriter"),
    "hdf5": _WriterSpec("qortex.convert.formats.hdf5", "HDF5Writer"),
    "webdataset": _WriterSpec("qortex.convert.formats.webdataset", "WebDatasetWriter"),
    "huggingface": _WriterSpec(
        "qortex.convert.formats.huggingface",
        "HuggingFaceWriter",
    ),
    "tfrecord": _WriterSpec("qortex.convert.formats.tfrecord", "TFRecordWriter"),
}


def get_writer(fmt: str) -> FormatWriter:
    """Return a FormatWriter instance for the given format name."""
    try:
        spec = _REGISTRY[fmt.lower()]
    except KeyError:
        from qortex.core.exceptions import FormatNotSupportedError

        raise FormatNotSupportedError(
            f"Unknown output format '{fmt}'. "
            f"Available: {sorted(_REGISTRY)}"
        ) from None

    try:
        module = import_module(spec.module)
    except ImportError as exc:
        from qortex.core.exceptions import ConversionError

        raise ConversionError(
            f"Output format '{fmt}' requires an optional dependency that is not "
            f"installed. Install the dependency for '{fmt}' or choose one of: "
            f"{sorted(_REGISTRY)}. Original import error: {exc}"
        ) from exc

    return getattr(module, spec.class_name)()
