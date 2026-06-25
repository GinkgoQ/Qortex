"""BaseAdapter protocol — every training adapter satisfies this interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class BaseAdapter(Protocol):
    """Convert a Qortex output directory into a framework-native dataset object."""

    framework: str

    def from_dir(self, data_dir: Path, split: str | None = None) -> Any:
        """Load the artifact at data_dir and return a framework dataset.

        Parameters
        ----------
        data_dir:
            Root output directory from ConversionPipeline.run().
        split:
            Optional split name ("train", "val", "test").  When provided,
            only samples with that split tag are returned.
        """
        ...
