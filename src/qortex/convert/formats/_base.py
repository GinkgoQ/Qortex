"""FormatWriter protocol — every format writer must satisfy this interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

from qortex.core.entities import SampleRecord


@runtime_checkable
class FormatWriter(Protocol):
    """Write an iterable of SampleRecords to a target directory."""

    format_name: str
    file_extension: str

    def write(
        self,
        samples: Iterator[SampleRecord],
        output_dir: Path,
        *,
        shard_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Write samples to output_dir.  Returns the root path of the output."""
        ...

    def estimate_size(self, n_samples: int, sample_shape: tuple[int, ...]) -> int:
        """Return a rough byte estimate for n_samples of the given shape."""
        ...
