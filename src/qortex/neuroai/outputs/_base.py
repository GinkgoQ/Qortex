"""OutputAdapter abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput


class OutputAdapter(ABC):
    """Write standardized model outputs to a destination.

    Every concrete adapter **must** maintain a ``_n_written`` counter and
    expose it via the ``n_written`` property so the runtime engine can
    accurately report total outputs written in ``PipelineRunReport``.
    """

    _n_written: int = 0

    @abstractmethod
    def open(self) -> None:
        """Open the output destination (file, stream, etc.)."""

    @abstractmethod
    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        """Write one model output record."""

    @abstractmethod
    def close(self) -> None:
        """Flush and close the output destination."""

    @property
    def n_written(self) -> int:
        """Number of records successfully written so far."""
        return self._n_written

    def write_marker(self, marker: Any) -> None:
        """Write a structured EventMarkerOutput (trigger event).

        Override in adapters that support closed-loop trigger events.
        The default implementation is a no-op so adapters that don't support
        markers don't need to implement it.
        """

    def __enter__(self) -> "OutputAdapter":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
