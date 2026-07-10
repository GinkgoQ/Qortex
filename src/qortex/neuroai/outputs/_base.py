"""OutputAdapter abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qortex.core.exceptions import QortexError
from qortex.neuroai.models._base import ModelOutput


class OutputAdapterError(QortexError):
    """Raised when an output adapter cannot write the requested format."""

    default_code = "neuroai.output_adapter_error"


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

    @property
    def n_prediction_records(self) -> int:
        """Number of primary prediction records written."""
        return self._n_written

    @property
    def n_marker_records(self) -> int:
        """Number of trigger/event marker records written."""
        return 0

    @property
    def n_output_records_total(self) -> int:
        """Total records written, including markers when supported."""
        return self.n_prediction_records + self.n_marker_records

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


__all__ = ["OutputAdapter", "OutputAdapterError"]
