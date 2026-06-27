"""OutputAdapter abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput


class OutputAdapter(ABC):
    """Write standardized model outputs to a destination."""

    @abstractmethod
    def open(self) -> None:
        """Open the output destination (file, stream, etc.)."""

    @abstractmethod
    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        """Write one model output record."""

    @abstractmethod
    def close(self) -> None:
        """Flush and close the output destination."""

    def __enter__(self) -> "OutputAdapter":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
