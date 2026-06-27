"""SourceAdapter abstract base class.

Every source adapter must:
  - probe() → SourceProfile        (no data loaded, header-only)
  - read_batch() → QortexData       (load all in-memory)
  - stream() → Iterator[QortexData] (lazy iteration)
  - replay() → Iterator[QortexData] (time-accurate replay at given speed)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from qortex.neuroai.contracts import QortexTimeSeries, QortexVolume, SourceProfile


QortexData = QortexTimeSeries | QortexVolume | Any


class SourceAdapter(ABC):
    """Abstract base for all NeuroAI source adapters."""

    @abstractmethod
    def probe(self) -> SourceProfile:
        """Probe the source without loading data.

        Returns a ``SourceProfile`` describing what the source can provide.
        Should be fast — header-only reads; no full data load.
        """

    @abstractmethod
    def read_batch(self) -> list[Any]:
        """Load the full source into memory.

        Returns a list of QortexData items (one per subject/run/window).
        """

    @abstractmethod
    def stream(self) -> Iterator[Any]:
        """Yield data items lazily.

        For file sources: one item per window or per file.
        For live sources: one item per incoming data record.
        """

    def replay(self, speed: float = 1.0) -> Iterator[Any]:
        """Replay a recorded source at the given speed multiplier.

        Default implementation calls ``stream()`` without timing.
        Real-time adapters should override with accurate sleep-based timing.
        """
        import time
        profile = self.probe()
        rate_hz = getattr(profile, "sampling_rate_hz", None)
        for item in self.stream():
            yield item
            if speed != 0 and rate_hz:
                win_dur = getattr(item, "shape", [0])[-1] / rate_hz if hasattr(item, "shape") else 0
                if win_dur > 0:
                    time.sleep(win_dur / speed)

    @property
    def source_id(self) -> str:
        return self.__class__.__name__
