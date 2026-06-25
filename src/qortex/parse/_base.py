"""BaseLoader protocol — every modality loader must satisfy this interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol, Union, runtime_checkable

import numpy as np

from qortex.core.entities import EventsRecord, FileRecord, ImageRecord, SampleRecord, SignalRecord

AnyRecord = Union[SignalRecord, ImageRecord, EventsRecord]


@runtime_checkable
class BaseLoader(Protocol):
    """Protocol every Qortex modality loader must implement.

    Lifecycle
    ---------
    1. ``can_load(file)``  — routing: is this loader the right one?
    2. ``inspect(file, path)``  — fast metadata without RAM allocation
    3. ``load(file, path)``     — eager: full data in memory
       ``lazy_load(file, path)`` — lazy: header only, data on-demand
    4. ``to_numpy(record)``     — extract array from any record type
    5. ``to_sample_records(record)`` — yield ML-ready SampleRecords

    Loaders registered via ``entry_points["qortex.loaders"]`` are
    discovered automatically by ``LoaderRegistry.discover()``.
    """

    modality: str
    supported_extensions: frozenset[str]

    def can_load(self, file: FileRecord) -> bool: ...

    def inspect(self, file: FileRecord, local_path: Path) -> dict[str, Any]: ...

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> AnyRecord: ...

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> AnyRecord: ...

    def to_numpy(self, record: AnyRecord, **kwargs) -> np.ndarray: ...

    def to_sample_records(
        self,
        record: AnyRecord,
        **kwargs,
    ) -> Iterator[SampleRecord]: ...
