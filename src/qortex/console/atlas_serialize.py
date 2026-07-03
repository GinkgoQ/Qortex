"""JSON-safe serialization for the mixed Pydantic/dataclass/enum model zoo
used across Qortex's decision, inspect, and neuroai subsystems.

Qortex intentionally does not standardize every return type on Pydantic —
``DatasetFitness``/``LabelLandscape``/``SignalBudget`` are plain
``@dataclass`` objects, while ``ReadinessReport``/``DoctorReport`` etc. are
Pydantic v2 models. The Atlas console API needs one function that turns any
of them (nested, arbitrarily deep) into plain JSON-serializable Python.
"""

from __future__ import annotations

import dataclasses
import enum
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from pydantic import BaseModel as _PydanticModel
except ImportError:  # pragma: no cover - pydantic is a hard Qortex dependency
    _PydanticModel = None  # type: ignore[assignment]


def _own_computed_properties(obj: Any) -> dict[str, Any]:
    """Computed ``@property`` attributes declared on Qortex's own domain
    classes (``ManifestSummary.n_subjects``, ``DownloadResult.success``,
    ``DoctorReport.n_errors``, ``DiffReport.consistent``, etc).

    Neither ``BaseModel.model_dump()`` nor ``dataclasses.fields()`` include
    plain ``@property`` members — only declared fields — so every one of
    these computed values was silently missing from serialized JSON, and
    every reader (Python ``.get(...)`` or JS ``response.field``) saw a
    default/undefined instead of the real number. This was caught for one
    field (``ManifestSummary.n_subjects`` rendering as 0 for a 17-subject
    dataset) but is a property of the serializer itself, not that one call
    site, so it's fixed once here for every domain class.

    Restricted to classes defined in ``qortex.*`` (checked via
    ``__module__``) so this never pulls in Pydantic's own public-but-
    internal properties (``model_extra``, ``model_fields_set``, ...).
    """
    out: dict[str, Any] = {}
    for cls in type(obj).__mro__:
        if not cls.__module__.startswith("qortex"):
            continue
        for name, member in vars(cls).items():
            if name.startswith("_") or name in out or not isinstance(member, property):
                continue
            try:
                out[name] = getattr(obj, name)
            except Exception:
                continue
    return out


def to_jsonable(obj: Any) -> Any:
    """Recursively convert Qortex domain objects into JSON-safe Python."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if _PydanticModel is not None and isinstance(obj, _PydanticModel):
        data = obj.model_dump(mode="python")
        data.update(_own_computed_properties(obj))
        return to_jsonable(data)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        data = {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        data.update({k: to_jsonable(v) for k, v in _own_computed_properties(obj).items()})
        return data
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)
