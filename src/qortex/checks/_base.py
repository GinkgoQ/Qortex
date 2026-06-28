"""Abstract base class for all Qortex checks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from qortex.checks._report import CheckReport


class BaseChecker(ABC):
    """Smallest composable unit of validation.

    Subclasses implement exactly one validation concern.  They must:
    - separate data collection from decision logic;
    - avoid loading full data unless the check explicitly requires it;
    - return UNKNOWN instead of guessing;
    - return BLOCK only when evidence is strong.
    """

    #: Human-readable name used in reports and CLI output
    name: str = "unnamed_check"

    #: Which goal(s) this check is required for (visualize, convert, train, neuroai-run)
    required_for: frozenset[str] = frozenset()

    @abstractmethod
    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        """Execute the check and return a finalized CheckReport."""

    @classmethod
    def check_name(cls) -> str:
        return getattr(cls, "name", cls.__name__.lower().replace("checker", ""))
