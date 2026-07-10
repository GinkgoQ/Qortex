"""Repair options emitted by the NeuroAI compiler."""

from __future__ import annotations

from typing import Literal

from qortex.neuroai.contracts import BaseModel, Field


RepairSeverity = Literal["info", "warning", "blocking"]


class RepairOption(BaseModel):
    code: str
    severity: RepairSeverity
    title: str
    detail: str
    command: list[str] = Field(default_factory=list)
    affects: list[str] = Field(default_factory=list)


__all__ = ["RepairOption", "RepairSeverity"]
