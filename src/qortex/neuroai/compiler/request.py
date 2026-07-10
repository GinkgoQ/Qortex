"""Request schema for the Qortex NeuroAI compiler."""

from __future__ import annotations

from qortex.neuroai.contracts import BaseModel


class CompilationRequest(BaseModel):
    source: str
    task: str
    device: str = "cpu"
    max_download_gb: float | None = None
    max_vram_gb: float | None = None
    accept_unknown_license_risk: bool = False
    allow_remote_code: bool = False
    require_open_license: bool = True
    include_plan_only: bool = True


__all__ = ["CompilationRequest"]
