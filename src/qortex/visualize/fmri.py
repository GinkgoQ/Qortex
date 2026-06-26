"""fMRI-specific QC entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.visualize.volume import VolumeViewer


def fmri_summary(
    bold_path: Path | str,
    *,
    max_frames: int = 50,
    title: str = "",
    events_path: Path | str | None = None,
    confounds_path: Path | str | None = None,
) -> Any:
    """Return a Plotly fMRI QC summary for a 4D BOLD NIfTI.

    The figure includes mean EPI, middle frame, temporal standard deviation,
    tSNR, global signal, framewise slice-time intensity, and optional
    events/confounds traces when sidecars are present or passed explicitly.
    """
    return VolumeViewer(bold_path, modality="fmri").fmri_summary(
        max_frames=max_frames,
        title=title,
        events_path=events_path,
        confounds_path=confounds_path,
    )
