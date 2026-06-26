"""Universal visual asset descriptor — the core language of qortex.visualize.

VisualAsset is the typed answer to "what is this file and what should I show?"
It is produced cheaply (header-only reads) and consumed by every renderer.
No pixel data is loaded to build a VisualAsset.

Hierarchy
---------
VisualFamily   — file-format family: "nifti", "dicom", "eeg", "gifti", …
VisualIntent   — what the data represents: "anatomical_volume", "bold_fmri", …
VisualModality — imaging physics: "mri", "ct", "pet", "eeg", "meg", …

Together these three axes drive the rendering decision.

VisualWarning  — structured problem/observation; always exposed to the caller.
VisualPlan     — pre-render summary: what will be rendered, which backend, why.
VisualResult   — post-render output: html/png bytes + provenance metadata.
"""

from __future__ import annotations

import json
import logging
import tempfile
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

log = logging.getLogger(__name__)

# ── Enumerations (kept as plain strings so downstream code needs no imports) ──

VisualFamily = str    # "nifti" | "dicom" | "eeg" | "gifti" | "cifti" | "unknown"
VisualIntent = str    # see INTENT_* constants below
VisualModality = str  # "mri" | "fmri" | "ct" | "pet" | "dwi" | "eeg" | "meg" | …

# Visual intent constants
INTENT_ANATOMICAL = "anatomical_volume"
INTENT_BOLD = "bold_fmri"
INTENT_DWI = "diffusion_volume"
INTENT_PET = "pet_volume"
INTENT_CT = "ct_volume"
INTENT_FIELDMAP = "fieldmap"
INTENT_MASK = "mask"
INTENT_LABELMAP = "labelmap"
INTENT_STAT_MAP = "statistical_map"
INTENT_SURFACE = "surface"
INTENT_SERIES_BROWSER = "dicom_series_browser"
INTENT_RAW_SIGNAL = "raw_electrophysiology"
INTENT_UNKNOWN = "unknown"

# Rendering modes
MODE_THUMBNAIL = "thumbnail"
MODE_STATIC = "static"
MODE_INTERACTIVE = "interactive_html"
MODE_DESKTOP = "desktop"
MODE_SUMMARY = "summary"


# ── VisualWarning ─────────────────────────────────────────────────────────────

@dataclass
class VisualWarning:
    """Structured problem, observation, or recommendation about a visual asset."""

    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    path: str | None = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}:{self.code}] {self.message}"


def _warn(code: str, msg: str, sev: str = "warning", path: str | None = None) -> VisualWarning:
    return VisualWarning(code=code, message=msg, severity=sev, path=path)


# ── VisualAsset ───────────────────────────────────────────────────────────────

@dataclass
class VisualAsset:
    """Everything known about a visualizable file before any pixel is loaded.

    Built cheaply by reading only file headers / directory metadata.
    This is the *universal language* — the structured answer to "what is this?"

    Parameters
    ----------
    path:
        Absolute path to the file or directory.
    family:
        File-format family: "nifti", "dicom", "eeg", "gifti", "unknown".
    intent:
        What the data represents.  See INTENT_* constants.
    modality:
        Imaging modality: "mri", "ct", "pet", "fmri", "dwi", "eeg", "meg".
    shape:
        Voxel/sample dimensions.  Empty tuple if unknown.
    ndim:
        Number of dimensions.
    spacing:
        Voxel sizes in mm (or sample interval in ms for signals).
    orientation:
        RAS/LAS/… orientation string.  None if unknown.
    affine:
        4×4 world-to-voxel affine.  None if not applicable.
    dtype:
        Numpy dtype string for the data array.
    n_timepoints:
        Number of 4th-dimension frames (fMRI, dynamic PET, EEG).
    n_channels:
        Number of EEG/MEG channels or diffusion directions.
    is_mask:
        True if the data appears to be a binary or label mask.
    is_stat_map:
        True if the data appears to be a statistical map (z/t/F).
    is_large:
        True if full in-memory loading would exceed ~4 GB.
    recommended_view:
        Suggested rendering strategy for this asset.
    warnings:
        Ordered list of VisualWarning items detected during inspection.
    metadata:
        Free-form dict of extra metadata (DICOM tags, sidecar JSON, etc.).
    companion_paths:
        Known companion files (bvec/bval, json sidecar, events.tsv, etc.).
    """

    path: Path
    family: VisualFamily = INTENT_UNKNOWN
    intent: VisualIntent = INTENT_UNKNOWN
    modality: VisualModality = "unknown"
    shape: tuple = ()
    ndim: int = 0
    spacing: tuple[float, ...] | None = None
    orientation: str | None = None
    affine: np.ndarray | None = None
    dtype: str = "unknown"
    n_timepoints: int = 1
    n_channels: int = 0
    is_mask: bool = False
    is_stat_map: bool = False
    is_large: bool = False
    recommended_view: str = MODE_STATIC
    warnings: list[VisualWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    companion_paths: list[Path] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────

    @property
    def is_4d(self) -> bool:
        return self.ndim == 4

    @property
    def is_dicom_series(self) -> bool:
        return self.family == "dicom" and self.path.is_dir()

    @property
    def size_voxels(self) -> int:
        if not self.shape:
            return 0
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def estimated_memory_mb(self) -> float:
        bytes_per_voxel = {"float32": 4, "float64": 8, "int16": 2, "uint8": 1}.get(self.dtype, 4)
        return self.size_voxels * bytes_per_voxel / 1e6

    @property
    def has_errors(self) -> bool:
        return any(w.severity == "error" for w in self.warnings)

    @property
    def voxel_size_str(self) -> str:
        if not self.spacing:
            return "unknown"
        return " × ".join(f"{s:.2f}" for s in self.spacing) + " mm"

    @property
    def shape_str(self) -> str:
        return " × ".join(str(s) for s in self.shape)

    def warn(self, code: str, msg: str, sev: str = "warning") -> None:
        self.warnings.append(_warn(code, msg, sev, str(self.path)))

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Path       : {self.path}",
            f"Family     : {self.family}",
            f"Intent     : {self.intent}",
            f"Modality   : {self.modality}",
            f"Shape      : {self.shape_str}",
            f"Voxel size : {self.voxel_size_str}",
            f"Orientation: {self.orientation or 'unknown'}",
            f"Timepoints : {self.n_timepoints}",
            f"Memory est : {self.estimated_memory_mb:.0f} MB",
            f"Recommended: {self.recommended_view}",
        ]
        if self.warnings:
            lines.append("Warnings   :")
            for w in self.warnings:
                lines.append(f"  {w}")
        return "\n".join(lines)

    # ── Plan / render ─────────────────────────────────────────────────────

    def plan(self, mode: str = "auto", **kwargs) -> "VisualPlan":
        """Derive a VisualPlan from this asset."""
        from qortex.visualize._dispatch import plan_from_asset
        return plan_from_asset(self, mode=mode, **kwargs)

    def render(self, mode: str = "auto", **kwargs) -> "VisualResult":
        """Render this asset and return a VisualResult."""
        from qortex.visualize._dispatch import render_asset
        return render_asset(self, mode=mode, **kwargs)

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "family": self.family,
            "intent": self.intent,
            "modality": self.modality,
            "shape": list(self.shape),
            "ndim": self.ndim,
            "spacing": list(self.spacing) if self.spacing else None,
            "orientation": self.orientation,
            "dtype": self.dtype,
            "n_timepoints": self.n_timepoints,
            "n_channels": self.n_channels,
            "is_mask": self.is_mask,
            "is_stat_map": self.is_stat_map,
            "is_large": self.is_large,
            "recommended_view": self.recommended_view,
            "estimated_memory_mb": self.estimated_memory_mb,
            "warnings": [
                {"code": w.code, "message": w.message, "severity": w.severity}
                for w in self.warnings
            ],
            "metadata": {k: str(v) for k, v in self.metadata.items()},
        }


# ── VisualPlan ────────────────────────────────────────────────────────────────

@dataclass
class VisualPlan:
    """Pre-render decision: what will be rendered and why.

    The plan is produced before any pixel data is loaded.  It exposes the
    reasoning so users can override before committing to rendering.
    """

    asset: VisualAsset
    mode: str                    # "static" | "interactive_html" | "thumbnail"
    backend: str                 # "pure_python" | "plotly" | "nilearn" | "mne"
    views: list[str]             # e.g. ["orthogonal", "mosaic", "timeseries"]
    window_preset: str | None    # e.g. "brain", "bone", "auto"
    colormap: str                # "gray", "hot", "plasma", "RdBu_r"
    overlay_path: Path | None    # companion overlay, if any
    requires_companions: list[str]  # companion file types needed
    warnings: list[VisualWarning] = field(default_factory=list)

    @property
    def estimated_memory_mb(self) -> float:
        return self.asset.estimated_memory_mb

    def describe(self) -> str:
        lines = [
            f"  Mode     : {self.mode}",
            f"  Backend  : {self.backend}",
            f"  Views    : {', '.join(self.views)}",
            f"  Window   : {self.window_preset or 'auto'}",
            f"  Colormap : {self.colormap}",
            f"  Memory   : ~{self.estimated_memory_mb:.0f} MB",
        ]
        if self.overlay_path:
            lines.append(f"  Overlay  : {self.overlay_path}")
        if self.requires_companions:
            lines.append(f"  Needs    : {', '.join(self.requires_companions)}")
        return "\n".join(lines)


# ── VisualResult ──────────────────────────────────────────────────────────────

@dataclass
class VisualResult:
    """Post-render output: rendered figures + provenance.

    Every rendering path in qortex.visualize produces a VisualResult so that
    callers have a consistent interface regardless of modality.
    """

    asset: VisualAsset
    plan: VisualPlan
    html: str | None = None            # self-contained HTML string
    png_bytes: bytes | None = None     # static PNG bytes
    figures: list[Any] = field(default_factory=list)   # plotly Figure objects
    warnings: list[VisualWarning] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def show(self) -> None:
        """Open the result in the default browser."""
        content = self.html or self._figures_to_html()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write(content)
            webbrowser.open(f"file://{f.name}")

    def to_html(self, path: Path | str, *, write_sidecar: bool = True) -> Path:
        """Write HTML to file and optionally write a JSON sidecar."""
        out = Path(path)
        content = self.html or self._figures_to_html()
        out.write_text(content, encoding="utf-8")
        if write_sidecar:
            sidecar = out.with_suffix(".json")
            sidecar.write_text(
                json.dumps(self.to_provenance_dict(), indent=2),
                encoding="utf-8",
            )
        return out

    def to_png(self, path: Path | str) -> Path:
        """Write PNG to file (if available)."""
        out = Path(path)
        if self.png_bytes:
            out.write_bytes(self.png_bytes)
        elif self.figures:
            try:
                import plotly.io as pio
                pio.write_image(self.figures[0], str(out))
            except ImportError:
                raise RuntimeError("PNG export requires kaleido: pip install kaleido")
        else:
            raise RuntimeError("No renderable output available")
        return out

    def to_provenance_dict(self) -> dict:
        return {
            "asset": self.asset.to_dict(),
            "plan": {
                "mode": self.plan.mode,
                "backend": self.plan.backend,
                "views": self.plan.views,
                "window_preset": self.plan.window_preset,
                "colormap": self.plan.colormap,
            },
            "warnings": [
                {"code": w.code, "message": w.message, "severity": w.severity}
                for w in self.warnings
            ],
            **self.provenance,
        }

    def _figures_to_html(self) -> str:
        if not self.figures:
            return "<html><body><p>No visual output generated.</p></body></html>"
        try:
            import plotly.io as pio
            parts = []
            for i, fig in enumerate(self.figures):
                kwargs = {"full_html": i == 0, "include_plotlyjs": "cdn" if i == 0 else False}
                parts.append(pio.to_html(fig, **kwargs))
            if len(parts) == 1:
                return parts[0]
            body = "\n".join(f'<div style="margin-bottom:24px">{p}</div>' for p in parts)
            return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>body{{background:#111;color:#ccc;font-family:sans-serif;margin:20px}}</style>
</head><body>{body}</body></html>"""
        except ImportError:
            return "<html><body><p>plotly not installed — cannot render figures.</p></body></html>"

    def __repr__(self) -> str:
        has_html = "html" if self.html else ""
        has_png = "png" if self.png_bytes else ""
        n_figs = f"{len(self.figures)} figs" if self.figures else ""
        content = ", ".join(x for x in [has_html, has_png, n_figs] if x)
        return f"VisualResult({self.asset.intent}, {content}, {len(self.warnings)} warnings)"
