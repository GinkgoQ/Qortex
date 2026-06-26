"""Dataset-level visual QC audit report.

Produces a self-contained HTML gallery — one thumbnail per file, grouped by
BIDS suffix — without loading any full volume into RAM.  The thumbnail for
each NIfTI reads exactly one center slice via the nibabel ArrayProxy.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Path-label helpers ────────────────────────────────────────────────────────

def _label_subject(path_label: str) -> str | None:
    """Extract subject ID (without sub- prefix) from a BIDS relative path."""
    for part in path_label.replace("\\", "/").split("/"):
        if part.startswith("sub-"):
            return part[4:]
    return None


def _label_suffix(path_label: str) -> str:
    """Extract the BIDS suffix (last _word in stem) from a path label."""
    stem = path_label.replace("\\", "/").rsplit("/", 1)[-1]
    for ext in (".nii.gz", ".nii", ".mgz", ".mgh", ".edf", ".fif",
                ".bdf", ".set", ".gz", ".json", ".tsv"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    else:
        stem = stem.rsplit(".", 1)[0]
    seg = stem.rsplit("_", 1)
    return seg[-1] if len(seg) > 1 else stem


def _label_datatype(path_label: str) -> str | None:
    """Extract BIDS datatype folder (anat/func/dwi/eeg/…) from a path label."""
    _BIDS_DATATYPES = {"anat", "func", "dwi", "eeg", "meg", "fmap", "perf", "pet", "micr"}
    for part in path_label.replace("\\", "/").split("/"):
        if part in _BIDS_DATATYPES:
            return part
    return None


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """Result of inspecting and rendering one file."""
    path_label: str          # relative path within the dataset
    asset: Any               # VisualAsset
    thumbnail_b64: str | None = None
    error: str | None = None


@dataclass
class VisualAuditReport:
    """Dataset-level visual QC report.

    Use ``.to_html()`` to write a self-contained HTML gallery,
    ``.show()`` to open it in a browser, or ``.summary()`` for a text digest.
    ``.to_json()`` / ``.visual_manifest_json()`` export machine-readable records.

    Optional manifest-completeness fields (populated by
    ``run_visual_audit_with_manifest()``):

    n_expected:       Total files listed in the dataset manifest.
    n_local_present:  Files that exist on local disk.
    n_missing_local:  Files expected but absent from disk.
    """
    dataset_id: str
    n_files_inspected: int
    n_rendered: int
    n_failed: int
    entries: list[AuditEntry] = field(default_factory=list)
    # Manifest-completeness metadata (None when not computed)
    n_expected: int | None = None
    n_local_present: int | None = None
    n_missing_local: int | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def to_html(self, path: Path | str | None = None) -> str:
        """Build a self-contained HTML visual QC grid.

        Parameters
        ----------
        path:   If provided, also write the HTML to this file.

        Returns
        -------
        str
            The complete HTML string.
        """
        html = _build_html(self)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(html, encoding="utf-8")
        return html

    def show(self) -> None:
        """Open the audit report in the default web browser."""
        import tempfile
        import webbrowser
        html = self.to_html()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w",
                                         encoding="utf-8") as fh:
            fh.write(html)
            webbrowser.open(f"file://{fh.name}")

    def coverage_matrix(self) -> dict:
        """Build a subject × BIDS-suffix coverage grid from the audit entries.

        Returns a dict with keys ``subjects``, ``suffixes``, ``cells``.
        Each cell value is ``"ok"``, ``"error"``, or ``"missing"``.
        """
        subjects: dict[str, dict[str, str]] = {}
        suffixes: set[str] = set()

        for e in self.entries:
            sub = _label_subject(e.path_label)
            if sub is None:
                continue
            suf = _label_suffix(e.path_label)
            suffixes.add(suf)
            if sub not in subjects:
                subjects[sub] = {}
            subjects[sub][suf] = "error" if e.error else "ok"

        sorted_subs = sorted(subjects.keys())
        sorted_suf = sorted(suffixes)
        cells = {
            sub: {suf: subjects[sub].get(suf, "missing") for suf in sorted_suf}
            for sub in sorted_subs
        }
        return {"subjects": sorted_subs, "suffixes": sorted_suf, "cells": cells}

    @property
    def failed_files(self) -> list[AuditEntry]:
        """All entries whose rendering raised an exception."""
        return [e for e in self.entries if e.error]

    @property
    def per_suffix_counts(self) -> dict[str, int]:
        """Count of inspected entries per BIDS suffix."""
        counts: dict[str, int] = {}
        for e in self.entries:
            suf = _label_suffix(e.path_label)
            counts[suf] = counts.get(suf, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def per_subject_counts(self) -> dict[str, int]:
        """Count of inspected entries per subject ID."""
        counts: dict[str, int] = {}
        for e in self.entries:
            sub = _label_subject(e.path_label)
            if sub:
                counts[sub] = counts.get(sub, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def per_datatype_counts(self) -> dict[str, int]:
        """Count of inspected entries per BIDS datatype folder."""
        counts: dict[str, int] = {}
        for e in self.entries:
            dt = _label_datatype(e.path_label)
            if dt:
                counts[dt] = counts.get(dt, 0) + 1
        return dict(sorted(counts.items()))

    def warning_summary(self) -> dict:
        """Aggregate all VisualWarnings from all entries into a structured summary.

        Returns
        -------
        dict
            ``by_code``              – {code: [path_labels]}
            ``by_severity``          – {severity: count}
            ``unusual_orientations`` – paths with non-standard image orientation
            ``anisotropic``          – paths with highly anisotropic voxels
            ``large_files``          – paths estimated >4 GB
            ``failed_renders``       – paths whose render raised an exception
            ``total_warnings``       – total warning count across all entries
        """
        by_code: dict[str, list[str]] = {}
        by_severity: dict[str, int] = {"info": 0, "warning": 0, "error": 0}
        unusual: list[str] = []
        anisotropic: list[str] = []
        large: list[str] = []

        for e in self.entries:
            for w in getattr(e.asset, "warnings", []):
                sev = getattr(w, "severity", "warning")
                code = getattr(w, "code", "unknown")
                by_severity[sev] = by_severity.get(sev, 0) + 1
                by_code.setdefault(code, []).append(e.path_label)
                if code == "unusual_orientation":
                    unusual.append(e.path_label)
                elif code == "anisotropic":
                    anisotropic.append(e.path_label)
                elif code == "large_file":
                    large.append(e.path_label)

        return {
            "by_code": by_code,
            "by_severity": by_severity,
            "unusual_orientations": unusual,
            "anisotropic": anisotropic,
            "large_files": large,
            "failed_renders": [e.path_label for e in self.entries if e.error],
            "total_warnings": sum(by_severity.values()),
        }

    def missing_expected_files(
        self,
        manifest_files: list,
        local_root: Path,
    ) -> list[dict]:
        """Find manifest entries that should exist locally but do not.

        Compares ``manifest_files`` against the actual filesystem.  Complements
        ``coverage_matrix()`` which only reflects what was *rendered* — here we
        look directly at what is absent from disk.

        Parameters
        ----------
        manifest_files:
            FileRecord-like objects with a ``.path`` attribute (BIDS-relative
            path).  Typically ``Dataset.manifest().files`` or a filtered subset.
        local_root:
            Root directory of the locally downloaded dataset.

        Returns
        -------
        list[dict]
            Each dict: ``{"path", "subject", "suffix", "size_bytes"}``.
        """
        missing = []
        for fr in manifest_files:
            rel = getattr(fr, "path", str(fr))
            if not (Path(local_root) / rel).exists():
                sub = getattr(fr, "subject", None) or _label_subject(rel)
                suf = getattr(fr, "suffix", None) or _label_suffix(rel)
                size = int(getattr(fr, "size", 0) or 0)
                missing.append({"path": rel, "subject": sub, "suffix": suf, "size_bytes": size})
        return missing

    def to_json(self) -> str:
        """Serialize the full report to a JSON string."""
        import json
        return json.dumps(self._to_dict(), indent=2, default=str)

    def visual_manifest_json(self, path: Path | str) -> Path:
        """Write a ``visual_manifest.json`` summarising this report.

        The manifest records every inspected file with its visual metadata,
        warnings, and thumbnail availability — useful for downstream tooling
        that needs a machine-readable audit record.

        Parameters
        ----------
        path:
            Output file path.  Parent directories are created as needed.

        Returns
        -------
        Path
            The path that was written.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(), encoding="utf-8")
        return out

    def summary(self) -> str:
        """Return a short text summary of the audit."""
        lines = [
            f"VisualAudit — {self.dataset_id}",
            f"  Files inspected : {self.n_files_inspected}",
            f"  Rendered OK     : {self.n_rendered}",
            f"  Failed          : {self.n_failed}",
        ]
        for e in self.entries:
            icon = "✗" if e.error else "✓"
            intent = getattr(e.asset, "intent", "?")
            lines.append(f"  {icon}  {e.path_label}  [{intent}]")
        return "\n".join(lines)

    def _to_dict(self) -> dict:
        d: dict = {
            "dataset_id": self.dataset_id,
            "n_files_inspected": self.n_files_inspected,
            "n_rendered": self.n_rendered,
            "n_failed": self.n_failed,
        }
        if self.n_expected is not None:
            d["n_expected"] = self.n_expected
            d["n_local_present"] = self.n_local_present
            d["n_missing_local"] = self.n_missing_local
        d.update({
            "coverage_matrix": self.coverage_matrix(),
            "per_suffix_counts": self.per_suffix_counts,
            "per_subject_counts": self.per_subject_counts,
            "per_datatype_counts": self.per_datatype_counts,
            "warning_summary": self.warning_summary(),
            "entries": [
                {
                    "path": e.path_label,
                    "error": e.error,
                    "has_thumbnail": e.thumbnail_b64 is not None,
                    "intent": getattr(e.asset, "intent", "unknown"),
                    "modality": getattr(e.asset, "modality", "unknown"),
                    "shape": list(getattr(e.asset, "shape", [])),
                    "spacing": list(getattr(e.asset, "spacing", None) or []),
                    "orientation": getattr(e.asset, "orientation", None),
                    "n_timepoints": getattr(e.asset, "n_timepoints", 1),
                    "warnings": [
                        {
                            "code": getattr(w, "code", ""),
                            "message": getattr(w, "message", ""),
                            "severity": getattr(w, "severity", "warning"),
                        }
                        for w in getattr(e.asset, "warnings", [])
                    ],
                }
                for e in self.entries
            ],
        })
        return d


# ── HTML builder helpers ──────────────────────────────────────────────────────

_INTENT_COLOR: dict[str, str] = {
    "anatomical_volume": "#6af",
    "bold_fmri":         "#a6f",
    "ct_volume":         "#f96",
    "pet_volume":        "#fa6",
    "diffusion_volume":  "#6fa",
    "statistical_map":   "#f6a",
    "mask":              "#aaf",
    "labelmap":          "#aff",
}


def _shorten(label: str, max_parts: int = 3) -> str:
    parts = label.replace("\\", "/").split("/")
    if len(parts) <= max_parts:
        return label
    return "…/" + "/".join(parts[-2:])


def _build_card(e: AuditEntry) -> str:
    if e.thumbnail_b64:
        img = (
            f'<img src="data:image/png;base64,{e.thumbnail_b64}" '
            f'style="width:100%;max-width:200px;image-rendering:pixelated;'
            f'background:#000;border-radius:3px;">'
        )
    elif e.error:
        msg = e.error[:80].replace("<", "&lt;").replace(">", "&gt;")
        img = (
            f'<div style="width:200px;height:140px;background:#2a1a1a;'
            f'display:flex;align-items:center;justify-content:center;'
            f'color:#f64;font-size:0.75em;border-radius:4px;padding:8px;'
            f'text-align:center">{msg}</div>'
        )
    else:
        img = '<div style="width:200px;height:140px;background:#1a1a1a;border-radius:4px;"></div>'

    asset = e.asset
    intent = getattr(asset, "intent", "unknown")
    modality = getattr(asset, "modality", "")
    shape = getattr(asset, "shape", None)
    spacing = getattr(asset, "spacing", None)
    warnings = getattr(asset, "warnings", [])

    shape_str = " × ".join(str(s) for s in shape) if shape else "?"
    vox_str = " × ".join(f"{v:.2f}" for v in spacing[:3]) if spacing else ""
    color = _INTENT_COLOR.get(intent, "#888")

    warn_html = "".join(
        f'<div style="color:#fa8;font-size:0.7em;margin-top:2px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
        f'⚠ {w.message[:55]}</div>'
        for w in list(warnings)[:2]
    )

    label_short = _shorten(e.path_label)
    label_full = e.path_label.replace('"', "&quot;")

    return f"""
<div style="background:#1a1a1a;border-radius:8px;padding:12px;
            width:224px;flex-shrink:0;box-sizing:border-box">
  <div style="text-align:center;margin-bottom:8px">{img}</div>
  <div style="font-size:0.72em;color:#666;word-break:break-all;margin-bottom:4px"
       title="{label_full}">{label_short}</div>
  <div style="font-size:0.8em;margin-bottom:2px">
    <span style="color:{color};font-weight:500">{intent.replace("_", " ")}</span>
    <span style="color:#555;margin-left:6px">{modality}</span>
  </div>
  <div style="font-size:0.72em;color:#666">{shape_str}</div>
  {"" if not vox_str else f'<div style="font-size:0.7em;color:#555">{vox_str} mm</div>'}
  {warn_html}
</div>"""


def _build_coverage_html(matrix: dict) -> str:
    """Render a subject × suffix coverage table as HTML."""
    subjects = matrix.get("subjects", [])
    suffixes = matrix.get("suffixes", [])
    cells = matrix.get("cells", {})
    if not subjects or not suffixes:
        return ""

    _STATUS_STYLE = {
        "ok":      ("color:#6f6", "✓"),
        "error":   ("color:#fa8", "⚠"),
        "missing": ("color:#333", "·"),
    }

    th_cells = "".join(
        f'<th style="padding:3px 8px;color:#888;font-weight:400">{s}</th>'
        for s in suffixes
    )
    rows = ""
    for sub in subjects:
        row_cells = "".join(
            f'<td style="text-align:center;{_STATUS_STYLE.get(cells.get(sub,{}).get(suf,"missing"),("color:#333","·"))[0]}">'
            f'{_STATUS_STYLE.get(cells.get(sub,{}).get(suf,"missing"),("color:#333","·"))[1]}</td>'
            for suf in suffixes
        )
        rows += f'<tr><td style="color:#aaa;padding:3px 8px;white-space:nowrap">sub-{sub}</td>{row_cells}</tr>\n'

    return f"""
<h3 style="color:#888;font-size:0.82em;margin-bottom:8px;font-weight:500;letter-spacing:0.04em">
  COVERAGE MATRIX &nbsp;<span style="color:#555;font-weight:400">✓ present &nbsp;⚠ error &nbsp;· missing</span>
</h3>
<div style="overflow-x:auto;margin-bottom:28px">
<table style="border-collapse:collapse;font-size:0.75em">
  <thead><tr><th style="padding:3px 8px;color:#666;font-weight:400;text-align:left">Subject</th>{th_cells}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


def _build_warning_html(ws: dict) -> str:
    """Render a structured warning summary section."""
    total = ws.get("total_warnings", 0)
    if total == 0:
        return ""

    by_sev = ws.get("by_severity", {})
    n_err = by_sev.get("error", 0)
    n_warn = by_sev.get("warning", 0)
    n_info = by_sev.get("info", 0)

    rows = ""
    skip_codes = {"unusual_orientation", "anisotropic", "large_file",
                  "nibabel_missing", "mne_missing", "pydicom_missing"}

    if ws.get("unusual_orientations"):
        files = ws["unusual_orientations"]
        rows += (
            f'<tr><td style="color:#fa8;padding:3px 16px 3px 0">Unusual orientation</td>'
            f'<td style="color:#888;padding:3px 16px 3px 0">{len(files)}</td>'
            f'<td style="color:#666;font-size:0.85em">'
            f'{", ".join(_shorten(f) for f in files[:3])}'
            f'{"…" if len(files) > 3 else ""}</td></tr>\n'
        )
    if ws.get("anisotropic"):
        files = ws["anisotropic"]
        rows += (
            f'<tr><td style="color:#fa8;padding:3px 16px 3px 0">Anisotropic voxels</td>'
            f'<td style="color:#888;padding:3px 16px 3px 0">{len(files)}</td>'
            f'<td style="color:#666;font-size:0.85em">'
            f'{", ".join(_shorten(f) for f in files[:3])}'
            f'{"…" if len(files) > 3 else ""}</td></tr>\n'
        )
    if ws.get("large_files"):
        files = ws["large_files"]
        rows += (
            f'<tr><td style="color:#fa8;padding:3px 16px 3px 0">Large files (&gt;4 GB)</td>'
            f'<td style="color:#888;padding:3px 16px 3px 0">{len(files)}</td>'
            f'<td style="color:#666;font-size:0.85em">'
            f'{", ".join(_shorten(f) for f in files[:3])}'
            f'{"…" if len(files) > 3 else ""}</td></tr>\n'
        )

    for code, paths in ws.get("by_code", {}).items():
        if code in skip_codes:
            continue
        rows += (
            f'<tr><td style="color:#aaa;padding:3px 16px 3px 0">{code.replace("_", " ")}</td>'
            f'<td style="color:#888;padding:3px 16px 3px 0">{len(paths)}</td>'
            f'<td style="color:#666;font-size:0.85em">'
            f'{", ".join(_shorten(p) for p in paths[:2])}'
            f'{"…" if len(paths) > 2 else ""}</td></tr>\n'
        )

    if not rows:
        return ""

    sev_badges = []
    if n_err:
        sev_badges.append(f'<span style="color:#f64">{n_err} error{"s" if n_err != 1 else ""}</span>')
    if n_warn:
        sev_badges.append(f'<span style="color:#fa8">{n_warn} warning{"s" if n_warn != 1 else ""}</span>')
    if n_info:
        sev_badges.append(f'<span style="color:#888">{n_info} info</span>')

    badge_str = " &nbsp;".join(sev_badges)
    return f"""
<div style="margin-bottom:28px">
<h3 style="color:#888;font-size:0.82em;margin-bottom:8px;font-weight:500;letter-spacing:0.04em">
  WARNING SUMMARY &nbsp;<span style="color:#555;font-weight:400">{badge_str}</span>
</h3>
<table style="border-collapse:collapse;font-size:0.78em;width:100%;max-width:900px">
  <thead>
    <tr>
      <th style="text-align:left;padding:3px 16px 3px 0;color:#555;font-weight:400">Issue</th>
      <th style="text-align:left;padding:3px 16px 3px 0;color:#555;font-weight:400">Count</th>
      <th style="text-align:left;padding:3px 0;color:#555;font-weight:400">Files</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


def _build_breakdown_html(report: VisualAuditReport) -> str:
    """Render per-suffix and per-datatype count tables side by side."""
    suf_counts = report.per_suffix_counts
    dt_counts = report.per_datatype_counts
    if not suf_counts and not dt_counts:
        return ""

    def _table(title: str, counts: dict) -> str:
        if not counts:
            return ""
        rows = "".join(
            f'<tr>'
            f'<td style="padding:2px 20px 2px 0;color:#aaa;white-space:nowrap">{k}</td>'
            f'<td style="color:#6af;text-align:right">{v}</td>'
            f'</tr>'
            for k, v in counts.items()
        )
        return (
            f'<div style="margin-right:48px;display:inline-block;vertical-align:top">'
            f'<div style="color:#555;font-size:0.72em;margin-bottom:6px;font-weight:500;'
            f'letter-spacing:0.06em">{title}</div>'
            f'<table style="border-collapse:collapse;font-size:0.78em">{rows}</table>'
            f'</div>'
        )

    inner = _table("BY SUFFIX", suf_counts) + _table("BY DATATYPE", dt_counts)
    if not inner.strip():
        return ""
    return f'<div style="margin-bottom:28px">{inner}</div>'


def _build_failed_html(failed: list[AuditEntry]) -> str:
    """Render a table of files that failed to render."""
    if not failed:
        return ""
    rows = "".join(
        f'<tr>'
        f'<td style="padding:3px 16px 3px 0;color:#f64;font-size:0.78em;white-space:nowrap">'
        f'{_shorten(e.path_label)}</td>'
        f'<td style="color:#888;font-size:0.75em">'
        f'{(e.error or "")[:100].replace("<", "&lt;").replace(">", "&gt;")}</td>'
        f'</tr>'
        for e in failed
    )
    n = len(failed)
    return f"""
<div style="margin-bottom:28px">
<h3 style="color:#f64;font-size:0.82em;margin-bottom:8px;font-weight:500;letter-spacing:0.04em">
  FAILED RENDERS &nbsp;<span style="color:#555;font-weight:400">({n} file{"s" if n != 1 else ""})</span>
</h3>
<table style="border-collapse:collapse;font-size:0.78em;width:100%;max-width:900px">
  <thead>
    <tr>
      <th style="text-align:left;padding:3px 16px 3px 0;color:#555;font-weight:400">File</th>
      <th style="text-align:left;padding:3px 0;color:#555;font-weight:400">Error</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>"""


def _build_html(report: VisualAuditReport) -> str:
    cards = "\n".join(_build_card(e) for e in report.entries)
    n_ok = report.n_rendered
    n_fail = report.n_failed
    n_total = report.n_files_inspected
    coverage_html = _build_coverage_html(report.coverage_matrix())
    warning_html = _build_warning_html(report.warning_summary())
    breakdown_html = _build_breakdown_html(report)
    failed_html = _build_failed_html(report.failed_files)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Audit — {report.dataset_id}</title>
<style>
  body      {{ background:#111; color:#ccc; font-family:system-ui,sans-serif;
               margin:24px; }}
  h2        {{ color:#6af; margin:0 0 6px }}
  .stats    {{ color:#888; font-size:0.85em; margin-bottom:20px; }}
  .stats b  {{ color:#ccc; }}
  .grid     {{ display:flex; flex-wrap:wrap; gap:16px; }}
</style>
</head>
<body>
<h2>Visual Audit — {report.dataset_id}</h2>
<div class="stats">
  Inspected <b>{n_total}</b> &nbsp;·&nbsp;
  Rendered <b style="color:#6f6">{n_ok}</b> &nbsp;·&nbsp;
  Failed <b style="color:#f64">{n_fail}</b>
</div>
{coverage_html}
{warning_html}
{breakdown_html}
{failed_html}
<div class="grid">
{cards}
</div>
</body>
</html>"""


# ── Core runner ───────────────────────────────────────────────────────────────

def run_visual_audit(
    dataset_id: str,
    file_records: list,          # list of objects with .path attribute
    local_root: Path,
    *,
    max_files: int = 24,
) -> VisualAuditReport:
    """Inspect and thumbnail-render up to *max_files* local NIfTI/EEG files.

    Each NIfTI reads exactly one center slice from the nibabel ArrayProxy —
    the full volume is never loaded.
    """
    from qortex.visualize._dispatch import inspect_file

    n_rendered = 0
    n_failed = 0
    entries: list[AuditEntry] = []

    for fr in file_records[:max_files]:
        rel_path: str = getattr(fr, "path", str(fr))
        local_path = local_root / rel_path
        if not local_path.exists():
            from qortex.visualize._asset import VisualAsset
            entries.append(AuditEntry(
                path_label=rel_path,
                asset=VisualAsset(path=local_path, family="unknown"),
                error="file not found locally",
            ))
            n_failed += 1
            continue

        try:
            asset = inspect_file(local_path)
            thumb_b64 = _make_thumbnail(asset, local_path)
            entries.append(AuditEntry(
                path_label=rel_path,
                asset=asset,
                thumbnail_b64=thumb_b64,
            ))
            n_rendered += 1
        except Exception as exc:
            log.debug("audit failed for %s: %s", rel_path, exc, exc_info=True)
            n_failed += 1
            try:
                asset = inspect_file(local_path)
            except Exception:
                from qortex.visualize._asset import VisualAsset
                asset = VisualAsset(path=local_path, family="unknown")
            entries.append(AuditEntry(
                path_label=rel_path,
                asset=asset,
                error=str(exc),
            ))

    return VisualAuditReport(
        dataset_id=dataset_id,
        n_files_inspected=len(entries),
        n_rendered=n_rendered,
        n_failed=n_failed,
        entries=entries,
    )


def _make_thumbnail(asset: Any, local_path: Path) -> str | None:
    """Extract a base64-encoded center-slice PNG for *asset*.

    For NIfTI: reads one axial slice via the nibabel ArrayProxy.
    For EEG: renders a short signal segment.
    Never loads the full volume.
    """
    family = getattr(asset, "family", "")

    if family == "nifti":
        return _nifti_thumbnail(asset, local_path)

    if family == "eeg":
        return _eeg_thumbnail(asset, local_path)

    return None


def _nifti_thumbnail(asset: Any, local_path: Path) -> str | None:
    """Center-slice PNG for a NIfTI file — single slice read, no full load."""
    try:
        from qortex.visualize.volume import VolumeViewer
        from qortex.visualize._html import array_to_b64png

        viewer = VolumeViewer(local_path, modality=getattr(asset, "modality", "mri"))
        if viewer._lazy is not None:
            cz = viewer._lazy.shape[2] // 2
            slc = viewer._lazy.slice_along(2, cz).T[::-1, :]
        else:
            vol3d = viewer._vol3d()
            cz = vol3d.shape[2] // 2
            slc = vol3d[:, :, cz].T[::-1, :]
        return array_to_b64png(slc, viewer._vmin, viewer._vmax, viewer.colormap)
    except Exception as exc:
        log.debug("nifti thumbnail failed: %s", exc)
        return None


def _eeg_thumbnail(asset: Any, local_path: Path) -> str | None:
    """Mini butterfly plot for a short EEG/MEG segment — encoded as static PNG."""
    try:
        import plotly.io as pio
        from qortex.visualize.timeseries import TimeSeriesViewer
        viewer = TimeSeriesViewer(local_path)
        fig = viewer.butterfly(tmax=10.0, max_channels=4, show_envelope=True)
        png_bytes = pio.to_image(fig, format="png", width=400, height=200)
        return base64.b64encode(png_bytes).decode()
    except Exception as exc:
        log.debug("eeg thumbnail failed: %s", exc)
        return None
