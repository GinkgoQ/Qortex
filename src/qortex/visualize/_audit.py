"""Dataset-level visual QC audit report.

Produces a self-contained HTML gallery — one thumbnail per file, grouped by
BIDS suffix — without loading any full volume into RAM.  The thumbnail for
each NIfTI reads exactly one center slice via the nibabel ArrayProxy.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any

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
    missing_local_files: list[dict[str, Any]] = field(default_factory=list)

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

    def action_items(self) -> list[dict[str, Any]]:
        """Return prioritized next actions for a curator or ML user."""
        actions: list[dict[str, Any]] = []
        if self.n_missing_local:
            actions.append(
                {
                    "severity": "error",
                    "code": "missing_local_files",
                    "message": f"{self.n_missing_local} expected manifest file(s) are absent locally.",
                    "recommendation": "Download the missing files or narrow the visual audit filters before trusting coverage.",
                }
            )
        if self.n_failed:
            actions.append(
                {
                    "severity": "error",
                    "code": "render_failures",
                    "message": f"{self.n_failed} inspected file(s) failed visual rendering.",
                    "recommendation": "Inspect failed file paths and optional dependencies before conversion or model training.",
                }
            )
        ws = self.warning_summary()
        if ws.get("anisotropic"):
            actions.append(
                {
                    "severity": "warning",
                    "code": "anisotropic_voxels",
                    "message": f"{len(ws['anisotropic'])} image file(s) have anisotropic voxel spacing.",
                    "recommendation": "Review slice direction and resampling policy before quantitative modeling.",
                }
            )
        if ws.get("unusual_orientations"):
            actions.append(
                {
                    "severity": "warning",
                    "code": "unusual_orientation",
                    "message": f"{len(ws['unusual_orientations'])} image file(s) use non-standard orientation.",
                    "recommendation": "Confirm orientation handling before overlays, registration checks, or ML preprocessing.",
                }
            )
        if not actions:
            actions.append(
                {
                    "severity": "info",
                    "code": "no_blocking_visual_issues",
                    "message": "No blocking visual audit issues were detected in the rendered sample.",
                    "recommendation": "Proceed to modality-specific QC or artifact verification for the intended analysis.",
                }
            )
        return actions

    def to_markdown(self, path: Path | str | None = None) -> str:
        """Build a concise Markdown report for lab notes or pull requests."""
        lines = [
            f"# Visual Audit: {self.dataset_id}",
            "",
            f"- Files inspected: {self.n_files_inspected}",
            f"- Rendered OK: {self.n_rendered}",
            f"- Failed: {self.n_failed}",
        ]
        if self.n_expected is not None:
            lines.extend(
                [
                    f"- Expected from manifest: {self.n_expected}",
                    f"- Present locally: {self.n_local_present}",
                    f"- Missing locally: {self.n_missing_local}",
                ]
            )
        lines.extend(["", "## Action Items", ""])
        for item in self.action_items():
            lines.append(f"- **{item['severity']} / {item['code']}**: {item['message']} {item['recommendation']}")
        if self.per_suffix_counts:
            lines.extend(["", "## Suffix Counts", ""])
            for suffix, count in self.per_suffix_counts.items():
                lines.append(f"- `{suffix}`: {count}")
        if self.failed_files:
            lines.extend(["", "## Failed Files", ""])
            for entry in self.failed_files:
                lines.append(f"- `{entry.path_label}`: {entry.error}")
        if self.missing_local_files:
            lines.extend(["", "## Missing Local Files", ""])
            for item in self.missing_local_files[:50]:
                size = item.get("size_bytes", 0) or 0
                lines.append(f"- `{item.get('path')}` ({size} bytes)")
            if len(self.missing_local_files) > 50:
                lines.append(f"- ... {len(self.missing_local_files) - 50} more")
        text = "\n".join(lines) + "\n"
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
        return text

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
            d["missing_local_files"] = self.missing_local_files
        d.update({
            "coverage_matrix": self.coverage_matrix(),
            "per_suffix_counts": self.per_suffix_counts,
            "per_subject_counts": self.per_subject_counts,
            "per_datatype_counts": self.per_datatype_counts,
            "warning_summary": self.warning_summary(),
            "action_items": self.action_items(),
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


def _esc(value: Any) -> str:
    return escape(str(value), quote=True)


def _build_card(e: AuditEntry) -> str:
    asset = e.asset
    intent = getattr(asset, "intent", "unknown")
    modality = getattr(asset, "modality", "")
    shape = getattr(asset, "shape", None)
    spacing = getattr(asset, "spacing", None)
    warnings = getattr(asset, "warnings", [])
    status = "error" if e.error else "ok"
    label_short = _shorten(e.path_label)
    label_full = e.path_label

    if e.thumbnail_b64:
        alt = f"Thumbnail preview for {label_short}, {intent.replace('_', ' ')}"
        img = (
            f'<img src="data:image/png;base64,{e.thumbnail_b64}" '
            f'alt="{_esc(alt)}" loading="lazy">'
        )
    elif e.error:
        msg = _esc(e.error[:120])
        img = (
            f'<div class="thumb thumb-error" role="img" aria-label="Render failed: {msg}">'
            f'{msg}</div>'
        )
    else:
        img = '<div class="thumb thumb-empty" role="img" aria-label="No thumbnail available"></div>'

    shape_str = " × ".join(str(s) for s in shape) if shape else "?"
    vox_str = " × ".join(f"{v:.2f}" for v in spacing[:3]) if spacing else ""
    color = _INTENT_COLOR.get(intent, "#888")

    warn_html = "".join(
        f'<li>{_esc(getattr(w, "message", "")[:90])}</li>'
        for w in list(warnings)[:2]
    )
    warn_block = f'<ul class="warnings" aria-label="Warnings">{warn_html}</ul>' if warn_html else ""

    return f"""
<article class="audit-card" tabindex="0"
         data-status="{status}" data-intent="{_esc(intent)}" data-modality="{_esc(modality)}"
         data-path="{_esc(label_full).lower()}"
         aria-label="{_esc(label_full)}; status {status}; intent {intent.replace('_', ' ')}">
  <div class="thumb-wrap">{img}</div>
  <div class="path" title="{_esc(label_full)}">{_esc(label_short)}</div>
  <div class="intent">
    <span style="color:{color}">{_esc(intent.replace("_", " "))}</span>
    <span class="modality">{_esc(modality)}</span>
  </div>
  <div class="shape">{_esc(shape_str)}</div>
  {"" if not vox_str else f'<div class="spacing">{_esc(vox_str)} mm</div>'}
  {warn_block}
</article>"""


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
        f'<th scope="col">{_esc(s)}</th>'
        for s in suffixes
    )
    rows = ""
    for sub in subjects:
        row_cells = "".join(
            f'<td class="coverage-{_esc(cells.get(sub,{}).get(suf,"missing"))}" '
            f'aria-label="sub-{_esc(sub)} {_esc(suf)} {cells.get(sub,{}).get(suf,"missing")}">'
            f'{_STATUS_STYLE.get(cells.get(sub,{}).get(suf,"missing"),("color:#333","·"))[1]}</td>'
            for suf in suffixes
        )
        rows += f'<tr><th scope="row">sub-{_esc(sub)}</th>{row_cells}</tr>\n'

    return f"""
<section class="panel" aria-labelledby="coverage-title">
<h3 id="coverage-title">Coverage Matrix <span>present / error / missing by subject and suffix</span></h3>
<div class="table-scroll">
<table class="coverage-table">
  <caption>Subject by suffix visual coverage status</caption>
  <thead><tr><th scope="col">Subject</th>{th_cells}</tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
</section>"""


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
            f'<tr><td>Unusual orientation</td>'
            f'<td>{len(files)}</td>'
            f'<td>'
            f'{_esc(", ".join(_shorten(f) for f in files[:3]))}'
            f'{"…" if len(files) > 3 else ""}</td></tr>\n'
        )
    if ws.get("anisotropic"):
        files = ws["anisotropic"]
        rows += (
            f'<tr><td>Anisotropic voxels</td>'
            f'<td>{len(files)}</td>'
            f'<td>'
            f'{_esc(", ".join(_shorten(f) for f in files[:3]))}'
            f'{"…" if len(files) > 3 else ""}</td></tr>\n'
        )
    if ws.get("large_files"):
        files = ws["large_files"]
        rows += (
            f'<tr><td>Large files (&gt;4 GB)</td>'
            f'<td>{len(files)}</td>'
            f'<td>'
            f'{_esc(", ".join(_shorten(f) for f in files[:3]))}'
            f'{"…" if len(files) > 3 else ""}</td></tr>\n'
        )

    for code, paths in ws.get("by_code", {}).items():
        if code in skip_codes:
            continue
        rows += (
            f'<tr><td>{_esc(code.replace("_", " "))}</td>'
            f'<td>{len(paths)}</td>'
            f'<td>'
            f'{_esc(", ".join(_shorten(p) for p in paths[:2]))}'
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
<section class="panel" aria-labelledby="warnings-title">
<h3 id="warnings-title">Warning Summary <span>{badge_str}</span></h3>
<table>
  <caption>Visual audit warning categories and example files</caption>
  <thead>
    <tr>
      <th scope="col">Issue</th>
      <th scope="col">Count</th>
      <th scope="col">Files</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</section>"""


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
            f'<td>{_esc(k)}</td>'
            f'<td class="num">{v}</td>'
            f'</tr>'
            for k, v in counts.items()
        )
        return (
            f'<div class="mini-table">'
            f'<div class="mini-title">{_esc(title)}</div>'
            f'<table><tbody>{rows}</tbody></table>'
            f'</div>'
        )

    inner = _table("BY SUFFIX", suf_counts) + _table("BY DATATYPE", dt_counts)
    if not inner.strip():
        return ""
    return f'<section class="panel breakdown" aria-label="Audit breakdown counts">{inner}</section>'


def _build_failed_html(failed: list[AuditEntry]) -> str:
    """Render a table of files that failed to render."""
    if not failed:
        return ""
    rows = "".join(
        f'<tr>'
        f'<td>{_esc(_shorten(e.path_label))}</td>'
        f'<td>{_esc((e.error or "")[:140])}</td>'
        f'</tr>'
        for e in failed
    )
    n = len(failed)
    return f"""
<section class="panel failed" aria-labelledby="failed-title">
<h3 id="failed-title">Failed Renders <span>({n} file{"s" if n != 1 else ""})</span></h3>
<table>
  <caption>Files that could not be rendered</caption>
  <thead>
    <tr>
      <th scope="col">File</th>
      <th scope="col">Error</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</section>"""


def _build_action_html(report: "VisualAuditReport") -> str:
    items = report.action_items()
    rows = "".join(
        f'<li class="action-{_esc(item["severity"])}">'
        f'<strong>{_esc(item["code"].replace("_", " "))}</strong>'
        f'<span>{_esc(item["message"])} {_esc(item["recommendation"])}</span>'
        f'</li>'
        for item in items
    )
    return f"""
<section class="panel actions" aria-labelledby="actions-title">
  <h3 id="actions-title">Action Items <span>prioritized next steps</span></h3>
  <ol>{rows}</ol>
</section>"""


def _build_completeness_html(report: "VisualAuditReport") -> str:
    """Render expected-vs-local completeness stats when manifest data is present."""
    if report.n_expected is None:
        return ""
    n_exp = report.n_expected
    n_pres = report.n_local_present or 0
    n_miss = report.n_missing_local or 0
    pct = int(100 * n_pres / max(1, n_exp))
    bar_color = "#6f6" if pct >= 90 else "#fa8" if pct >= 50 else "#f64"
    missing_html = ""
    if report.missing_local_files:
        rows = "".join(
            f'<tr><td>{_esc(_shorten(str(item.get("path", "")), max_parts=4))}</td>'
            f'<td>{_esc(item.get("subject") or "")}</td>'
            f'<td>{_esc(item.get("suffix") or "")}</td>'
            f'<td class="num">{int(item.get("size_bytes", 0) or 0)}</td></tr>'
            for item in report.missing_local_files[:25]
        )
        more = ""
        if len(report.missing_local_files) > 25:
            more = f'<p class="missing-more">Showing 25 of {len(report.missing_local_files)} missing files.</p>'
        missing_html = f"""
  <details class="missing-files">
    <summary>Missing local file paths</summary>
    <table>
      <caption>Manifest files absent from the local dataset root</caption>
      <thead><tr><th scope="col">Path</th><th scope="col">Subject</th><th scope="col">Suffix</th><th scope="col">Bytes</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    {more}
  </details>"""
    return f"""
<section class="completeness" aria-labelledby="completeness-title">
  <h3 id="completeness-title">Manifest Completeness</h3>
  <div class="metric-row">
    <span>Expected <b>{n_exp}</b></span>
    <span>Local <b style="color:{bar_color}">{n_pres}</b> <em>({pct}%)</em></span>
    <span>Missing <b class="bad">{n_miss}</b></span>
    <span>Rendered <b class="good">{report.n_rendered}</b></span>
    <span>Failed <b class="bad">{report.n_failed}</b></span>
  </div>
  <div class="progress" role="progressbar" aria-label="Local manifest completeness"
       aria-valuemin="0" aria-valuemax="100" aria-valuenow="{pct}">
    <div style="background:{bar_color};width:{pct}%"></div>
  </div>
  {missing_html}
</section>"""


def _build_filter_html(report: "VisualAuditReport") -> str:
    intents = sorted({getattr(e.asset, "intent", "unknown") for e in report.entries})
    options = "".join(f'<option value="{_esc(intent)}">{_esc(intent.replace("_", " "))}</option>' for intent in intents)
    return f"""
<section class="filters" aria-label="Filter visual audit entries">
  <label>
    <span>Search files</span>
    <input id="audit-search" type="search" placeholder="subject, suffix, modality, path"
           aria-controls="audit-grid">
  </label>
  <label>
    <span>Status</span>
    <select id="audit-status" aria-controls="audit-grid">
      <option value="">All</option>
      <option value="ok">Rendered</option>
      <option value="error">Failed</option>
    </select>
  </label>
  <label>
    <span>Intent</span>
    <select id="audit-intent" aria-controls="audit-grid">
      <option value="">All</option>
      {options}
    </select>
  </label>
  <div id="audit-count" class="filter-count" aria-live="polite"></div>
</section>"""


def _build_html(report: "VisualAuditReport") -> str:
    cards = "\n".join(_build_card(e) for e in report.entries)
    n_ok = report.n_rendered
    n_fail = report.n_failed
    n_total = report.n_files_inspected
    coverage_html = _build_coverage_html(report.coverage_matrix())
    warning_html = _build_warning_html(report.warning_summary())
    breakdown_html = _build_breakdown_html(report)
    failed_html = _build_failed_html(report.failed_files)
    completeness_html = _build_completeness_html(report)
    action_html = _build_action_html(report)
    filter_html = _build_filter_html(report)

    # Simple stats line when no manifest data is available
    if report.n_expected is None:
        stats_line = (
            f'  Inspected <b>{n_total}</b> &nbsp;·&nbsp;'
            f'  Rendered <b style="color:#6f6">{n_ok}</b> &nbsp;·&nbsp;'
            f'  Failed <b style="color:#f64">{n_fail}</b>'
        )
        stats_block = f'<div class="stats">{stats_line}</div>'
    else:
        stats_block = ""  # completeness_html already has all counts

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Audit — {report.dataset_id}</title>
<style>
  :root {{ color-scheme: dark; --bg:#111; --panel:#1a1a1a; --text:#d8d8d8; --muted:#8d8d8d; --line:#303030; --blue:#7db7ff; --good:#8af58a; --bad:#ff7a7a; --warn:#ffbf80; }}
  * {{ box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,sans-serif; margin:24px; line-height:1.45; }}
  a.skip {{ position:absolute; left:-999px; top:8px; background:#fff; color:#000; padding:8px 10px; border-radius:4px; }}
  a.skip:focus {{ left:8px; z-index:10; }}
  h1 {{ color:var(--blue); margin:0 0 6px; font-size:1.35rem; letter-spacing:0; }}
  h3 {{ color:var(--muted); font-size:0.84rem; margin:0 0 10px; font-weight:650; letter-spacing:0; }}
  h3 span {{ color:#666; font-weight:400; margin-left:6px; }}
  caption {{ position:absolute; left:-10000px; }}
  table {{ border-collapse:collapse; font-size:0.8rem; width:100%; max-width:920px; }}
  th, td {{ padding:4px 12px 4px 0; text-align:left; border-bottom:1px solid #222; color:#aaa; }}
  th {{ color:#c6c6c6; font-weight:550; }}
  .stats {{ color:var(--muted); font-size:0.88rem; margin-bottom:20px; }}
  .stats b {{ color:var(--text); }}
  .panel, .completeness {{ margin:0 0 24px; background:var(--panel); border:1px solid #252525; border-radius:8px; padding:14px 16px; max-width:980px; }}
  .completeness .metric-row, .filters {{ display:flex; gap:18px; flex-wrap:wrap; align-items:end; color:var(--muted); font-size:0.9rem; }}
  .metric-row b {{ color:var(--text); }}
  .metric-row em {{ color:#666; font-style:normal; }}
  .good {{ color:var(--good) !important; }} .bad {{ color:var(--bad) !important; }}
  .progress {{ margin-top:10px; height:6px; background:#242424; border-radius:3px; max-width:420px; overflow:hidden; }}
  .progress div {{ height:6px; border-radius:3px; }}
  details.missing-files {{ margin-top:12px; color:#aaa; }}
  details.missing-files summary {{ cursor:pointer; color:var(--warn); font-size:0.86rem; }}
  .missing-more {{ margin:8px 0 0; color:#777; font-size:0.8rem; }}
  .filters {{ margin:0 0 20px; align-items:center; }}
  .filters label {{ display:flex; flex-direction:column; gap:4px; color:#777; font-size:0.78rem; }}
  .filters input, .filters select {{ background:#181818; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:8px 10px; min-width:160px; }}
  .filters input {{ min-width:280px; }}
  .filter-count {{ color:#777; font-size:0.82rem; padding-bottom:8px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:16px; align-items:stretch; }}
  .audit-card {{ background:var(--panel); border:1px solid #252525; border-radius:8px; padding:12px; width:224px; flex-shrink:0; }}
  .audit-card:focus {{ outline:2px solid var(--blue); outline-offset:3px; }}
  .audit-card img, .thumb {{ width:100%; max-width:200px; height:140px; object-fit:contain; background:#000; border-radius:4px; image-rendering:pixelated; }}
  .thumb-wrap {{ text-align:center; margin-bottom:8px; }}
  .thumb {{ display:flex; align-items:center; justify-content:center; padding:8px; font-size:0.76rem; text-align:center; }}
  .thumb-error {{ background:#2a1717; color:var(--bad); }}
  .thumb-empty {{ background:#1f1f1f; }}
  .path {{ font-size:0.74rem; color:#7a7a7a; word-break:break-all; margin-bottom:4px; }}
  .intent {{ font-size:0.82rem; margin-bottom:2px; font-weight:550; }}
  .modality {{ color:#777; margin-left:6px; font-weight:400; }}
  .shape {{ font-size:0.74rem; color:#777; }}
  .spacing {{ font-size:0.72rem; color:#666; }}
  .warnings {{ margin:6px 0 0; padding-left:16px; color:var(--warn); font-size:0.72rem; }}
  .coverage-ok {{ color:var(--good); }} .coverage-error {{ color:var(--warn); }} .coverage-missing {{ color:#555; }}
  .table-scroll {{ overflow-x:auto; }}
  .mini-table {{ margin-right:48px; display:inline-block; vertical-align:top; }}
  .mini-title {{ color:#777; font-size:0.74rem; margin-bottom:6px; font-weight:650; }}
  .num {{ color:var(--blue); text-align:right; }}
  .actions ol {{ margin:0; padding-left:22px; }}
  .actions li {{ margin:6px 0; }}
  .actions strong {{ display:inline-block; min-width:170px; color:#ddd; }}
  .action-error strong {{ color:var(--bad); }} .action-warning strong {{ color:var(--warn); }} .action-info strong {{ color:var(--blue); }}
  @media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior:auto !important; }} }}
</style>
</head>
<body>
<a class="skip" href="#audit-grid">Skip to visual entries</a>
<header>
<h1>Visual Audit — {_esc(report.dataset_id)}</h1>
{stats_block}
</header>
{completeness_html}
{action_html}
{coverage_html}
{warning_html}
{breakdown_html}
{failed_html}
{filter_html}
<main id="audit-grid" class="grid" tabindex="-1" aria-label="Visual audit entries">
{cards}
</main>
<script>
const cards = Array.from(document.querySelectorAll('.audit-card'));
const q = document.getElementById('audit-search');
const status = document.getElementById('audit-status');
const intent = document.getElementById('audit-intent');
const count = document.getElementById('audit-count');
function applyFilters() {{
  const text = (q.value || '').trim().toLowerCase();
  const st = status.value;
  const it = intent.value;
  let shown = 0;
  for (const card of cards) {{
    const matchesText = !text || card.dataset.path.includes(text) || card.dataset.intent.includes(text) || card.dataset.modality.includes(text);
    const matchesStatus = !st || card.dataset.status === st;
    const matchesIntent = !it || card.dataset.intent === it;
    const visible = matchesText && matchesStatus && matchesIntent;
    card.hidden = !visible;
    if (visible) shown += 1;
  }}
  count.textContent = `${{shown}} of ${{cards.length}} entries shown`;
}}
[q, status, intent].forEach(el => el.addEventListener('input', applyFilters));
applyFilters();
</script>
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


# ── File-selection helper ─────────────────────────────────────────────────────

def select_visual_files(
    manifest_files: list,
    *,
    subjects: list[str] | None = None,
    suffixes: list[str] | None = None,
    datatypes: list[str] | None = None,
    max_size_mb: float | None = None,
    n_per_suffix: int | None = None,
) -> list:
    """Filter a list of FileRecord-like objects for visual audit selection.

    Applies structured field filters (subject, suffix, datatype, size) rather
    than path-string matching so the logic is consistent between
    ``Dataset.visualize()`` and ``visualize-openneuro``.

    Parameters
    ----------
    manifest_files:
        List of objects with at minimum a ``.path`` attribute and optionally
        ``.subject``, ``.suffix``, ``.datatype``, ``.size`` attributes.
        Bare strings are also accepted (path-label only).
    subjects:
        Keep only files belonging to these subject IDs (without 'sub-' prefix).
        If None, all subjects are kept.
    suffixes:
        Keep only files with these BIDS suffixes (e.g. ``["T1w", "bold"]``).
        If None, all suffixes are kept.
    datatypes:
        Keep only files inside these BIDS datatype folders
        (e.g. ``["anat", "func"]``).  If None, all datatypes are kept.
    max_size_mb:
        Drop files whose ``.size`` attribute (bytes) exceeds this limit.
        Files without a size attribute pass this filter.
    n_per_suffix:
        Retain at most this many files per unique suffix.  Applied after all
        other filters so the cap never distorts subject coverage.

    Returns
    -------
    list
        Filtered subset of *manifest_files* preserving original order.
    """
    def _get(fr, attr: str, fallback_fn=None):
        val = getattr(fr, attr, None)
        if val is not None:
            return val
        if fallback_fn is not None:
            return fallback_fn(getattr(fr, "path", str(fr)))
        return None

    out = []
    for fr in manifest_files:
        path_str = getattr(fr, "path", str(fr))

        # Subject filter
        if subjects is not None:
            sub = _get(fr, "subject") or _label_subject(path_str)
            if sub not in subjects:
                continue

        # Suffix filter
        if suffixes is not None:
            suf = _get(fr, "suffix") or _label_suffix(path_str)
            if suf not in suffixes:
                continue

        # Datatype filter
        if datatypes is not None:
            dt = _get(fr, "datatype") or _label_datatype(path_str)
            if dt not in datatypes:
                continue

        # Size filter
        if max_size_mb is not None:
            size_bytes = int(_get(fr, "size") or 0)
            if size_bytes > max_size_mb * 1024 * 1024:
                continue

        out.append(fr)

    # n_per_suffix cap — applied after all other filters
    if n_per_suffix is not None and n_per_suffix > 0:
        suffix_counts: dict[str, int] = {}
        capped = []
        for fr in out:
            path_str = getattr(fr, "path", str(fr))
            suf = getattr(fr, "suffix", None) or _label_suffix(path_str)
            if suffix_counts.get(suf, 0) < n_per_suffix:
                capped.append(fr)
                suffix_counts[suf] = suffix_counts.get(suf, 0) + 1
        return capped

    return out


def select_visual_file_records(
    manifest: Any,
    *,
    subjects: list[str] | None = None,
    suffixes: list[str] | None = None,
    datatypes: list[str] | None = None,
    max_size_mb: float | None = None,
    n_per_suffix: int | None = None,
) -> list:
    """Select manifest file records through the shared structured audit filter.

    Accepts either a manifest-like object with a ``.files`` attribute or a raw
    file-record iterable.  This is the canonical helper for Dataset-level visual
    selection, OpenNeuro preview, and CLI audit commands.
    """
    manifest_files = getattr(manifest, "files", manifest)
    return select_visual_files(
        list(manifest_files),
        subjects=subjects,
        suffixes=suffixes,
        datatypes=datatypes,
        max_size_mb=max_size_mb,
        n_per_suffix=n_per_suffix,
    )


# ── Manifest-aware audit runner ───────────────────────────────────────────────

def run_visual_audit_with_manifest(
    dataset_id: str,
    manifest_files: list,
    local_root: Path,
    *,
    max_files: int = 24,
    subjects: list[str] | None = None,
    suffixes: list[str] | None = None,
    datatypes: list[str] | None = None,
    max_size_mb: float | None = None,
    n_per_suffix: int | None = None,
) -> VisualAuditReport:
    """Run a visual audit and automatically compute manifest-completeness stats.

    Unlike ``run_visual_audit()`` which only inspects already-present files,
    this function also counts how many manifest entries exist locally vs.
    are missing — giving the curator a direct answer to "how complete is my
    local copy?".

    Completeness fields populated on the returned report:

    * ``n_expected``      — total entries in *manifest_files* after filtering
    * ``n_local_present`` — entries that exist on disk
    * ``n_missing_local`` — entries absent from disk

    Parameters
    ----------
    dataset_id:
        Human-readable identifier shown in the HTML header and JSON.
    manifest_files:
        All FileRecord-like objects from the dataset manifest (unfiltered).
        The helper applies ``select_visual_files`` internally.
    local_root:
        Root of the locally downloaded dataset.
    max_files:
        Cap on the number of files to render thumbnails for.
    subjects, suffixes, datatypes, max_size_mb, n_per_suffix:
        Forwarded to ``select_visual_files()`` for pre-filtering.
    """
    # Pre-filter with the shared helper
    filtered = select_visual_file_records(
        manifest_files,
        subjects=subjects,
        suffixes=suffixes,
        datatypes=datatypes,
        max_size_mb=max_size_mb,
        n_per_suffix=n_per_suffix,
    )

    # Completeness accounting (fast: just stat() each path)
    local_root = Path(local_root)
    n_expected = len(filtered)
    n_local_present = 0
    n_missing_local = 0
    local_files: list = []
    missing_local_files: list[dict[str, Any]] = []
    for fr in filtered:
        rel = getattr(fr, "path", str(fr))
        if (local_root / rel).exists():
            n_local_present += 1
            local_files.append(fr)
        else:
            n_missing_local += 1
            missing_local_files.append({
                "path": rel,
                "subject": getattr(fr, "subject", None) or _label_subject(rel),
                "suffix": getattr(fr, "suffix", None) or _label_suffix(rel),
                "datatype": getattr(fr, "datatype", None) or _label_datatype(rel),
                "size_bytes": int(getattr(fr, "size", 0) or 0),
            })

    # Run the core audit on locally-present files only
    report = run_visual_audit(dataset_id, local_files, local_root, max_files=max_files)

    # Attach completeness metadata
    report.n_expected = n_expected
    report.n_local_present = n_local_present
    report.n_missing_local = n_missing_local
    report.missing_local_files = missing_local_files

    return report
