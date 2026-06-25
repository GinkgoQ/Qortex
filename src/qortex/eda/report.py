"""EDA report builder — assembles all metrics + figures into an EDAReport."""

from __future__ import annotations

from pathlib import Path

from qortex.core.entities import EDAReport, Manifest
from qortex.eda.events import summarize_events
from qortex.eda.quality import compute_quality_metrics
from qortex.eda.summary import (
    build_dataset_summary,
    build_modality_summaries,
    coverage_matrix,
    file_table,
)


class EDAEngine:
    """Produces an EDAReport from a Manifest and optional local dataset path."""

    def __init__(self, manifest: Manifest) -> None:
        self._manifest = manifest

    def run(self, local_path: Path | None = None) -> EDAReport:
        manifest = self._manifest

        dataset_summary = build_dataset_summary(manifest)
        modality_summaries = build_modality_summaries(manifest)
        quality = compute_quality_metrics(manifest, local_path)
        event_summaries = summarize_events(manifest, local_path) if local_path else []
        if event_summaries:
            risks = list(quality.risks)
            missing_label_files = [
                event.path for event in event_summaries
                if event.label_column is None or event.n_missing_labels > 0
            ]
            imbalanced = [
                event for event in event_summaries
                if event.imbalance_ratio is not None and event.imbalance_ratio >= 5.0
            ]
            if missing_label_files:
                risks.append(
                    f"{len(missing_label_files)} local events file(s) have missing or undetected labels."
                )
            if imbalanced:
                risks.append(
                    f"{len(imbalanced)} local events file(s) show class imbalance ratio >= 5:1."
                )
            quality = quality.model_copy(update={"risks": risks})

        report = EDAReport(
            dataset_id=manifest.dataset_id,
            snapshot=manifest.snapshot,
            dataset_path=local_path,
            summary=dataset_summary,
            modality_summaries=modality_summaries,
            event_summaries=event_summaries,
            quality=quality,
        )

        # Attempt to render figures (requires plotly)
        try:
            report = self._attach_figures(report)
        except ImportError:
            pass

        # Render HTML
        report.html = self._render_html(report)

        return report

    # ── Figures ───────────────────────────────────────────────────────────

    def _attach_figures(self, report: EDAReport) -> EDAReport:
        from qortex.eda.plots import (
            modality_bar,
            size_distribution,
            subject_coverage_heatmap,
        )

        df_files = file_table(self._manifest)
        df_coverage = coverage_matrix(self._manifest)

        mod_counts = {
            mod: s.n_files
            for mod, s in report.modality_summaries.items()
        }

        figs: dict = {}
        try:
            figs["modality_bar"] = modality_bar(mod_counts)
        except Exception:
            pass
        try:
            figs["coverage_heatmap"] = subject_coverage_heatmap(df_coverage)
        except Exception:
            pass
        try:
            figs["size_dist"] = size_distribution(df_files)
        except Exception:
            pass

        report.figures = figs
        return report

    # ── HTML rendering ────────────────────────────────────────────────────

    def _render_html(self, report: EDAReport) -> str:
        s = report.summary
        q = report.quality

        # Figures as plotly JSON (embedded, no external CDN required)
        fig_html = ""
        for name, fig in report.figures.items():
            try:
                fig_html += (
                    f'<div class="figure">'
                    f'<div id="{name}"></div>'
                    f'<script>'
                    f'Plotly.newPlot("{name}", {fig.to_json()});'
                    f'</script>'
                    f'</div>\n'
                )
            except Exception:
                pass

        issues_html = "".join(f"<li>{i}</li>" for i in q.issues) or "<li>None</li>"
        risks_html = "".join(f"<li>{r}</li>" for r in q.risks) or "<li>None</li>"

        modality_rows = "".join(
            f"<tr><td>{m}</td><td>{ms.n_files}</td>"
            f"<td>{ms.n_subjects}</td>"
            f"<td>{ms.total_size / 1e6:.1f} MB</td></tr>"
            for m, ms in sorted(report.modality_summaries.items())
        )
        event_rows = "".join(
            f"<tr><td>{event.path}</td><td>{event.n_events}</td>"
            f"<td>{event.label_column or 'N/A'}</td>"
            f"<td>{event.n_classes}</td>"
            f"<td>{event.imbalance_ratio or 'N/A'}</td></tr>"
            for event in report.event_summaries
        )

        plotly_cdn = (
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            if report.figures else ""
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Qortex EDA — {report.dataset_id}</title>
  {plotly_cdn}
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ color: #2c3e50; }}
    h2 {{ color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #eee; }}
    th {{ background: #f5f5f5; font-weight: 600; }}
    .score-block {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }}
    .score-card {{ background: #f0f4ff; border-radius: 8px; padding: 16px 24px; min-width: 160px; }}
    .score-value {{ font-size: 2em; font-weight: 700; color: #3b5bdb; }}
    .score-label {{ color: #555; font-size: 0.9em; }}
    .figure {{ margin: 24px 0; }}
    ul {{ margin: 8px 0; padding-left: 20px; }}
    .warn {{ color: #c0392b; }}
    .ok {{ color: #27ae60; }}
  </style>
</head>
<body>
  <h1>Qortex EDA Report</h1>
  <p><strong>Dataset:</strong> {report.dataset_id} &nbsp; | &nbsp;
     <strong>Snapshot:</strong> {report.snapshot or "latest"} &nbsp; | &nbsp;
     <strong>DOI:</strong> {s.doi or "N/A"}</p>
  <p><strong>Generated:</strong> {report.generated_at.isoformat()}</p>

  <h2>Dataset Overview</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Files</td><td>{s.n_files}</td></tr>
    <tr><td>Subjects</td><td>{s.n_subjects}</td></tr>
    <tr><td>Sessions</td><td>{s.n_sessions}</td></tr>
    <tr><td>Tasks</td><td>{s.n_tasks}</td></tr>
    <tr><td>Total size</td><td>{s.total_size / 1e9:.2f} GB</td></tr>
    <tr><td>Modalities</td><td>{', '.join(s.modalities) or 'N/A'}</td></tr>
    <tr><td>Has derivatives</td><td>{'Yes' if s.has_derivatives else 'No'}</td></tr>
    <tr><td>Has events files</td><td>{'Yes' if s.has_events else 'No'}</td></tr>
  </table>

  <h2>Quality Scores</h2>
  <div class="score-block">
    <div class="score-card">
      <div class="score-value">{q.bids_score:.0f}</div>
      <div class="score-label">BIDS Score / 100</div>
    </div>
    <div class="score-card">
      <div class="score-value">{q.ml_readiness_score:.0f}</div>
      <div class="score-label">ML-Readiness / 100</div>
    </div>
    <div class="score-card">
      <div class="score-value">{q.loadability_score:.0f}</div>
      <div class="score-label">Loadability / 100</div>
    </div>
  </div>

  <h3>Issues</h3>
  <ul class="warn">{issues_html}</ul>

  <h3>ML Risks</h3>
  <ul class="warn">{risks_html}</ul>

  <h2>Modality Breakdown</h2>
  <table>
    <tr><th>Modality</th><th>Files</th><th>Subjects</th><th>Size</th></tr>
    {modality_rows}
  </table>

  <h2>Local Events and Labels</h2>
  <table>
    <tr><th>Events file</th><th>Rows</th><th>Label column</th><th>Classes</th><th>Imbalance ratio</th></tr>
    {event_rows if event_rows else '<tr><td colspan="5">No local events files were summarized.</td></tr>'}
  </table>

  <h2>Figures</h2>
  {fig_html if fig_html else '<p>Install plotly for interactive figures: <code>pip install plotly</code></p>'}

  <footer style="margin-top:40px;color:#999;font-size:0.85em;">
    Generated by Qortex by GinkgoQ
  </footer>
</body>
</html>"""
