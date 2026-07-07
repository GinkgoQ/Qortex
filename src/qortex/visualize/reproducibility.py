"""Reproducibility / provenance figure: pipeline DAG, environment, artifact hashes.

Every value shown is introspected at call time — real Python/package
versions, real file hashes for files that exist on disk — never a
placeholder or an assumed version string.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def environment_snapshot() -> dict[str, str]:
    """Real, introspected environment info — no assumed/fabricated versions."""
    from qortex.neuroai.artifact import _get_qortex_version

    info = {
        "qortex": _get_qortex_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": _package_version("numpy") or "not installed",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "cpu-only"
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except ImportError:
        info["torch"] = "not installed"
    return info


def artifact_hash_table(artifact_dir: str | Path) -> list[dict[str, str]]:
    """Real SHA-256 + size for every file under *artifact_dir*.

    Returns an empty list (not fabricated rows) when the directory has no
    files or does not exist.
    """
    root = Path(artifact_dir)
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "artifact": str(path.relative_to(root)),
                "size_kb": f"{path.stat().st_size / 1024:.1f}",
                "sha256": _sha256_file(path)[:16] + "…",
            })
    return rows


def reproducibility_figure(
    *,
    pipeline_steps: list[str],
    dataset_id: str = "",
    artifact_dir: str | Path | None = None,
    seed: int | None = None,
    title: str | None = None,
):
    """Pipeline DAG + environment panel + artifact hash table.

    Parameters
    ----------
    pipeline_steps:
        Ordered stage names actually run (e.g. ``["download", "convert",
        "eda", "export"]``). Rendered as boxes with arrows — no assumed
        pipeline, the caller states what ran.
    artifact_dir:
        Directory of real output files to hash. When omitted or empty,
        the hash panel says so rather than showing placeholder rows.
    seed:
        Random seed actually used, if any — shown verbatim, never inferred.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch
    except ImportError as exc:
        raise ImportError(
            "reproducibility_figure() requires matplotlib: pip install matplotlib"
        ) from exc

    from qortex.visualize.design import (
        INK, SUBINK, apply_theme, figure_title, section_title, style_table,
    )

    apply_theme()

    env = environment_snapshot()
    hash_rows = artifact_hash_table(artifact_dir) if artifact_dir else []

    fig = plt.figure(figsize=(13.0, 9.0))
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[0.55, 1.0, 1.6], hspace=0.75, wspace=0.3, figure=fig,
        top=0.87, bottom=0.05, left=0.055, right=0.97,
    )

    figure_title(fig, title or "Reproducibility & provenance", subtitle=dataset_id or None)

    # ── Pipeline DAG ─────────────────────────────────────────────────────────
    ax_dag = fig.add_subplot(gs[0, :])
    ax_dag.axis("off")
    section_title(ax_dag, "Pipeline", y=1.02)
    n = max(len(pipeline_steps), 1)
    slot = 0.92 / n
    box_w = slot * 0.75
    for i, step in enumerate(pipeline_steps):
        x = 0.03 + i * slot
        ax_dag.add_patch(FancyBboxPatch(
            (x, 0.28), box_w, 0.42, boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#eef0ff", edgecolor="#4f46e5", linewidth=1.3, transform=ax_dag.transAxes,
        ))
        ax_dag.text(x + box_w / 2, 0.49, step, transform=ax_dag.transAxes,
                    ha="center", va="center", fontsize=9.5, fontweight="bold", color=INK)
        if i < len(pipeline_steps) - 1:
            arrow_x0 = x + box_w
            arrow_x1 = x + slot
            ax_dag.annotate(
                "", xy=(arrow_x1, 0.49), xytext=(arrow_x0, 0.49),
                xycoords=ax_dag.transAxes, textcoords=ax_dag.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=SUBINK, linewidth=1.5, mutation_scale=14),
            )

    # ── Environment panel ────────────────────────────────────────────────────
    ax_env = fig.add_subplot(gs[1, 0])
    ax_env.axis("off")
    section_title(ax_env, "Environment", y=1.02)
    env_lines = "\n".join(f"{k:<10}: {v}" for k, v in env.items())
    ax_env.text(0.0, 0.88, env_lines, fontsize=9, family="monospace", color=INK,
                va="top", transform=ax_env.transAxes)

    # ── Reproducibility details ──────────────────────────────────────────────
    ax_repro = fig.add_subplot(gs[1, 1])
    ax_repro.axis("off")
    section_title(ax_repro, "Reproducibility details", y=1.02)
    details = {
        "dataset": dataset_id or "n/a",
        "seed": str(seed) if seed is not None else "not fixed",
        "hostname": platform.node(),
    }
    ax_repro.text(0.0, 0.88, "\n".join(f"{k:<10}: {v}" for k, v in details.items()),
                  fontsize=9, family="monospace", color=INK, va="top", transform=ax_repro.transAxes)

    # ── Artifact hash table ───────────────────────────────────────────────────
    ax_hash = fig.add_subplot(gs[2, :])
    ax_hash.axis("off")
    section_title(ax_hash, "Artifacts & hashes", y=1.01)
    if hash_rows:
        max_rows = 6
        rows = [[r["artifact"], r["size_kb"], r["sha256"]] for r in hash_rows[:max_rows]]
        tbl = ax_hash.table(cellText=rows, colLabels=["Artifact", "Size (KB)", "SHA-256"],
                             loc="upper center", cellLoc="left", bbox=[0.0, 0.05, 1.0, 0.85])
        style_table(tbl, fontsize=8)
        if len(hash_rows) > max_rows:
            ax_hash.text(0.0, -0.02, f"... +{len(hash_rows) - max_rows} more files",
                         fontsize=8, color=SUBINK, transform=ax_hash.transAxes)
    else:
        ax_hash.text(0.0, 0.5, "No artifact_dir given, or it contains no files.",
                     fontsize=9.5, color=SUBINK, transform=ax_hash.transAxes)

    return fig


__all__ = ["reproducibility_figure", "environment_snapshot", "artifact_hash_table"]
