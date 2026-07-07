"""NeuroAI showcase artifact rendering.

This module turns a completed segmentation run into visual evidence boards:
source image, prediction mask, optional ground truth, overlay, contour/error
view, class metrics, mask area profile, and a small artifact manifest.

It deliberately does not run clinical inference by itself.  Model execution
belongs to the NeuroAI runtime/adapters; this layer renders the outputs and
validates that the result is inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


QORTEX_COLORS: tuple[str, ...] = (
    "#79863c",
    "#6574a6",
    "#c07a5a",
    "#b94949",
    "#4f8f8b",
    "#8d6a9f",
    "#d49a3a",
)


@dataclass(frozen=True)
class ShowcaseInput:
    """Inputs required to render a segmentation evidence board."""

    image: Any
    prediction_mask: Any
    output_dir: str | Path
    case_id: str
    model_id: str
    source_id: str
    class_labels: dict[int, str] = field(default_factory=lambda: {0: "background", 1: "foreground"})
    truth_mask: Any | None = None
    affine: Any | None = None
    voxel_sizes: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    slice_index: int | None = None


@dataclass(frozen=True)
class ShowcaseArtifacts:
    """Files written by :func:`render_segmentation_showcase`."""

    output_dir: Path
    board: Path
    overlay: Path
    mask: Path
    metrics: Path
    manifest: Path
    area_plot: Path
    source_slice: Path
    error_map: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "output_dir": str(self.output_dir),
            "board": str(self.board),
            "overlay": str(self.overlay),
            "mask": str(self.mask),
            "metrics": str(self.metrics),
            "manifest": str(self.manifest),
            "area_plot": str(self.area_plot),
            "source_slice": str(self.source_slice),
            "error_map": str(self.error_map) if self.error_map else None,
        }


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    class_name: str
    confidence: float


@dataclass(frozen=True)
class DetectionShowcaseInput:
    """Inputs required to render a detection evidence board."""

    image: Any                       # [H, W] or [H, W, 3]
    detections: list[Detection]
    output_dir: str | Path
    case_id: str
    model_id: str
    source_id: str
    threshold: float = 0.5
    nms_iou: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionShowcaseArtifacts:
    """Files written by :func:`render_detection_showcase`."""

    output_dir: Path
    board: Path
    detections_json: Path
    manifest: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "board": str(self.board),
            "detections_json": str(self.detections_json),
            "manifest": str(self.manifest),
        }


def render_detection_showcase(payload: DetectionShowcaseInput) -> DetectionShowcaseArtifacts:
    """Render detection evidence: annotated image + confidence table + legend + caption.

    ``neuroai.outputs.overlay_out.OverlayOutputAdapter`` annotates individual
    streamed frames one at a time (no legend, no threshold/NMS caption, no
    composition). This renders one composed evidence board — the same design
    as :func:`render_segmentation_showcase` — for archiving or sharing a
    single detection run.
    """
    image = _as_image(payload.image)
    out_dir = Path(payload.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    board_path = out_dir / "detection-board.png"
    detections_path = out_dir / "detections.json"
    manifest_path = out_dir / "showcase-manifest.json"

    class_names = sorted({d.class_name for d in payload.detections})
    color_map = {name: QORTEX_COLORS[i % len(QORTEX_COLORS)] for i, name in enumerate(class_names)}

    _save_detection_board(image, payload, color_map, board_path)

    ranked = sorted(payload.detections, key=lambda d: d.confidence, reverse=True)
    detections_path.write_text(
        json.dumps(
            [{"bbox": list(d.bbox), "class_name": d.class_name, "confidence": d.confidence} for d in ranked],
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": payload.case_id,
        "model_id": payload.model_id,
        "source_id": payload.source_id,
        "threshold": payload.threshold,
        "nms_iou": payload.nms_iou,
        "n_detections": len(payload.detections),
        "classes": class_names,
        "metadata": _json_safe(payload.metadata),
        "artifacts": {"board": board_path.name, "detections": detections_path.name},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return DetectionShowcaseArtifacts(
        output_dir=out_dir, board=board_path,
        detections_json=detections_path, manifest=manifest_path,
    )


@dataclass(frozen=True)
class ModelZooArtifacts:
    """Files written by :func:`render_model_zoo_showcase`."""

    output_dir: Path
    board: Path
    manifest: Path

    def to_dict(self) -> dict[str, str]:
        return {"output_dir": str(self.output_dir), "board": str(self.board), "manifest": str(self.manifest)}


def render_model_zoo_showcase(
    output_dir: str | Path,
    *,
    example_model: str = "resnet18",
    pretrained: bool = False,
    device: str = "cpu",
) -> ModelZooArtifacts:
    """Render a model-zoo reference board: backend availability, curated
    example models, and one real inference example.

    ``pretrained=True`` downloads real weights for *only* ``example_model*``
    (never implicit, never for the other adapters) so the inference panel
    shows genuine ImageNet-trained probabilities rather than an untrained
    architecture. With the default ``pretrained=False`` the forward pass is
    still real — a real architecture, real matrix multiplications — but the
    weights are randomly initialised, so the panel is labelled as an
    architecture demo rather than presented as a real classification.
    """
    from qortex.neuroai.models import list_models, get_model_card
    from qortex.neuroai.models._registry import make_model_adapter
    from qortex.neuroai.models.zoo import backend_availability
    from qortex.neuroai.spec import ModelSpec, RuntimeSpec

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    board_path = out_dir / "model-zoo-board.png"
    manifest_path = out_dir / "model-zoo-manifest.json"

    backends = backend_availability()
    models = list_models()

    top5: list[tuple[str, float]] = []
    entry = next((m for m in models if m.model_id == example_model or example_model in m.aliases), None)
    if entry is not None and entry.output_contract is not None and entry.output_contract.output_type in (
        "image_classification", "classification",
    ):
        spec = ModelSpec(provider=entry.provider, id=entry.model_id, task="classification",
                          extra={"pretrained": pretrained})
        adapter = make_model_adapter(spec)
        adapter.load(RuntimeSpec(device=device))
        rng = np.random.default_rng(0)
        shape = entry.input_contract.spatial_shape or (224, 224)
        if entry.provider == "keras":
            image = rng.random((*shape, 3)).astype("float32")
        else:
            image = rng.random((3, *shape)).astype("float32")
        out = adapter.predict(image)
        top5 = sorted(out.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:5]

    _save_model_zoo_board(board_path, backends, models, example_model, top5, pretrained)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backends": backends,
        "models": [
            {"model_id": m.model_id, "provider": m.provider,
             "task": m.output_contract.output_type if m.output_contract else None,
             "estimated_memory_mb": m.estimated_memory_mb}
            for m in models
        ],
        "example_model": example_model,
        "pretrained": pretrained,
        "top5": top5,
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")

    return ModelZooArtifacts(output_dir=out_dir, board=board_path, manifest=manifest_path)


_TASK_COLORS: dict[str, str] = {
    "image_classification": "#4f46e5",
    "classification": "#4f46e5",
    "eeg_classification": "#0891b2",
    "audio_transcription": "#7c3aed",
    "segmentation": "#059669",
    "detection": "#d97706",
    "embedding": "#db2777",
}


def _save_model_zoo_board(
    board_path: Path,
    backends: dict[str, bool],
    models: list[Any],
    example_model: str,
    top5: list[tuple[str, float]],
    pretrained: bool,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import FancyBboxPatch

    from qortex.visualize.design import (
        BORDER, CARD_BG, INK, SUBINK, STATUS,
        apply_theme, figure_title, section_title, style_table,
    )

    apply_theme()
    fig = plt.figure(figsize=(13.0, 10.2))
    gs = gridspec.GridSpec(
        3, 1, figure=fig, height_ratios=[0.34, 1.55, 1.15],
        hspace=0.62, top=0.86, bottom=0.06, left=0.055, right=0.97,
    )

    n_available = sum(backends.values())
    figure_title(
        fig, "NeuroAI model registry",
        subtitle=f"{n_available}/{len(backends)} backends importable in this environment  ·  "
                 f"{len(models)} curated example models",
    )

    # ── Backend availability, as pill badges (not plain checkmark text) ──────
    ax_backends = fig.add_subplot(gs[0])
    ax_backends.axis("off")
    section_title(ax_backends, "Available backends", y=1.15)
    ordered = sorted(backends.items(), key=lambda kv: (not kv[1], kv[0]))
    n_cols = len(ordered)
    slot = 1.0 / n_cols
    for i, (name, ok) in enumerate(ordered):
        color = STATUS["success"] if ok else SUBINK
        mark = "●" if ok else "○"
        cx = i * slot + slot * 0.06
        ax_backends.text(
            cx, 0.42, f"{mark}  {name}", fontsize=10.5, fontweight="bold" if ok else "normal",
            color=color if ok else "#9ca3af", transform=ax_backends.transAxes, va="center",
            bbox=dict(boxstyle="round,pad=0.42", facecolor=color, alpha=0.10 if ok else 0.05,
                      edgecolor=color, linewidth=1.0 if ok else 0.8),
        )

    # ── Example models table ──────────────────────────────────────────────────
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")
    section_title(ax_table, "Example models", y=1.05)
    rows = [
        [
            m.model_id,
            (m.output_contract.output_type if m.output_contract else "?"),
            m.provider,
            f"{m.estimated_memory_mb:,.0f} MB" if m.estimated_memory_mb else "n/a",
        ]
        for m in models
    ]
    tbl = ax_table.table(cellText=rows, colLabels=["Model", "Task", "Backend", "Est. memory"],
                          loc="upper center", cellLoc="left", bbox=[0.0, 0.0, 1.0, 0.98],
                          colWidths=[0.42, 0.24, 0.18, 0.16])
    style_table(tbl)
    task_col = 1
    for (r, c), cell in tbl.get_celld().items():
        if r > 0 and c == task_col:
            task = rows[r - 1][task_col]
            cell.set_text_props(color=_TASK_COLORS.get(task, INK), fontweight="bold")

    # ── Inference example ─────────────────────────────────────────────────────
    ax_inf = fig.add_subplot(gs[2])
    if top5:
        status = "ImageNet-pretrained weights" if pretrained else "untrained weights — architecture demo only"
        status_color = STATUS["success"] if pretrained else STATUS["warning"]
        names, vals = zip(*top5)
        y_pos = np.arange(len(names))
        max_val = max(vals) or 1.0
        shades = np.linspace(1.0, 0.55, len(names))
        colors = [(0.310, 0.275, 0.898, s) for s in shades]  # indigo, fading alpha by rank
        bars = ax_inf.barh(y_pos, vals, color=colors, edgecolor="none", height=0.62)
        for bar, val in zip(bars, vals):
            ax_inf.text(bar.get_width() + max_val * 0.015, bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", fontsize=8.5, color=INK)
        ax_inf.set_yticks(y_pos)
        ax_inf.set_yticklabels(names, fontsize=9)
        ax_inf.invert_yaxis()
        ax_inf.set_xlabel("probability", fontsize=9, color=SUBINK)
        ax_inf.set_xlim(0, max_val * 1.18)
        ax_inf.grid(axis="x", color="#eef1f4", linewidth=0.9)
        ax_inf.grid(axis="y", visible=False)
        section_title(ax_inf, f"Inference example — {example_model}", y=1.14)
        ax_inf.text(1.0, 1.14, status, transform=ax_inf.transAxes, ha="right", va="bottom",
                    fontsize=8.5, fontweight="bold", color=status_color)
    else:
        ax_inf.axis("off")
        ax_inf.text(0.0, 0.5, f"No classification example available for {example_model!r}.",
                     fontsize=9.5, color=SUBINK, transform=ax_inf.transAxes)

    fig.savefig(board_path, dpi=200, facecolor="white")
    plt.close(fig)


def render_segmentation_showcase(payload: ShowcaseInput) -> ShowcaseArtifacts:
    """Render segmentation evidence from real model outputs.

    Parameters
    ----------
    payload:
        Source image, predicted mask, optional ground-truth mask, and run
        metadata.

    Returns
    -------
    ShowcaseArtifacts
        Paths to all generated files.
    """

    image = _as_volume(payload.image, name="image")
    pred = _as_mask(payload.prediction_mask, name="prediction_mask")
    truth = None if payload.truth_mask is None else _as_mask(payload.truth_mask, name="truth_mask")
    _validate_geometry(image, pred, truth)

    out_dir = Path(payload.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = _select_slice(image, pred, truth, payload.slice_index)
    image_slice = _slice(image, z)
    pred_slice = _slice(pred, z)
    truth_slice = None if truth is None else _slice(truth, z)

    source_slice_path = out_dir / "source-slice.png"
    mask_path = out_dir / "prediction-mask.png"
    overlay_path = out_dir / "prediction-overlay.png"
    area_plot_path = out_dir / "mask-area-profile.png"
    board_path = out_dir / "segmentation-board.png"
    metrics_path = out_dir / "metrics.json"
    manifest_path = out_dir / "showcase-manifest.json"
    error_path = out_dir / "error-map.png" if truth is not None else None

    _save_source_slice(image_slice, source_slice_path, payload, z)
    _save_mask(pred_slice, mask_path, payload.class_labels)
    _save_overlay(image_slice, pred_slice, overlay_path, payload.class_labels)
    if truth_slice is not None and error_path is not None:
        _save_error_map(truth_slice, pred_slice, error_path)
    _save_area_profile(pred, area_plot_path, payload.class_labels)

    metrics = segmentation_metrics(pred, truth, payload.class_labels)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    _save_board(
        image_slice=image_slice,
        pred_slice=pred_slice,
        truth_slice=truth_slice,
        board_path=board_path,
        payload=payload,
        z=z,
        metrics=metrics,
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_id": payload.case_id,
        "model_id": payload.model_id,
        "source_id": payload.source_id,
        "slice_index": z,
        "image_shape": list(image.shape),
        "prediction_shape": list(pred.shape),
        "truth_shape": list(truth.shape) if truth is not None else None,
        "class_labels": {str(k): v for k, v in payload.class_labels.items()},
        "voxel_sizes": list(payload.voxel_sizes) if payload.voxel_sizes else None,
        "metadata": _json_safe(payload.metadata),
        "artifacts": {
            "board": board_path.name,
            "overlay": overlay_path.name,
            "mask": mask_path.name,
            "source_slice": source_slice_path.name,
            "area_plot": area_plot_path.name,
            "metrics": metrics_path.name,
            "error_map": error_path.name if error_path else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return ShowcaseArtifacts(
        output_dir=out_dir,
        board=board_path,
        overlay=overlay_path,
        mask=mask_path,
        metrics=metrics_path,
        manifest=manifest_path,
        area_plot=area_plot_path,
        source_slice=source_slice_path,
        error_map=error_path,
    )


def render_segmentation_showcase_from_files(
    *,
    image_path: str | Path,
    prediction_mask_path: str | Path,
    output_dir: str | Path,
    case_id: str,
    model_id: str,
    source_id: str | None = None,
    truth_mask_path: str | Path | None = None,
    class_labels: dict[int, str] | None = None,
    slice_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ShowcaseArtifacts:
    """Load NIfTI image/mask files and render a segmentation showcase.

    This helper is the file boundary for real model outputs: the model runner
    writes a prediction volume, then Qortex renders the exact file that would be
    archived or shared.
    """

    image_path = Path(image_path)
    prediction_mask_path = Path(prediction_mask_path)
    truth_path = Path(truth_mask_path) if truth_mask_path is not None else None
    image, affine, voxel_sizes = _load_nifti_volume(image_path, name="image_path")
    pred, _pred_affine, _pred_voxels = _load_nifti_volume(prediction_mask_path, name="prediction_mask_path")
    truth = None
    if truth_path is not None:
        truth, _truth_affine, _truth_voxels = _load_nifti_volume(truth_path, name="truth_mask_path")

    return render_segmentation_showcase(
        ShowcaseInput(
            image=image,
            prediction_mask=pred,
            truth_mask=truth,
            output_dir=output_dir,
            case_id=case_id,
            model_id=model_id,
            source_id=source_id or str(image_path),
            class_labels=class_labels or {0: "background", 1: "foreground"},
            affine=affine,
            voxel_sizes=voxel_sizes,
            metadata={
                "image_path": str(image_path),
                "prediction_mask_path": str(prediction_mask_path),
                "truth_mask_path": str(truth_path) if truth_path else None,
                **(metadata or {}),
            },
            slice_index=slice_index,
        )
    )


def segmentation_metrics(
    prediction_mask: Any,
    truth_mask: Any | None,
    class_labels: dict[int, str],
) -> dict[str, Any]:
    """Compute per-class segmentation metrics when truth is available."""

    pred = _as_mask(prediction_mask, name="prediction_mask")
    if truth_mask is None:
        return {
            "has_ground_truth": False,
            "predicted_voxels": {
                str(label): int(np.count_nonzero(pred == label))
                for label in sorted(class_labels)
                if label != 0
            },
        }
    truth = _as_mask(truth_mask, name="truth_mask")
    if truth.shape != pred.shape:
        raise ValueError(f"truth_mask shape {truth.shape} does not match prediction_mask {pred.shape}")

    per_class: dict[str, dict[str, float | int | str]] = {}
    for label in sorted(class_labels):
        if label == 0:
            continue
        p = pred == label
        t = truth == label
        tp = int(np.count_nonzero(p & t))
        fp = int(np.count_nonzero(p & ~t))
        fn = int(np.count_nonzero(~p & t))
        denom_dice = (2 * tp + fp + fn)
        denom_iou = (tp + fp + fn)
        per_class[str(label)] = {
            "name": class_labels[label],
            "dice": float((2 * tp) / denom_dice) if denom_dice else 1.0,
            "iou": float(tp / denom_iou) if denom_iou else 1.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predicted_voxels": int(np.count_nonzero(p)),
            "truth_voxels": int(np.count_nonzero(t)),
        }
    return {"has_ground_truth": True, "per_class": per_class}


def _save_board(
    *,
    image_slice: np.ndarray,
    pred_slice: np.ndarray,
    truth_slice: np.ndarray | None,
    board_path: Path,
    payload: ShowcaseInput,
    z: int,
    metrics: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec

    sns.set_theme(style="white", context="notebook")
    fig = plt.figure(figsize=(15.5, 9.2), constrained_layout=True)
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[0.18, 1.0, 0.42])

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.72,
        payload.case_id,
        fontsize=20,
        fontweight="bold",
        color="#20242c",
        ha="left",
        va="center",
    )
    ax_title.text(
        0.0,
        0.2,
        f"source: {payload.source_id}   model: {payload.model_id}   slice: {z}",
        fontsize=10.5,
        color="#51555f",
        ha="left",
        va="center",
    )

    axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    _imshow_gray(axes[0], image_slice, "source")
    _imshow_mask(axes[1], pred_slice, payload.class_labels, "prediction mask")
    _imshow_overlay(axes[2], image_slice, pred_slice, payload.class_labels, "overlay")
    if truth_slice is not None:
        _imshow_error(axes[3], truth_slice, pred_slice, "error map")
    else:
        _imshow_contour(axes[3], image_slice, pred_slice, "prediction contour")

    ax_metrics = fig.add_subplot(gs[2, :2])
    _draw_metric_panel(ax_metrics, metrics)
    ax_legend = fig.add_subplot(gs[2, 2:])
    _draw_legend_panel(ax_legend, payload.class_labels)

    fig.savefig(board_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_source_slice(image_slice: np.ndarray, path: Path, payload: ShowcaseInput, z: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _imshow_gray(ax, image_slice, f"{payload.case_id} · source slice {z}")
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_mask(mask_slice: np.ndarray, path: Path, labels: dict[int, str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _imshow_mask(ax, mask_slice, labels, "prediction mask")
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_overlay(image_slice: np.ndarray, mask_slice: np.ndarray, path: Path, labels: dict[int, str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _imshow_overlay(ax, image_slice, mask_slice, labels, "prediction overlay")
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_error_map(truth_slice: np.ndarray, pred_slice: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _imshow_error(ax, truth_slice, pred_slice, "prediction error map")
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_area_profile(mask: np.ndarray, path: Path, labels: dict[int, str]) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    z_axis = np.arange(mask.shape[0])
    for idx, label in enumerate(sorted(labels)):
        if label == 0:
            continue
        area = np.count_nonzero(mask == label, axis=(1, 2))
        ax.plot(z_axis, area, lw=2.0, color=QORTEX_COLORS[idx % len(QORTEX_COLORS)], label=labels[label])
    ax.set_xlabel("slice index")
    ax.set_ylabel("predicted voxels")
    ax.set_title("Mask area by slice", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _imshow_gray(ax: Any, arr: np.ndarray, title: str) -> None:
    lo, hi = _window(arr)
    ax.imshow(arr, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_mask(ax: Any, mask: np.ndarray, labels: dict[int, str], title: str) -> None:
    from matplotlib.colors import ListedColormap, BoundaryNorm

    max_label = max(max(labels), int(mask.max(initial=0)))
    colors = ["#111111"] + [QORTEX_COLORS[(i - 1) % len(QORTEX_COLORS)] for i in range(1, max_label + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, max_label + 1.5), cmap.N)
    ax.imshow(mask, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_overlay(ax: Any, image: np.ndarray, mask: np.ndarray, labels: dict[int, str], title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=_window(image)[0], vmax=_window(image)[1])
    for idx, label in enumerate(sorted(labels)):
        if label == 0:
            continue
        region = np.ma.masked_where(mask != label, mask)
        ax.imshow(region, cmap=_single_color_cmap(QORTEX_COLORS[idx % len(QORTEX_COLORS)]), alpha=0.48)
    _draw_contours(ax, mask)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_contour(ax: Any, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=_window(image)[0], vmax=_window(image)[1])
    _draw_contours(ax, mask)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _imshow_error(ax: Any, truth: np.ndarray, pred: np.ndarray, title: str) -> None:
    from matplotlib.colors import ListedColormap, BoundaryNorm

    truth_fg = truth > 0
    pred_fg = pred > 0
    error = np.zeros_like(pred, dtype=np.uint8)
    error[pred_fg & truth_fg] = 1
    error[pred_fg & ~truth_fg] = 2
    error[~pred_fg & truth_fg] = 3
    cmap = ListedColormap(["#111111", "#79863c", "#b94949", "#6574a6"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(error, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_contours(ax: Any, mask: np.ndarray) -> None:
    for idx, label in enumerate(sorted(int(v) for v in np.unique(mask) if int(v) != 0)):
        ax.contour(mask == label, levels=[0.5], colors=[QORTEX_COLORS[idx % len(QORTEX_COLORS)]], linewidths=1.8)


def _draw_metric_panel(ax: Any, metrics: dict[str, Any]) -> None:
    ax.axis("off")
    if not metrics.get("has_ground_truth"):
        rows = metrics.get("predicted_voxels", {})
        text = "\n".join(f"class {k}: {v:,} voxels" for k, v in rows.items()) or "no foreground voxels"
        ax.text(0, 0.75, "prediction volume", fontsize=12, fontweight="bold", color="#20242c")
        ax.text(0, 0.35, text, fontsize=10, color="#51555f", family="monospace")
        return
    lines = []
    for label, values in metrics["per_class"].items():
        lines.append(
            f"{values['name']}: Dice {values['dice']:.3f}  IoU {values['iou']:.3f}  TP {values['tp']:,}  FP {values['fp']:,}  FN {values['fn']:,}"
        )
    ax.text(0, 0.78, "segmentation metrics", fontsize=12, fontweight="bold", color="#20242c")
    ax.text(0, 0.28, "\n".join(lines), fontsize=10, color="#51555f", family="monospace")


def _draw_legend_panel(ax: Any, labels: dict[int, str]) -> None:
    from matplotlib.patches import Rectangle

    ax.axis("off")
    ax.text(0, 0.78, "classes", fontsize=12, fontweight="bold", color="#20242c")
    y = 0.48
    for idx, label in enumerate(sorted(labels)):
        if label == 0:
            continue
        ax.add_patch(
            Rectangle(
                (0, y - 0.04),
                0.055,
                0.08,
                color=QORTEX_COLORS[idx % len(QORTEX_COLORS)],
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(0.08, y, f"{label}: {labels[label]}", transform=ax.transAxes, va="center", fontsize=10, color="#51555f")
        y -= 0.18
    ax.text(0, 0.02, "green=true positive, red=false positive, blue=false negative", fontsize=8.5, color="#6b6f7b")


def _save_detection_board(
    image: np.ndarray,
    payload: "DetectionShowcaseInput",
    color_map: dict[str, str],
    board_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.patches import Rectangle

    from qortex.visualize.design import INK, SUBINK, apply_theme

    apply_theme()
    fig = plt.figure(figsize=(13.5, 8.8))
    gs = gridspec.GridSpec(
        3, 3, figure=fig, height_ratios=[0.18, 1.0, 0.55], hspace=0.35,
        top=0.93, bottom=0.05, left=0.03, right=0.98,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0.0, 0.72, payload.case_id, fontsize=18, fontweight="bold", color=INK, ha="left", va="center")
    nms_txt = f"   NMS IoU: {payload.nms_iou:.2f}" if payload.nms_iou is not None else ""
    ax_title.text(
        0.0, 0.2,
        f"source: {payload.source_id}   model: {payload.model_id}   "
        f"threshold: {payload.threshold:.2f}{nms_txt}   detections: {len(payload.detections)}",
        fontsize=10, color=SUBINK, ha="left", va="center",
    )

    ax_img = fig.add_subplot(gs[1, :])
    if image.ndim == 2:
        ax_img.imshow(image, cmap="gray")
    else:
        ax_img.imshow(image)
    for det in payload.detections:
        x1, y1, x2, y2 = det.bbox
        color = color_map[det.class_name]
        ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=2))
        ax_img.text(
            x1, max(y1 - 4, 0), f"{det.class_name} {det.confidence:.2f}",
            fontsize=8.5, color="white", va="bottom", ha="left",
            bbox=dict(facecolor=color, edgecolor="none", pad=1.5, alpha=0.9),
        )
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_title("detections", fontsize=12, fontweight="bold")

    ax_table = fig.add_subplot(gs[2, :2])
    _draw_detection_table(ax_table, payload.detections)
    ax_legend = fig.add_subplot(gs[2, 2])
    _draw_detection_legend(ax_legend, color_map)

    fig.savefig(board_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _draw_detection_table(ax: Any, detections: list["Detection"]) -> None:
    ax.axis("off")
    ax.text(0, 0.94, "detections (by confidence)", fontsize=12, fontweight="bold", color="#20242c")
    ranked = sorted(detections, key=lambda d: d.confidence, reverse=True)[:10]
    lines = [
        f"{d.class_name:<16} {d.confidence:.2f}   "
        f"[{d.bbox[0]:.0f},{d.bbox[1]:.0f},{d.bbox[2]:.0f},{d.bbox[3]:.0f}]"
        for d in ranked
    ]
    if len(detections) > 10:
        lines.append(f"... +{len(detections) - 10} more")
    ax.text(
        0, 0.74, "\n".join(lines) or "no detections above threshold",
        fontsize=9.5, color="#51555f", family="monospace", va="top",
    )


def _draw_detection_legend(ax: Any, color_map: dict[str, str]) -> None:
    from matplotlib.patches import Rectangle

    ax.axis("off")
    ax.text(0, 0.94, "classes", fontsize=12, fontweight="bold", color="#20242c")
    y = 0.74
    for name, color in color_map.items():
        ax.add_patch(Rectangle((0, y - 0.04), 0.1, 0.08, color=color, transform=ax.transAxes, clip_on=False))
        ax.text(0.14, y, name, transform=ax.transAxes, va="center", fontsize=10, color="#51555f")
        y -= 0.16


def _as_image(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim not in (2, 3):
        raise ValueError(f"image must be 2D (H,W) or 3D (H,W,C), got shape {arr.shape}")
    return arr


def _as_volume(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim == 4:
        # Accept C/Z/Y/X or Z/Y/X/C by taking the first channel for rendering.
        arr = arr[0] if arr.shape[0] <= 8 else arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D, 3D, or 4D, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _as_mask(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim == 4:
        arr = np.argmax(arr, axis=0) if arr.shape[0] <= 8 else np.argmax(arr, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 2D, 3D, or 4D, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def _validate_geometry(image: np.ndarray, pred: np.ndarray, truth: np.ndarray | None) -> None:
    if image.shape != pred.shape:
        raise ValueError(f"image shape {image.shape} does not match prediction_mask shape {pred.shape}")
    if truth is not None and truth.shape != pred.shape:
        raise ValueError(f"truth_mask shape {truth.shape} does not match prediction_mask shape {pred.shape}")


def _select_slice(image: np.ndarray, pred: np.ndarray, truth: np.ndarray | None, preferred: int | None) -> int:
    if preferred is not None:
        return int(np.clip(preferred, 0, image.shape[0] - 1))
    area = np.count_nonzero(pred > 0, axis=(1, 2))
    if truth is not None:
        area = area + np.count_nonzero(truth > 0, axis=(1, 2))
    if np.any(area > 0):
        return int(np.argmax(area))
    energy = np.mean(np.abs(image), axis=(1, 2))
    return int(np.argmax(energy))


def _slice(volume: np.ndarray, z: int) -> np.ndarray:
    return np.asarray(volume[int(z)])


def _window(arr: np.ndarray) -> tuple[float, float]:
    values = np.asarray(arr, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _single_color_cmap(color: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("qortex_single", [(1, 1, 1, 0), color])


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _load_nifti_volume(path: Path, *, name: str) -> tuple[np.ndarray, list[list[float]], tuple[float, ...] | None]:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "NIfTI showcase rendering requires nibabel. Install with: pip install 'qortex[mri]'"
        ) from exc
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=float).tolist()
    voxel_sizes = tuple(float(v) for v in img.header.get_zooms()[: min(3, arr.ndim)])
    return arr, affine, voxel_sizes


__all__ = [
    "QORTEX_COLORS",
    "ShowcaseArtifacts",
    "ShowcaseInput",
    "render_segmentation_showcase",
    "render_segmentation_showcase_from_files",
    "segmentation_metrics",
    "Detection",
    "DetectionShowcaseInput",
    "DetectionShowcaseArtifacts",
    "render_detection_showcase",
    "ModelZooArtifacts",
    "render_model_zoo_showcase",
]
