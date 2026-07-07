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
]
