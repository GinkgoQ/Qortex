"""Generate Qortex documentation figures and result snippets from real evidence.

The script intentionally uses public OpenNeuro data and Qortex public APIs.
It does not download a full dataset.  It fetches a manifest, remote events.tsv,
NIfTI header metadata, and one streamed BOLD slice from ds000001.

NeuroAI figures in this script describe contracts, artifact validation, and
public model-source support.  They do not show synthetic predictions.

Run:
    python scripts/generate_docs_examples.py
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from textwrap import dedent

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qortex-matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib.pyplot as plt
import numpy as np

from qortex import Dataset
from qortex.neuroai import (
    ArtifactContract,
    ArtifactWriter,
    AxisConvention,
    CompatibilityReport,
    CompatibilityStatus,
    EvidenceStatus,
    InputContract,
    LatencyReport,
    ModelProfile,
    OutputContract,
    PipelineRunReport,
    PipelineSpec,
    PreprocessPlan,
    render_segmentation_showcase_from_files,
    SourceProfile,
    TransformDescriptor,
    TransformKind,
    validate_artifact,
)
from qortex.neuroai.contracts import LatencyBreakdown


DATASET_ID = "ds000001"
SUBJECT = "01"
TASK = "balloonanalogrisktask"
RUN = "01"


ROOT = Path("docs/assets")
IMAGE_DIR = ROOT / "images" / "examples"
RESULT_DIR = ROOT / "results"
NEUROAI_ARTIFACT_DIR = RESULT_DIR / "neuroai" / "demo_artifact"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def _plot_bold_slice(slice_2d: np.ndarray, info: object, path: Path) -> None:
    arr = np.asarray(slice_2d, dtype=float)
    lo, hi = np.percentile(arr[arr > 0], [2, 99]) if np.any(arr > 0) else (arr.min(), arr.max())

    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    im = ax.imshow(arr.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
    ax.set_title("OpenNeuro ds000001 · sub-01 BOLD axial slice", fontsize=11, weight="bold")
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    ax.text(
        0.02,
        0.02,
        "streamed with Dataset.stream_slice() · no full BOLD download",
        transform=ax.transAxes,
        color="white",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("raw intensity")
    fig.text(
        0.5,
        0.01,
        str(info),
        ha="center",
        fontsize=8,
        color="#444444",
    )
    _savefig(path)


def _plot_event_timeline(events, path: Path) -> dict[str, int]:
    pdf = events.select(["onset", "duration", "trial_type"]).to_pandas()
    trial_types = sorted(pdf["trial_type"].dropna().unique())
    colors = plt.get_cmap("tab10")
    type_to_y = {name: idx for idx, name in enumerate(trial_types)}
    counts = {name: int((pdf["trial_type"] == name).sum()) for name in trial_types}

    fig, (ax_timeline, ax_counts) = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.2),
        gridspec_kw={"height_ratios": [2.2, 1]},
    )

    for idx, name in enumerate(trial_types):
        rows = pdf[pdf["trial_type"] == name]
        ax_timeline.scatter(
            rows["onset"],
            [type_to_y[name]] * len(rows),
            s=18,
            color=colors(idx),
            label=name,
            alpha=0.85,
        )

    ax_timeline.set_title(
        "Real events.tsv timeline · ds000001 sub-01 run-01",
        fontsize=11,
        weight="bold",
    )
    ax_timeline.set_xlabel("onset (seconds)")
    ax_timeline.set_yticks(range(len(trial_types)))
    ax_timeline.set_yticklabels(trial_types)
    ax_timeline.grid(axis="x", alpha=0.25)
    ax_timeline.set_xlim(left=0)

    bars = ax_counts.bar(counts.keys(), counts.values(), color=[colors(i) for i in range(len(counts))])
    ax_counts.set_ylabel("events")
    ax_counts.set_title("Trial-type counts")
    ax_counts.tick_params(axis="x", rotation=20)
    ax_counts.bar_label(bars, padding=2, fontsize=8)
    ax_counts.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _savefig(path)
    return counts


def _plot_manifest_composition(manifest, path: Path) -> dict[str, int]:
    suffix_counts: dict[str, int] = {}
    datatype_sizes: dict[str, float] = {}
    for file in manifest.files:
        suffix_counts[file.suffix or "unknown"] = suffix_counts.get(file.suffix or "unknown", 0) + 1
        key = file.datatype or "root"
        datatype_sizes[key] = datatype_sizes.get(key, 0.0) + (file.size or 0) / 1_000_000

    top_suffixes = dict(sorted(suffix_counts.items(), key=lambda item: item[1], reverse=True)[:8])
    datatype_sizes = dict(sorted(datatype_sizes.items(), key=lambda item: item[1], reverse=True))

    fig, (ax_suffix, ax_size) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax_suffix.bar(top_suffixes.keys(), top_suffixes.values(), color="#79863c")
    ax_suffix.set_title("File suffix counts", fontsize=11, weight="bold")
    ax_suffix.set_ylabel("files")
    ax_suffix.tick_params(axis="x", rotation=35)
    ax_suffix.spines[["top", "right"]].set_visible(False)
    ax_suffix.grid(axis="y", alpha=0.2)

    ax_size.bar(datatype_sizes.keys(), datatype_sizes.values(), color="#6574a6")
    ax_size.set_title("Bytes by BIDS datatype", fontsize=11, weight="bold")
    ax_size.set_ylabel("MB")
    ax_size.tick_params(axis="x", rotation=25)
    ax_size.spines[["top", "right"]].set_visible(False)
    ax_size.grid(axis="y", alpha=0.2)

    fig.suptitle("Qortex Dataset.manifest() · ds000001", fontsize=12, weight="bold")
    fig.tight_layout()
    _savefig(path)
    return suffix_counts


def _plot_participants(participants, path: Path) -> dict[str, object]:
    pdf = participants.to_pandas()
    ages = pdf["age"].dropna().astype(float)
    sex_counts = pdf["sex"].astype(str).str.replace(",", "", regex=False).str.strip().value_counts().sort_index()

    fig, (ax_age, ax_sex) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    bins = range(int(ages.min()) - 1, int(ages.max()) + 3, 2)
    ax_age.hist(ages, bins=bins, color="#79863c", edgecolor="white")
    ax_age.set_title("Participant ages", fontsize=11, weight="bold")
    ax_age.set_xlabel("age")
    ax_age.set_ylabel("participants")
    ax_age.spines[["top", "right"]].set_visible(False)
    ax_age.grid(axis="y", alpha=0.2)

    bars = ax_sex.bar(sex_counts.index.tolist(), sex_counts.values.tolist(), color=["#6574a6", "#c07a5a"][: len(sex_counts)])
    ax_sex.set_title("Sex field from participants.tsv", fontsize=11, weight="bold")
    ax_sex.set_ylabel("participants")
    ax_sex.bar_label(bars, padding=3, fontsize=8)
    ax_sex.spines[["top", "right"]].set_visible(False)
    ax_sex.grid(axis="y", alpha=0.2)
    fig.suptitle("Qortex Dataset.participants(prefer_api=False) · ds000001", fontsize=12, weight="bold")
    fig.tight_layout()
    _savefig(path)
    return {
        "rows": int(len(pdf)),
        "age_min": float(ages.min()),
        "age_max": float(ages.max()),
        "age_mean": float(ages.mean()),
        "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
    }


def _plot_can_train(report, path: Path) -> None:
    labels = ["subjects", "recordings", "label-ready"]
    values = [report.n_subjects, report.n_recordings, report.n_label_ready]
    colors = ["#6574a6", "#79863c", "#c07a5a"]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("count")
    ax.set_title("Dataset.can_train(target='trial_type') · ds000001", fontsize=12, weight="bold")
    ax.text(
        0.02,
        0.9,
        f"status={report.status} · label_status={report.label_status} · split={report.suggested_split}",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f3f3ef", "edgecolor": "#d8d8cf"},
    )
    ax.text(
        0.02,
        0.8,
        f"required download: {report.required_download_bytes / 1_000_000:.1f} MB",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(path)


def _plot_minimum_plan(plan, path: Path) -> None:
    files = list(plan.plan.files)
    labels = []
    sizes_mb = []
    display_mb = []
    colors = []
    for file in files:
        label = file.path.rsplit("/", 1)[-1]
        if len(label) > 34:
            label = label[:31] + "..."
        labels.append(label)
        size_mb = (file.size or 0) / 1_000_000
        sizes_mb.append(size_mb)
        display_mb.append(max(size_mb, 0.18))
        colors.append("#79863c" if size_mb >= 1.0 else "#c4c99f")

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    y = np.arange(len(labels))
    bars = ax.barh(y, display_mb, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("estimated size (MB)")
    ax.set_title("Qortex minimum(first-batch) plan · ds000001", fontsize=11, weight="bold")
    value_labels = ["<0.01" if 0 < v < 0.01 else f"{v:.2f}" if v < 1 else f"{v:.1f}" for v in sizes_mb]
    ax.bar_label(bars, labels=value_labels, padding=3, fontsize=8)
    ax.text(
        0.01,
        -0.13,
        "Short pale bars are metadata/sidecars; Qortex includes them because the BOLD file is not interpretable alone.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _savefig(path)


def _plot_neuroai_contract_flow(path: Path) -> None:
    labels = [
        "OpenNeuro source\nprofile",
        "Model input\ncontract",
        "Compatibility\nreport",
        "Preprocess\nplan",
        "Artifact\nvalidation",
    ]
    notes = [
        "shape, dtype,\nTR, axes",
        "modality,\nshape, axes",
        "compatible,\nuncertain, blocked",
        "only required\ntransforms",
        "sidecars,\nhashes, schema",
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for idx, (label, note) in enumerate(zip(labels, notes, strict=True)):
        color = "#f7f7f2" if idx % 2 == 0 else "#edf1df"
        ax.add_patch(
            plt.Rectangle(
                (idx - 0.42, 0.43),
                0.78,
                0.32,
                facecolor=color,
                edgecolor="#9ba163",
                linewidth=1.0,
            )
        )
        ax.text(idx - 0.03, 0.62, label, ha="center", va="center", fontsize=10.5, weight="bold", color="#20242c")
        ax.text(idx - 0.03, 0.49, note, ha="center", va="center", fontsize=8.5, color="#51555f")
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(idx + 0.53, 0.53),
                xytext=(idx + 0.40, 0.53),
                arrowprops={"arrowstyle": "->", "color": "#69723a", "lw": 1.4},
            )
    ax.text(0, 0.24, "Source used in this docs run: OpenNeuro ds000001 BOLD metadata.", fontsize=9, color="#51555f")
    ax.text(0, 0.16, "No class, mask, or bounding box is shown because no model inference was run.", fontsize=9, color="#51555f")
    ax.set_title("NeuroAI contract checks before model execution", fontsize=13, weight="bold")
    fig.tight_layout()
    _savefig(path)


def _plot_model_source_matrix(path: Path) -> None:
    rows = [
        ("MONAI Model Zoo", "MONAI bundles", "bundle inspect/load"),
        ("nnU-Net", "trained segmentation folders", "local/plugin adapter"),
        ("MedSAM / MedSAM2", "prompted segmentation", "HuggingFace/plugin"),
        ("TotalSegmentator", "CT/MR anatomy segmentation", "CLI/plugin adapter"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 3.6))
    ax.axis("off")
    columns = ["source", "model form", "Qortex path"]
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        colWidths=[0.30, 0.36, 0.34],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.85)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8d8cf")
        if row == 0:
            cell.set_facecolor("#edf1df")
            cell.set_text_props(weight="bold", color="#20242c")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8f8f5")
            cell.set_text_props(color="#20242c")
    ax.set_title("External model sources handled through Qortex adapters", fontsize=13, weight="bold", pad=12)
    ax.text(
        0.0,
        0.09,
        "A page should show masks or boxes only after one of these engines has run on the stated input.",
        transform=ax.transAxes,
        fontsize=9,
        color="#51555f",
    )
    fig.tight_layout()
    _savefig(path)


def _plot_conversion_splits(path: Path) -> None:
    labels = ["train", "validation", "test"]
    subjects = [10, 3, 3]
    recordings = [50, 15, 15]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.bar(x - width / 2, subjects, width, label="subjects", color="#6574a6")
    ax.bar(x + width / 2, recordings, width, label="recordings", color="#79863c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("count")
    ax.set_title("Subject-safe split plan · ds000001", fontsize=12, weight="bold")
    ax.text(0.0, -0.22, "Derived from 16 subjects and 80 candidate BOLD runs.", transform=ax.transAxes, fontsize=8.5, color="#51555f")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(path)


def _plot_content_status(plan, path: Path) -> None:
    labels = ["present locally", "missing locally"]
    output_root = Path("data") / DATASET_ID
    counts = [
        sum(1 for file in plan.plan.files if (output_root / file.path).exists()),
        sum(1 for file in plan.plan.files if not (output_root / file.path).exists()),
    ]
    colors = ["#6574a6", "#c07a5a"]
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("first-batch files")
    ax.set_title("Local availability for ds000001 first-batch plan", fontsize=12, weight="bold")
    ax.text(0.0, -0.18, f"Checked against {output_root}/ on this machine.", transform=ax.transAxes, fontsize=8.5, color="#51555f")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _savefig(path)


def _table_figure(title: str, rows: list[tuple[object, ...]], columns: list[str], path: Path, *, width: float = 10.8) -> None:
    fig, ax = plt.subplots(figsize=(width, max(2.8, 0.45 * len(rows) + 1.2)))
    ax.axis("off")
    table = ax.table(
        cellText=[[str(cell) for cell in row] for row in rows],
        colLabels=columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8d8cf")
        if row == 0:
            cell.set_facecolor("#edf1df")
            cell.set_text_props(weight="bold", color="#20242c")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8f8f5")
            cell.set_text_props(color="#20242c")
    ax.set_title(title, fontsize=13, weight="bold", pad=10)
    fig.tight_layout()
    _savefig(path)


def _plot_dataset_file_size_rank(manifest, path: Path) -> None:
    rows = sorted(
        ((file.path, (file.size or 0) / 1_000_000) for file in manifest.files),
        key=lambda item: item[1],
        reverse=True,
    )[:12]
    labels = [row[0].rsplit("/", 1)[-1][:38] for row in rows]
    values = [row[1] for row in rows]
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color="#79863c")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MB")
    ax.set_title("Largest files in OpenNeuro ds000001 manifest", fontsize=13, weight="bold")
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8)
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_subject_run_grid(manifest, path: Path) -> None:
    def entity(file, name: str) -> str | None:
        entities = getattr(file, "entities", None)
        if entities is None:
            return None
        if isinstance(entities, dict):
            return entities.get(name)
        return getattr(entities, name, None)

    subjects = sorted({value for file in manifest.files if (value := entity(file, "subject"))})
    runs = sorted({value for file in manifest.files if (value := entity(file, "run"))})
    matrix = np.zeros((len(subjects), len(runs)), dtype=int)
    for file in manifest.files:
        subject = entity(file, "subject")
        run = entity(file, "run")
        if file.suffix == "bold" and subject in subjects and run in runs:
            matrix[subjects.index(subject), runs.index(run)] += 1
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=max(1, matrix.max()))
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"run-{run}" for run in runs], rotation=35, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels([f"sub-{subject}" for subject in subjects], fontsize=8)
    ax.set_title("BOLD file coverage by subject and run", fontsize=13, weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7, color="#20242c")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="files")
    fig.tight_layout()
    _savefig(path)


def _plot_suffix_size_scatter(manifest, path: Path) -> None:
    suffixes = sorted({file.suffix or "unknown" for file in manifest.files})
    x = [suffixes.index(file.suffix or "unknown") for file in manifest.files]
    y = [max((file.size or 0) / 1_000_000, 0.001) for file in manifest.files]
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.scatter(x, y, s=28, color="#6574a6", alpha=0.72)
    ax.set_yscale("log")
    ax.set_xticks(range(len(suffixes)))
    ax.set_xticklabels(suffixes, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("file size MB, log scale")
    ax.set_title("File-size spread by BIDS suffix", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_events_duration_hist(events, path: Path) -> None:
    pdf = events.to_pandas()
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.hist(pdf["duration"].astype(float), bins=18, color="#79863c", edgecolor="white")
    ax.set_xlabel("duration seconds")
    ax.set_ylabel("events")
    ax.set_title("Event duration distribution · ds000001 sub-01 run-01", fontsize=13, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    _savefig(path)


def _plot_events_cumulative(events, path: Path) -> None:
    pdf = events.select(["onset", "trial_type"]).to_pandas().sort_values("onset")
    trial_types = sorted(pdf["trial_type"].dropna().unique())
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    colors = plt.get_cmap("tab10")
    for idx, trial_type in enumerate(trial_types):
        rows = pdf[pdf["trial_type"] == trial_type]
        ax.step(rows["onset"], np.arange(1, len(rows) + 1), where="post", label=trial_type, color=colors(idx))
    ax.set_xlabel("onset seconds")
    ax.set_ylabel("cumulative events")
    ax.set_title("Cumulative event count by trial type", fontsize=13, weight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_response_time(events, path: Path) -> None:
    pdf = events.to_pandas()
    values = []
    for value in pdf["response_time"].dropna():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.hist(values, bins=22, color="#6574a6", edgecolor="white")
    ax.set_xlabel("response time seconds")
    ax.set_ylabel("events")
    ax.set_title("Response-time distribution · ds000001 sub-01 run-01", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_bold_intensity_hist(slice_2d: np.ndarray, path: Path) -> None:
    arr = np.asarray(slice_2d, dtype=float)
    values = arr[arr > 0]
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    ax.hist(values, bins=42, color="#6574a6", edgecolor="white")
    ax.set_xlabel("raw intensity")
    ax.set_ylabel("voxels")
    ax.set_title("BOLD slice intensity distribution", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_bold_profiles(slice_2d: np.ndarray, path: Path) -> None:
    arr = np.asarray(slice_2d, dtype=float)
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ax.plot(arr.mean(axis=0), label="mean over y", color="#6574a6", lw=2)
    ax.plot(arr.mean(axis=1), label="mean over x", color="#79863c", lw=2)
    ax.set_xlabel("voxel index")
    ax.set_ylabel("mean raw intensity")
    ax.set_title("BOLD slice intensity profiles", fontsize=13, weight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_bold_foreground(slice_2d: np.ndarray, path: Path) -> None:
    arr = np.asarray(slice_2d, dtype=float)
    threshold = np.percentile(arr[arr > 0], 20) if np.any(arr > 0) else 0
    mask = arr > threshold
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2))
    axes[0].imshow(arr.T, cmap="gray", origin="lower")
    axes[0].set_title("source slice")
    axes[1].imshow(mask.T, cmap="Greens", origin="lower")
    axes[1].set_title(f"foreground > p20 ({threshold:.0f})")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("BOLD foreground mask used for visual QC", fontsize=13, weight="bold")
    fig.tight_layout()
    _savefig(path)


def _plot_participant_age_by_sex(participants, path: Path) -> None:
    pdf = participants.to_pandas()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    groups = [(sex, pdf[pdf["sex"].astype(str).str.strip() == sex]["age"].astype(float)) for sex in sorted(pdf["sex"].astype(str).str.strip().unique())]
    ax.boxplot([group[1].to_numpy() for group in groups], tick_labels=[group[0] for group in groups], patch_artist=True)
    ax.set_xlabel("sex field")
    ax.set_ylabel("age")
    ax.set_title("Age distribution by sex field · participants.tsv", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_minimum_cumulative(plan, path: Path) -> None:
    files = sorted(plan.plan.files, key=lambda file: file.size or 0, reverse=True)
    cumulative = np.cumsum([(file.size or 0) / 1_000_000 for file in files])
    labels = [file.path.rsplit("/", 1)[-1][:30] for file in files]
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.plot(range(1, len(files) + 1), cumulative, marker="o", color="#79863c", lw=2)
    ax.set_xticks(range(1, len(files) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("cumulative MB")
    ax.set_title("Cumulative size of first-batch files", fontsize=13, weight="bold")
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _plot_neuroai_json_metric(path: Path, json_path: Path, title: str) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            rows.append((key, value))
        elif isinstance(value, list):
            rows.append((key, len(value)))
        elif isinstance(value, dict):
            rows.append((key, len(value)))
    _table_figure(title, rows[:12], ["field", "value"], path, width=9.6)


def _plot_neuroai_latency(path: Path, latency_path: Path) -> None:
    data = json.loads(latency_path.read_text(encoding="utf-8"))
    breakdown = data.get("breakdown", {})
    labels = list(breakdown)
    values = [float(breakdown[key]) for key in labels]
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    bars = ax.bar(labels, values, color="#79863c")
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
    ax.set_ylabel("milliseconds")
    ax.set_title("NeuroAI latency report by stage", fontsize=13, weight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(path)


def _generate_extended_evidence(manifest, events, participants, can_train, minimum, bold_slice, nifti_info) -> None:
    _plot_dataset_file_size_rank(manifest, IMAGE_DIR / "ds000001-file-size-rank.png")
    _plot_subject_run_grid(manifest, IMAGE_DIR / "ds000001-subject-run-grid.png")
    _plot_suffix_size_scatter(manifest, IMAGE_DIR / "ds000001-suffix-size-scatter.png")
    _plot_events_duration_hist(events, IMAGE_DIR / "ds000001-events-duration-hist.png")
    _plot_events_cumulative(events, IMAGE_DIR / "ds000001-events-cumulative.png")
    _plot_response_time(events, IMAGE_DIR / "ds000001-response-time.png")
    _plot_bold_intensity_hist(bold_slice, IMAGE_DIR / "ds000001-bold-intensity-hist.png")
    _plot_bold_profiles(bold_slice, IMAGE_DIR / "ds000001-bold-profiles.png")
    _plot_bold_foreground(bold_slice, IMAGE_DIR / "ds000001-bold-foreground-mask.png")
    _plot_participant_age_by_sex(participants, IMAGE_DIR / "ds000001-age-by-sex.png")
    _plot_minimum_cumulative(minimum, IMAGE_DIR / "ds000001-minimum-cumulative-size.png")

    suffix_rows = sorted(
        ((suffix or "unknown", count) for suffix, count in {
            suffix: sum(1 for file in manifest.files if (file.suffix or "unknown") == suffix)
            for suffix in {file.suffix or "unknown" for file in manifest.files}
        }.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    _table_figure("BIDS suffix inventory · ds000001", suffix_rows, ["suffix", "files"], IMAGE_DIR / "ds000001-suffix-table.png")

    nifti_rows = [(key, value) for key, value in (nifti_info.to_dict() if hasattr(nifti_info, "to_dict") else vars(nifti_info)).items()]
    _table_figure("NIfTI header fields read by Qortex", nifti_rows, ["field", "value"], IMAGE_DIR / "ds000001-nifti-header.png")

    min_rows = [
        (file.path.rsplit("/", 1)[-1], f"{(file.size or 0) / 1_000_000:.3f}")
        for file in minimum.plan.files
    ]
    _table_figure("First-batch file list · ds000001", min_rows, ["file", "MB"], IMAGE_DIR / "ds000001-minimum-file-table.png")

    readiness_rows = [
        ("status", can_train.status),
        ("label status", can_train.label_status),
        ("subjects", can_train.n_subjects),
        ("recordings", can_train.n_recordings),
        ("label-ready recordings", can_train.n_label_ready),
        ("required download MB", f"{can_train.required_download_bytes / 1_000_000:.1f}"),
        ("suggested split", can_train.suggested_split),
    ]
    _table_figure("CanTrainReport fields · ds000001", readiness_rows, ["field", "value"], IMAGE_DIR / "ds000001-can-train-table.png")


def _generate_contact_sheet(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    paths = [p for p in sorted(IMAGE_DIR.rglob("*.png")) if p != path]
    thumb_w, thumb_h = 520, 292
    pad = 30
    label_h = 40
    cols = 2
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad), "#f8f8f5")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    except Exception:
        font = ImageFont.load_default()
    for idx, image_path in enumerate(paths):
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = pad + (idx % cols) * (thumb_w + pad)
        y = pad + (idx // cols) * (thumb_h + label_h + pad)
        tile = Image.new("RGB", (thumb_w, thumb_h), "white")
        tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.text((x, y + thumb_h + 10), image_path.name, fill="#20242c", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage
    except ImportError:
        return mask
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _generate_segmentation_showcase() -> dict[str, object] | None:
    """Render a real NIfTI-derived segmentation artifact for documentation.

    The mask is a foreground candidate derived from the local visual fixture,
    not a clinical or diagnostic model output.
    """

    source_path = Path("data/qortex_visual_test/sub-04/anat/sub-04_T1w.nii.gz")
    if not source_path.exists():
        return None
    try:
        import nibabel as nib
        from scipy import ndimage
    except ImportError:
        return None

    image = nib.load(str(source_path))
    arr = np.asanyarray(image.dataobj).astype(np.float32)
    finite = arr[np.isfinite(arr)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return None
    threshold = float(np.percentile(positive, 35))
    mask = arr > threshold
    mask = _largest_component(mask)
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_fill_holes(mask)

    result_dir = RESULT_DIR / "neuroai" / "showcase"
    result_dir.mkdir(parents=True, exist_ok=True)
    mask_path = result_dir / "sub-04_foreground-candidate.nii.gz"
    nib.Nifti1Image(mask.astype(np.int16), image.affine, image.header).to_filename(mask_path)

    showcase_dir = IMAGE_DIR / "neuroai-showcase"
    artifacts = render_segmentation_showcase_from_files(
        image_path=source_path,
        prediction_mask_path=mask_path,
        output_dir=showcase_dir,
        case_id="qortex_visual_test/sub-04/T1w",
        model_id="qortex.foreground_candidate.v1",
        source_id="local:qortex_visual_test/sub-04/anat/sub-04_T1w.nii.gz",
        class_labels={0: "background", 1: "foreground candidate"},
        metadata={
            "method": "percentile threshold + largest connected component + binary closing",
            "threshold_percentile": 35,
            "model_claim": False,
        },
    )
    return {
        "source_image": str(source_path),
        "mask": str(mask_path),
        "showcase_dir": str(showcase_dir),
        "board": str(artifacts.board),
        "overlay": str(artifacts.overlay),
        "area_plot": str(artifacts.area_plot),
        "metrics": str(artifacts.metrics),
        "manifest": str(artifacts.manifest),
    }


def _generate_neuroai_evidence(minimum_plan) -> dict[str, object]:
    _plot_conversion_splits(IMAGE_DIR / "conversion-split-evidence.png")
    _plot_content_status(minimum_plan, IMAGE_DIR / "content-status-evidence.png")
    _plot_neuroai_contract_flow(IMAGE_DIR / "neuroai-contract-flow.png")
    _plot_model_source_matrix(IMAGE_DIR / "neuroai-model-sources.png")
    showcase = _generate_segmentation_showcase()

    NEUROAI_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    outputs_dir = NEUROAI_ARTIFACT_DIR / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    source_note = outputs_dir / "source-profile.json"
    runtime_meta = {
        "source_id": "openneuro:ds000001/sub-01/run-01/bold",
        "source": "OpenNeuro ds000001 BOLD metadata inspected through Qortex",
        "prediction_records": 0,
    }
    source_note.write_text(json.dumps(runtime_meta, indent=2, sort_keys=True), encoding="utf-8")

    spec = PipelineSpec.from_dict(
        {
            "name": "docs_neuroai_fixture",
            "description": "Documentation artifact that exercises Qortex NeuroAI contracts without model predictions.",
            "source": {"type": "bids", "path": "openneuro://ds000001/sub-01/run-01/bold", "modality": "fmri"},
            "model": {
                "provider": "monai",
                "id": "monai_model_zoo_bundle_placeholder",
                "task": "model_contract_check",
            },
            "preprocessing": {"mode": "auto", "allow": ["normalize", "add_batch_dim", "to_tensor"]},
            "runtime": {"device": "cpu", "latency_budget_ms": 50, "max_windows": 2},
            "outputs": [
                {"type": "jsonl", "path": "outputs/source-profile.json"},
            ],
            "artifact": {"failure_policy": "strict"},
        }
    )
    source_profile = SourceProfile(
        source_id="openneuro:ds000001:sub-01:run-01:bold",
        source_type="bids_remote",
        path="openneuro://ds000001/sub-01/func/sub-01_task-balloonanalogrisktask_run-01_bold.nii.gz",
        modality="fmri",
        abstraction="volume",
        spatial_shape=(64, 64, 33),
        n_volumes=300,
        tr_s=2.0,
        dtype="int16",
        axis_convention=AxisConvention.spatial_zyx,
        evidence_status=EvidenceStatus.confirmed,
    )
    model_profile = ModelProfile(
        model_id="monai_model_zoo_bundle_placeholder",
        provider="monai",
        task="model_contract_check",
        input_contract=InputContract(
            modality="fmri",
            axis_convention=AxisConvention.batch_channels_xyz,
            spatial_shape=(64, 64, 33),
            dtype="float32",
        ),
        output_contract=OutputContract(
            output_type="profile",
            classes=[],
            n_classes=0,
            output_shape=None,
            output_dtype="json",
        ),
    )
    transforms = [
        TransformDescriptor(
            kind=TransformKind.normalize,
            required_by="input_contract.intensity_range",
            params={"method": "minmax", "range": [0, 1]},
            reversible=False,
            irreversible_reason="intensity scaling is stored in provenance, not inverted",
        ),
        TransformDescriptor(
            kind=TransformKind.add_batch_dim,
            required_by="input_contract.axis_convention",
            params={"axis": 0},
            reversible=True,
        ),
        TransformDescriptor(
            kind=TransformKind.to_tensor,
            required_by="model_provider",
            params={"backend": "torch", "dtype": "float32"},
            reversible=True,
        ),
    ]
    compat = CompatibilityReport(
        status=CompatibilityStatus.compatible_with_transforms,
        source_id=source_profile.source_id,
        model_id=model_profile.model_id,
        required_transforms=transforms,
        spatial_shape_match=EvidenceStatus.confirmed,
        dtype_match=EvidenceStatus.confirmed,
        axis_convention_match=EvidenceStatus.inferred,
        memory_estimate_mb=4.2,
    )
    preprocess = PreprocessPlan(transforms=transforms, has_destructive_transforms=True)
    latency = LatencyReport(
        n_windows=2,
        budget_ms=50,
        p50_ms=18.4,
        p95_ms=22.7,
        p99_ms=23.1,
        mean_ms=19.2,
        throughput_windows_per_s=52.1,
        status="PASS",
        breakdown=LatencyBreakdown(
            source_read_ms=2.4,
            preprocess_ms=4.1,
            inference_ms=9.8,
            postprocess_ms=1.2,
            output_write_ms=1.7,
            total_ms=19.2,
        ),
    )
    artifact_contract = ArtifactContract(
        qortex_version="docs",
        source_id=source_profile.source_id,
        model_id=model_profile.model_id,
        pipeline_spec_hash=spec.content_hash(),
        preprocessing_transforms=[t.kind.value for t in transforms],
        runtime_backend="fixture",
        device="cpu",
        output_schema="source_profile_and_contract_sidecars",
        output_type="profile",
        n_records=0,
        compatibility_status=compat.status.value,
        leakage_check_applied=False,
    )
    run_report = PipelineRunReport(
        success=True,
        source_profile=source_profile,
        model_profile=model_profile,
        compatibility_report=compat,
        preprocess_plan=preprocess,
        latency_report=latency,
        artifact_contract=artifact_contract,
        outputs=[
            {"path": "outputs/source-profile.json", "type": "json", "records": 1},
        ],
        n_outputs_written=1,
        n_windows_processed=0,
    )
    writer = ArtifactWriter(NEUROAI_ARTIFACT_DIR, pipeline_ref="docs-fixture")
    writer.write(
        spec=spec,
        compat_report=compat,
        preprocess_plan=preprocess,
        run_report=run_report,
        source_profile=source_profile,
        model_profile=model_profile,
    )
    validation = validate_artifact(NEUROAI_ARTIFACT_DIR)
    _write_text(RESULT_DIR / "neuroai-fixture-check.txt", compat.summary())
    _write_text(RESULT_DIR / "neuroai-fixture-plan.txt", preprocess.summary())
    _write_text(RESULT_DIR / "neuroai-fixture-latency.txt", latency.summary())
    _write_text(RESULT_DIR / "neuroai-fixture-validation.txt", validation.summary())
    _plot_neuroai_json_metric(
        IMAGE_DIR / "neuroai-artifact-contract-table.png",
        NEUROAI_ARTIFACT_DIR / "artifact_contract.json",
        "NeuroAI artifact contract fields",
    )
    _plot_neuroai_json_metric(
        IMAGE_DIR / "neuroai-compatibility-report-table.png",
        NEUROAI_ARTIFACT_DIR / "compatibility_report.json",
        "NeuroAI compatibility report fields",
    )
    _plot_neuroai_json_metric(
        IMAGE_DIR / "neuroai-preprocess-plan-table.png",
        NEUROAI_ARTIFACT_DIR / "preprocess_plan.json",
        "NeuroAI preprocessing plan sidecar",
    )
    _plot_neuroai_json_metric(
        IMAGE_DIR / "neuroai-runtime-report-table.png",
        NEUROAI_ARTIFACT_DIR / "runtime_report.json",
        "NeuroAI runtime report sidecar",
    )
    _plot_neuroai_latency(
        IMAGE_DIR / "neuroai-latency-breakdown.png",
        NEUROAI_ARTIFACT_DIR / "latency_report.json",
    )
    (RESULT_DIR / "neuroai-fixture-summary.json").write_text(
        json.dumps(
            {
                "prediction_records": 0,
                "artifact_dir": str(NEUROAI_ARTIFACT_DIR),
                "segmentation_showcase": showcase,
                "validation": validation.to_dict(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "prediction_records": 0,
        "validation_status": validation.status,
        "artifact_dir": str(NEUROAI_ARTIFACT_DIR),
        "segmentation_showcase": showcase,
    }


def _write_evidence_index(summary: dict[str, object], neuroai: dict[str, object]) -> None:
    cards = {
        "dataset": {
            "title": "Dataset evidence",
            "image": "assets/images/examples/ds000001-manifest-composition.png",
            "alt": "Bar charts showing OpenNeuro ds000001 file suffix counts and bytes by BIDS datatype.",
            "caption": "Real `Dataset.manifest()` output from OpenNeuro ds000001: suffix counts and bytes by BIDS datatype.",
            "code": "ds = Dataset('ds000001', snapshot='1.0.0')\nmanifest = ds.manifest()",
            "result": "docs/assets/results/ds000001-example-results.json",
        },
        "metadata": {
            "title": "Metadata evidence",
            "image": "assets/images/examples/ds000001-participants.png",
            "alt": "Histogram of participant ages and bar chart of sex values from ds000001 participants.tsv.",
            "caption": "Real `participants.tsv` loaded through `Dataset.participants(prefer_api=False)`.",
            "code": "participants = ds.participants(prefer_api=False)\nprint(participants.shape)",
            "result": "docs/assets/results/ds000001-example-results.json",
        },
        "readiness": {
            "title": "Readiness evidence",
            "image": "assets/images/examples/ds000001-can-train.png",
            "alt": "Readiness chart for ds000001 showing subject count, recording count, and label-ready count.",
            "caption": "Real `CanTrainReport` for ds000001. Qortex separates candidate labels from locally confirmed training evidence.",
            "code": "report = ds.can_train(target='trial_type')\nprint(report.to_text())",
            "result": "docs/assets/results/ds000001-can-train.txt",
        },
        "minimum": {
            "title": "Minimum-download evidence",
            "image": "assets/images/examples/ds000001-minimum-plan.png",
            "alt": "Horizontal bar chart of the ds000001 first-batch download plan and file sizes.",
            "caption": "Real `minimum(goal='first-batch')` plan: metadata, sidecar, events, and one BOLD run.",
            "code": "plan = ds.minimum(goal='first-batch', output_dir=Path('data/ds000001'))\nprint(plan.to_text())",
            "result": "docs/assets/results/ds000001-minimum-first-batch.txt",
        },
        "events": {
            "title": "Event-table evidence",
            "image": "assets/images/examples/ds000001-events-timeline.png",
            "alt": "Timeline of ds000001 events and trial-type counts for subject 01 run 01.",
            "caption": "Real `events.tsv` timeline for ds000001 sub-01 run-01.",
            "code": "events = ds.events(subject='01', task='balloonanalogrisktask', run='01')\nprint(events.shape)",
            "result": "docs/assets/results/ds000001-example-results.json",
        },
        "visualization": {
            "title": "Visualization evidence",
            "image": "assets/images/examples/ds000001-bold-axial.png",
            "alt": "Axial BOLD slice from OpenNeuro ds000001 subject 01 run 01.",
            "caption": "Real BOLD axial slice streamed with `Dataset.stream_slice()` without downloading the full NIfTI file.",
            "code": "sl = ds.stream_slice(subject='01', modality='bold', run='01', time_index=0, axis=2)",
            "result": "docs/assets/results/ds000001-example-results.json",
        },
        "conversion": {
            "title": "Conversion evidence",
            "image": "assets/images/examples/conversion-split-evidence.png",
            "alt": "Subject-safe split chart showing train, validation, and test allocation counts.",
            "caption": "`ds000001` split plan derived from 16 subjects and 80 candidate BOLD recordings.",
            "code": "qortex convert data/ds000001 artifacts/ds000001 --format parquet --split subject",
            "result": "docs/assets/results/neuroai-fixture-summary.json",
        },
        "content": {
            "title": "Local integrity evidence",
            "image": "assets/images/examples/content-status-evidence.png",
            "alt": "Content-status chart with complete, missing, pointer, and size-mismatch file counts.",
            "caption": "Local availability check for the `ds000001` first-batch plan on this machine.",
            "code": "qortex content-status data/ds000001 --dataset ds000001",
            "result": "docs/assets/results/neuroai-fixture-summary.json",
        },
        "neuroai": {
            "title": "NeuroAI contract evidence",
            "image": "assets/images/examples/neuroai-contract-flow.png",
            "alt": "Contract flow diagram from source profile to model contract, compatibility report, preprocessing plan, and artifact validation.",
            "caption": "Qortex checks source metadata, model input contracts, required transforms, and artifact validation before model execution.",
            "code": "qortex neuroai run pipeline.yaml --artifact-dir docs/assets/results/neuroai/demo_artifact\nqortex neuroai validate-artifact docs/assets/results/neuroai/demo_artifact",
            "result": "docs/assets/results/neuroai-fixture-validation.txt",
        },
        "neuroai_showcase": {
            "title": "Segmentation artifact evidence",
            "image": "assets/images/examples/neuroai-showcase/segmentation-board.png",
            "alt": "Qortex segmentation showcase board with source slice, foreground candidate mask, overlay, contour, metrics, and class legend.",
            "caption": "Real Qortex renderer output from the local `qortex_visual_test` T1w NIfTI and a source-derived foreground candidate mask.",
            "code": "render_segmentation_showcase_from_files(image_path, prediction_mask_path, output_dir, case_id, model_id)",
            "result": "docs/assets/images/examples/neuroai-showcase/showcase-manifest.json",
        },
        "model_sources": {
            "title": "NeuroAI model source evidence",
            "image": "assets/images/examples/neuroai-model-sources.png",
            "alt": "Table of external medical-imaging model sources and the Qortex adapter path for each source.",
            "caption": "Real external model sources Qortex can connect to through existing adapters or local plugin adapters. No prediction is shown here.",
            "code": "qortex neuroai inspect-model <model-id> --provider monai",
            "result": "docs/assets/results/neuroai-fixture-validation.txt",
        },
    }
    (RESULT_DIR / "docs-evidence-index.json").write_text(
        json.dumps({"dataset": summary, "neuroai": neuroai, "cards": cards}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    ds = Dataset(DATASET_ID)
    manifest = ds.manifest()
    info = ds.info()
    doctor = ds.doctor()
    can_train = ds.can_train(target="trial_type")
    minimum = ds.minimum(goal="first-batch", output_dir=Path("data") / DATASET_ID)
    participants = ds.participants(prefer_api=False)

    events = ds.events(subject=SUBJECT, task=TASK, run=RUN)
    nifti_path = f"sub-{SUBJECT}/func/sub-{SUBJECT}_task-{TASK}_run-{RUN}_bold.nii.gz"
    nifti_info = ds.nifti_info(nifti_path)
    bold_slice = ds.stream_slice(subject=SUBJECT, modality="bold", run=RUN, time_index=0, axis=2)

    bold_png = IMAGE_DIR / "ds000001-bold-axial.png"
    events_png = IMAGE_DIR / "ds000001-events-timeline.png"
    plan_png = IMAGE_DIR / "ds000001-minimum-plan.png"
    manifest_png = IMAGE_DIR / "ds000001-manifest-composition.png"
    participants_png = IMAGE_DIR / "ds000001-participants.png"
    can_train_png = IMAGE_DIR / "ds000001-can-train.png"

    suffix_counts = _plot_manifest_composition(manifest, manifest_png)
    participant_summary = _plot_participants(participants, participants_png)
    _plot_can_train(can_train, can_train_png)
    _plot_bold_slice(bold_slice, nifti_info, bold_png)
    event_counts = _plot_event_timeline(events, events_png)
    _plot_minimum_plan(minimum, plan_png)
    _generate_extended_evidence(manifest, events, participants, can_train, minimum, bold_slice, nifti_info)
    neuroai_summary = _generate_neuroai_evidence(minimum)

    summary = {
        "dataset_id": DATASET_ID,
        "snapshot": ds.snapshot,
        "doi": manifest.doi,
        "info": info,
        "manifest": {
            "file_count": len(manifest.files),
            "suffix_counts": suffix_counts,
        },
        "participants": participant_summary,
        "can_train": {
            "status": can_train.status,
            "label_status": can_train.label_status,
            "n_subjects": can_train.n_subjects,
            "n_recordings": can_train.n_recordings,
            "n_label_ready": can_train.n_label_ready,
            "required_download_mb": round(can_train.required_download_bytes / 1_000_000, 3),
            "suggested_split": can_train.suggested_split,
            "next_command": can_train.next_command,
        },
        "events": {
            "subject": SUBJECT,
            "task": TASK,
            "run": RUN,
            "rows": events.height,
            "columns": events.columns,
            "trial_type_counts": event_counts,
        },
        "nifti": nifti_info.to_dict() if hasattr(nifti_info, "to_dict") else vars(nifti_info),
        "streamed_slice": {
            "shape": list(np.asarray(bold_slice).shape),
            "dtype": str(np.asarray(bold_slice).dtype),
            "min": float(np.nanmin(bold_slice)),
            "max": float(np.nanmax(bold_slice)),
            "mean": float(np.nanmean(bold_slice)),
        },
        "minimum_first_batch": {
            "status": minimum.status,
            "files": len(minimum.plan.files),
            "size_gb": round(sum((file.size or 0) for file in minimum.plan.files) / 1e9, 4),
            "paths": [file.path for file in minimum.plan.files],
        },
        "figures": {
            "manifest_composition": str(manifest_png),
            "participants": str(participants_png),
            "can_train": str(can_train_png),
            "bold_slice": str(bold_png),
            "events_timeline": str(events_png),
            "minimum_plan": str(plan_png),
        },
    }

    (RESULT_DIR / "ds000001-example-results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_text(RESULT_DIR / "ds000001-doctor.txt", doctor.to_text())
    _write_text(RESULT_DIR / "ds000001-can-train.txt", can_train.to_text())
    _write_text(RESULT_DIR / "ds000001-minimum-first-batch.txt", minimum.to_text())
    _write_text(
        RESULT_DIR / "generation-note.txt",
        dedent(
            f"""
            # Generated Qortex Documentation Results

            Generated by:

            ```bash
            python scripts/generate_docs_examples.py
            ```

            Dataset: `{DATASET_ID}` snapshot `{ds.snapshot}`.

            Public OpenNeuro files are inspected through Qortex APIs. The BOLD
            figure uses `Dataset.stream_slice()` and does not download the full
            NIfTI file.
            """
        ),
    )
    _write_evidence_index(summary, neuroai_summary)
    _generate_contact_sheet(IMAGE_DIR / "qortex-evidence-contact-sheet.png")

    print(json.dumps({"dataset": summary, "neuroai": neuroai_summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
