"""project_20_visualization_advanced

Deep integration tests for the full qortex.visualize feature set.
Each test exercises a real workflow rather than mocked internals.

Test catalogue
--------------
 1.  Lazy NIfTI proxy — inspect header without loading pixels
 2.  Render thumbnail without full volume load (single-slice ArrayProxy)
 3.  Static ortho figure — correct panel count, shape, voxel-aspect titles
 4.  Overlay affine mismatch raises OverlayGeometryError by default
 5.  Overlay resample=True — resamples into base space without error
 6.  Visual audit report — per_suffix_counts, warning_summary, to_json(), HTML
 7.  fMRI QC summary — 6-panel figure, Welford streaming stats, streaming shapes
 8.  fMRI streaming stats — mean/std/tSNR/global_signal/framewise on synthetic 4D
 9.  DWI summary — b0/high-b slices, bval histogram, gradient sphere panel
10.  DWI bval/bvec parsing — shells dict, b0_indices, high_b_indices
11.  overlay_contour — contour is strictly inside mask boundary
12.  compare_masks — Dice ≈ 1 for identical masks, ≈ 0 for disjoint masks
13.  overlay_edges — gradient-magnitude edge detection on synthetic mask
14.  VisualAuditReport to_json() — all top-level keys present and types correct
15.  DICOM PHI hidden by default — DicomSeriesBrowser redacts patient fields
16.  EEG thumbnail — _eeg_thumbnail produces a non-empty base64 PNG
17.  coverage_matrix — subject × suffix cross-table structure
18.  visualize-openneuro selection — shared select_visual_files helper
19.  Artifact.compare_splits() — returns VisualAuditReport with correct structure
20.  Visual audit n_expected / n_local_present auto-population
21.  DWI contact_sheet — shell-specific montage with real plotted slice panels
22.  TimeSeriesViewer.topomap — sensor interpolation and channel marker output
23.  fMRI summary companions — events and confounds become explicit QC traces
24.  Pathless VisualResult.to_png — clear error for non-file-backed NIfTI result
25.  Visualization edge cases — robust BIDS suffix, smooth aspect resize, empty Dice

All tests use synthetic in-memory or temp-file data.
nibabel, plotly, and MNE are each skipped gracefully when absent.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, require, require_gt, require_in, require_type, passed,
)


# ── Fixture builders ──────────────────────────────────────────────────────────

def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


def _write_synthetic_nifti(
    path: Path,
    shape: tuple,
    affine: np.ndarray | None = None,
    *,
    modality: str = "mri",
) -> Path:
    """Write a synthetic NIfTI-1 file to disk using nibabel.

    The volume contains a bright sphere in a dark background so windowing
    logic has something real to work with.  Returns *path*.
    """
    nib = _try_import("nibabel")
    if nib is None:
        raise ImportError("nibabel required for NIfTI fixture")

    vol = np.zeros(shape, dtype=np.float32)
    if len(shape) >= 3:
        cx, cy, cz = [s // 2 for s in shape[:3]]
        r = min(shape[:3]) // 3
        xs, ys, zs = np.ogrid[:shape[0], :shape[1], :shape[2]]
        sphere = (xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2 < r ** 2
        if modality == "mri" and vol.ndim == 3:
            n_sphere = int(sphere.sum())
            vol[sphere] = (900.0 + np.random.uniform(0, 100, n_sphere)).astype(np.float32)
        elif modality == "bold":
            # 4D: add physiological drift + noise per timepoint
            rng = np.random.RandomState(42)
            vol = rng.randn(*shape).astype(np.float32) * 30 + 1000
            for t in range(shape[3]):
                vol[sphere, t] += float(np.sin(t * 0.3) * 60)
        elif modality == "dwi":
            rng = np.random.RandomState(7)
            for t in range(shape[3]):
                decay = float(np.exp(-t * 0.02))
                frame = rng.randn(*shape[:3]).astype(np.float32) * 20 + 800 * decay
                frame[sphere] = 900 * decay + rng.randn(int(sphere.sum())).astype(np.float32) * 10
                vol[..., t] = frame

    if affine is None:
        affine = np.diag([1.5, 1.5, 3.0, 1.0])

    img = nib.Nifti1Image(vol, affine)
    nib.save(img, str(path))
    return path


def _write_bval(path: Path, bvals: list[float]) -> Path:
    path.write_text(" ".join(str(int(b)) for b in bvals) + "\n")
    return path


def _write_bvec(path: Path, n_vols: int) -> Path:
    rng = np.random.RandomState(1)
    dirs = rng.randn(3, n_vols).astype(np.float32)
    norms = np.linalg.norm(dirs, axis=0)
    norms[norms < 1e-6] = 1.0
    dirs /= norms
    lines = [" ".join(f"{v:.6f}" for v in dirs[row]) for row in range(3)]
    path.write_text("\n".join(lines) + "\n")
    return path


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_01_lazy_nifti_proxy():
    banner("1 — Lazy NIfTI proxy: inspect header without loading pixels")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize.volume import _LazyNIfTI

    with tempfile.TemporaryDirectory() as tmp:
        p = _write_synthetic_nifti(Path(tmp) / "sub-01_T1w.nii.gz", (48, 48, 30))
        lazy = _LazyNIfTI(p)

        require(lazy.shape[:3] == (48, 48, 30), f"Wrong shape: {lazy.shape}")
        require(lazy.affine.shape == (4, 4), "affine must be 4×4")
        require(len(lazy.zooms) == 3, "zooms must have 3 elements")
        require(lazy.zooms[2] > lazy.zooms[0], "z-voxel should be larger than x-voxel")
        require(lazy._proxy is not None, "proxy must not be None")

        # Confirm the proxy is a nibabel ArrayProxy, NOT a loaded array
        require(not isinstance(lazy._proxy, np.ndarray),
                "proxy must be ArrayProxy, not a preloaded ndarray")

        print_kv("shape", lazy.shape)
        print_kv("zooms_mm", lazy.zooms)
        print_kv("dtype", lazy.dtype)


def test_02_thumbnail_no_full_load():
    banner("2 — Thumbnail extraction: single slice, no full volume materialisation")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize.volume import _LazyNIfTI
    from qortex.visualize._html import array_to_b64png

    with tempfile.TemporaryDirectory() as tmp:
        p = _write_synthetic_nifti(Path(tmp) / "sub-01_T1w.nii.gz", (64, 64, 40))
        lazy = _LazyNIfTI(p)
        cz = lazy.shape[2] // 2

        # Read exactly one axial slice
        slc = lazy.slice_along(2, cz)
        require(slc.shape == (64, 64), f"Slice shape wrong: {slc.shape}")
        require(slc.dtype == np.float32, f"Slice dtype wrong: {slc.dtype}")

        # Encode to PNG without loading any other data
        vmin, vmax = lazy.sample_window("mri")
        b64 = array_to_b64png(slc.T, vmin, vmax, "gray")
        require(isinstance(b64, str) and len(b64) > 200, "base64 PNG too short or wrong type")

        import base64
        png_bytes = base64.b64decode(b64)
        require(png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Invalid PNG magic bytes")

        print_kv("slice_shape", slc.shape)
        print_kv("window", f"[{vmin:.0f}, {vmax:.0f}]")
        print_kv("png_bytes", len(png_bytes))


def test_03_static_ortho():
    banner("3 — Static ortho figure: panel count, subplot titles, aspect ratio")
    nib = _try_import("nibabel")
    plotly = _try_import("plotly")
    if nib is None or plotly is None:
        print("  SKIP: nibabel or plotly not installed")
        return

    from qortex.visualize.volume import VolumeViewer

    with tempfile.TemporaryDirectory() as tmp:
        p = _write_synthetic_nifti(Path(tmp) / "T1w.nii.gz", (48, 48, 30))
        viewer = VolumeViewer(p, modality="mri")
        fig = viewer.ortho(title="Test Ortho")

    require(hasattr(fig, "data"), "ortho() must return a Figure")
    require(len(fig.data) == 3, f"Expected 3 traces, got {len(fig.data)}")

    # All three must be Heatmaps
    from plotly.graph_objects import Heatmap
    for i, trace in enumerate(fig.data):
        require(isinstance(trace, Heatmap), f"Trace {i} is {type(trace).__name__}, not Heatmap")

    # Title must contain the override
    require("Test Ortho" in (fig.layout.title.text or ""), "Title not set")

    # Subplot titles: Axial, Coronal, Sagittal
    ann_texts = [a.text for a in (fig.layout.annotations or [])]
    require(any("Axial" in t for t in ann_texts), "Missing Axial subplot title")
    require(any("Coronal" in t for t in ann_texts), "Missing Coronal subplot title")
    require(any("Sagittal" in t for t in ann_texts), "Missing Sagittal subplot title")

    print_kv("traces", len(fig.data))
    print_kv("subplot_titles", [t for t in ann_texts if t])


def test_04_overlay_affine_mismatch_raises():
    banner("4 — Overlay affine mismatch raises OverlayGeometryError by default")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize.overlay import overlay_mask, OverlayGeometryError

    with tempfile.TemporaryDirectory() as tmp:
        base_path = _write_synthetic_nifti(
            Path(tmp) / "T1w.nii.gz", (32, 32, 20),
            affine=np.diag([1.0, 1.0, 3.0, 1.0]),
        )
        mask_path = _write_synthetic_nifti(
            Path(tmp) / "brain_mask.nii.gz", (32, 32, 20),
            affine=np.diag([2.0, 2.0, 4.0, 1.0]),  # different voxel sizes → different affine
        )

        try:
            overlay_mask(base_path, mask_path)
            raise AssertionError("Expected OverlayGeometryError was not raised")
        except OverlayGeometryError as exc:
            print_kv("raised", str(exc)[:80])
            print_kv("status", "PASS — OverlayGeometryError raised as expected")


def test_05_overlay_resample_works():
    banner("5 — Overlay resample=True resamples mask into base space")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize.overlay import overlay_mask
    from qortex.visualize._asset import VisualResult

    with tempfile.TemporaryDirectory() as tmp:
        # Base: 1.5mm isotropic 48³
        base_path = _write_synthetic_nifti(
            Path(tmp) / "T1w.nii.gz", (32, 32, 20),
            affine=np.diag([1.5, 1.5, 3.0, 1.0]),
        )
        # Mask: different voxel size — nibabel resampling required
        mask_path = _write_synthetic_nifti(
            Path(tmp) / "mask_2mm.nii.gz", (24, 24, 15),
            affine=np.diag([2.0, 2.0, 4.0, 1.0]),
        )

        result = overlay_mask(base_path, mask_path, resample=True, allow_affine_mismatch=True)
        require_type(result, VisualResult, "overlay_mask(resample=True)")
        require(isinstance(result.html, str) and len(result.html) > 1000, "HTML too short")
        require(result.provenance.get("type") == "mask_overlay", "Wrong provenance type")

        print_kv("result_type", type(result).__name__)
        print_kv("html_chars", len(result.html))


def test_06_visual_audit_report():
    banner("6 — VisualAuditReport: per_suffix_counts, warning_summary, HTML, JSON")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize._audit import run_visual_audit, VisualAuditReport

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Create a realistic BIDS-like structure
        (root / "sub-01" / "anat").mkdir(parents=True)
        (root / "sub-01" / "func").mkdir(parents=True)
        (root / "sub-02" / "anat").mkdir(parents=True)

        _write_synthetic_nifti(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz", (32, 32, 20))
        _write_synthetic_nifti(root / "sub-01" / "func" / "sub-01_task-rest_bold.nii.gz",
                               (20, 20, 12, 10), modality="bold")
        _write_synthetic_nifti(root / "sub-02" / "anat" / "sub-02_T1w.nii.gz", (32, 32, 20))

        class _FR:
            def __init__(self, path):
                self.path = path

        file_records = [
            _FR("sub-01/anat/sub-01_T1w.nii.gz"),
            _FR("sub-01/func/sub-01_task-rest_bold.nii.gz"),
            _FR("sub-02/anat/sub-02_T1w.nii.gz"),
        ]

        report = run_visual_audit("test-dataset", file_records, root, max_files=10)

    require_type(report, VisualAuditReport, "run_visual_audit")
    require(report.n_files_inspected == 3, f"Expected 3 inspected, got {report.n_files_inspected}")
    require(report.n_failed == 0, f"Expected 0 failures, got {report.n_failed}")

    # per_suffix_counts
    suf = report.per_suffix_counts
    require("T1w" in suf, f"T1w missing from per_suffix_counts: {suf}")
    require("bold" in suf, f"bold missing from per_suffix_counts: {suf}")
    require(suf["T1w"] == 2, f"Expected T1w=2, got {suf['T1w']}")
    print_kv("per_suffix_counts", suf)

    # per_subject_counts
    sub_counts = report.per_subject_counts
    require("01" in sub_counts and "02" in sub_counts, f"subjects: {sub_counts}")
    print_kv("per_subject_counts", sub_counts)

    # per_datatype_counts
    dt = report.per_datatype_counts
    require("anat" in dt and "func" in dt, f"datatypes: {dt}")
    print_kv("per_datatype_counts", dt)

    # warning_summary
    ws = report.warning_summary()
    require(isinstance(ws, dict), "warning_summary must return dict")
    for k in ("by_code", "by_severity", "failed_renders", "total_warnings"):
        require(k in ws, f"warning_summary missing key: {k}")

    # coverage_matrix
    cm = report.coverage_matrix()
    require("subjects" in cm and "suffixes" in cm and "cells" in cm, "coverage_matrix keys")
    require(len(cm["subjects"]) == 2, f"Expected 2 subjects, got {cm['subjects']}")

    # HTML output
    html = report.to_html()
    require(len(html) > 2000, "HTML too short")
    require("COVERAGE MATRIX" in html, "HTML missing coverage matrix section")
    require("BY SUFFIX" in html, "HTML missing BY SUFFIX section")
    require("test-dataset" in html, "HTML missing dataset ID")

    # JSON export
    js = report.to_json()
    data = json.loads(js)
    for k in ("dataset_id", "n_files_inspected", "coverage_matrix", "per_suffix_counts", "entries"):
        require(k in data, f"JSON missing key: {k}")
    require(len(data["entries"]) == 3, f"JSON entries count wrong: {len(data['entries'])}")

    # visual_manifest_json write
    with tempfile.TemporaryDirectory() as tmp2:
        manifest_path = Path(tmp2) / "visual_manifest.json"
        out = report.visual_manifest_json(manifest_path)
        require(out.exists(), "visual_manifest.json not written")
        loaded = json.loads(out.read_text())
        require(loaded["dataset_id"] == "test-dataset", "manifest dataset_id wrong")
    print_kv("visual_manifest.json", "OK — written and validated")


def test_07_fmri_summary_figure():
    banner("7 — fMRI QC summary: 6-panel figure, correct traces and titles")
    nib = _try_import("nibabel")
    plotly = _try_import("plotly")
    if nib is None or plotly is None:
        print("  SKIP: nibabel or plotly not installed")
        return

    from qortex.visualize.volume import VolumeViewer

    with tempfile.TemporaryDirectory() as tmp:
        # 4D BOLD: 20³ × 30 timepoints — small enough to be fast
        p = _write_synthetic_nifti(
            Path(tmp) / "sub-01_task-rest_bold.nii.gz",
            (20, 20, 12, 30),
            modality="bold",
        )
        viewer = VolumeViewer(p, modality="fmri")
        require(viewer.n_volumes == 30, f"n_volumes={viewer.n_volumes}")

        fig = viewer.fmri_summary(max_frames=20, title="fMRI QC")

    require(hasattr(fig, "data"), "fmri_summary must return a Figure")

    # 6 panels: 4 Heatmaps + 1 Scatter (global signal) + 1 Heatmap (framewise)
    from plotly.graph_objects import Heatmap, Scatter
    heatmaps = [t for t in fig.data if isinstance(t, Heatmap)]
    scatters = [t for t in fig.data if isinstance(t, Scatter)]
    require(len(heatmaps) >= 4, f"Expected ≥4 Heatmaps, got {len(heatmaps)}")
    require(len(scatters) >= 1, f"Expected ≥1 Scatter (global signal), got {len(scatters)}")

    ann_texts = [a.text or "" for a in (fig.layout.annotations or [])]
    require(any("Mean EPI" in t for t in ann_texts), "Missing 'Mean EPI' panel")
    require(any("tSNR" in t for t in ann_texts), "Missing 'tSNR' panel")
    require(any("Global Signal" in t for t in ann_texts), "Missing 'Global Signal' panel")

    require("fMRI QC" in (fig.layout.title.text or ""), "Title override not set")

    print_kv("heatmap_traces", len(heatmaps))
    print_kv("scatter_traces", len(scatters))
    print_kv("subplot_titles", [t for t in ann_texts if t][:6])


def test_08_fmri_streaming_stats():
    banner("8 — fMRI streaming: mean/std/tSNR/global_signal/framewise on synthetic 4D")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize.volume import _LazyNIfTI

    with tempfile.TemporaryDirectory() as tmp:
        shape4d = (24, 24, 16, 40)
        p = _write_synthetic_nifti(
            Path(tmp) / "bold.nii.gz", shape4d, modality="bold"
        )
        lazy = _LazyNIfTI(p)
        nx, ny, nz, nt = shape4d

        # mean_volume
        mean_vol = lazy.mean_volume(max_frames=20)
        require(mean_vol.shape == (nx, ny, nz), f"mean_volume shape: {mean_vol.shape}")
        require(mean_vol.dtype == np.float32, f"mean_volume dtype: {mean_vol.dtype}")
        require(np.isfinite(mean_vol).all(), "mean_volume has NaN/Inf")

        # std_volume — Welford streaming
        std_vol = lazy.std_volume(max_frames=20)
        require(std_vol.shape == (nx, ny, nz), f"std_volume shape: {std_vol.shape}")
        require((std_vol >= 0).all(), "std_volume must be non-negative")

        # tsnr_volume — mean/std
        tsnr_vol = lazy.tsnr_volume(max_frames=20)
        require(tsnr_vol.shape == (nx, ny, nz), f"tsnr_volume shape: {tsnr_vol.shape}")
        # tSNR clipped to [0, 500]
        require(float(tsnr_vol.max()) <= 500.01, f"tSNR > 500: {tsnr_vol.max()}")
        require(float(tsnr_vol.min()) >= 0, f"tSNR < 0: {tsnr_vol.min()}")

        # global_signal — brain-masked mean per timepoint
        gsig = lazy.global_signal()
        require(gsig.ndim == 1, f"global_signal must be 1D, got {gsig.ndim}D")
        require(gsig.shape[0] == nt, f"global_signal length {gsig.shape[0]} != {nt}")
        require(np.isfinite(gsig).all(), "global_signal has NaN/Inf")

        # framewise_intensity_map — (n_slices, n_frames) matrix
        matrix, t_idxs = lazy.framewise_intensity_map(n_frames=20, n_slices=8)
        require(matrix.ndim == 2, f"framewise_intensity_map must be 2D, got {matrix.ndim}D")
        require(matrix.shape[0] == 8, f"rows != n_slices: {matrix.shape[0]}")
        require(matrix.shape[1] <= 20, f"cols > n_frames: {matrix.shape[1]}")
        require(np.isfinite(matrix).all(), "framewise matrix has NaN/Inf")

        print_kv("mean_vol_range", f"[{mean_vol.min():.0f}, {mean_vol.max():.0f}]")
        print_kv("tsnr_range", f"[{tsnr_vol[tsnr_vol>0].min():.1f}, {tsnr_vol.max():.1f}]")
        print_kv("global_signal_range", f"[{gsig.min():.0f}, {gsig.max():.0f}]")
        print_kv("framewise_shape", matrix.shape)


def test_09_dwi_summary_figure():
    banner("9 — DWI summary: 4-panel figure from synthetic DWI + bval/bvec")
    nib = _try_import("nibabel")
    plotly = _try_import("plotly")
    if nib is None or plotly is None:
        print("  SKIP: nibabel or plotly not installed")
        return

    from qortex.visualize.dwi import DWIViewer

    # Shell structure: 1 b0 + 6 b1000 + 6 b2000
    n_vols = 13
    bvals = [0] + [1000] * 6 + [2000] * 6

    with tempfile.TemporaryDirectory() as tmp:
        dwi_path = _write_synthetic_nifti(
            Path(tmp) / "sub-01_dwi.nii.gz", (24, 24, 16, n_vols), modality="dwi"
        )
        bval_path = _write_bval(Path(tmp) / "sub-01_dwi.bval", bvals)
        bvec_path = _write_bvec(Path(tmp) / "sub-01_dwi.bvec", n_vols)

        viewer = DWIViewer(dwi_path, bval_path=bval_path, bvec_path=bvec_path)

        # Properties
        b0_idxs = viewer.b0_indices
        require(b0_idxs == [0], f"b0_indices wrong: {b0_idxs}")
        require(len(viewer.high_b_indices) == 6, f"high_b_indices: {viewer.high_b_indices}")

        shells = viewer.shells
        require(0 in shells and 1000 in shells and 2000 in shells,
                f"shells dict wrong: {shells}")
        require(shells[0] == 1, f"b0 count wrong: {shells[0]}")
        require(shells[2000] == 6, f"b2000 count wrong: {shells[2000]}")
        print_kv("shells", shells)

        # 4-panel summary figure
        fig = viewer.dwi_summary(title="DWI QC")

    require(hasattr(fig, "data"), "dwi_summary() must return a Figure")
    require(len(fig.data) >= 3, f"Expected ≥3 traces, got {len(fig.data)}")

    from plotly.graph_objects import Heatmap, Bar
    heatmaps = [t for t in fig.data if isinstance(t, Heatmap)]
    bars = [t for t in fig.data if isinstance(t, Bar)]
    require(len(heatmaps) >= 2, f"Expected ≥2 Heatmaps (b0, high-b), got {len(heatmaps)}")
    require(len(bars) >= 1, f"Expected ≥1 Bar (bval histogram), got {len(bars)}")

    ann_texts = [a.text or "" for a in (fig.layout.annotations or [])]
    require(any("b0" in t.lower() or "mean" in t.lower() for t in ann_texts),
            f"Missing b0 panel title. Got: {ann_texts}")
    print_kv("traces", len(fig.data))
    print_kv("heatmaps", len(heatmaps))
    print_kv("bars", len(bars))


def test_10_dwi_bval_bvec_parsing():
    banner("10 — DWI bval/bvec parsing: shells, b0_indices, high_b_indices")

    from qortex.visualize.dwi import _load_bvals, _load_bvecs, _find_companions

    bvals_list = [0, 0, 1000, 1000, 1000, 2000, 2000, 2000, 2000]
    with tempfile.TemporaryDirectory() as tmp:
        bval_path = _write_bval(Path(tmp) / "sub-01_dwi.bval", bvals_list)
        bvec_path = _write_bvec(Path(tmp) / "sub-01_dwi.bvec", len(bvals_list))

        bvals = _load_bvals(bval_path)
        bvecs = _load_bvecs(bvec_path)

    require(bvals.shape == (9,), f"bvals shape: {bvals.shape}")
    require(bvecs.shape == (3, 9), f"bvecs shape: {bvecs.shape}")
    require(float(bvals[0]) < 50, f"First bval should be b0: {bvals[0]}")
    require(float(bvals[2]) > 900, f"Third bval should be b1000: {bvals[2]}")

    # Norms of diffusion directions should be ~1
    norms = np.linalg.norm(bvecs, axis=0)
    require(np.allclose(norms, 1.0, atol=0.01), f"bvec norms not unit: {norms}")

    # _find_companions auto-detection
    with tempfile.TemporaryDirectory() as tmp2:
        stem = "sub-01_dwi"
        bval_auto = _write_bval(Path(tmp2) / f"{stem}.bval", [0, 1000])
        bvec_auto = _write_bvec(Path(tmp2) / f"{stem}.bvec", 2)
        fake_nii = Path(tmp2) / f"{stem}.nii.gz"
        fake_nii.write_bytes(b"")  # placeholder

        found_bval, found_bvec = _find_companions(fake_nii, None, None)
        require(found_bval is not None and found_bval.exists(), "bval not auto-detected")
        require(found_bvec is not None and found_bvec.exists(), "bvec not auto-detected")

    print_kv("bvals", bvals.tolist())
    print_kv("bvec_norms_range", f"[{norms.min():.3f}, {norms.max():.3f}]")
    print_kv("companion_autodetect", "OK")


def test_11_overlay_contour():
    banner("11 — overlay_contour: contour pixels are on mask boundary")

    from qortex.visualize.overlay import _binary_contour_2d

    # Create a 20×20 square mask in the center of a 50×50 image
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[15:35, 15:35] = 1.0  # solid 20×20 square

    contour = _binary_contour_2d(mask)

    # The contour must be a strict subset of the mask boundary
    require(contour.dtype == bool, f"contour dtype: {contour.dtype}")
    require(contour.sum() > 0, "contour must have at least one pixel")

    # Interior pixels should NOT be contour
    require(not contour[20, 20], "Interior pixel [20,20] should not be in contour")
    require(not contour[25, 25], "Interior pixel [25,25] should not be in contour")

    # Border pixels (first row of the square) MUST be in contour
    require(contour[15, 20], "Boundary pixel [15,20] must be in contour")
    require(contour[34, 20], "Boundary pixel [34,20] must be in contour")

    # The contour count should approximate the perimeter: ~4×20 = 80 pixels
    perimeter_approx = float(contour.sum())
    require(60 <= perimeter_approx <= 100,
            f"Contour pixel count {perimeter_approx} unexpected (expected ~80)")

    # Mask pixels NOT in mask must never appear in contour
    require(not contour[10, 10], "Background pixel [10,10] in contour (wrong!)")

    print_kv("contour_pixels", int(contour.sum()))
    print_kv("interior_[20,20]", bool(contour[20, 20]))
    print_kv("boundary_[15,20]", bool(contour[15, 20]))

    # Full overlay_contour on NIfTI
    nib = _try_import("nibabel")
    if nib is None:
        print("  overlay_contour() NIfTI test: SKIP (nibabel not installed)")
        return

    from qortex.visualize.overlay import overlay_contour
    from qortex.visualize._asset import VisualResult

    with tempfile.TemporaryDirectory() as tmp:
        base = _write_synthetic_nifti(Path(tmp) / "T1w.nii.gz", (32, 32, 20))
        mask_p = _write_synthetic_nifti(Path(tmp) / "mask.nii.gz", (32, 32, 20))
        result = overlay_contour(base, mask_p, color=(255, 80, 80))
        require_type(result, VisualResult, "overlay_contour result")
        require(len(result.html) > 1000, "HTML too short")
    print_kv("overlay_contour", "OK")


def test_12_compare_masks_dice():
    banner("12 — compare_masks: Dice≈1 for identical, Dice≈0 for disjoint masks")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize.overlay import compare_masks
    from qortex.visualize._asset import VisualResult

    with tempfile.TemporaryDirectory() as tmp:
        base = _write_synthetic_nifti(Path(tmp) / "T1w.nii.gz", (32, 32, 20))

        # Identical masks → Dice ≈ 1
        mask1 = _write_synthetic_nifti(Path(tmp) / "pred.nii.gz", (32, 32, 20))
        result_same = compare_masks(base, mask1, mask1)
        require_type(result_same, VisualResult, "compare_masks(identical)")
        dice_same = result_same.provenance.get("dice_approx", 0.0)
        require(dice_same > 0.9, f"Dice for identical masks should be ~1, got {dice_same:.3f}")
        print_kv("Dice (identical masks)", f"{dice_same:.3f}")

        # Disjoint masks (left half vs right half of the volume)
        vol_pred = np.zeros((32, 32, 20), dtype=np.float32)
        vol_pred[:16, :, :] = 1.0     # left half
        vol_truth = np.zeros((32, 32, 20), dtype=np.float32)
        vol_truth[16:, :, :] = 1.0    # right half

        import nibabel as nib_mod
        aff = np.eye(4)
        nib_mod.save(nib_mod.Nifti1Image(vol_pred, aff), str(Path(tmp) / "pred2.nii.gz"))
        nib_mod.save(nib_mod.Nifti1Image(vol_truth, aff), str(Path(tmp) / "truth2.nii.gz"))

        result_disjoint = compare_masks(
            base, Path(tmp) / "pred2.nii.gz", Path(tmp) / "truth2.nii.gz",
            allow_affine_mismatch=True,
        )
        dice_disjoint = result_disjoint.provenance.get("dice_approx", 1.0)
        require(dice_disjoint < 0.05,
                f"Dice for disjoint masks should be ~0, got {dice_disjoint:.3f}")
        print_kv("Dice (disjoint masks)", f"{dice_disjoint:.3f}")

        # Verify legend embedded in HTML
        require("Dice" in result_disjoint.html, "Dice score missing from HTML")
        require("TP" in result_disjoint.html or "True Positive" in result_disjoint.html,
                "TP legend missing from HTML")


def test_13_overlay_edges():
    banner("13 — overlay_edges: gradient-magnitude edges on synthetic mask")

    from qortex.visualize.overlay import _binary_contour_2d

    # Gradient edges on a smooth circular mask
    r = 15
    cx, cy = 30, 30
    Y, X = np.ogrid[:60, :60]
    mask = ((X - cx) ** 2 + (Y - cy) ** 2 < r ** 2).astype(np.float32)

    dy, dx = np.gradient(mask)
    grad_mag = np.sqrt(dx ** 2 + dy ** 2)
    thresh = float(grad_mag.max()) * 0.30
    edges = grad_mag > thresh if thresh > 1e-6 else np.zeros_like(grad_mag, dtype=bool)

    require(edges.sum() > 0, "Edge detection found no edge pixels")
    # Edges should be around the circle perimeter
    require(not edges[30, 30], "Center should NOT be an edge")
    require(edges.sum() < mask.sum(), "Edge pixels should be fewer than mask pixels")

    print_kv("edge_pixels", int(edges.sum()))
    print_kv("mask_pixels", int(mask.sum()))

    nib = _try_import("nibabel")
    if nib is None:
        print("  overlay_edges() NIfTI test: SKIP (nibabel not installed)")
        return

    from qortex.visualize.overlay import overlay_edges
    from qortex.visualize._asset import VisualResult

    with tempfile.TemporaryDirectory() as tmp:
        base = _write_synthetic_nifti(Path(tmp) / "T1w.nii.gz", (32, 32, 20))
        mask_p = _write_synthetic_nifti(Path(tmp) / "mask.nii.gz", (32, 32, 20))
        result = overlay_edges(base, mask_p)
        require_type(result, VisualResult, "overlay_edges result")
    print_kv("overlay_edges", "OK")


def test_14_to_json_structure():
    banner("14 — VisualAuditReport.to_json(): all required top-level keys and types")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize._audit import run_visual_audit

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sub-01" / "anat").mkdir(parents=True)
        _write_synthetic_nifti(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz", (24, 24, 16))

        class _FR:
            def __init__(self, path): self.path = path

        report = run_visual_audit("ds-json-test", [_FR("sub-01/anat/sub-01_T1w.nii.gz")],
                                  root, max_files=5)

    js = report.to_json()
    data = json.loads(js)

    required_top = {
        "dataset_id": str,
        "n_files_inspected": int,
        "n_rendered": int,
        "n_failed": int,
        "coverage_matrix": dict,
        "per_suffix_counts": dict,
        "per_subject_counts": dict,
        "per_datatype_counts": dict,
        "warning_summary": dict,
        "entries": list,
    }
    for key, expected_type in required_top.items():
        require(key in data, f"JSON missing key: {key!r}")
        require(isinstance(data[key], expected_type),
                f"JSON[{key!r}] type: expected {expected_type.__name__}, "
                f"got {type(data[key]).__name__}")

    # entries inner structure
    entry = data["entries"][0]
    for k in ("path", "error", "has_thumbnail", "intent", "modality", "shape", "warnings"):
        require(k in entry, f"entry missing key: {k!r}")

    # warning_summary inner structure
    ws = data["warning_summary"]
    for k in ("by_code", "by_severity", "failed_renders", "total_warnings"):
        require(k in ws, f"warning_summary missing key: {k!r}")

    print_kv("top-level keys", list(data.keys()))
    print_kv("entry keys", list(entry.keys()))
    print_kv("warning_summary keys", list(ws.keys()))


def test_15_dicom_phi_hidden():
    banner("15 — DicomSeriesBrowser: patient fields hidden by default")

    pydicom = _try_import("pydicom")
    if pydicom is None:
        print("  SKIP: pydicom not installed")
        return

    # Write a minimal DICOM file with identifiable patient data
    import tempfile
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, UID
    import pydicom.uid as dcm_uid

    with tempfile.TemporaryDirectory() as tmp:
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        ds.file_meta.MediaStorageSOPInstanceUID = dcm_uid.generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_implicit_VR = False
        ds.is_little_endian = True

        ds.PatientName = "Smith^John"
        ds.PatientID = "PAT-0001"
        ds.PatientBirthDate = "19800101"
        ds.PatientSex = "M"
        ds.StudyDate = "20240101"
        ds.Modality = "CT"
        ds.SeriesDescription = "Head CT"
        ds.SeriesInstanceUID = dcm_uid.generate_uid()
        ds.StudyInstanceUID = dcm_uid.generate_uid()
        ds.SOPInstanceUID = dcm_uid.generate_uid()
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        ds.InstanceNumber = 1
        ds.SliceThickness = 5.0
        ds.PixelSpacing = [0.5, 0.5]
        ds.Rows = 4
        ds.Columns = 4
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.zeros((4, 4), dtype=np.int16).tobytes()

        dcm_path = Path(tmp) / "series01" / "slice_001.dcm"
        dcm_path.parent.mkdir(parents=True)
        ds.save_as(str(dcm_path), write_like_original=False)

        from qortex.visualize.dicom import DicomSeriesBrowser
        browser = DicomSeriesBrowser(Path(tmp))
        html = browser.to_html()

    # By default, patient identifiable fields must be absent or redacted
    require(isinstance(html, str) and len(html) > 100, "DicomSeriesBrowser.to_html() empty")

    phi_strings = ["Smith", "PAT-0001", "19800101"]
    for phi in phi_strings:
        require(phi not in html,
                f"PHI field {phi!r} present in HTML — must be redacted by default")

    # Non-PHI metadata should appear
    require("CT" in html or "Head" in html or "Series" in html,
            "Expected series metadata missing from HTML")

    print_kv("html_chars", len(html))
    print_kv("phi_fields_checked", phi_strings)
    print_kv("phi_redacted", "OK")


def test_16_eeg_thumbnail():
    banner("16 — EEG thumbnail: butterfly figure renders without error")
    plotly = _try_import("plotly")
    if plotly is None:
        print("  SKIP: plotly not installed")
        return

    from qortex.visualize.timeseries import TimeSeriesViewer
    import base64

    rng = np.random.RandomState(99)
    eeg = rng.randn(8, 1024).astype(np.float32) * 1e-5
    viewer = TimeSeriesViewer(eeg, sfreq=256.0, modality="eeg")
    fig = viewer.butterfly(tmax=4.0, max_channels=8, show_envelope=False)

    require(hasattr(fig, "data"), "butterfly() must return a plotly Figure")
    require(len(fig.data) > 0, "butterfly figure has no traces")

    # Check HTML export (does not require kaleido)
    import plotly.io as pio
    html_str = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    require(len(html_str) > 500, "butterfly figure HTML too short")
    print_kv("n_traces", len(fig.data))
    print_kv("html_chars", len(html_str))

    # PNG export only when kaleido is available and compatible with Plotly.
    try:
        import kaleido  # noqa: F401
        png_bytes = pio.to_image(fig, format="png", width=400, height=200)
        b64 = base64.b64encode(png_bytes).decode()
        require(len(b64) > 200, "EEG thumbnail base64 too short")
        png_raw = base64.b64decode(b64)
        require(png_raw[:8] == b"\x89PNG\r\n\x1a\n", "EEG thumbnail not a valid PNG")
        print_kv("png_bytes", len(png_raw))
    except (ImportError, ValueError) as exc:
        print(f"  PNG export: SKIP ({type(exc).__name__}: {str(exc).splitlines()[0]})")


def test_17_coverage_matrix():
    banner("17 — coverage_matrix: subject × suffix cross-table structure")

    from qortex.visualize._audit import VisualAuditReport, AuditEntry
    from qortex.visualize._asset import VisualAsset

    def _fake_asset():
        return VisualAsset(path=Path("/tmp/x"), family="nifti", intent="anatomical_volume")

    entries = [
        AuditEntry("sub-01/anat/sub-01_T1w.nii.gz", _fake_asset(), thumbnail_b64="abc"),
        AuditEntry("sub-01/func/sub-01_task-rest_bold.nii.gz", _fake_asset()),
        AuditEntry("sub-02/anat/sub-02_T1w.nii.gz", _fake_asset(), thumbnail_b64="def"),
        AuditEntry("sub-02/dwi/sub-02_dwi.nii.gz", _fake_asset()),
        AuditEntry("sub-03/anat/sub-03_T1w.nii.gz", _fake_asset(), error="nibabel missing"),
    ]
    report = VisualAuditReport(
        dataset_id="ds-coverage-test",
        n_files_inspected=5,
        n_rendered=4,
        n_failed=1,
        entries=entries,
    )

    cm = report.coverage_matrix()
    require("subjects" in cm and "suffixes" in cm and "cells" in cm, "coverage_matrix keys")
    require(sorted(cm["subjects"]) == ["01", "02", "03"], f"subjects: {cm['subjects']}")
    require("T1w" in cm["suffixes"], f"T1w missing from suffixes: {cm['suffixes']}")
    require("bold" in cm["suffixes"], f"bold missing from suffixes: {cm['suffixes']}")

    # sub-01 T1w should be 'ok'
    require(cm["cells"]["01"]["T1w"] == "ok", f"sub-01 T1w: {cm['cells']['01']['T1w']}")
    # sub-02 bold should be 'missing' (no bold for sub-02)
    require(cm["cells"]["02"].get("bold", "missing") == "missing",
            f"sub-02 bold should be missing")
    # sub-03 T1w should be 'error'
    require(cm["cells"]["03"]["T1w"] == "error", f"sub-03 T1w: {cm['cells']['03']['T1w']}")

    print_kv("subjects", cm["subjects"])
    print_kv("suffixes", cm["suffixes"])
    print_kv("sub-01", cm["cells"]["01"])
    print_kv("sub-03", cm["cells"]["03"])


def test_18_select_visual_files():
    banner("18 — select_visual_files: shared file-selection helper")

    from qortex.visualize._audit import select_visual_files

    class _FR:
        def __init__(self, path, subject=None, suffix=None, datatype=None, size=0):
            self.path = path
            self.subject = subject
            self.suffix = suffix
            self.datatype = datatype
            self.size = size

    files = [
        _FR("sub-01/anat/sub-01_T1w.nii.gz",    "01", "T1w",  "anat",  1_000_000),
        _FR("sub-01/func/sub-01_bold.nii.gz",    "01", "bold", "func",  50_000_000),
        _FR("sub-02/anat/sub-02_T1w.nii.gz",     "02", "T1w",  "anat",  1_000_000),
        _FR("sub-02/anat/sub-02_T2w.nii.gz",     "02", "T2w",  "anat",  2_000_000),
        _FR("sub-03/dwi/sub-03_dwi.nii.gz",      "03", "dwi",  "dwi",   200_000_000),
        _FR("sub-03/anat/sub-03_T1w.nii.gz",     "03", "T1w",  "anat",  1_000_000),
        _FR("sub-04/anat/sub-04_T1w.nii.gz",     "04", "T1w",  "anat",  1_000_000),
    ]

    # Filter by suffix
    t1w_only = select_visual_files(files, suffixes=["T1w"])
    require(all(f.suffix == "T1w" for f in t1w_only), "suffix filter failed")
    require(len(t1w_only) == 4, f"Expected 4 T1w files, got {len(t1w_only)}")
    print_kv("T1w filter", len(t1w_only))

    # Filter by subject
    sub01_files = select_visual_files(files, subjects=["01"])
    require(all(f.subject == "01" for f in sub01_files), "subject filter failed")
    require(len(sub01_files) == 2, f"Expected 2 files for sub-01, got {len(sub01_files)}")
    print_kv("sub-01 filter", len(sub01_files))

    # Filter by datatype
    anat_files = select_visual_files(files, datatypes=["anat"])
    require(all(f.datatype == "anat" for f in anat_files), "datatype filter failed")
    print_kv("anat filter", len(anat_files))

    # Filter by max_size_mb
    small_files = select_visual_files(files, max_size_mb=5.0)
    require(all(f.size <= 5 * 1024 * 1024 for f in small_files), "max_size_mb filter failed")
    require(not any(f.suffix == "bold" for f in small_files), "bold (50MB) should be excluded")
    print_kv("small (<5MB) filter", len(small_files))

    # n_per_suffix cap
    capped = select_visual_files(files, suffixes=["T1w"], n_per_suffix=2)
    require(len(capped) <= 2, f"n_per_suffix=2 not respected: {len(capped)} T1w files returned")
    print_kv("n_per_suffix=2 cap", len(capped))


def test_19_artifact_compare_splits():
    banner("19 — Artifact.compare_splits(): VisualAuditReport with split-level structure")
    plotly = _try_import("plotly")
    polars = _try_import("polars")
    if plotly is None or polars is None:
        print("  SKIP: plotly or polars not installed")
        return

    import polars as pl
    from qortex.artifact import Artifact
    from qortex.visualize._audit import VisualAuditReport

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Build a minimal Parquet artifact
        rng = np.random.RandomState(42)
        for split in ("train", "val"):
            split_dir = root / split
            split_dir.mkdir()
            data = {
                "label": [str(i % 3) for i in range(8)],
                "subject": [f"0{(i % 4) + 1}" for i in range(8)],
                "signal": [rng.randn(64).astype(np.float32).tolist() for _ in range(8)],
            }
            df = pl.DataFrame(data)
            df.write_parquet(split_dir / "shard_00.parquet")

        manifest = {
            "artifact_id": "art-compare-test",
            "dataset_id": "ds-test",
            "snapshot": "1.0.0",
            "output_format": "parquet",
            "output_path": str(root),
            "n_samples": 16,
            "n_subjects": 4,
            "splits": {"train": 8, "val": 8},
            "source_files": [],
        }
        (root / "artifact_manifest.json").write_text(json.dumps(manifest))

        art = Artifact(root)

        # compare_splits: renders from train AND val, returns combined report
        report = art.compare_splits(n=4)
        require_type(report, VisualAuditReport, "compare_splits() result")
        require(report.n_files_inspected >= 4, f"Expected ≥4 entries, got {report.n_files_inspected}")

        # HTML should contain both splits
        html = report.to_html()
        require("train" in html.lower() or "val" in html.lower(),
                "HTML should mention split names")
        print_kv("n_entries", report.n_files_inspected)
        print_kv("n_rendered", report.n_rendered)
        print_kv("splits_in_html", "train" in html.lower() and "val" in html.lower())


def test_20_audit_expected_vs_local():
    banner("20 — Visual audit: n_expected / n_local_present in HTML header")
    nib = _try_import("nibabel")
    if nib is None:
        print("  SKIP: nibabel not installed")
        return

    from qortex.visualize._audit import run_visual_audit_with_manifest

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "sub-01" / "anat").mkdir(parents=True)
        (root / "sub-02" / "anat").mkdir(parents=True)

        _write_synthetic_nifti(root / "sub-01" / "anat" / "sub-01_T1w.nii.gz", (24, 24, 16))
        # sub-02 NOT downloaded

        class _FR:
            def __init__(self, path): self.path = path

        manifest_files = [
            _FR("sub-01/anat/sub-01_T1w.nii.gz"),
            _FR("sub-02/anat/sub-02_T1w.nii.gz"),   # expected but absent
            _FR("sub-03/anat/sub-03_T1w.nii.gz"),   # expected but absent
        ]

        report = run_visual_audit_with_manifest(
            "ds-completeness-test",
            manifest_files,
            root,
            max_files=10,
        )

    require(report.n_files_inspected == 1, f"Expected 1 local file, got {report.n_files_inspected}")
    require(hasattr(report, "n_expected"), "report missing n_expected attribute")
    require(report.n_expected == 3, f"n_expected should be 3, got {report.n_expected}")
    require(report.n_local_present == 1, f"n_local_present should be 1, got {report.n_local_present}")
    require(report.n_missing_local == 2, f"n_missing_local should be 2, got {report.n_missing_local}")

    html = report.to_html()
    require("Expected" in html or "expected" in html, "HTML should show expected count")
    require("Missing" in html or "missing" in html, "HTML should show missing count")

    print_kv("n_expected", report.n_expected)
    print_kv("n_local_present", report.n_local_present)
    print_kv("n_missing_local", report.n_missing_local)


def test_21_dwi_contact_sheet():
    banner("21 — DWI contact_sheet(): shell-specific montage with slice panels")
    nib = _try_import("nibabel")
    plotly = _try_import("plotly")
    if nib is None or plotly is None:
        print("  SKIP: nibabel or plotly not installed")
        return

    from qortex.visualize.dwi import DWIViewer

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dwi_path = _write_synthetic_nifti(
            root / "sub-01_dir-AP_dwi.nii.gz",
            (24, 24, 12, 8),
            modality="dwi",
        )
        bvals = [0, 0, 1000, 1000, 1000, 2000, 2000, 2000]
        bval_path = _write_bval(root / "sub-01_dir-AP_dwi.bval", bvals)
        bvec_path = _write_bvec(root / "sub-01_dir-AP_dwi.bvec", len(bvals))

        viewer = DWIViewer(dwi_path, bval_path=bval_path, bvec_path=bvec_path)
        fig = viewer.contact_sheet(shell="high", n_slices=6, n_cols=3)

    require(hasattr(fig, "data"), "contact_sheet() must return a Plotly figure")
    require(len(fig.data) == 6, f"Expected 6 slice panels, got {len(fig.data)}")
    require("high-b" in (fig.layout.title.text or ""), "contact sheet title should mention selected shell")
    print_kv("shells", viewer.shells)
    print_kv("contact_sheet_traces", len(fig.data))
    print_kv("title", fig.layout.title.text)


def test_22_timeseries_topomap():
    banner("22 — TimeSeriesViewer.topomap(): interpolated scalp map plus channel markers")
    plotly = _try_import("plotly")
    if plotly is None:
        print("  SKIP: plotly not installed")
        return

    from qortex.visualize.timeseries import TimeSeriesViewer

    sfreq = 100.0
    t = np.arange(0, 2.0, 1.0 / sfreq)
    data = np.column_stack([
        np.sin(2 * np.pi * 8 * t),
        np.cos(2 * np.pi * 8 * t),
        0.5 * np.sin(2 * np.pi * 12 * t),
        0.5 * np.cos(2 * np.pi * 12 * t),
    ]).astype(np.float32)
    channels = ["Fz", "Cz", "Pz", "Oz"]

    viewer = TimeSeriesViewer(data, sfreq=sfreq, ch_names=channels)
    fig = viewer.topomap(t=0.5, title="Synthetic topography")

    trace_types = [trace.type for trace in fig.data]
    require("heatmap" in trace_types, f"Topomap missing heatmap trace: {trace_types}")
    require("scatter" in trace_types, f"Topomap missing sensor marker trace: {trace_types}")
    require("Synthetic topography" in (fig.layout.title.text or ""), "Topomap title not applied")
    print_kv("trace_types", trace_types)
    print_kv("channels", channels)


def test_23_fmri_summary_events_confounds():
    banner("23 — fMRI summary: events.tsv markers and confounds traces")
    nib = _try_import("nibabel")
    plotly = _try_import("plotly")
    if nib is None or plotly is None:
        print("  SKIP: nibabel or plotly not installed")
        return

    from qortex.visualize.volume import VolumeViewer

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        bold_path = _write_synthetic_nifti(
            root / "sub-01_task-rest_bold.nii.gz",
            (20, 20, 10, 16),
            modality="bold",
        )
        (root / "sub-01_task-rest_events.tsv").write_text(
            "onset\tduration\ttrial_type\n"
            "4.0\t1.0\tstim\n"
            "12.0\t1.0\tstim\n",
            encoding="utf-8",
        )
        (root / "sub-01_task-rest_desc-confounds_timeseries.tsv").write_text(
            "framewise_displacement\tdvars\tstd_dvars\n"
            + "\n".join(f"{0.01 * i:.4f}\t{2.0 + i:.3f}\t{0.1 * i:.3f}" for i in range(16))
            + "\n",
            encoding="utf-8",
        )

        fig = VolumeViewer(bold_path, modality="fmri").fmri_summary(title="BOLD QC")

    names = [getattr(trace, "name", "") for trace in fig.data]
    require("Global signal" in names, "Global signal trace missing")
    require("FD" in names, "Framewise displacement trace missing")
    require("DVARS" in names, "DVARS trace missing")
    require(any(name == "stim" for name in names), f"Event marker trace missing: {names}")
    print_kv("trace_names", [name for name in names if name])
    print_kv("layout_height", fig.layout.height)


def test_24_pathless_nifti_png_error():
    banner("24 — VisualResult.to_png(): clear failure for pathless NIfTI fallback")
    from qortex.visualize._asset import MODE_STATIC, VisualAsset, VisualPlan, VisualResult

    asset = VisualAsset(path=Path("<array>"), family="nifti", modality="mri", shape=(8, 8, 8), ndim=3)
    plan = VisualPlan(
        asset=asset,
        mode=MODE_STATIC,
        backend="pure_python",
        views=["orthogonal"],
        window_preset="auto",
        colormap="gray",
        overlay_path=None,
        requires_companions=[],
    )
    result = VisualResult(asset=asset, plan=plan)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            result.to_png(Path(tmp) / "pathless.png")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected ValueError for pathless NIfTI PNG fallback")

    require("no filesystem path" in message, f"Unexpected error message: {message}")
    print_kv("error", message)


def test_25_visualization_edge_cases():
    banner("25 — Visualization edge cases: BIDS suffix, aspect resize, empty-mask Dice")
    import base64
    import struct

    from qortex.visualize._dispatch import _bids_suffix
    from qortex.visualize._html import array_to_b64png
    from qortex.visualize.overlay import compare_masks

    suffix = _bids_suffix(Path("sub-01_ses-02_task-rest_run-03_desc-preproc_bold.nii.gz"))
    require(suffix == "bold", f"Expected robust BIDS suffix 'bold', got {suffix!r}")

    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    png_b64 = array_to_b64png(arr, float(arr.min()), float(arr.max()), aspect=(2.0, 1.0))
    png = base64.b64decode(png_b64)
    width, height = struct.unpack(">II", png[16:24])
    require(width == 4 and height == 6, f"Aspect-corrected PNG dimensions wrong: {width}x{height}")

    base = np.ones((12, 12, 8), dtype=np.float32)
    empty = np.zeros_like(base)
    result = compare_masks(base, empty, empty, title="Empty mask comparison")
    dice = result.provenance.get("dice_approx")
    require(dice == 1.0, f"Empty-vs-empty Dice should be 1.0, got {dice}")
    print_kv("bids_suffix", suffix)
    print_kv("aspect_png_size", f"{width}x{height}")
    print_kv("empty_mask_dice", dice)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    test_01_lazy_nifti_proxy()
    test_02_thumbnail_no_full_load()
    test_03_static_ortho()
    test_04_overlay_affine_mismatch_raises()
    test_05_overlay_resample_works()
    test_06_visual_audit_report()
    test_07_fmri_summary_figure()
    test_08_fmri_streaming_stats()
    test_09_dwi_summary_figure()
    test_10_dwi_bval_bvec_parsing()
    test_11_overlay_contour()
    test_12_compare_masks_dice()
    test_13_overlay_edges()
    test_14_to_json_structure()
    test_15_dicom_phi_hidden()
    test_16_eeg_thumbnail()
    test_17_coverage_matrix()
    test_18_select_visual_files()
    test_19_artifact_compare_splits()
    test_20_audit_expected_vs_local()
    test_21_dwi_contact_sheet()
    test_22_timeseries_topomap()
    test_23_fmri_summary_events_confounds()
    test_24_pathless_nifti_png_error()
    test_25_visualization_edge_cases()
    passed("project_20_visualization_advanced")


if __name__ == "__main__":
    main()
