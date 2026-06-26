"""project_19_visualization

Tests the qortex.visualize module end-to-end:
  - auto-dispatch via qortex.visualize.open()
  - VolumeViewer: NIfTI slice rendering, windowing presets, interactive HTML
  - TimeSeriesViewer: butterfly/PSD/spectrogram from synthetic data
  - pure-Python PNG encoder sanity check
  - self-contained HTML output verification

All tests use synthetic data (no network, no nibabel/MNE required for core
HTML generation tests; MNE/nibabel tests are skipped when not installed).
"""

from __future__ import annotations

import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, require, require_gt, require_in, require_type, passed,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_mri(shape=(64, 64, 40)) -> np.ndarray:
    """Synthetic MRI-like volume: bright sphere in dark background."""
    vol = np.zeros(shape, dtype=np.float32)
    cx, cy, cz = [s // 2 for s in shape]
    r = min(shape) // 3
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < r ** 2:
                    vol[x, y, z] = 800 + 200 * np.random.rand()
    return vol


def _make_synthetic_bold(shape=(32, 32, 20, 50)) -> np.ndarray:
    """4D BOLD-like volume: resting-state drift + noise."""
    rng = np.random.RandomState(42)
    vol = rng.randn(*shape).astype(np.float32) * 50 + 1000
    # Add a slow drift in one region
    vol[10:20, 10:20, 8:12, :] += np.sin(np.linspace(0, 4 * np.pi, shape[3])) * 80
    return vol


def _make_synthetic_eeg(n_ch=32, n_samples=2048, sfreq=256.0) -> np.ndarray:
    """Synthetic EEG: alpha band (10 Hz) + noise."""
    rng = np.random.RandomState(7)
    t = np.arange(n_samples) / sfreq
    eeg = rng.randn(n_ch, n_samples).astype(np.float32) * 1e-5
    # Add 10 Hz alpha to all channels
    eeg += np.sin(2 * np.pi * 10 * t)[np.newaxis, :] * 5e-6
    return eeg


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_import():
    banner("1 — module import and auto-dispatch detection")

    import qortex
    require(hasattr(qortex, "visualize"), "qortex.visualize not attached")

    from qortex import visualize
    require(hasattr(visualize, "open"), "visualize.open() missing")
    require(hasattr(visualize, "volume"), "visualize.volume() missing")
    require(hasattr(visualize, "timeseries"), "visualize.timeseries() missing")

    from qortex.visualize.volume import VolumeViewer
    from qortex.visualize.timeseries import TimeSeriesViewer
    print_kv("imports", "OK")


def test_colormap_luts():
    banner("2 — colormap LUTs (pure numpy)")

    from qortex.visualize._colors import get_lut, apply_window, auto_window

    for name in ("gray", "hot", "plasma", "RdBu_r"):
        lut = get_lut(name)
        require(lut.shape == (256, 3), f"LUT {name} has wrong shape {lut.shape}")
        require(lut.dtype == np.uint8, f"LUT {name} dtype is {lut.dtype}")
    print_kv("LUTs", "gray, hot, plasma, RdBu_r — all (256,3) uint8")

    vol = np.random.rand(10, 10, 10).astype(np.float32) * 1000
    vmin, vmax = auto_window(vol, "mri")
    require(vmin < vmax, f"auto_window returned vmin={vmin} >= vmax={vmax}")
    normed = apply_window(vol, vmin, vmax)
    require(normed.min() >= 0.0 and normed.max() <= 1.0, "apply_window out of [0,1]")
    print_kv("auto_window (mri)", {"vmin": f"{vmin:.1f}", "vmax": f"{vmax:.1f}"})

    ct_vol = np.random.rand(10, 10, 10).astype(np.float32) * 2000 - 1000
    ct_vmin, ct_vmax = auto_window(ct_vol, "ct")
    print_kv("auto_window (ct)", {"vmin": f"{ct_vmin:.1f}", "vmax": f"{ct_vmax:.1f}"})


def test_png_encoder():
    banner("3 — pure-Python PNG encoder")

    from qortex.visualize._html import array_to_b64png

    import base64

    arr = np.zeros((32, 32), dtype=np.float32)
    arr[8:24, 8:24] = 1.0  # white square in dark background

    b64 = array_to_b64png(arr, 0.0, 1.0, "gray")
    require(isinstance(b64, str), "array_to_b64png must return str")
    require(len(b64) > 100, "base64 PNG too short")

    png_bytes = base64.b64decode(b64)
    # PNG magic: 8 bytes
    require(png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "PNG magic header missing")
    print_kv("PNG encoder", {"b64_len": len(b64), "bytes": len(png_bytes)})

    # Test with different colormaps
    for cmap in ("hot", "plasma"):
        b = array_to_b64png(arr, 0.0, 1.0, cmap)
        require(len(b) > 50, f"colormap {cmap} produced empty PNG")
    print_kv("colormaps", "hot, plasma — OK")


def test_volume_viewer_synthetic():
    banner("4 — VolumeViewer with synthetic MRI volume")

    from qortex.visualize.volume import VolumeViewer

    vol = _make_synthetic_mri((48, 48, 30))
    # Wrap in a nibabel-like duck-typed object
    class _FakeNib:
        affine = np.diag([1.0, 1.0, 3.0, 1.0])
        def get_fdata(self, dtype=None):
            return vol.copy()

    viewer = VolumeViewer(_FakeNib(), modality="mri")
    require_type(viewer, VolumeViewer, "VolumeViewer")
    require(viewer.shape == (48, 48, 30), f"Wrong shape {viewer.shape}")
    require(viewer.n_volumes == 1, "Expected 1 volume for 3D data")
    vox = viewer.voxel_sizes
    require(len(vox) == 3, "voxel_sizes must have 3 elements")
    print_kv("viewer", repr(viewer))


def test_volume_viewer_4d():
    banner("5 — VolumeViewer with 4D BOLD (temporal mean + slices)")

    from qortex.visualize.volume import VolumeViewer

    bold = _make_synthetic_bold((32, 32, 20, 50))

    class _FakeNib4D:
        affine = np.diag([2.0, 2.0, 3.0, 1.0])
        def get_fdata(self, dtype=None):
            return bold.copy()

    viewer = VolumeViewer(_FakeNib4D(), modality="fmri")
    require(viewer.n_volumes == 50, f"Expected 50 volumes, got {viewer.n_volumes}")
    vol3d = viewer._vol3d()
    require(vol3d.shape == (32, 32, 20), f"Mean volume shape wrong: {vol3d.shape}")
    print_kv("4D BOLD viewer", repr(viewer))


def test_volume_html_output():
    banner("6 — VolumeViewer interactive HTML (synthetic, no nibabel)")

    from qortex.visualize.volume import VolumeViewer
    from qortex.visualize._html import render_axis_slices, build_interactive_html

    vol = np.random.rand(32, 32, 20).astype(np.float32)

    class _FakeNib:
        affine = np.eye(4)
        def get_fdata(self, dtype=None):
            return vol.copy()

    viewer = VolumeViewer(_FakeNib(), modality="mri")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_viewer.html"
        html = viewer.interactive_html(out_path, title="Synthetic MRI", max_slices_per_axis=10)

        require(isinstance(html, str), "interactive_html must return str")
        require(len(html) > 1000, f"HTML too short: {len(html)} chars")
        require("<html" in html.lower(), "HTML missing <html> tag")
        require("base64" in html or "data:image" in html, "HTML has no embedded images")
        require(out_path.exists(), "HTML file was not written to disk")
        file_size = out_path.stat().st_size

    print_kv("HTML output", {
        "html_chars": len(html),
        "file_bytes": file_size,
        "has_title": "Synthetic MRI" in html,
    })
    require("Synthetic MRI" in html, "HTML missing title")


def test_timeseries_viewer():
    banner("7 — TimeSeriesViewer with synthetic EEG array")

    from qortex.visualize.timeseries import TimeSeriesViewer

    eeg = _make_synthetic_eeg(n_ch=16, n_samples=1024, sfreq=256.0)
    viewer = TimeSeriesViewer(eeg, sfreq=256.0, modality="eeg")

    require(viewer.n_channels == 16, f"Wrong n_channels: {viewer.n_channels}")
    require(viewer.n_samples == 1024, f"Wrong n_samples: {viewer.n_samples}")
    require(abs(viewer.sfreq - 256.0) < 0.01, f"Wrong sfreq: {viewer.sfreq}")
    require(abs(viewer.duration_s - 4.0) < 0.1, f"Wrong duration: {viewer.duration_s}")
    print_kv("timeseries viewer", repr(viewer))


def test_timeseries_psd():
    banner("8 — TimeSeriesViewer PSD (Welch, pure numpy)")

    from qortex.visualize.timeseries import _welch_psd

    sfreq = 256.0
    n = 2048
    t = np.arange(n) / sfreq
    signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)

    freqs, psd = _welch_psd(signal, sfreq, nperseg=512)
    require(freqs.shape == psd.shape, "freqs and psd must have same shape")
    require(freqs.max() <= sfreq / 2 + 1, "Frequencies exceed Nyquist")
    require(psd.min() >= 0, "PSD must be non-negative")

    # 10 Hz peak should dominate
    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    require(abs(peak_freq - 10.0) < 2.0, f"PSD peak at {peak_freq:.1f} Hz, expected ~10 Hz")
    print_kv("PSD peak", f"{peak_freq:.2f} Hz (expected 10 Hz)")


def test_timeseries_spectrogram():
    banner("9 — TimeSeriesViewer STFT spectrogram (pure numpy)")

    from qortex.visualize.timeseries import _stft

    sfreq = 256.0
    n = 2048
    t = np.arange(n) / sfreq
    # Chirp: frequency increases over time
    signal = np.sin(2 * np.pi * (5 + 10 * t / t[-1]) * t).astype(np.float32)

    freqs, t_arr, power_db = _stft(signal, sfreq, nperseg=128)
    require(freqs.ndim == 1, "freqs must be 1D")
    require(t_arr.ndim == 1, "t_arr must be 1D")
    require(power_db.ndim == 2, "power_db must be 2D")
    require(power_db.shape[0] == len(freqs), "power_db rows must match freqs")
    require(power_db.shape[1] == len(t_arr), "power_db cols must match t_arr")
    print_kv("STFT", {
        "n_freqs": len(freqs),
        "n_timepoints": len(t_arr),
        "power_db_range": f"[{power_db.min():.1f}, {power_db.max():.1f}]",
    })


def test_timeseries_html():
    banner("10 — TimeSeriesViewer dashboard HTML")

    from qortex.visualize.timeseries import TimeSeriesViewer

    eeg = _make_synthetic_eeg(n_ch=8, n_samples=512, sfreq=256.0)

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("  SKIP: plotly not installed")
        return

    viewer = TimeSeriesViewer(eeg, sfreq=256.0, modality="eeg")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = viewer.to_html(Path(tmpdir) / "eeg_dashboard.html", title="EEG Test")
        require(out_path.exists(), "Dashboard HTML not written")
        html = out_path.read_text()
        require(len(html) > 500, "Dashboard HTML too short")
        require("EEG Test" in html, "Dashboard missing title")
    print_kv("EEG dashboard", {"chars": len(html), "has_title": True})


def test_auto_dispatch_numpy():
    banner("11 — auto-dispatch: inspect + VisualAsset language")

    from qortex import visualize
    from qortex.visualize._asset import (
        VisualAsset, VisualPlan, VisualResult, INTENT_BOLD, INTENT_RAW_SIGNAL
    )
    import numpy as np

    # inspect a 3D volume
    vol = _make_synthetic_mri((32, 32, 20))
    asset = visualize.inspect(vol)
    require_type(asset, VisualAsset, "inspect(3d array) → VisualAsset")
    require(asset.ndim == 3, f"ndim {asset.ndim} != 3")
    print_kv("inspect(3d array)", {"intent": asset.intent, "ndim": asset.ndim})

    # inspect a 4D BOLD array
    bold = _make_synthetic_bold((16, 16, 10, 20))
    asset4d = visualize.inspect(bold)
    require(asset4d.intent == INTENT_BOLD, f"4D intent={asset4d.intent!r}")
    require(asset4d.n_timepoints == 20, f"n_timepoints={asset4d.n_timepoints}")
    print_kv("inspect(4d bold)", {"intent": asset4d.intent, "n_timepoints": asset4d.n_timepoints})

    # plan() derives rendering decision
    plan = asset.plan()
    require_type(plan, VisualPlan, "asset.plan() → VisualPlan")
    require(plan.backend, "plan.backend is empty")
    require(plan.views, "plan.views is empty")
    print_kv("VisualPlan", {"mode": plan.mode, "backend": plan.backend, "views": plan.views})

    # asset summary
    summary = asset.summary()
    require(isinstance(summary, str) and len(summary) > 30, "summary() too short")
    print_kv("asset.summary()", summary.split("\n")[0])


def test_ct_windowing_presets():
    banner("12 — CT windowing presets (HU)")

    from qortex.visualize._colors import CT_PRESETS, auto_window

    for preset_name, preset in CT_PRESETS.items():
        require(preset.center is not None, f"CT preset {preset_name} missing center")
        require(preset.width is not None, f"CT preset {preset_name} missing width")
        vmin, vmax = preset.vmin, preset.vmax
        require(vmin < vmax, f"CT preset {preset_name} vmin >= vmax")

    print_kv("CT presets", list(CT_PRESETS.keys()))

    # Verify brain preset values (clinical standard: c=40, w=80)
    brain = CT_PRESETS["brain"]
    require(brain.center == 40, f"brain center={brain.center} (expected 40)")
    require(brain.width == 80, f"brain width={brain.width} (expected 80)")
    require(brain.vmin == 0.0, f"brain vmin={brain.vmin} (expected 0.0)")
    require(brain.vmax == 80.0, f"brain vmax={brain.vmax} (expected 80.0)")
    print_kv("brain preset", f"c={brain.center} w={brain.width} → [{brain.vmin}, {brain.vmax}] HU")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    test_import()
    test_colormap_luts()
    test_png_encoder()
    test_volume_viewer_synthetic()
    test_volume_viewer_4d()
    test_volume_html_output()
    test_timeseries_viewer()
    test_timeseries_psd()
    test_timeseries_spectrogram()
    test_timeseries_html()
    test_auto_dispatch_numpy()
    test_ct_windowing_presets()
    passed("project_19_visualization")


if __name__ == "__main__":
    main()
