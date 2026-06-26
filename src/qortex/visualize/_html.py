"""HTML generation utilities for the Qortex visualizer.

Provides:
  - Pure-Python base64 PNG encoder (no Pillow/matplotlib required)
  - Array-to-image conversion with colormap LUT
  - Slice-renderer for pre-encoding all slices of a volume axis
  - HTML template builder for the interactive orthogonal viewer
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any

import numpy as np

from qortex.visualize._colors import apply_window, get_lut


# ── Pure-Python PNG encoder ───────────────────────────────────────────────────

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def _rgb_to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode (H, W, 3) uint8 array to PNG bytes without any external dependency."""
    H, W = rgb.shape[:2]
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=2 (RGB)
    ihdr = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    # IDAT: prepend filter byte 0 (None) to each row
    raw = bytearray()
    for row in rgb:
        raw.append(0)
        raw.extend(row.tobytes())
    idat = _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=1))
    iend = _png_chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def array_to_b64png(
    arr2d: np.ndarray,
    vmin: float,
    vmax: float,
    colormap: str = "gray",
    *,
    flip_ud: bool = True,
    aspect: tuple[float, float] | None = None,
) -> str:
    """Encode a 2D float array to a base64 PNG string.

    Parameters
    ----------
    arr2d:
        Shape (H, W). Values are windowed to [vmin, vmax].
    vmin, vmax:
        Intensity window.
    colormap:
        Named colormap: "gray", "hot", "plasma", "RdBu_r".
    flip_ud:
        Flip vertically so image y-axis matches anatomical superior-up convention.
    aspect:
        (yx_ratio, ) — if the voxels are anisotropic, repeat rows/cols to
        correct the aspect ratio. Pass (row_mm, col_mm).
    """
    normed = apply_window(arr2d.astype(np.float32), vmin, vmax)
    lut = get_lut(colormap)

    if flip_ud:
        normed = normed[::-1, :]

    if aspect is not None:
        row_mm, col_mm = aspect
        if row_mm > 0 and col_mm > 0 and abs(row_mm - col_mm) > 0.05:
            scale = max(1, round(row_mm / col_mm))
            if row_mm > col_mm:
                normed = np.repeat(normed, scale, axis=0)
            else:
                normed = np.repeat(normed, scale, axis=1)

    indices = (normed * 255).astype(np.uint8)
    rgb = lut[indices]  # (H, W, 3)
    return base64.b64encode(_rgb_to_png_bytes(rgb)).decode("ascii")


def render_axis_slices(
    vol3d: np.ndarray,
    axis: int,
    vmin: float,
    vmax: float,
    colormap: str = "gray",
    *,
    voxel_sizes: tuple[float, float, float] | None = None,
    max_slices: int = 200,
) -> list[str]:
    """Render all slices along one axis as base64 PNGs.

    Returns a list of base64 PNG strings, one per slice position.
    For large volumes (> max_slices), linearly sub-samples.
    """
    n = vol3d.shape[axis]
    if n > max_slices:
        indices = np.round(np.linspace(0, n - 1, max_slices)).astype(int).tolist()
    else:
        indices = list(range(n))

    results: list[str] = []

    # Determine voxel aspect for the displayed slice
    if voxel_sizes:
        vx, vy, vz = voxel_sizes
        if axis == 0:      # sagittal: display axes are y, z
            aspect = (vy, vz)
        elif axis == 1:    # coronal: display axes are x, z
            aspect = (vx, vz)
        else:              # axial: display axes are x, y
            aspect = (vx, vy)
    else:
        aspect = None

    for i in indices:
        slc = np.take(vol3d, i, axis=axis)
        results.append(array_to_b64png(slc, vmin, vmax, colormap, aspect=aspect))

    return results


# ── Overlay blending ──────────────────────────────────────────────────────────

def blend_overlay(
    base_rgb: np.ndarray,
    overlay_arr: np.ndarray,
    threshold: float,
    vmin: float,
    vmax: float,
    colormap: str = "hot",
    alpha: float = 0.6,
) -> np.ndarray:
    """Blend a statistical map overlay onto a grayscale background.

    Parameters
    ----------
    base_rgb: (H, W, 3) uint8
    overlay_arr: (H, W) float — the stat map slice
    threshold: values below this absolute value are transparent
    """
    mask = np.abs(overlay_arr) >= threshold
    if not mask.any():
        return base_rgb.copy()

    normed = apply_window(overlay_arr, vmin, vmax)
    lut = get_lut(colormap)
    ov_rgb = lut[(normed * 255).astype(np.uint8)]

    result = base_rgb.copy().astype(np.float32)
    result[mask] = (
        result[mask] * (1.0 - alpha) + ov_rgb[mask].astype(np.float32) * alpha
    )
    return result.astype(np.uint8)


# ── Interactive HTML viewer ───────────────────────────────────────────────────

_VIEWER_CSS = """
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:#111; color:#ccc; font-family:system-ui,-apple-system,sans-serif; }
  .header { background:#1e1e1e; padding:10px 18px; border-bottom:1px solid #333;
            display:flex; justify-content:space-between; align-items:center; }
  .header h1 { font-size:1em; color:#ddd; font-weight:600; }
  .header .meta { font-size:0.8em; color:#888; }
  .viewer-wrap { display:grid; grid-template-columns:repeat(3,1fr);
                 gap:10px; padding:12px; }
  .panel { background:#1a1a1a; border-radius:6px; overflow:hidden; border:1px solid #2a2a2a; }
  .panel-hdr { background:#242424; padding:5px 10px; font-size:0.75em; color:#999;
               display:flex; justify-content:space-between; }
  .panel-hdr .axis-label { color:#6af; font-weight:600; }
  .panel img { width:100%; display:block; image-rendering:pixelated; cursor:crosshair; }
  .ctrl-row { padding:6px 10px; display:flex; align-items:center; gap:8px;
              border-top:1px solid #222; }
  input[type=range] { flex:1; -webkit-appearance:none; height:3px; background:#444;
                      border-radius:2px; outline:none; cursor:pointer; }
  input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:12px; height:12px;
    background:#6af; border-radius:50%; }
  .slice-idx { font-size:0.72em; color:#777; min-width:52px; text-align:right; }
  .footer { padding:8px 18px; background:#161616; border-top:1px solid #2a2a2a;
            display:flex; gap:20px; font-size:0.75em; color:#666; flex-wrap:wrap; }
  .footer .hi { color:#999; }
  .window-presets { display:flex; gap:6px; flex-wrap:wrap; }
  .preset-btn { background:#2a2a2a; border:1px solid #444; color:#aaa; padding:3px 9px;
    font-size:0.72em; border-radius:3px; cursor:pointer; }
  .preset-btn:hover { background:#333; color:#ddd; }
  .time-panel { padding:8px 12px; background:#181818; border-top:1px solid #222; }
  .time-panel label { font-size:0.75em; color:#888; display:block; margin-bottom:4px; }
  .kbd-hint { font-size:0.68em; color:#555; }
"""

_VIEWER_JS = r"""
  var DATA = {DATA_JSON};
  var state = { x: DATA.cx, y: DATA.cy, z: DATA.cz, t: 0 };
  var hasCT = DATA.modality === 'ct';

  function b64img(b64) { return 'data:image/png;base64,' + b64; }

  function getSlice(axis, idx) {
    var arr = DATA['slices_' + axis];
    var si = DATA['si_' + axis];
    // find closest pre-rendered slice index
    var best = 0, bestDist = Math.abs(si[0] - idx);
    for (var i = 1; i < si.length; i++) {
      var d = Math.abs(si[i] - idx);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return arr[best];
  }

  function getTimeSlice(t) {
    if (!DATA.slices_t || !DATA.si_t) return null;
    var best = 0, bestDist = Math.abs(DATA.si_t[0] - t);
    for (var i = 1; i < DATA.si_t.length; i++) {
      var d = Math.abs(DATA.si_t[i] - t);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return DATA.slices_t[best];
  }

  function updateView() {
    document.getElementById('img-axial').src    = b64img(getSlice('z', state.z));
    document.getElementById('img-coronal').src  = b64img(getSlice('y', state.y));
    document.getElementById('img-sagittal').src = b64img(getSlice('x', state.x));
    document.getElementById('lbl-z').textContent = 'z=' + state.z;
    document.getElementById('lbl-y').textContent = 'y=' + state.y;
    document.getElementById('lbl-x').textContent = 'x=' + state.x;
    if (DATA.n_volumes > 1) {
      var tSlice = getTimeSlice(state.t);
      if (tSlice) document.getElementById('img-time').src = b64img(tSlice);
      document.getElementById('lbl-t').textContent = 't=' + state.t +
        (DATA.tr ? '  (' + (state.t * DATA.tr).toFixed(1) + 's)' : '');
    }
  }

  function onSlider(axis, val) {
    state[axis] = parseInt(val);
    updateView();
  }

  function applyPreset(name) {
    if (!DATA.windows || !DATA.windows[name]) return;
    var w = DATA.windows[name];
    DATA.slices_x = w.slices_x; DATA.si_x = w.si_x;
    DATA.slices_y = w.slices_y; DATA.si_y = w.si_y;
    DATA.slices_z = w.slices_z; DATA.si_z = w.si_z;
    document.querySelectorAll('.preset-btn').forEach(function(b) {
      b.style.background = (b.dataset.preset === name) ? '#1a3a5a' : '#2a2a2a';
      b.style.color = (b.dataset.preset === name) ? '#6af' : '#aaa';
    });
    var info = document.getElementById('window-info');
    if (info) info.textContent = name;
    updateView();
  }

  document.addEventListener('keydown', function(e) {
    var axisMap = {'ArrowLeft': ['x',-1], 'ArrowRight': ['x',1],
                   'ArrowUp':   ['z', 1], 'ArrowDown':  ['z',-1],
                   'PageUp':    ['y', 1], 'PageDown':   ['y',-1]};
    var m = axisMap[e.key];
    if (!m) return;
    e.preventDefault();
    var axis = m[0], delta = m[1];
    var max = DATA['n_' + axis] - 1;
    state[axis] = Math.max(0, Math.min(max, state[axis] + delta));
    document.getElementById('sl-' + axis).value = state[axis];
    updateView();
  });

  window.addEventListener('load', updateView);
"""

def build_interactive_html(
    *,
    title: str,
    dataset_info: str,
    modality: str,
    shape: tuple,
    voxel_sizes: tuple,
    vmin: float,
    vmax: float,
    window_str: str,
    slices_x: list[str],    # pre-rendered sagittal slices
    slices_y: list[str],    # pre-rendered coronal slices
    slices_z: list[str],    # pre-rendered axial slices
    si_x: list[int],        # slice indices for x
    si_y: list[int],        # slice indices for y
    si_z: list[int],        # slice indices for z
    cx: int, cy: int, cz: int,  # center voxel
    n_volumes: int = 1,
    tr: float | None = None,
    slices_t: list[str] | None = None,
    si_t: list[int] | None = None,
    ct_window_stacks: dict | None = None,
) -> str:
    """Build a standalone interactive orthogonal viewer HTML page."""
    nx, ny, nz = shape[:3]

    # Embed all data as JSON (slices + metadata)
    data_obj = {
        "slices_x": slices_x, "slices_y": slices_y, "slices_z": slices_z,
        "si_x": si_x, "si_y": si_y, "si_z": si_z,
        "cx": cx, "cy": cy, "cz": cz,
        "n_x": nx, "n_y": ny, "n_z": nz,
        "n_volumes": n_volumes,
        "tr": tr,
        "modality": modality,
    }
    if slices_t:
        data_obj["slices_t"] = slices_t
    if si_t is not None:
        data_obj["si_t"] = si_t
    if ct_window_stacks:
        data_obj["windows"] = ct_window_stacks

    data_json = json.dumps(data_obj)

    ct_buttons = ""
    if modality == "ct":
        presets = [
            ("Brain", "brain"),
            ("Soft tissue", "soft_tissue"),
            ("Bone", "bone"),
            ("Lung", "lung"),
            ("Subdural", "subdural"),
        ]
        buttons = " ".join(
            f'<button class="preset-btn" data-preset="{key}" onclick="applyPreset(\'{key}\')">{label}</button>'
            for label, key in presets
        )
        ct_buttons = f"""
        <div class="footer">
          <span class="hi">CT Windowing:</span>
          <div class="window-presets">{buttons}</div>
          <span id="window-info" class="hi">{window_str}</span>
        </div>"""

    time_panel = ""
    if n_volumes > 1:
        time_panel = f"""
        <div class="time-panel">
          <label>Time (volumes: {n_volumes}{f', TR={tr}s' if tr else ''})</label>
          <div class="ctrl-row">
            <input type="range" id="sl-t" min="0" max="{n_volumes-1}" value="0"
              oninput="onSlider('t', this.value)">
            <span class="slice-idx" id="lbl-t">t=0</span>
          </div>
        </div>"""

    # Time image row (optional)
    time_img = ""
    if slices_t:
        time_img = '<img id="img-time" src="" style="display:block;max-height:120px;object-fit:contain;">'

    vox_str = " × ".join(f"{v:.2f}" for v in voxel_sizes[:3]) + " mm"
    js = _VIEWER_JS.replace("{DATA_JSON}", data_json)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qortex Viewer — {title}</title>
  <style>{_VIEWER_CSS}</style>
</head>
<body>
  <div class="header">
    <h1>Qortex Viewer &nbsp;|&nbsp; {title}</h1>
    <span class="meta">{dataset_info}</span>
  </div>

  <div class="viewer-wrap">
    <!-- Axial (z) -->
    <div class="panel">
      <div class="panel-hdr">
        <span><span class="axis-label">AX</span> Axial</span>
        <span class="kbd-hint">↑↓ keys</span>
      </div>
      <img id="img-axial" src="">
      <div class="ctrl-row">
        <input type="range" id="sl-z" min="0" max="{nz-1}" value="{cz}"
          oninput="onSlider('z', this.value)">
        <span class="slice-idx" id="lbl-z">z={cz}</span>
      </div>
    </div>

    <!-- Coronal (y) -->
    <div class="panel">
      <div class="panel-hdr">
        <span><span class="axis-label">COR</span> Coronal</span>
        <span class="kbd-hint">PgUp/Dn keys</span>
      </div>
      <img id="img-coronal" src="">
      <div class="ctrl-row">
        <input type="range" id="sl-y" min="0" max="{ny-1}" value="{cy}"
          oninput="onSlider('y', this.value)">
        <span class="slice-idx" id="lbl-y">y={cy}</span>
      </div>
    </div>

    <!-- Sagittal (x) -->
    <div class="panel">
      <div class="panel-hdr">
        <span><span class="axis-label">SAG</span> Sagittal</span>
        <span class="kbd-hint">←→ keys</span>
      </div>
      <img id="img-sagittal" src="">
      <div class="ctrl-row">
        <input type="range" id="sl-x" min="0" max="{nx-1}" value="{cx}"
          oninput="onSlider('x', this.value)">
        <span class="slice-idx" id="lbl-x">x={cx}</span>
      </div>
    </div>
  </div>

  {time_panel}
  {time_img}
  {ct_buttons}

  <div class="footer">
    <span>Shape: <span class="hi">{" × ".join(str(s) for s in shape)}</span></span>
    <span>Voxel: <span class="hi">{vox_str}</span></span>
    <span>Window: <span class="hi">{window_str}</span></span>
    <span>Modality: <span class="hi">{modality.upper()}</span></span>
  </div>

  <script>{js}</script>
</body>
</html>"""
