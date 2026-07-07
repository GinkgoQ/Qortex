"""NiiVue-backed interactive volume viewer.

``volume.py``'s ``build_interactive_html`` pre-renders every slice as a
base64 PNG and drives a hand-rolled JS slider — a DIY reimplementation of
what a real WebGL volume renderer already does natively (true 3D texture
sampling, real-time windowing, multiplanar reformatting at arbitrary
angles). This module embeds NiiVue — a WebGL2 viewer built specifically for
NIfTI/DICOM — directly into a self-contained HTML page instead.

The NiiVue bundle is vendored (``_vendor/niivue.umd.js``, BSD-2-Clause) and
inlined, so the exported HTML has no CDN dependency and works offline,
matching the rest of this package's "self-contained HTML" convention.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
from typing import Any

_VENDOR_DIR = Path(__file__).parent / "_vendor"
_NIIVUE_JS_PATH = _VENDOR_DIR / "niivue.umd.js"


def _read_niivue_bundle() -> str:
    if not _NIIVUE_JS_PATH.exists():
        raise FileNotFoundError(
            f"Vendored NiiVue bundle not found at {_NIIVUE_JS_PATH}. "
            "Re-vendor it with: npm pack @niivue/niivue@latest && "
            "tar -xzOf niivue-niivue-*.tgz package/dist/niivue.umd.js > "
            f"{_NIIVUE_JS_PATH}"
        )
    return _NIIVUE_JS_PATH.read_text(encoding="utf-8")


def niivue_html(
    source: str | Path,
    *,
    title: str = "",
    colormap: str = "gray",
    back_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> str:
    """Build a self-contained HTML page with a real WebGL2 NiiVue viewer.

    Parameters
    ----------
    source:
        Path to a NIfTI file (``.nii`` / ``.nii.gz``). The raw bytes are
        base64-embedded directly in the page — no server, no CDN fetch.
    title:
        Page/viewer title.
    colormap:
        Any NiiVue-supported colormap name (``"gray"``, ``"red"``, ``"hot"``, ...).
    back_color:
        RGBA background (0-1 floats) for the render canvas.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    data_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    niivue_js = _read_niivue_bundle()
    safe_title = escape(title or path.name)
    safe_name = escape(path.name)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{safe_title}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: #0b0b0f; height: 100%; }}
  #header {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; color: #e5e7eb;
             padding: 10px 16px; font-size: 15px; font-weight: 600; }}
  #gl {{ width: 100%; height: calc(100vh - 44px); display: block; }}
</style>
</head>
<body>
<div id="header">{safe_title}</div>
<canvas id="gl"></canvas>
<script>{niivue_js}</script>
<script>
  const b64 = "{data_b64}";
  const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  const blobUrl = URL.createObjectURL(new Blob([bytes], {{ type: "application/octet-stream" }}));
  const nv = new niivue.Niivue({{ backColor: [{back_color[0]}, {back_color[1]}, {back_color[2]}, {back_color[3]}] }});
  nv.attachToCanvas(document.getElementById("gl"));
  nv.loadVolumes([{{ url: blobUrl, name: "{safe_name}", colormap: "{colormap}" }}]).then(() => {{
    nv.setSliceType(nv.sliceTypeMultiplanar);
  }});
</script>
</body>
</html>
"""


__all__ = ["niivue_html"]
