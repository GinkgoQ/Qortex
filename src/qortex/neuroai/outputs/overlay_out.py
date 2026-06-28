"""Image/video overlay output adapter.

Renders bounding boxes, segmentation masks, and classification labels directly
onto source images and writes the annotated frames to disk.

Requires either Pillow (pure Python, recommended for stills) or OpenCV
(for video and faster rendering).  Falls back gracefully between the two.

YAML config::

    outputs:
      - type: overlay
        path: annotated_frames/
        format: png          # png | jpg
        alpha: 0.5           # mask overlay transparency (0–1)
        line_width: 2        # bounding box line width in pixels
        font_size: 12        # label font size (Pillow only)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qortex.core.exceptions import OutputAdapterError
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.outputs.types import (
    DetectionOutput,
    SegmentationOutput,
    ClassificationOutput,
    EventMarkerOutput,
)
from qortex.neuroai.spec import OutputSpec

log = logging.getLogger(__name__)

# Default palette: 20 distinct colours for class visualisation
_PALETTE = [
    (220,  20,  60), (119,  11,  32), (  0,   0, 142), (  0,   0, 230),
    (106,   0, 228), (  0,  60, 100), (  0,  80, 100), (  0,   0, 192),
    (250, 170,  30), (100, 170,  30), (220, 220,   0), (175, 116, 175),
    (250, 120, 194), (125, 125, 125), (255,  69,   0), (  0, 125, 155),
    (209, 169, 109), (128,  64, 128), (244,  35, 232), ( 70,  70,  70),
]


class OverlayOutputAdapter(OutputAdapter):
    """Render model predictions on top of source images and save to disk.

    Parameters
    ----------
    spec:
        ``OutputSpec`` with ``type="overlay"`` and ``path=<output_dir>``.
    alpha:
        Segmentation mask transparency (0 = invisible, 1 = opaque).
    line_width:
        Bounding-box border width in pixels.
    fmt:
        Output image format: ``"png"`` (lossless) or ``"jpg"``.
    """

    def __init__(
        self,
        spec: OutputSpec | None = None,
        *,
        output_dir: str | Path = "annotated_frames",
        alpha: float = 0.45,
        line_width: int = 2,
        fmt: str = "png",
        pipeline_ref: str | None = None,
    ) -> None:
        if spec is not None:
            output_dir = spec.path or output_dir
            alpha = float(spec.extra.get("alpha", alpha))
            line_width = int(spec.extra.get("line_width", line_width))
            fmt = str(spec.extra.get("format", fmt)).lower()

        self._dir = Path(output_dir)
        self._alpha = max(0.0, min(1.0, alpha))
        self._line_width = max(1, line_width)
        self._fmt = fmt.lstrip(".")
        self._pipeline_ref = pipeline_ref
        self._frame_idx = 0
        self._n_written = 0

    # ── OutputAdapter interface ───────────────────────────────────────────────

    def open(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        log.info("OverlayOutputAdapter: writing to %s", self._dir)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        meta = metadata or {}
        source_image = meta.get("source_image")  # numpy [H,W] or [H,W,C]
        if source_image is None:
            # No image to draw on — write a metadata-only JSON sidecar
            self._write_sidecar(output, meta)
            self._frame_idx += 1
            return

        try:
            annotated = self._render(source_image, output, meta)
            out_path = self._dir / f"frame_{self._frame_idx:06d}.{self._fmt}"
            _save_image(annotated, out_path, fmt=self._fmt)
            self._n_written += 1
            log.debug("Overlay written: %s", out_path.name)
        except Exception as exc:
            raise OutputAdapterError(
                f"Failed to write overlay frame {self._frame_idx}: {exc}",
                output_type="overlay",
                path=str(self._dir),
            ) from exc
        finally:
            self._frame_idx += 1

    def write_marker(self, marker: EventMarkerOutput) -> None:
        """Write trigger event as a JSON sidecar next to the current frame."""
        import json
        sidecar = self._dir / f"trigger_{self._frame_idx:06d}.json"
        sidecar.write_text(
            json.dumps({
                "event_type": marker.event_type,
                "label": marker.label,
                "confidence": marker.confidence,
                "window_index": marker.window_index,
                "timestamp_utc": marker.timestamp_utc,
                "emit_payload": marker.emit_payload,
            }, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        log.info("OverlayOutputAdapter closed — %d frames written", self.n_written)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render(self, image, output: ModelOutput, meta: dict) -> Any:
        import numpy as np

        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)  # grayscale → RGB
        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)

        # Detect which output type we have
        raw = getattr(output, "raw", output)
        if isinstance(raw, DetectionOutput):
            img = self._draw_detections(img, raw)
        elif isinstance(raw, SegmentationOutput):
            img = self._draw_segmentation(img, raw)
        elif isinstance(raw, ClassificationOutput):
            img = self._draw_classification(img, raw)

        return img

    def _draw_detections(self, img, det: DetectionOutput):
        try:
            from PIL import ImageDraw, Image
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            for box in det.boxes:
                colour = _PALETTE[box.class_index % len(_PALETTE)]
                draw.rectangle(
                    [box.x1, box.y1, box.x2, box.y2],
                    outline=colour,
                    width=self._line_width,
                )
                label = f"{box.class_name} {box.confidence:.2f}"
                draw.text((box.x1 + 2, box.y1 + 2), label, fill=colour)
            return _pil_to_numpy(pil)
        except ImportError:
            return _cv2_draw_boxes(img, det, self._line_width)

    def _draw_segmentation(self, img, seg: SegmentationOutput):
        import numpy as np
        mask = np.asarray(seg.mask)
        if mask.shape != img.shape[:2]:
            mask = _resize_mask(mask, img.shape[:2])

        overlay = img.copy()
        for class_idx, class_name in seg.class_labels.items():
            colour = _PALETTE[class_idx % len(_PALETTE)]
            region = mask == class_idx
            overlay[region] = colour

        return (img * (1 - self._alpha) + overlay * self._alpha).astype(img.dtype)

    def _draw_classification(self, img, clf: ClassificationOutput):
        try:
            from PIL import ImageDraw, Image
            pil = Image.fromarray(img)
            draw = ImageDraw.Draw(pil)
            label = f"{clf.class_name}: {clf.confidence:.2%}"
            draw.rectangle([4, 4, len(label) * 8, 22], fill=(0, 0, 0, 160))
            draw.text((6, 6), label, fill=(255, 255, 255))
            return _pil_to_numpy(pil)
        except ImportError:
            return img

    def _write_sidecar(self, output: ModelOutput, meta: dict) -> None:
        import json
        sidecar = self._dir / f"prediction_{self._frame_idx:06d}.json"
        record: dict[str, Any] = {
            "window_index": meta.get("window_index", self._frame_idx),
            "source": meta.get("source"),
            "class_name": getattr(output, "class_name", None),
            "confidence": getattr(output, "confidence", None),
        }
        sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.n_written += 1


# ── Image helpers ──────────────────────────────────────────────────────────────

def _normalize_to_uint8(arr):
    import numpy as np
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr.clip(0, 255).astype(np.uint8)


def _pil_to_numpy(pil_image):
    import numpy as np
    return np.asarray(pil_image)


def _resize_mask(mask, target_hw: tuple):
    import numpy as np
    th, tw = target_hw
    mh, mw = mask.shape[:2]
    if (mh, mw) == (th, tw):
        return mask
    try:
        import cv2
        return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        # Nearest-neighbour resize via numpy index scaling
        row_idx = (np.arange(th) * mh / th).astype(int)
        col_idx = (np.arange(tw) * mw / tw).astype(int)
        return mask[np.ix_(row_idx, col_idx)]


def _save_image(arr, path: Path, fmt: str = "png") -> None:
    try:
        from PIL import Image
        Image.fromarray(arr).save(str(path))
        return
    except ImportError:
        pass
    try:
        import cv2
        import numpy as np
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        return
    except ImportError:
        pass
    raise OutputAdapterError(
        "Saving overlay images requires Pillow or OpenCV. "
        "Install with: pip install pillow  or  pip install opencv-python-headless",
        output_type="overlay",
        path=str(path),
    )


def _cv2_draw_boxes(img, det: DetectionOutput, line_width: int):
    try:
        import cv2
        out = img.copy()
        for box in det.boxes:
            colour = _PALETTE[box.class_index % len(_PALETTE)][::-1]  # RGB→BGR
            cv2.rectangle(
                out, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)),
                colour, line_width,
            )
            label = f"{box.class_name} {box.confidence:.2f}"
            cv2.putText(
                out, label, (int(box.x1) + 2, int(box.y1) + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA,
            )
        return out
    except ImportError:
        return img
