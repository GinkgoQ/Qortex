"""YOLO txt output adapter.

Writes detection results in YOLO label format:
  ``{class_id} {cx} {cy} {w} {h}``  (all values normalized 0–1)

One .txt file per image, plus a ``classes.txt`` mapping index → class name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)


class YOLOOutputAdapter(OutputAdapter):
    """Output adapter that writes detections in YOLO txt format.

    Parameters
    ----------
    path:
        Output directory (one ``<image_name>.txt`` per call to write()).
    pipeline_ref:
        Short pipeline reference for provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        self._out_dir = Path(path)
        self._pipeline_ref = pipeline_ref
        self._class_map: dict[int, str] = {}
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        self._out_dir.mkdir(parents=True, exist_ok=True)
        log.info("YOLO txt output ready: %s", self._out_dir)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        meta = metadata or {}
        img_w = float(meta.get("image_width", 640))
        img_h = float(meta.get("image_height", 640))
        image_name = meta.get("image_name") or f"image_{self._n_written:05d}"

        detections = meta.get("detections") or []
        if not detections and output.bbox is not None:
            detections = [
                {
                    "x1": output.bbox[0], "y1": output.bbox[1],
                    "x2": output.bbox[2], "y2": output.bbox[3],
                    "class_name": output.class_name or "unknown",
                    "class_index": output.class_index or 0,
                }
            ]

        lines: list[str] = []
        for det in detections:
            class_idx = int(det.get("class_index", 0))
            class_name = str(det.get("class_name", f"class_{class_idx}"))
            self._class_map[class_idx] = class_name

            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", img_w))
            y2 = float(det.get("y2", img_h))

            cx = (x1 + x2) / 2.0 / img_w
            cy = (y1 + y2) / 2.0 / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        txt_path = self._out_dir / f"{image_name}.txt"
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        self._n_written += 1

    def close(self) -> None:
        # Write classes.txt
        if self._class_map:
            max_idx = max(self._class_map)
            classes_lines = [
                self._class_map.get(i, f"class_{i}") for i in range(max_idx + 1)
            ]
            (self._out_dir / "classes.txt").write_text(
                "\n".join(classes_lines), encoding="utf-8"
            )
        log.info("YOLO txt output closed (%d label files written)", self._n_written)
