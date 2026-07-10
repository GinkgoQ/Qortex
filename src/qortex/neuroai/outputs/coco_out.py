"""COCO JSON output adapter.

Accumulates detection results across writes and serializes them as a
COCO-format JSON file on close().
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter, OutputAdapterError

log = logging.getLogger(__name__)


class COCOOutputAdapter(OutputAdapter):
    """Output adapter that serializes detections in COCO JSON format.

    Parameters
    ----------
    path:
        Output JSON file path.
    pipeline_ref:
        Short pipeline reference for provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._pipeline_ref = pipeline_ref
        self._data: dict = {}
        self._image_id_counter = 0
        self._annotation_id_counter = 0
        self._category_map: dict[str, int] = {}

    @property
    def n_written(self) -> int:
        return len(self._data.get("images", []))

    @property
    def n_prediction_records(self) -> int:
        return len(self._data.get("annotations", []))

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data = {
            "info": {
                "description": "Qortex NeuroAI detection results",
                "url": "",
                "version": "1.0",
                "year": datetime.now(timezone.utc).year,
                "contributor": "Qortex",
                "date_created": datetime.now(timezone.utc).isoformat(),
                "pipeline_ref": self._pipeline_ref,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        log.info("COCO output ready: %s", self._path)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        meta = metadata or {}

        detections = meta.get("detections") or []
        if not detections and output.bbox is not None:
            detections = [
                {
                    "x1": output.bbox[0], "y1": output.bbox[1],
                    "x2": output.bbox[2], "y2": output.bbox[3],
                    "class_name": output.class_name or "unknown",
                    "class_index": output.class_index or 0,
                    "confidence": max(output.probabilities.values()) if output.probabilities else 0.0,
                }
            ]
        if not detections:
            raise OutputAdapterError(
                "COCO output requires detection records or output.bbox; "
                f"received output_type={output.output_type!r}."
            )

        # Image entry
        image_id = meta.get("image_id") or (self._image_id_counter + 1)
        self._image_id_counter = int(image_id)
        w = int(meta.get("image_width", 640))
        h = int(meta.get("image_height", 640))
        file_name = meta.get("image_name") or meta.get("source_id") or f"image_{image_id:06d}"

        self._data["images"].append({
            "id": image_id,
            "file_name": str(file_name),
            "width": w,
            "height": h,
        })

        for det in detections:
            class_name = str(det.get("class_name", "unknown"))

            # Register category
            if class_name not in self._category_map:
                cat_id = len(self._category_map) + 1
                self._category_map[class_name] = cat_id
                self._data["categories"].append({
                    "id": cat_id,
                    "name": class_name,
                    "supercategory": "object",
                })
            cat_id = self._category_map[class_name]

            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", w))
            y2 = float(det.get("y2", h))
            bw = x2 - x1
            bh = y2 - y1

            self._annotation_id_counter += 1
            self._data["annotations"].append({
                "id": self._annotation_id_counter,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "score": float(det.get("confidence", 0.0)),
            })

        self._image_id_counter += 1

    def close(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        n_ann = len(self._data["annotations"])
        n_img = len(self._data["images"])
        log.info("COCO JSON saved: %s (%d images, %d annotations)", self._path.name, n_img, n_ann)
