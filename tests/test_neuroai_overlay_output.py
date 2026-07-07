from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qortex.neuroai.models import ModelOutput
from qortex.neuroai.outputs import ClassificationOutput, SegmentationOutput
from qortex.neuroai.outputs.overlay_out import OverlayOutputAdapter


def test_overlay_output_writes_sidecar_without_source_image(tmp_path: Path):
    adapter = OverlayOutputAdapter(output_dir=tmp_path)
    adapter.open()

    adapter.write(
        ModelOutput(
            output_type="classification",
            raw=ClassificationOutput(
                class_name="left_hand",
                class_index=1,
                confidence=0.87,
                probabilities={"rest": 0.13, "left_hand": 0.87},
            ),
        ),
        metadata={"window_index": 3, "source": "unit-test"},
    )
    adapter.close()

    sidecar = tmp_path / "prediction_000000.json"
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["window_index"] == 3
    assert record["source"] == "unit-test"
    assert record["class_name"] == "left_hand"
    assert adapter.n_written == 1


def test_overlay_output_renders_segmentation_frame(tmp_path: Path):
    image = np.zeros((32, 40), dtype=np.float32)
    image[8:24, 12:28] = 0.8
    mask = np.zeros_like(image, dtype=np.int16)
    mask[10:22, 15:30] = 1

    output = ModelOutput(
        output_type="segmentation",
        raw=SegmentationOutput(
            mask=mask,
            n_classes=2,
            class_labels={0: "background", 1: "target"},
        ),
    )

    adapter = OverlayOutputAdapter(output_dir=tmp_path, fmt="png", alpha=0.5)
    adapter.open()
    adapter.write(output, metadata={"source_image": image, "window_index": 0})
    adapter.close()

    frame = tmp_path / "frame_000000.png"
    assert frame.exists()
    assert frame.stat().st_size > 0
    assert adapter.n_written == 1
