"""Factory for OutputAdapter instances."""

from __future__ import annotations

from pathlib import Path

from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.spec import OutputSpec


def make_output_adapter(spec: OutputSpec, *, pipeline_ref: str | None = None) -> OutputAdapter:
    """Return the correct OutputAdapter for the given OutputSpec.

    Raises
    ------
    ValueError
        When the output type is unknown.
    ImportError
        When the required optional dependency is missing.
    """
    out_type = (spec.type or "").lower().strip()
    path = spec.path

    if out_type in ("jsonl", "json_lines", "json"):
        from qortex.neuroai.outputs.jsonl_out import JSONLOutputAdapter
        return JSONLOutputAdapter(
            path or "predictions.jsonl",
            append=spec.append,
            pipeline_ref=pipeline_ref,
        )

    if out_type in ("parquet",):
        from qortex.neuroai.outputs.parquet_out import ParquetOutputAdapter
        return ParquetOutputAdapter(path or "predictions.parquet", pipeline_ref=pipeline_ref)

    if out_type in ("csv",):
        from qortex.neuroai.outputs.csv_out import CSVOutputAdapter
        return CSVOutputAdapter(
            path or "predictions.csv",
            append=spec.append,
            pipeline_ref=pipeline_ref,
        )

    if out_type in ("lsl_marker", "lsl"):
        from qortex.neuroai.outputs.lsl_out import LSLMarkerOutputAdapter
        return LSLMarkerOutputAdapter(
            stream_name=spec.stream_name or "qortex_predictions",
            pipeline_ref=pipeline_ref,
        )

    if out_type in ("nifti", "nii", "nifti_mask"):
        from qortex.neuroai.outputs.nifti_out import NIfTIOutputAdapter
        return NIfTIOutputAdapter(path or "mask.nii.gz", pipeline_ref=pipeline_ref)

    if out_type in ("dicom_seg", "dicomseg"):
        from qortex.neuroai.outputs.dicom_seg_out import DICOMSEGOutputAdapter
        return DICOMSEGOutputAdapter(path or "output_seg/", pipeline_ref=pipeline_ref)

    if out_type in ("dicom_sr", "dicomsr"):
        from qortex.neuroai.outputs.dicom_sr_out import DICOMSROutputAdapter
        return DICOMSROutputAdapter(path or "output_sr/", pipeline_ref=pipeline_ref)

    if out_type in ("bids", "bids_derivative"):
        from qortex.neuroai.outputs.bids_out import BIDSDerivativeOutputAdapter
        return BIDSDerivativeOutputAdapter(
            path or "derivatives/",
            pipeline_ref=pipeline_ref,
        )

    if out_type in ("coco", "coco_json"):
        from qortex.neuroai.outputs.coco_out import COCOOutputAdapter
        return COCOOutputAdapter(
            path or "predictions_coco.json",
            pipeline_ref=pipeline_ref,
        )

    if out_type in ("yolo", "yolo_txt"):
        from qortex.neuroai.outputs.yolo_out import YOLOOutputAdapter
        return YOLOOutputAdapter(path or "yolo_labels/", pipeline_ref=pipeline_ref)

    if out_type in ("websocket", "ws"):
        if not path:
            raise ValueError("WebSocket output requires a URL in spec.path")
        from qortex.neuroai.outputs.websocket_out import WebSocketOutputAdapter
        return WebSocketOutputAdapter(path, pipeline_ref=pipeline_ref)

    if out_type in ("http", "http_callback", "webhook"):
        if not path:
            raise ValueError("HTTP callback output requires a URL in spec.path")
        from qortex.neuroai.outputs.http_out import HTTPCallbackOutputAdapter
        return HTTPCallbackOutputAdapter(path, pipeline_ref=pipeline_ref)

    if out_type in ("overlay", "image_overlay", "video_overlay"):
        from qortex.neuroai.outputs.overlay_out import OverlayOutputAdapter
        return OverlayOutputAdapter(
            spec,
            output_dir=path or "annotated_frames",
            pipeline_ref=pipeline_ref,
        )

    raise ValueError(
        f"Unknown output type: {out_type!r}. "
        f"Supported: 'jsonl', 'parquet', 'csv', 'lsl_marker', 'nifti', "
        f"'dicom_seg', 'dicom_sr', 'bids', 'coco', 'yolo', 'websocket', 'http', 'overlay'."
    )
