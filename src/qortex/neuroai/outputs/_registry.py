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
    """
    out_type = (spec.type or "").lower().strip()

    if out_type in ("jsonl", "json_lines", "json"):
        from qortex.neuroai.outputs.jsonl_out import JSONLOutputAdapter
        path = spec.path or "predictions.jsonl"
        return JSONLOutputAdapter(path, append=spec.append, pipeline_ref=pipeline_ref)

    if out_type in ("parquet",):
        from qortex.neuroai.outputs.parquet_out import ParquetOutputAdapter
        path = spec.path or "predictions.parquet"
        return ParquetOutputAdapter(path, pipeline_ref=pipeline_ref)

    if out_type in ("lsl_marker", "lsl"):
        try:
            from qortex.neuroai.outputs.lsl_out import LSLMarkerOutputAdapter
            return LSLMarkerOutputAdapter(
                stream_name=spec.stream_name or "qortex_predictions",
                pipeline_ref=pipeline_ref,
            )
        except ImportError:
            raise ImportError(
                "LSL output requires pylsl. "
                "Install with: pip install 'qortex[lsl]'"
            )

    raise ValueError(
        f"Unknown output type: {out_type!r}. "
        f"Supported: 'jsonl', 'parquet', 'lsl_marker'."
    )
