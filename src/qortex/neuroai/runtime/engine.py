"""Local inference runtime engine.

Ties together source → preprocessing → model → output in a single
deterministic execution loop.  Every stage is timed by the ``PipelineProfiler``
and every failure is surfaced as a structured warning rather than a silent skip.

The engine does NOT make decisions — it executes what the ``Pipeline.check()``
path already verified.  Calling ``run()`` without a prior ``check()`` is
allowed but will log a warning.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

from qortex.neuroai.benchmark import PipelineProfiler
from qortex.neuroai.contracts import (
    ArtifactContract,
    CompatibilityReport,
    PipelineRunReport,
    PreprocessPlan,
    SourceProfile,
    WarningItem,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.preprocess.planner import TransformExecutor
from qortex.neuroai.sources._base import SourceAdapter
from qortex.neuroai.spec import PipelineSpec, RuntimeSpec, TriggerSpec

log = logging.getLogger(__name__)


class RuntimeEngine:
    """Execute a verified NeuroAI pipeline locally.

    Parameters
    ----------
    spec:
        The pipeline spec.
    source:
        A probed source adapter.
    model:
        A loaded model adapter.
    plan:
        A validated preprocessing plan.
    outputs:
        List of opened output adapters.
    compat_report:
        The CompatibilityReport from the check phase.
    profiler:
        Optional latency profiler (created internally if None).
    """

    def __init__(
        self,
        spec: PipelineSpec,
        source: SourceAdapter,
        model: ModelAdapter,
        plan: PreprocessPlan,
        outputs: list[OutputAdapter],
        compat_report: CompatibilityReport | None = None,
        profiler: PipelineProfiler | None = None,
    ) -> None:
        self._spec = spec
        self._source = source
        self._model = model
        self._plan = plan
        self._outputs = outputs
        self._compat = compat_report
        self._profiler = profiler or PipelineProfiler(
            budget_ms=spec.runtime.latency_budget_ms
        )
        self._executor = TransformExecutor(plan)

    def run(self) -> PipelineRunReport:
        """Execute the full pipeline and return a run report.

        Returns
        -------
        PipelineRunReport
            Includes latency report, artifact contract, and any errors.
        """
        errors: list[str] = []
        warnings: list[WarningItem] = []
        n_ok = 0

        trigger = self._spec.trigger
        _trigger_streak: int = 0
        _trigger_required: int = int(trigger.when.get("stable_for", 1)) if trigger else 1

        try:
            for idx, data_item in enumerate(self._source.stream()):
                # ── Source read done ────────────────────────────────────────
                # (timing was started by the source iter itself here we measure post-hoc)
                self._profiler.start_source_read()
                self._profiler.end_source_read()  # zero-cost marker; real source was timed externally

                # ── Preprocessing ───────────────────────────────────────────
                self._profiler.start_preprocess()
                try:
                    arr = self._executor.apply(_extract_array(data_item))
                except Exception as exc:
                    err_msg = f"Preprocess error on window {idx}: {exc}"
                    log.warning(err_msg)
                    errors.append(err_msg)
                    self._profiler.commit_window(dropped=True, error=str(exc))
                    continue
                self._profiler.end_preprocess()

                # ── Inference ───────────────────────────────────────────────
                self._profiler.start_inference()
                try:
                    output: ModelOutput = self._model.predict(arr)
                except Exception as exc:
                    err_msg = f"Inference error on window {idx}: {exc}"
                    log.warning(err_msg)
                    errors.append(err_msg)
                    self._profiler.commit_window(dropped=True, error=str(exc))
                    continue
                self._profiler.end_inference()

                # ── Postprocess (trigger evaluation) ────────────────────────
                self._profiler.start_postprocess()
                trigger_fired = False
                if trigger is not None:
                    pred_dict = {
                        "class": output.class_name,
                        "probabilities": output.probabilities,
                    }
                    if trigger.evaluate(pred_dict):
                        _trigger_streak += 1
                    else:
                        _trigger_streak = 0
                    if _trigger_streak >= _trigger_required:
                        trigger_fired = True
                        log.info("Trigger fired at window %d: %s", idx, trigger.emit)
                        _trigger_streak = 0
                self._profiler.end_postprocess()

                # ── Output write ────────────────────────────────────────────
                self._profiler.start_output_write()
                meta = {
                    "window_index": idx,
                    "trigger_fired": trigger_fired,
                    "source": self._source.source_id,
                }
                for out_adapter in self._outputs:
                    try:
                        out_adapter.write(output, metadata=meta)
                    except Exception as exc:
                        err_msg = f"Output write error on window {idx}: {exc}"
                        log.warning(err_msg)
                        errors.append(err_msg)

                # Emit structured EventMarker when trigger fires
                if trigger_fired and trigger is not None:
                    _emit_trigger_event(
                        trigger, idx, output, self._outputs, self._source.source_id
                    )
                self._profiler.end_output_write()

                self._profiler.commit_window()
                n_ok += 1

        except KeyboardInterrupt:
            log.info("Pipeline interrupted by user after %d windows.", n_ok)

        latency_report = self._profiler.report()
        artifact_contract = self._make_artifact_contract(latency_report)

        n_outputs_written = sum(getattr(o, "n_written", 0) for o in self._outputs)
        success = n_ok > 0 and not any(e for e in errors)
        return PipelineRunReport(
            success=success,
            compatibility_report=self._compat,
            preprocess_plan=self._plan,
            latency_report=latency_report,
            artifact_contract=artifact_contract,
            outputs=[{"n_written": getattr(o, "n_written", 0)} for o in self._outputs],
            errors=errors,
            warnings=warnings,
            n_windows_processed=n_ok,
            n_outputs_written=n_outputs_written,
        )

    def _make_artifact_contract(self, latency_report) -> ArtifactContract:
        from qortex import __version__
        from datetime import datetime, timezone
        return ArtifactContract(
            qortex_version=__version__,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_id=self._source.source_id,
            model_id=self._spec.model.id,
            model_revision=self._spec.model.revision,
            pipeline_spec_hash=self._spec.content_hash(),
            preprocessing_transforms=[
                str(t.kind.value if hasattr(t.kind, "value") else t.kind)
                for t in self._plan.transforms
            ],
            runtime_backend=self._spec.runtime.device,
            device=self._spec.runtime.device,
            output_type=self._spec.model.task,
            compatibility_status=(
                self._compat.status.value
                if self._compat and hasattr(self._compat.status, "value")
                else str(self._compat.status) if self._compat else None
            ),
            unknowns=list(self._compat.unknowns if self._compat else []),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _emit_trigger_event(
    trigger: "TriggerSpec",
    window_idx: int,
    output: "ModelOutput",
    adapters: list["OutputAdapter"],
    source_id: str,
) -> None:
    """Write a structured EventMarkerOutput to all output adapters that support it.

    Called only when the trigger condition is satisfied.  The marker includes
    the trigger class, probability, and emit payload so downstream consumers
    (BCI decoders, alert systems) can act on it without inspecting every window.
    """
    from qortex.neuroai.outputs.types import EventMarkerOutput
    import datetime

    probs = output.probabilities or {}
    triggered_class = trigger.when.get("class", "")

    marker = EventMarkerOutput(
        event_type="trigger",
        label=triggered_class,
        confidence=float(probs.get(triggered_class, 0.0)),
        window_index=window_idx,
        source_id=source_id,
        emit_payload=trigger.emit,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    for adapter in adapters:
        write_marker = getattr(adapter, "write_marker", None)
        if callable(write_marker):
            try:
                write_marker(marker)
            except Exception as exc:
                log.warning("EventMarker write failed on %s: %s", type(adapter).__name__, exc)


def _extract_array(data_item: Any):
    """Extract the underlying numpy array from a QortexData object.

    Source adapters set QortexAbstraction.data to the actual numpy array.
    Raw numpy arrays are passed through directly.

    For QortexEventTable, the .data field is a Polars DataFrame.  Numeric
    columns are stacked into a (n_rows, n_numeric_cols) float32 array so the
    standard preprocessing chain can process tabular data without special-casing
    every downstream transform.
    """
    import numpy as np
    if isinstance(data_item, np.ndarray):
        return data_item

    # QortexTimeSeries / QortexVolume carry their numpy array in .data
    raw = getattr(data_item, "data", None)
    if isinstance(raw, np.ndarray):
        return raw

    # QortexEventTable.data is a Polars DataFrame — extract numeric columns
    if raw is not None and type(raw).__name__ == "DataFrame":
        try:
            numeric_cols = [c for c in raw.columns if raw[c].dtype.is_numeric()]
            if not numeric_cols:
                raise TypeError(
                    "QortexEventTable has no numeric columns; cannot convert to array. "
                    "Only numeric columns are extracted for model inference."
                )
            return raw.select(numeric_cols).to_numpy(allow_copy=True).astype(np.float32)
        except Exception as exc:
            raise TypeError(
                f"Failed to convert QortexEventTable DataFrame to numpy array: {exc}"
            ) from exc

    # Torch tensor
    if hasattr(data_item, "numpy"):
        return data_item.numpy()
    if hasattr(data_item, "detach"):
        return data_item.detach().cpu().numpy()

    # Fall through — TransformExecutor._coerce_numpy will raise with a clear message
    return data_item
