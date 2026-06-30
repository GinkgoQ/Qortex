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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from qortex.neuroai.benchmark import PipelineProfiler
from qortex.neuroai.contracts import (
    ArtifactContract,
    CompatibilityReport,
    PipelineRunReport,
    PreprocessPlan,
    WarningItem,
)
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.preprocess.planner import TransformExecutor
from qortex.neuroai.sources._base import SourceAdapter
from qortex.neuroai.spec import PipelineSpec, TriggerSpec

log = logging.getLogger(__name__)


@dataclass
class WindowRecord:
    """One source window plus auditable metadata carried through runtime."""

    index: int
    data_item: Any
    array: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _NullExecutor:
    """Context-compatible sequential executor used when num_workers=0."""

    def __enter__(self) -> "_NullExecutor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def map(self, fn: Any, items: list[Any]) -> list[Any]:
        return [fn(item) for item in items]


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
        n_seen = 0

        trigger = self._spec.trigger
        _trigger_streak: int = 0
        _trigger_required: int = int(trigger.when.get("stable_for", 1)) if trigger else 1
        batch_size = max(1, int(getattr(self._spec.runtime, "batch_size", 1) or 1))
        num_workers = max(0, int(getattr(self._spec.runtime, "num_workers", 0) or 0))
        source_failure_policy = getattr(self._spec.runtime, "source_failure_policy", "strict")
        preprocess_failure_policy = getattr(self._spec.runtime, "preprocess_failure_policy", "strict")
        max_windows = getattr(self._spec.runtime, "max_windows", None)
        max_duration_s = getattr(self._spec.runtime, "max_duration_s", None)
        idle_timeout_s = getattr(self._spec.runtime, "idle_timeout_s", None)
        started_at = time.monotonic()
        last_item_at = started_at

        with ThreadPoolExecutor(max_workers=num_workers) if num_workers > 0 else _NullExecutor() as pool:
            stream = iter(self._source.stream())
            while True:
                now = time.monotonic()
                if max_windows is not None and n_seen >= int(max_windows):
                    break
                if max_duration_s is not None and (now - started_at) >= float(max_duration_s):
                    warnings.append(WarningItem(
                        code="RUNTIME_MAX_DURATION_REACHED",
                        message=f"Stopped after runtime.max_duration_s={max_duration_s}.",
                        severity="info",
                    ))
                    break
                if (
                    idle_timeout_s is not None
                    and n_seen == 0
                    and (now - last_item_at) >= float(idle_timeout_s)
                ):
                    errors.append(
                        f"No source windows received before idle_timeout_s={idle_timeout_s}"
                    )
                    break
                batch: list[WindowRecord] = []
                abort_source = False
                for _ in range(batch_size):
                    if max_windows is not None and n_seen >= int(max_windows):
                        break
                    self._profiler.start_source_read()
                    try:
                        data_item = next(stream)
                    except StopIteration:
                        self._profiler.end_source_read()
                        break
                    except Exception as exc:
                        self._profiler.end_source_read()
                        err_msg = f"Source read error at window {n_seen}: {exc}"
                        log.warning(err_msg)
                        errors.append(err_msg)
                        self._profiler.commit_window(dropped=True, error=err_msg)
                        n_seen += 1
                        if source_failure_policy == "strict":
                            abort_source = True
                            break
                        continue
                    self._profiler.end_source_read()
                    last_item_at = time.monotonic()
                    batch.append(
                        WindowRecord(
                            index=n_seen,
                            data_item=data_item,
                            metadata=_extract_metadata(
                                data_item,
                                window_index=n_seen,
                                source_id=self._source.source_id,
                                source_profile=getattr(self, "_compat", None),
                            ),
                        )
                    )
                    n_seen += 1

                if abort_source and not batch:
                    break
                stop_after_batch = abort_source
                if not batch:
                    break

                # ── Preprocessing ───────────────────────────────────────────
                self._profiler.start_preprocess()
                preprocessed: list[WindowRecord] = []
                dropped_preprocess: list[tuple[WindowRecord, Exception]] = []
                try:
                    preprocessed, dropped_preprocess = self._preprocess_batch(
                        batch,
                        pool=pool,
                        num_workers=num_workers,
                        failure_policy=preprocess_failure_policy,
                    )
                except Exception as exc:
                    self._profiler.end_preprocess()
                    err_msg = f"Preprocess error on batch starting at window {batch[0].index}: {exc}"
                    log.warning(err_msg)
                    errors.append(err_msg)
                    self._profiler.commit_batch(len(batch), dropped=True, error=err_msg)
                    if stop_after_batch:
                        break
                    continue
                self._profiler.end_preprocess()
                for record, exc in dropped_preprocess:
                    err_msg = f"Preprocess error on window {record.index}: {exc}"
                    log.warning(err_msg)
                    errors.append(err_msg)
                if not preprocessed:
                    first_error = str(dropped_preprocess[0][1]) if dropped_preprocess else "no preprocessed windows"
                    self._profiler.commit_batch(len(batch), dropped=True, error=first_error)
                    if stop_after_batch:
                        break
                    continue

                # ── Inference ───────────────────────────────────────────────
                self._profiler.start_inference()
                try:
                    arrays = [record.array for record in preprocessed]
                    outputs: list[ModelOutput]
                    if len(arrays) > 1:
                        outputs = self._model.predict_batch(arrays)
                    else:
                        outputs = [self._model.predict(arrays[0])]
                    if len(outputs) != len(preprocessed):
                        raise RuntimeError(
                            f"predict_batch returned {len(outputs)} output(s) for "
                            f"{len(preprocessed)} input window(s)"
                        )
                except Exception as exc:
                    self._profiler.end_inference()
                    err_msg = f"Inference error on batch starting at window {preprocessed[0].index}: {exc}"
                    log.warning(err_msg)
                    errors.append(err_msg)
                    self._profiler.commit_batch(len(preprocessed), dropped=True, error=err_msg)
                    for record, dropped_exc in dropped_preprocess:
                        self._profiler.commit_window(
                            dropped=True,
                            error=f"Preprocess error on window {record.index}: {dropped_exc}",
                        )
                    if stop_after_batch:
                        break
                    continue
                self._profiler.end_inference()

                self._profiler.start_postprocess()
                prepared_outputs: list[tuple[WindowRecord, ModelOutput, bool, dict[str, Any]]] = []
                for record, output in zip(preprocessed, outputs):
                    # ── Postprocess (trigger evaluation) ────────────────────
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
                            log.info("Trigger fired at window %d: %s", record.index, trigger.emit)
                            _trigger_streak = 0
                    meta = dict(record.metadata)
                    meta["trigger_fired"] = trigger_fired
                    prepared_outputs.append((record, output, trigger_fired, meta))
                self._profiler.end_postprocess()

                # ── Output write ────────────────────────────────────────────
                self._profiler.start_output_write()
                for record, output, trigger_fired, meta in prepared_outputs:
                    for out_adapter in self._outputs:
                        try:
                            out_adapter.write(output, metadata=meta)
                        except Exception as exc:
                            err_msg = f"Output write error on window {record.index}: {exc}"
                            log.warning(err_msg)
                            errors.append(err_msg)

                    # Emit structured EventMarker when trigger fires
                    if trigger_fired and trigger is not None:
                        _emit_trigger_event(
                            trigger, record.index, output, self._outputs, self._source.source_id
                        )
                    n_ok += 1
                self._profiler.end_output_write()
                self._profiler.commit_batch(len(prepared_outputs))
                for record, exc in dropped_preprocess:
                    self._profiler.commit_window(
                        dropped=True,
                        error=f"Preprocess error on window {record.index}: {exc}",
                    )
                if stop_after_batch:
                    break

        latency_report = self._profiler.report()
        artifact_contract = self._make_artifact_contract(latency_report)

        n_outputs_written = sum(getattr(o, "n_written", 0) for o in self._outputs)
        if n_ok == 0 and getattr(self._spec.runtime, "fail_on_no_windows", True):
            errors.append("No windows were successfully processed.")
        success = n_ok > 0 and not any(e for e in errors)
        return PipelineRunReport(
            success=success,
            compatibility_report=self._compat,
            preprocess_plan=self._plan,
            latency_report=latency_report,
            artifact_contract=artifact_contract,
            outputs=[
                {
                    "adapter": type(o).__name__,
                    "n_prediction_records": getattr(o, "n_prediction_records", getattr(o, "n_written", 0)),
                    "n_marker_records": getattr(o, "n_marker_records", 0),
                    "n_output_records_total": getattr(o, "n_output_records_total", getattr(o, "n_written", 0)),
                    "n_written": getattr(o, "n_written", 0),
                }
                for o in self._outputs
            ],
            errors=errors,
            warnings=warnings,
            n_windows_processed=n_ok,
            n_outputs_written=n_outputs_written,
        )

    def _preprocess_record(self, record: WindowRecord) -> WindowRecord:
        raw_array = _extract_array(record.data_item)
        record.metadata["input_shape"] = _shape_of(raw_array)
        record.metadata["input_dtype"] = _dtype_of(raw_array)
        record.array = self._executor.apply(raw_array)
        record.metadata["preprocessed_shape"] = _shape_of(record.array)
        record.metadata["preprocessed_dtype"] = _dtype_of(record.array)
        return record

    def _preprocess_batch(
        self,
        batch: list[WindowRecord],
        *,
        pool: Any,
        num_workers: int,
        failure_policy: str,
    ) -> tuple[list[WindowRecord], list[tuple[WindowRecord, Exception]]]:
        if failure_policy == "strict":
            if num_workers > 0 and len(batch) > 1:
                return list(pool.map(self._preprocess_record, batch)), []
            return [self._preprocess_record(record) for record in batch], []

        if failure_policy != "drop_failed":
            raise RuntimeError(
                f"Unsupported runtime.preprocess_failure_policy={failure_policy!r}"
            )

        ok: list[WindowRecord] = []
        dropped: list[tuple[WindowRecord, Exception]] = []
        if num_workers > 0 and len(batch) > 1 and hasattr(pool, "submit"):
            futures = [(record, pool.submit(self._preprocess_record, record)) for record in batch]
            for record, fut in futures:
                try:
                    ok.append(fut.result())
                except Exception as exc:
                    dropped.append((record, exc))
            return ok, dropped

        for record in batch:
            try:
                ok.append(self._preprocess_record(record))
            except Exception as exc:
                dropped.append((record, exc))
        return ok, dropped

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


def _extract_metadata(
    data_item: Any,
    *,
    window_index: int,
    source_id: str,
    source_profile: Any = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "window_index": window_index,
        "source": source_id,
        "source_id": source_id,
    }
    for attr in (
        "shape", "axes", "dtype", "units", "channel_names",
        "sampling_frequency_hz", "timebase", "reference",
        "voxel_sizes_mm", "affine", "coordinate_frame", "tr_s", "n_volumes",
        "columns", "n_events",
    ):
        value = getattr(data_item, attr, None)
        if value is not None:
            metadata[attr] = _json_safe(value)
    provenance = getattr(data_item, "source_provenance", None)
    if isinstance(provenance, dict):
        metadata["source_provenance"] = _json_safe(provenance)
        for key in (
            "path", "subject", "session", "task", "run", "suffix",
            "tmin", "tmax", "onset", "duration", "event_index",
            "series_uid", "study_uid", "timestamp",
        ):
            if key in provenance:
                metadata[key] = _json_safe(provenance[key])
    return metadata


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(v) for v in shape]
    except Exception:
        return [int(v) for v in tuple(shape)]


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    return str(value)
