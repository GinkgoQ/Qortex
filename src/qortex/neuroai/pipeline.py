"""Pipeline — top-level NeuroAI runtime facade.

``Pipeline`` is the primary user-facing object for the NeuroAI runtime.
It orchestrates check → plan → load → run in a coherent, contract-driven flow.

Usage::

    from qortex.neuroai import Pipeline

    pipe = Pipeline.from_yaml("pipeline.yaml")

    # Step 1: check compatibility (no model weights loaded)
    report = pipe.check()
    print(report.summary())

    if report.is_runnable:
        # Step 2: run — loads model, executes, writes outputs
        run_report = pipe.run()
        print(run_report.latency_report.summary())

    # Optional: benchmark latency without writing outputs
    bench = pipe.benchmark(n_windows=50)
    print(bench.summary())

    # Optional: replay from an XDF file
    pipe.replay("recording.xdf")
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from itertools import islice
from pathlib import Path
from typing import Any

from qortex.neuroai.benchmark import PipelineProfiler
from qortex.neuroai.compatibility import CompatibilityEngine
from qortex.neuroai.contracts import (
    CompatibilityReport,
    InputContract,
    LatencyReport,
    ModelProfile,
    OutputContract,
    PipelineRunReport,
    PreprocessPlan,
    SourceProfile,
)
from qortex.neuroai.models._registry import make_model_adapter
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter
from qortex.neuroai.outputs._registry import make_output_adapter
from qortex.neuroai.preprocess.planner import PreprocessPlanner
from qortex.neuroai.runtime.engine import RuntimeEngine
from qortex.neuroai.sources._registry import make_source_adapter
from qortex.neuroai.spec import PipelineSpec
from qortex.core.exceptions import ContractValidationError, RuntimeExecutionError

log = logging.getLogger(__name__)


class Pipeline:
    """Declarative NeuroAI pipeline from source → model → output.

    Do not construct directly — use ``from_yaml()`` or ``from_dict()``.

    Parameters
    ----------
    spec:
        Parsed pipeline specification.
    """

    def __init__(self, spec: PipelineSpec) -> None:
        self._spec = spec
        self._source_profile: SourceProfile | None = None
        self._model_profile: ModelProfile | None = None
        self._compat_report: CompatibilityReport | None = None
        self._preprocess_plan: PreprocessPlan | None = None
        self._model_adapter = None
        self._source_adapter = None
        self._checked = False

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Pipeline":
        """Load a pipeline from a YAML file.

        Parameters
        ----------
        path:
            Path to the pipeline YAML file.

        Returns
        -------
        Pipeline

        Raises
        ------
        FileNotFoundError
            When the YAML file does not exist.
        ContractValidationError
            When the spec is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline YAML not found: {p}")
        try:
            spec = PipelineSpec.from_yaml(p)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError(f"PipelineSpec({p})", [str(exc)]) from exc
        errors = spec.validate()
        if errors:
            raise ContractValidationError(f"PipelineSpec({p})", errors)
        return cls(spec)

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        """Construct a Pipeline from a dict."""
        try:
            spec = PipelineSpec.from_dict(d)
        except (TypeError, ValueError) as exc:
            raise ContractValidationError("PipelineSpec", [str(exc)]) from exc
        errors = spec.validate()
        if errors:
            raise ContractValidationError("PipelineSpec", errors)
        return cls(spec)

    # ── Check ────────────────────────────────────────────────────────────────

    def check(self) -> CompatibilityReport:
        """Verify source-model compatibility without loading model weights.

        This is the main safety gate.  Run it before ``run()`` to catch
        modality, channel, sampling rate, shape, and dtype mismatches early.

        Returns
        -------
        CompatibilityReport
            Status, required transforms, blockers, warnings, and unknowns.
        """
        log.info("Pipeline.check(): probing source and model…")

        # Probe source (header only)
        self._source_adapter = make_source_adapter(
            self._spec.source,
            window_spec=self._spec.window,
        )
        self._source_profile = self._source_adapter.probe()
        log.info("Source profile: %s", self._source_profile.source_id)

        # Inspect model (no weight download)
        self._model_adapter = make_model_adapter(self._spec.model)
        self._model_profile = self._model_adapter.inspect()
        _apply_model_contract_overrides(self._model_profile, self._spec.model)
        log.info("Model profile: %s", self._model_profile.model_id)

        # Compatibility check
        engine = CompatibilityEngine()
        self._compat_report = engine.check(
            self._source_profile,
            self._model_profile,
            self._spec.preprocessing,
            runtime=self._spec.runtime,
            window=self._spec.window,
        )

        # Build preprocessing plan — pass source_profile so the planner can
        # auto-insert rescale_intensity for DICOM/MRI modalities; pass
        # model_provider so to_tensor knows whether to emit torch or numpy.
        planner = PreprocessPlanner()
        self._preprocess_plan = planner.build_plan(
            self._compat_report,
            window_duration_s=(
                self._spec.window.duration_s if self._spec.window else None
            ),
            source_profile=self._source_profile,
            model_provider=self._spec.model.provider,
        )

        self._checked = True
        return self._compat_report

    # ── Plan preprocessing ────────────────────────────────────────────────────

    def plan_preprocessing(self) -> PreprocessPlan:
        """Return the deterministic preprocessing plan for this pipeline.

        Runs ``check()`` if not already done.  Returns the full transform
        chain with documentation of each step, why it is required, and whether
        it is reversible.

        Returns
        -------
        PreprocessPlan
            Ordered list of transforms with explanations.

        Examples
        --------
        >>> plan = pipe.plan_preprocessing()
        >>> for t in plan.transforms:
        ...     print(f"  {t.kind.value}: {t.required_by}")
        """
        if not self._checked:
            self.check()
        return self._preprocess_plan

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, *, artifact_dir: str | Path | None = None) -> PipelineRunReport:
        """Execute the full pipeline: source → preprocess → model → outputs.

        ``check()`` is automatically called if not already done.

        Returns
        -------
        PipelineRunReport
            Includes latency report, artifact contract, errors, and output counts.
        """
        if not self._checked:
            log.warning("Pipeline.run() called without prior check(). Running check now.")
            self.check()

        compat = self._compat_report
        if compat and not compat.is_runnable:
            raise _make_runtime_error(
                f"Pipeline {self._spec.name!r} is not runnable: "
                f"status={compat.status.value}. "
                f"Blockers: {[b.message for b in compat.blockers]}"
            )

        # Load model weights
        log.info("Pipeline.run(): loading model %s…", self._spec.model.id)
        self._model_adapter.load(self._spec.runtime)

        pipeline_ref = self._spec.content_hash()[:12]
        run_spec = self._spec
        if artifact_dir is not None:
            run_spec = _spec_with_artifact_outputs(self._spec, Path(artifact_dir))

        # Open output adapters
        output_adapters = [
            make_output_adapter(out_spec, pipeline_ref=pipeline_ref)
            for out_spec in run_spec.outputs
        ]
        for adapter in output_adapters:
            adapter.open()

        # Execute
        try:
            engine = RuntimeEngine(
                spec=run_spec,
                source=self._source_adapter,
                model=self._model_adapter,
                plan=self._preprocess_plan,
                outputs=output_adapters,
                compat_report=self._compat_report,
            )
            report = engine.run()
            report.source_profile = self._source_profile
            report.model_profile = self._model_profile
            _attach_output_contract_to_artifact(report, self._model_profile)
        finally:
            for adapter in output_adapters:
                try:
                    adapter.close()
                except Exception as exc:
                    log.warning("Error closing output adapter: %s", exc)
            self._model_adapter.unload()

        _write_artifact_sidecar(run_spec, report)

        if artifact_dir is not None:
            try:
                from qortex.neuroai.artifact import ArtifactWriter
                writer = ArtifactWriter(artifact_dir, pipeline_ref=pipeline_ref)
                writer.write(
                    spec=run_spec,
                    compat_report=self._compat_report,
                    preprocess_plan=self._preprocess_plan,
                    run_report=report,
                    source_profile=self._source_profile,
                    model_profile=self._model_profile,
                )
            except Exception as exc:
                failure_policy = str(getattr(self._spec.artifact, "failure_policy", "strict")).lower()
                if failure_policy == "warn":
                    log.warning("ArtifactWriter failed (non-fatal by policy): %s", exc)
                else:
                    raise _make_runtime_error(
                        f"ArtifactWriter failed for requested artifact_dir={artifact_dir}: {exc}"
                    ) from exc

        return report

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(self, n_windows: int = 20) -> LatencyReport:
        """Benchmark pipeline latency without writing real outputs.

        Loads the model, runs ``n_windows`` windows through the full chain
        (source → preprocess → inference) and returns latency statistics.

        Parameters
        ----------
        n_windows:
            Number of windows to time.

        Returns
        -------
        LatencyReport
            p50/p95/p99 and per-stage breakdown.
        """
        if not self._checked:
            self.check()

        compat = self._compat_report
        if compat and not compat.is_runnable:
            raise _make_runtime_error(
                f"Cannot benchmark — pipeline is not runnable: {compat.status.value}"
            )

        log.info("Benchmark: loading model %s…", self._spec.model.id)
        self._model_adapter.load(self._spec.runtime)

        profiler = PipelineProfiler(budget_ms=self._spec.runtime.latency_budget_ms)
        output_adapter = BenchmarkOutputAdapter()
        output_adapter.open()
        try:
            engine = RuntimeEngine(
                spec=self._spec,
                source=self._source_adapter,
                model=self._model_adapter,
                plan=self._preprocess_plan,
                outputs=[output_adapter],
                compat_report=self._compat_report,
                profiler=profiler,
                source_iterator=lambda: islice(self._source_adapter.stream(), n_windows),
            )
            engine.run()

        finally:
            output_adapter.close()
            self._model_adapter.unload()

        report = profiler.report()
        report.requested_windows = int(n_windows)
        report.source_exhausted = report.n_windows < int(n_windows)
        return report

    # ── Replay ────────────────────────────────────────────────────────────────

    def replay(
        self,
        source_path: str | Path,
        *,
        speed: float = 1.0,
        output_dir: Path | None = None,
    ) -> PipelineRunReport:
        """Replay a recorded session through the pipeline.

        Unlike ``run()``, replay swaps the source adapter so that the same model
        and output configuration is applied to a different (recorded) data file.
        Compatibility is re-checked against the replay source, so a new
        ``CompatibilityReport`` and ``PreprocessPlan`` are always computed.

        Parameters
        ----------
        source_path:
            Path to a recorded session file (XDF, EDF, or any supported format).
        speed:
            Playback speed multiplier (1.0 = real-time, 2.0 = 2× speed).
        output_dir:
            Optional output directory override.

        Returns
        -------
        PipelineRunReport
        """
        from qortex.neuroai.spec import SourceSpec
        if speed <= 0:
            raise ValueError("Replay speed must be greater than 0.")

        replay_spec = SourceSpec(
            type="local_file",
            path=str(source_path),
            modality=self._spec.source.modality,
        )

        original_outputs = self._spec.outputs
        replay_outputs = original_outputs

        if output_dir:
            from qortex.neuroai.spec import OutputSpec
            replay_outputs = [
                OutputSpec(
                    type=o.type,
                    path=str(output_dir / Path(o.path or "replay.jsonl").name),
                    stream_name=o.stream_name,
                )
                for o in original_outputs
            ]

        replay_pipeline = Pipeline(replace(
            self._spec,
            source=replay_spec,
            outputs=replay_outputs,
        ))
        replay_pipeline.check()
        log.info("Pipeline.replay(): replaying %s at %.3gx speed", source_path, speed)

        compat = replay_pipeline._compat_report
        if compat and not compat.is_runnable:
            raise _make_runtime_error(
                f"Replay pipeline is not runnable: status={compat.status.value}. "
                f"Blockers: {[b.message for b in compat.blockers]}"
            )

        replay_pipeline._model_adapter.load(replay_pipeline._spec.runtime)
        pipeline_ref = replay_pipeline._spec.content_hash()[:12]
        output_adapters = [
            make_output_adapter(out_spec, pipeline_ref=pipeline_ref)
            for out_spec in replay_pipeline._spec.outputs
        ]
        for adapter in output_adapters:
            adapter.open()

        try:
            engine = RuntimeEngine(
                spec=replay_pipeline._spec,
                source=replay_pipeline._source_adapter,
                model=replay_pipeline._model_adapter,
                plan=replay_pipeline._preprocess_plan,
                outputs=output_adapters,
                compat_report=replay_pipeline._compat_report,
                source_iterator=lambda: replay_pipeline._source_adapter.replay(speed=speed),
            )
            report = engine.run()
            report.source_profile = replay_pipeline._source_profile
            report.model_profile = replay_pipeline._model_profile
            _attach_output_contract_to_artifact(report, replay_pipeline._model_profile)
        finally:
            for adapter in output_adapters:
                try:
                    adapter.close()
                except Exception as exc:
                    log.warning("Error closing replay output adapter: %s", exc)
            replay_pipeline._model_adapter.unload()

        _write_artifact_sidecar(replay_pipeline._spec, report)

        return report

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def spec(self) -> PipelineSpec:
        return self._spec

    @property
    def source_profile(self) -> SourceProfile | None:
        return self._source_profile

    @property
    def model_profile(self) -> ModelProfile | None:
        return self._model_profile

    @property
    def compatibility_report(self) -> CompatibilityReport | None:
        return self._compat_report

    @property
    def preprocess_plan(self) -> PreprocessPlan | None:
        return self._preprocess_plan


# ── Helpers ───────────────────────────────────────────────────────────────────

class BenchmarkOutputAdapter(OutputAdapter):
    """In-memory output sink used by ``Pipeline.benchmark()``.

    The adapter exercises the same runtime output-writing path as file and
    streaming adapters while deliberately avoiding external I/O.
    """

    def __init__(self) -> None:
        self._n_written = 0
        self._is_open = False

    def open(self) -> None:
        self._is_open = True

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        if not self._is_open:
            raise RuntimeExecutionError("Benchmark output adapter is not open")
        self._n_written += 1

    def close(self) -> None:
        self._is_open = False


def _make_runtime_error(msg: str) -> Exception:
    return RuntimeExecutionError(msg)


def _spec_with_artifact_outputs(spec: PipelineSpec, artifact_dir: Path) -> PipelineSpec:
    """Return a shallow spec copy with file outputs routed into artifact_dir/outputs."""
    outputs_dir = artifact_dir / "outputs"
    routed_outputs = []
    seen_names: dict[str, int] = {}
    for out in spec.outputs:
        out_type = (out.type or "").lower().strip()
        if out_type in {"lsl_marker", "lsl", "websocket", "ws", "http", "http_callback", "webhook"}:
            routed_outputs.append(out)
            continue
        original = Path(out.path) if out.path else Path(_default_output_name(out_type))
        name = original.name or _default_output_name(out_type)
        count = seen_names.get(name, 0)
        seen_names[name] = count + 1
        if count:
            stem = Path(name).stem
            suffix = "".join(Path(name).suffixes)
            name = f"{stem}_{count}{suffix}" if suffix else f"{name}_{count}"
        routed_outputs.append(replace(out, path=str(outputs_dir / name)))
    return replace(spec, outputs=routed_outputs)


def _default_output_name(out_type: str) -> str:
    if out_type in {"parquet"}:
        return "predictions.parquet"
    if out_type in {"csv"}:
        return "predictions.csv"
    if out_type in {"nifti", "nii", "nifti_mask"}:
        return "mask.nii.gz"
    if out_type in {"coco", "coco_json"}:
        return "predictions_coco.json"
    if out_type in {"dicom_seg", "dicomseg"}:
        return "output_seg"
    if out_type in {"dicom_sr", "dicomsr"}:
        return "output_sr"
    if out_type in {"bids", "bids_derivative"}:
        return "derivatives"
    if out_type in {"yolo", "yolo_txt"}:
        return "yolo_labels"
    if out_type in {"overlay", "image_overlay", "video_overlay"}:
        return "annotated_frames"
    return "predictions.jsonl"


def _write_artifact_sidecar(spec: PipelineSpec, report: PipelineRunReport) -> None:
    """Write per-run provenance JSON sidecar next to the first output file."""
    try:
        if not spec.outputs or not spec.outputs[0].path:
            return
        out_path = Path(spec.outputs[0].path)
        sidecar = out_path.parent / f"{out_path.stem}_provenance.json"
        contract = report.artifact_contract
        if contract is None:
            return
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        data = contract.model_dump() if hasattr(contract, "model_dump") else contract.__dict__
        sidecar.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("Provenance written to %s", sidecar)
    except Exception as exc:
        log.debug("Could not write provenance sidecar: %s", exc)


def _attach_output_contract_to_artifact(
    report: PipelineRunReport,
    model_profile: ModelProfile | None,
) -> None:
    contract = getattr(report, "artifact_contract", None)
    output_contract = getattr(model_profile, "output_contract", None) if model_profile else None
    if contract is None or output_contract is None:
        return
    try:
        output_type = getattr(output_contract, "output_type", None)
        if output_type:
            contract.output_type = str(output_type)
        if hasattr(output_contract, "model_dump"):
            schema = output_contract.model_dump()
        else:
            schema = dict(getattr(output_contract, "__dict__", {}))
        contract.output_schema = json.dumps(_json_safe(schema), sort_keys=True)
    except Exception as exc:
        log.debug("Could not attach model output contract to artifact contract: %s", exc)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _apply_model_contract_overrides(profile: ModelProfile, model_spec: Any) -> None:
    input_override = getattr(model_spec, "input_contract", None)
    output_override = getattr(model_spec, "output_contract", None)
    if input_override:
        profile.input_contract = _coerce_input_contract(input_override)
    if output_override:
        profile.output_contract = _coerce_output_contract(output_override)


def _coerce_input_contract(value: Any) -> InputContract:
    if isinstance(value, InputContract):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise ContractValidationError(
            "model.input_contract",
            [f"expected mapping, got {type(value).__name__}"],
        )
    try:
        return InputContract(**value)
    except Exception as exc:
        raise ContractValidationError("model.input_contract", [str(exc)]) from exc


def _coerce_output_contract(value: Any) -> OutputContract:
    if isinstance(value, OutputContract):
        return value
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        raise ContractValidationError(
            "model.output_contract",
            [f"expected mapping, got {type(value).__name__}"],
        )
    try:
        return OutputContract(**value)
    except Exception as exc:
        raise ContractValidationError("model.output_contract", [str(exc)]) from exc
