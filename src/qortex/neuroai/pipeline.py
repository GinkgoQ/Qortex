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
from pathlib import Path
from typing import Any

from qortex.neuroai.benchmark import PipelineProfiler
from qortex.neuroai.compatibility import CompatibilityEngine
from qortex.neuroai.contracts import (
    ArtifactContract,
    CompatibilityReport,
    LatencyReport,
    ModelProfile,
    PipelineRunReport,
    PreprocessPlan,
    SourceProfile,
)
from qortex.neuroai.models._registry import make_model_adapter
from qortex.neuroai.outputs._registry import make_output_adapter
from qortex.neuroai.preprocess.planner import PreprocessPlanner
from qortex.neuroai.runtime.engine import RuntimeEngine
from qortex.neuroai.sources._registry import make_source_adapter
from qortex.neuroai.spec import PipelineSpec

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
        ValueError
            When the spec is invalid.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline YAML not found: {p}")
        spec = PipelineSpec.from_yaml(p)
        errors = spec.validate()
        if errors:
            raise ValueError(
                f"Invalid pipeline spec in {p}:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return cls(spec)

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        """Construct a Pipeline from a dict."""
        spec = PipelineSpec.from_dict(d)
        errors = spec.validate()
        if errors:
            raise ValueError("Invalid pipeline spec:\n" + "\n".join(f"  - {e}" for e in errors))
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
        log.info("Model profile: %s", self._model_profile.model_id)

        # Compatibility check
        engine = CompatibilityEngine()
        self._compat_report = engine.check(
            self._source_profile,
            self._model_profile,
            self._spec.preprocessing,
        )

        # Build preprocessing plan
        planner = PreprocessPlanner()
        self._preprocess_plan = planner.build_plan(
            self._compat_report,
            window_duration_s=(
                self._spec.window.duration_s if self._spec.window else None
            ),
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

        # Open output adapters
        pipeline_ref = self._spec.content_hash()[:12]
        output_adapters = [
            make_output_adapter(out_spec, pipeline_ref=pipeline_ref)
            for out_spec in self._spec.outputs
        ]
        for adapter in output_adapters:
            adapter.open()

        # Execute
        try:
            engine = RuntimeEngine(
                spec=self._spec,
                source=self._source_adapter,
                model=self._model_adapter,
                plan=self._preprocess_plan,
                outputs=output_adapters,
                compat_report=self._compat_report,
            )
            report = engine.run()
        finally:
            for adapter in output_adapters:
                try:
                    adapter.close()
                except Exception as exc:
                    log.warning("Error closing output adapter: %s", exc)
            self._model_adapter.unload()

        _write_artifact_sidecar(self._spec, report)

        if artifact_dir is not None:
            try:
                from qortex.neuroai.artifact import ArtifactWriter
                writer = ArtifactWriter(artifact_dir, pipeline_ref=pipeline_ref)
                writer.write(
                    spec=self._spec,
                    compat_report=self._compat_report,
                    preprocess_plan=self._preprocess_plan,
                    run_report=report,
                    source_profile=self._source_profile,
                    model_profile=self._model_profile,
                )
            except Exception as exc:
                log.warning("ArtifactWriter failed (non-fatal): %s", exc)

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

        from qortex.neuroai.outputs.jsonl_out import JSONLOutputAdapter
        from io import StringIO

        # Dummy null output — no real I/O
        class NullOutput:
            n_written = 0
            def open(self): pass
            def write(self, *a, **kw): self.n_written += 1
            def close(self): pass

        log.info("Benchmark: loading model %s…", self._spec.model.id)
        self._model_adapter.load(self._spec.runtime)

        profiler = PipelineProfiler(budget_ms=self._spec.runtime.latency_budget_ms)
        try:
            engine = RuntimeEngine(
                spec=self._spec,
                source=self._source_adapter,
                model=self._model_adapter,
                plan=self._preprocess_plan,
                outputs=[NullOutput()],
                compat_report=self._compat_report,
                profiler=profiler,
            )
            # Limit to n_windows
            from itertools import islice
            import types

            def _limited_stream(adapter, n):
                yield from islice(adapter.stream(), n)

            original_stream = self._source_adapter.stream
            self._source_adapter.stream = lambda: _limited_stream(self._source_adapter, n_windows)
            try:
                engine.run()
            finally:
                self._source_adapter.stream = original_stream

        finally:
            self._model_adapter.unload()

        return profiler.report()

    # ── Replay ────────────────────────────────────────────────────────────────

    def replay(
        self,
        source_path: str | Path,
        *,
        speed: float = 1.0,
        output_dir: Path | None = None,
    ) -> PipelineRunReport:
        """Replay a recorded session through the pipeline.

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

        replay_spec = SourceSpec(
            type="local_file",
            path=str(source_path),
            modality=self._spec.source.modality,
        )
        original_source = self._source_adapter
        self._source_adapter = make_source_adapter(
            replay_spec,
            window_spec=self._spec.window,
        )

        # Override outputs to replay dir if given
        original_outputs = self._spec.outputs
        if output_dir:
            from qortex.neuroai.spec import OutputSpec
            self._spec.outputs = [
                OutputSpec(
                    type=o.type,
                    path=str(output_dir / Path(o.path or "replay.jsonl").name),
                    stream_name=o.stream_name,
                )
                for o in original_outputs
            ]

        try:
            report = self.run()
        finally:
            self._source_adapter = original_source
            self._spec.outputs = original_outputs

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

def _make_runtime_error(msg: str) -> Exception:
    try:
        from qortex.core.exceptions import QortexError
        return type("RuntimeExecutionError", (QortexError,), {})(msg)
    except ImportError:
        return RuntimeError(msg)


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
