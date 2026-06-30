"""Pipeline specification — YAML-driven declarative pipeline description.

``PipelineSpec`` is the single source of truth for a Qortex NeuroAI pipeline.
It can be loaded from a YAML file, constructed programmatically, and serialised
back to YAML for reproducibility.

All fields map to the AGENT.md §8 YAML example.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal


_VALID_TRANSFORMS = frozenset({
    "resample", "channel_select", "channel_reorder", "channel_map",
    "bandpass", "normalize", "window", "cast_dtype", "rescale_intensity",
    "reorient", "resample_spatial", "pad_or_crop",
    "add_batch_dim", "add_channel_dim", "transpose_axes", "to_tensor",
})


# ── Sub-specs ──────────────────────────────────────────────────────────────────

@dataclass
class SourceSpec:
    """Declaration of the data source for the pipeline."""

    type: str                                   # "lsl" | "xdf" | "edf" | "bids" | "local_file" | "nifti" | ...
    path: str | None = None                     # for file-based sources
    query: dict[str, Any] = field(default_factory=dict)  # for LSL stream query
    subjects: list[str] | None = None
    sessions: list[str] | None = None
    modality: str | None = None
    suffix: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("SourceSpec.from_dict() requires a mapping")
        subjects = _as_string_list(d.get("subjects", d.get("subject")))
        sessions = _as_string_list(d.get("sessions", d.get("session")))
        return cls(
            type=d.get("type", ""),
            path=d.get("path"),
            query=d.get("query", {}),
            subjects=subjects,
            sessions=sessions,
            modality=d.get("modality"),
            suffix=d.get("suffix"),
            extra={k: v for k, v in d.items()
                   if k not in ("type", "path", "query", "subjects",
                                "subject", "sessions", "session",
                                "modality", "suffix")},
        )

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.path:
            d["path"] = self.path
        if self.query:
            d["query"] = self.query
        if self.subjects:
            d["subjects"] = self.subjects
        if self.sessions:
            d["sessions"] = self.sessions
        if self.modality:
            d["modality"] = self.modality
        if self.suffix:
            d["suffix"] = self.suffix
        d.update(self.extra)
        return d


@dataclass
class WindowSpec:
    """Sliding-window configuration for streaming sources."""

    duration_s: float | None = None         # e.g. 2.0
    step_s: float | None = None             # e.g. 0.25 (250ms)
    overlap_frac: float = 0.0              # fraction of window overlap
    tmin: float = 0.0
    event_aligned: bool = False
    drop_short: bool = True                 # drop windows shorter than duration

    @classmethod
    def from_dict(cls, d: dict) -> "WindowSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("WindowSpec.from_dict() requires a mapping")

        def _parse_time(v) -> float | None:
            if v is None:
                return None
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip()
            if s.endswith("ms"):
                return float(s[:-2]) / 1000.0
            if s.endswith("s"):
                return float(s[:-1])
            return float(s)

        return cls(
            duration_s=_parse_time(d.get("duration", d.get("duration_s"))),
            step_s=_parse_time(d.get("step", d.get("step_s"))),
            overlap_frac=float(d.get("overlap", d.get("overlap_frac", 0.0))),
            tmin=float(d.get("tmin", 0.0)),
            event_aligned=_as_bool(d.get("event_aligned", False)),
            drop_short=_as_bool(d.get("drop_short", True)),
        )

    def to_dict(self) -> dict:
        d: dict = {}
        if self.duration_s is not None:
            d["duration"] = f"{self.duration_s}s"
        if self.step_s is not None:
            d["step"] = f"{self.step_s}s"
        if self.overlap_frac:
            d["overlap"] = self.overlap_frac
        if self.tmin:
            d["tmin"] = self.tmin
        if self.event_aligned:
            d["event_aligned"] = True
        if not self.drop_short:
            d["drop_short"] = False
        return d


@dataclass
class ModelSpec:
    """Declaration of the model to use."""

    provider: str                           # "huggingface" | "onnx" | "torch" | "custom"
    id: str                                 # model ID or file path
    task: str | None = None                 # "eeg_classification" | "segmentation" | ...
    revision: str | None = None
    trust_remote_code: bool = False
    input_contract: dict[str, Any] | None = None
    output_contract: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("ModelSpec.from_dict() requires a mapping")
        return cls(
            provider=d.get("provider", "huggingface"),
            id=d.get("id", ""),
            task=d.get("task"),
            revision=d.get("revision"),
            trust_remote_code=_as_bool(d.get("trust_remote_code", False)),
            input_contract=d.get("input_contract"),
            output_contract=d.get("output_contract"),
            extra={k: v for k, v in d.items()
                   if k not in (
                       "provider", "id", "task", "revision", "trust_remote_code",
                       "input_contract", "output_contract",
                   )},
        )

    def to_dict(self) -> dict:
        d: dict = {"provider": self.provider, "id": self.id}
        if self.task:
            d["task"] = self.task
        if self.revision:
            d["revision"] = self.revision
        if self.trust_remote_code:
            d["trust_remote_code"] = True
        if self.input_contract:
            d["input_contract"] = _json_safe(self.input_contract)
        if self.output_contract:
            d["output_contract"] = _json_safe(self.output_contract)
        d.update(self.extra)
        return d


@dataclass
class PreprocessSpec:
    """What automatic preprocessing is allowed."""

    mode: Literal["auto", "explicit", "none"] = "auto"
    allow: list[str] = field(default_factory=list)   # transform kinds allowed
    deny: list[str] = field(default_factory=list)    # transform kinds forbidden
    normalize: bool = True
    resample: bool = True
    channel_select: bool = True
    channel_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("PreprocessSpec.from_dict() requires a mapping")
        return cls(
            mode=d.get("mode", "auto"),
            allow=list(d.get("allow", [])),
            deny=list(d.get("deny", [])),
            normalize=_as_bool(d.get("normalize", True)),
            resample=_as_bool(d.get("resample", True)),
            channel_select=_as_bool(d.get("channel_select", True)),
            channel_map={str(k): str(v) for k, v in (d.get("channel_map") or {}).items()},
        )

    def allows(self, transform_kind: str) -> bool:
        transform_kind = str(transform_kind)
        if self.mode == "none":
            return False
        if transform_kind == "normalize" and not self.normalize:
            return False
        if transform_kind in {"resample", "resample_spatial"} and not self.resample:
            return False
        if transform_kind == "channel_select" and not self.channel_select:
            return False
        if transform_kind in self.deny:
            return False
        if self.allow:
            return transform_kind in self.allow
        return True

    def to_dict(self) -> dict:
        d: dict = {"mode": self.mode}
        if self.allow:
            d["allow"] = self.allow
        if self.deny:
            d["deny"] = self.deny
        if not self.normalize:
            d["normalize"] = False
        if not self.resample:
            d["resample"] = False
        if not self.channel_select:
            d["channel_select"] = False
        if self.channel_map:
            d["channel_map"] = dict(self.channel_map)
        return d


@dataclass
class RuntimeSpec:
    """Execution environment specification."""

    device: str = "auto"           # "auto" | "cpu" | "cuda" | "mps" | "cuda:0"
    latency_budget_ms: float | None = None
    optimize: Literal["safe", "speed", "memory"] = "safe"
    num_workers: int = 0
    batch_size: int = 1
    fp16: bool = False             # requires explicit opt-in
    cache_model: bool = True
    source_failure_policy: Literal["strict", "skip_window", "continue_recording"] = "strict"
    preprocess_failure_policy: Literal["strict", "drop_failed"] = "strict"
    max_windows: int | None = None
    max_duration_s: float | None = None
    idle_timeout_s: float | None = None
    fail_on_no_windows: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("RuntimeSpec.from_dict() requires a mapping")
        return cls(
            device=str(d.get("device", "auto")),
            latency_budget_ms=float(d["latency_budget_ms"]) if "latency_budget_ms" in d else None,
            optimize=d.get("optimize", "safe"),
            num_workers=int(d.get("num_workers", 0)),
            batch_size=int(d.get("batch_size", 1)),
            fp16=_as_bool(d.get("fp16", False)),
            cache_model=_as_bool(d.get("cache_model", True)),
            source_failure_policy=d.get("source_failure_policy", "strict"),
            preprocess_failure_policy=d.get("preprocess_failure_policy", "strict"),
            max_windows=int(d["max_windows"]) if d.get("max_windows") is not None else None,
            max_duration_s=float(d["max_duration_s"]) if d.get("max_duration_s") is not None else None,
            idle_timeout_s=float(d["idle_timeout_s"]) if d.get("idle_timeout_s") is not None else None,
            fail_on_no_windows=_as_bool(d.get("fail_on_no_windows", True)),
        )

    def to_dict(self) -> dict:
        d: dict = {"device": self.device, "optimize": self.optimize}
        if self.latency_budget_ms is not None:
            d["latency_budget_ms"] = self.latency_budget_ms
        if self.num_workers:
            d["num_workers"] = self.num_workers
        if self.batch_size != 1:
            d["batch_size"] = self.batch_size
        if self.fp16:
            d["fp16"] = True
        if not self.cache_model:
            d["cache_model"] = False
        if self.source_failure_policy != "strict":
            d["source_failure_policy"] = self.source_failure_policy
        if self.preprocess_failure_policy != "strict":
            d["preprocess_failure_policy"] = self.preprocess_failure_policy
        if self.max_windows is not None:
            d["max_windows"] = self.max_windows
        if self.max_duration_s is not None:
            d["max_duration_s"] = self.max_duration_s
        if self.idle_timeout_s is not None:
            d["idle_timeout_s"] = self.idle_timeout_s
        if not self.fail_on_no_windows:
            d["fail_on_no_windows"] = False
        return d


@dataclass
class OutputSpec:
    """One output destination for pipeline results."""

    type: str                               # "jsonl" | "parquet" | "lsl_marker" | "nifti" | ...
    path: str | None = None
    stream_name: str | None = None          # for LSL
    append: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "OutputSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("OutputSpec.from_dict() requires a mapping")
        return cls(
            type=d.get("type", "jsonl"),
            path=d.get("path"),
            stream_name=d.get("stream_name"),
            append=_as_bool(d.get("append", False)),
            extra={k: v for k, v in d.items()
                   if k not in ("type", "path", "stream_name", "append")},
        )

    def to_dict(self) -> dict:
        d: dict = {"type": self.type}
        if self.path:
            d["path"] = self.path
        if self.stream_name:
            d["stream_name"] = self.stream_name
        if self.append:
            d["append"] = True
        d.update(self.extra)
        return d


@dataclass
class TriggerSpec:
    """Optional trigger rule for closed-loop emission."""

    when: dict[str, Any] = field(default_factory=dict)  # condition dict
    emit: dict[str, Any] = field(default_factory=dict)  # action dict

    @classmethod
    def from_dict(cls, d: dict) -> "TriggerSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("TriggerSpec.from_dict() requires a mapping")
        return cls(
            when=d.get("when", {}),
            emit=d.get("emit", {}),
        )

    def to_dict(self) -> dict:
        return {"when": self.when, "emit": self.emit}

    def evaluate(self, prediction: dict[str, Any]) -> bool:
        """Return True if the trigger condition is satisfied by this prediction."""
        class_name = self.when.get("class")
        prob_gte = self.when.get("probability_gte")

        if class_name is None:
            return False

        probs = prediction.get("probabilities", {})
        try:
            prob = float(probs.get(class_name, 0.0))
        except (TypeError, ValueError):
            return False
        if prob_gte is not None:
            try:
                prob_gte = float(prob_gte)
            except (TypeError, ValueError):
                return False
        if prob_gte is not None and prob < prob_gte:
            return False

        predicted_class = prediction.get("class")
        if predicted_class != class_name:
            return False

        return True


@dataclass
class ArtifactSpec:
    """Artifact writing policy for reproducible NeuroAI runs."""

    failure_policy: Literal["strict", "warn"] = "strict"

    @classmethod
    def from_dict(cls, d: dict) -> "ArtifactSpec":
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise TypeError("ArtifactSpec.from_dict() requires a mapping")
        return cls(failure_policy=str(d.get("failure_policy", "strict")).lower())

    def to_dict(self) -> dict:
        return {"failure_policy": self.failure_policy}


# ── Pipeline Spec ─────────────────────────────────────────────────────────────

@dataclass
class PipelineSpec:
    """Declarative description of a complete NeuroAI pipeline.

    Loaded from YAML or constructed programmatically.  All planning, checking,
    and execution uses this as the single source of truth.

    Usage::

        spec = PipelineSpec.from_yaml("pipeline.yaml")
        spec.validate()
        hash = spec.content_hash()
    """

    name: str = "unnamed_pipeline"
    source: SourceSpec = field(default_factory=lambda: SourceSpec(type="local_file"))
    window: WindowSpec | None = None
    model: ModelSpec = field(default_factory=lambda: ModelSpec(provider="huggingface", id=""))
    preprocessing: PreprocessSpec = field(default_factory=PreprocessSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    outputs: list[OutputSpec] = field(default_factory=list)
    trigger: TriggerSpec | None = None
    artifact: ArtifactSpec = field(default_factory=ArtifactSpec)
    description: str = ""
    version: str = "1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineSpec":
        """Load a PipelineSpec from a YAML file."""
        content = Path(path).read_text(encoding="utf-8")
        try:
            from ruamel.yaml import YAML
            d = YAML(typ="safe").load(content)
        except ImportError:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "YAML loading requires ruamel.yaml or PyYAML. "
                    "Install qortex with its declared runtime dependencies."
                ) from None
            d = yaml.safe_load(content)
        if d is None:
            d = {}
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineSpec":
        if not isinstance(d, dict):
            raise TypeError("PipelineSpec.from_dict() requires a mapping")
        source_d = d.get("source", {})
        window_d = d.get("window")
        model_d = d.get("model", {})
        preprocess_d = d.get("preprocessing", {})
        runtime_d = d.get("runtime", {})
        outputs_d = d.get("output", d.get("outputs", []))
        trigger_d = d.get("trigger")
        artifact_d = d.get("artifact", {})

        if isinstance(outputs_d, dict):
            outputs_d = [outputs_d]

        return cls(
            name=d.get("name", "unnamed_pipeline"),
            source=SourceSpec.from_dict(source_d),
            window=WindowSpec.from_dict(window_d) if window_d else None,
            model=ModelSpec.from_dict(model_d),
            preprocessing=PreprocessSpec.from_dict(preprocess_d),
            runtime=RuntimeSpec.from_dict(runtime_d),
            outputs=[OutputSpec.from_dict(o) for o in outputs_d],
            trigger=TriggerSpec.from_dict(trigger_d) if trigger_d else None,
            artifact=ArtifactSpec.from_dict(artifact_d),
            description=d.get("description", ""),
            version=str(d.get("version", "1")),
        )

    def to_dict(self) -> dict:
        d: dict = {
            "name": self.name,
            "version": self.version,
            "source": self.source.to_dict(),
            "model": self.model.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
            "runtime": self.runtime.to_dict(),
            "outputs": [o.to_dict() for o in self.outputs],
            "artifact": self.artifact.to_dict(),
        }
        if self.window:
            d["window"] = self.window.to_dict()
        if self.trigger:
            d["trigger"] = self.trigger.to_dict()
        if self.description:
            d["description"] = self.description
        return d

    def to_yaml(self) -> str:
        try:
            from ruamel.yaml import YAML
            from io import StringIO
            buf = StringIO()
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.dump(self.to_dict(), buf)
            return buf.getvalue()
        except ImportError:
            try:
                import yaml
            except ImportError:
                return json.dumps(self.to_dict(), indent=2)
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    def content_hash(self) -> str:
        """SHA-256 hash of the canonical spec content for provenance."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def validate(self) -> list[str]:
        """Return a list of validation errors; empty = valid.

        Checks:
        - Required fields (source.type, model.id, model.provider)
        - Local plugin security gate
        - At least one output declared
        - Output, runtime, trigger, and source sanity
        - Window timing sanity (duration_s > 0, step_s > 0, step_s <= duration_s)
        - All preprocessing.allow values are valid TransformKind names
        - Provider must be a known value
        """
        errors: list[str] = []

        # Required fields
        if not self.source.type:
            errors.append("source.type is required")
        if not self.model.id:
            errors.append("model.id is required")
        if not self.model.provider:
            errors.append("model.provider is required")

        # Known providers
        _KNOWN_PROVIDERS = {"huggingface", "onnx", "torch", "torchscript",
                            "monai", "braindecode", "ultralytics", "custom", "plugin"}
        if self.model.provider and self.model.provider.lower() not in _KNOWN_PROVIDERS:
            errors.append(
                f"model.provider {self.model.provider!r} is not a recognised provider. "
                f"Known: {', '.join(sorted(_KNOWN_PROVIDERS))}"
            )

        provider = (self.model.provider or "").lower()
        source_type = (self.source.type or "").lower()

        # Security gate for local executable plugins.
        if provider in {"plugin", "custom"} and not self.model.trust_remote_code:
            errors.append(
                "model.trust_remote_code=True is required for provider='plugin' or "
                "provider='custom' because local Python model code will be executed"
            )

        # File-backed sources should fail early when the path is missing.
        _FILE_SOURCE_TYPES = {
            "local_file", "file", "local", "bids", "dicom", "dicom_folder",
            "nwb", "xdf", "image", "video", "img",
        }
        if source_type in _FILE_SOURCE_TYPES:
            if not self.source.path:
                errors.append(f"source.path is required for source.type={self.source.type!r}")
            else:
                try:
                    src_path = Path(self.source.path).expanduser()
                    if not src_path.exists():
                        errors.append(f"source.path does not exist: {self.source.path!r}")
                except (OSError, RuntimeError) as exc:
                    errors.append(f"source.path is not readable: {self.source.path!r} ({exc})")

        # Outputs
        if not self.outputs:
            errors.append("At least one output must be specified in 'outputs'")
        _KNOWN_OUTPUTS = {
            "jsonl", "json_lines", "json", "parquet", "csv", "lsl_marker", "lsl",
            "nifti", "nii", "nifti_mask", "dicom_seg", "dicomseg",
            "dicom_sr", "dicomsr", "bids", "bids_derivative", "coco",
            "coco_json", "yolo", "yolo_txt", "websocket", "ws", "http",
            "http_callback", "webhook", "overlay", "image_overlay", "video_overlay",
        }
        _URL_OUTPUTS = {"websocket", "ws", "http", "http_callback", "webhook"}
        for idx, out in enumerate(self.outputs):
            out_type = (out.type or "").lower().strip()
            if not out_type:
                errors.append(f"outputs[{idx}].type is required")
                continue
            if out_type not in _KNOWN_OUTPUTS:
                errors.append(
                    f"outputs[{idx}].type {out.type!r} is not supported. "
                    f"Known: {', '.join(sorted(_KNOWN_OUTPUTS))}"
                )
            if out_type in _URL_OUTPUTS and not out.path:
                errors.append(f"outputs[{idx}].path must be a URL for output type {out.type!r}")
            if out_type in {"lsl_marker", "lsl"} and out.path:
                errors.append(
                    f"outputs[{idx}].path is ignored for LSL outputs; use stream_name instead"
                )

        # Window timing
        if self.window is not None:
            if self.window.duration_s is not None and self.window.duration_s <= 0:
                errors.append(
                    f"window.duration must be > 0 (got {self.window.duration_s})"
                )
            if self.window.step_s is not None and self.window.step_s <= 0:
                errors.append(
                    f"window.step must be > 0 (got {self.window.step_s})"
                )
            if (
                self.window.duration_s is not None
                and self.window.step_s is not None
                and self.window.step_s > self.window.duration_s
            ):
                errors.append(
                    f"window.step ({self.window.step_s}s) must be <= "
                    f"window.duration ({self.window.duration_s}s)"
                )
            if not 0.0 <= self.window.overlap_frac < 1.0:
                errors.append(
                    f"window.overlap must be >= 0 and < 1 (got {self.window.overlap_frac})"
                )
            if self.window.tmin < 0:
                errors.append(f"window.tmin must be >= 0 (got {self.window.tmin})")

        # Runtime sanity. Device availability is checked at adapter load time;
        # these are contract-level checks that do not touch hardware.
        if self.runtime.batch_size <= 0:
            errors.append(f"runtime.batch_size must be > 0 (got {self.runtime.batch_size})")
        if self.runtime.num_workers < 0:
            errors.append(f"runtime.num_workers must be >= 0 (got {self.runtime.num_workers})")
        if self.runtime.latency_budget_ms is not None and self.runtime.latency_budget_ms <= 0:
            errors.append(
                f"runtime.latency_budget_ms must be > 0 (got {self.runtime.latency_budget_ms})"
            )
        if self.runtime.optimize not in {"safe", "speed", "memory"}:
            errors.append(
                f"runtime.optimize must be one of safe, speed, memory "
                f"(got {self.runtime.optimize!r})"
            )
        if self.runtime.source_failure_policy not in {"strict", "skip_window", "continue_recording"}:
            errors.append(
                "runtime.source_failure_policy must be one of strict, skip_window, "
                f"continue_recording (got {self.runtime.source_failure_policy!r})"
            )
        if self.runtime.preprocess_failure_policy not in {"strict", "drop_failed"}:
            errors.append(
                "runtime.preprocess_failure_policy must be one of strict, drop_failed "
                f"(got {self.runtime.preprocess_failure_policy!r})"
            )
        if self.runtime.max_windows is not None and self.runtime.max_windows <= 0:
            errors.append(f"runtime.max_windows must be > 0 (got {self.runtime.max_windows})")
        if self.runtime.max_duration_s is not None and self.runtime.max_duration_s <= 0:
            errors.append(
                f"runtime.max_duration_s must be > 0 (got {self.runtime.max_duration_s})"
            )
        if self.runtime.idle_timeout_s is not None and self.runtime.idle_timeout_s <= 0:
            errors.append(
                f"runtime.idle_timeout_s must be > 0 (got {self.runtime.idle_timeout_s})"
            )
        if not str(self.runtime.device or "").strip():
            errors.append("runtime.device must not be empty")

        # preprocessing.allow — check against known TransformKind values
        for tk in self.preprocessing.allow:
            if tk not in _VALID_TRANSFORMS:
                errors.append(
                    f"preprocessing.allow contains unknown transform {tk!r}. "
                    f"Valid: {', '.join(sorted(_VALID_TRANSFORMS))}"
                )
        for tk in self.preprocessing.deny:
            if tk not in _VALID_TRANSFORMS:
                errors.append(
                    f"preprocessing.deny contains unknown transform {tk!r}. "
                    f"Valid: {', '.join(sorted(_VALID_TRANSFORMS))}"
                )

        allow_set = set(self.preprocessing.allow)
        deny_set = set(self.preprocessing.deny)
        overlap = sorted(allow_set & deny_set)
        if overlap:
            errors.append(
                "preprocessing.allow and preprocessing.deny contain the same "
                f"transform(s): {', '.join(overlap)}"
            )

        if self.preprocessing.mode not in {"auto", "explicit", "none"}:
            errors.append(
                f"preprocessing.mode must be one of auto, explicit, none "
                f"(got {self.preprocessing.mode!r})"
            )
        if not self.preprocessing.normalize and "normalize" in allow_set:
            errors.append(
                "preprocessing.normalize is False but preprocessing.allow includes 'normalize'"
            )
        if not self.preprocessing.resample and (
            "resample" in allow_set or "resample_spatial" in allow_set
        ):
            errors.append(
                "preprocessing.resample is False but preprocessing.allow includes "
                "'resample' or 'resample_spatial'"
            )
        if not self.preprocessing.channel_select and "channel_select" in allow_set:
            errors.append(
                "preprocessing.channel_select is False but preprocessing.allow includes "
                "'channel_select'"
            )

        if self.trigger is not None:
            when = self.trigger.when or {}
            emit = self.trigger.emit or {}
            if not when:
                errors.append("trigger.when must not be empty when trigger is provided")
            if "class" not in when:
                errors.append("trigger.when.class is required for class-based triggers")
            if "probability_gte" in when:
                try:
                    prob = float(when["probability_gte"])
                    if not 0.0 <= prob <= 1.0:
                        errors.append(
                            f"trigger.when.probability_gte must be between 0 and 1 (got {prob})"
                        )
                except (TypeError, ValueError):
                    errors.append("trigger.when.probability_gte must be numeric")
            if "stable_for" in when:
                try:
                    stable_for = int(when["stable_for"])
                    if stable_for <= 0:
                        errors.append(
                            f"trigger.when.stable_for must be > 0 (got {stable_for})"
                        )
                except (TypeError, ValueError):
                    errors.append("trigger.when.stable_for must be an integer")
            if not emit:
                errors.append("trigger.emit must describe the event payload to emit")

        if self.artifact.failure_policy not in {"strict", "warn"}:
            errors.append(
                f"artifact.failure_policy must be one of strict, warn "
                f"(got {self.artifact.failure_policy!r})"
            )

        return errors


def _as_bool(value: Any) -> bool:
    """Parse common config boolean encodings without Python truthiness traps."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation for spec hashing/serialization."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _as_string_list(value: Any) -> list[str] | None:
    """Normalize scalar-or-list schema fields while preserving absence."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]
