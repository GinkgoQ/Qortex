"""Runtime compatibility check domain.

Validates whether a source can satisfy a model or pipeline input contract:
modality, channel coverage, sampling rate, spatial shape, dtype, memory estimate,
and unsupported transform detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qortex.checks._base import BaseChecker
from qortex.checks._report import (
    CheckFinding,
    CheckReport,
    CheckSeverity,
    EvidenceRecord,
    EvidenceState,
    SuggestedFix,
)

# Memory estimate thresholds (bytes)
_WARN_MEMORY_BYTES = 2 * 1024**3   # 2 GiB
_BLOCK_MEMORY_BYTES = 8 * 1024**3  # 8 GiB


class RuntimeCompatibilityChecker(BaseChecker):
    """Validate source↔model contract compatibility before runtime execution."""

    name = "runtime_compatibility"
    required_for = frozenset({"neuroai-run"})

    def __init__(
        self,
        *,
        required_modality: str | None = None,
        required_channels: list[str] | None = None,
        required_sampling_hz: float | None = None,
        required_spatial_shape: tuple[int, ...] | None = None,
        required_dtype: str | None = None,
        required_window_s: float | None = None,
        pipeline_transforms: list[str] | None = None,
    ) -> None:
        self._modality = required_modality
        self._channels = required_channels
        self._sampling_hz = required_sampling_hz
        self._spatial_shape = required_spatial_shape
        self._dtype = required_dtype
        self._window_s = required_window_s
        self._transforms = pipeline_transforms or []

    def run(self, dataset_path: Path, **kwargs) -> CheckReport:
        report = CheckReport(
            name=self.name,
            scope=str(dataset_path),
            inputs={
                "dataset_path": str(dataset_path),
                "required_modality": self._modality,
                "required_channels": self._channels,
                "required_sampling_hz": self._sampling_hz,
                "required_spatial_shape": self._spatial_shape,
                "required_dtype": self._dtype,
                "required_window_s": self._window_s,
            },
        )

        source_profile = kwargs.get("source_profile")
        if source_profile is None:
            report.add(CheckFinding(
                code="RUNTIME.NO_SOURCE_PROFILE",
                severity=CheckSeverity.UNKNOWN,
                message=(
                    "No SourceProfile provided to RuntimeCompatibilityChecker. "
                    "Run probe() on the source adapter first."
                ),
                path=str(dataset_path),
                evidence=[EvidenceRecord(
                    field="source_profile",
                    state=EvidenceState.missing,
                    observed_source="kwargs",
                )],
            ))
            return report.finalize()

        self._check_modality(source_profile, report)
        self._check_channels(source_profile, report)
        self._check_sampling_rate(source_profile, report)
        self._check_spatial_shape(source_profile, report)
        self._check_dtype(source_profile, report)
        self._check_memory_estimate(source_profile, report)
        self._check_transforms(report)

        return report.finalize()

    def _check_modality(self, sp: Any, report: CheckReport) -> None:
        if not self._modality:
            return
        src_mod = getattr(sp, "modality", None)
        if src_mod is None:
            report.add(CheckFinding(
                code="RUNTIME.MODALITY_UNKNOWN",
                severity=CheckSeverity.UNKNOWN,
                message="Source modality is unknown; cannot verify model compatibility.",
                evidence=[EvidenceRecord(
                    field="modality",
                    state=EvidenceState.missing,
                    observed_source="SourceProfile",
                )],
            ))
        elif str(src_mod) != self._modality:
            report.add(CheckFinding(
                code="RUNTIME.MODALITY_MISMATCH",
                severity=CheckSeverity.BLOCK,
                message=(
                    f"Source modality '{src_mod}' does not match required '{self._modality}'."
                ),
                expected=self._modality,
                observed=str(src_mod),
                evidence=[EvidenceRecord(
                    field="modality",
                    state=EvidenceState.contradicted,
                    claimed_value=self._modality,
                    observed_value=str(src_mod),
                    claimed_source="model contract",
                    observed_source="SourceProfile",
                )],
            ))

    def _check_channels(self, sp: Any, report: CheckReport) -> None:
        if not self._channels:
            return
        src_names = getattr(sp, "channel_names", None) or []
        if not src_names:
            report.add(CheckFinding(
                code="RUNTIME.CHANNEL_NAMES_UNKNOWN",
                severity=CheckSeverity.UNKNOWN,
                message="Source channel names are unknown; cannot verify model channel coverage.",
                evidence=[EvidenceRecord(
                    field="channel_names",
                    state=EvidenceState.missing,
                    observed_source="SourceProfile",
                )],
            ))
            return
        missing = [c for c in self._channels if c not in src_names]
        if missing:
            report.add(CheckFinding(
                code="RUNTIME.MISSING_REQUIRED_CHANNELS",
                severity=CheckSeverity.BLOCK,
                message=(
                    f"{len(missing)} required channels are absent from source: {missing[:10]}."
                ),
                expected=self._channels,
                observed=src_names,
                evidence=[EvidenceRecord(
                    field="channel_names",
                    state=EvidenceState.contradicted,
                    claimed_value=self._channels,
                    observed_value=src_names,
                    claimed_source="model contract",
                    observed_source="SourceProfile",
                )],
                suggested_fix=SuggestedFix(
                    description="Use a channel_map transform or select a compatible source.",
                    safe=True,
                ),
            ))
        else:
            report.record_evidence(EvidenceRecord(
                field="channel_coverage",
                state=EvidenceState.confirmed,
                observed_value={"required": len(self._channels), "available": len(src_names)},
                observed_source="SourceProfile",
            ))

    def _check_sampling_rate(self, sp: Any, report: CheckReport) -> None:
        if not self._sampling_hz:
            return
        src_hz = getattr(sp, "sampling_rate_hz", None)
        if src_hz is None:
            report.add(CheckFinding(
                code="RUNTIME.SAMPLING_RATE_UNKNOWN",
                severity=CheckSeverity.UNKNOWN,
                message="Source sampling rate is unknown.",
                evidence=[EvidenceRecord(
                    field="sampling_rate_hz",
                    state=EvidenceState.missing,
                    observed_source="SourceProfile",
                )],
            ))
            return
        if abs(src_hz - self._sampling_hz) / max(self._sampling_hz, 1.0) > 0.01:
            report.add(CheckFinding(
                code="RUNTIME.SAMPLING_RATE_MISMATCH",
                severity=CheckSeverity.WARN,
                message=(
                    f"Source sampling rate {src_hz} Hz differs from required {self._sampling_hz} Hz. "
                    "A resample transform will be needed."
                ),
                expected=self._sampling_hz,
                observed=src_hz,
                evidence=[EvidenceRecord(
                    field="sampling_rate_hz",
                    state=EvidenceState.contradicted,
                    claimed_value=self._sampling_hz,
                    observed_value=src_hz,
                    claimed_source="model contract",
                    observed_source="SourceProfile",
                )],
                suggested_fix=SuggestedFix(
                    description=f"Add a resample transform to {self._sampling_hz} Hz in the pipeline.",
                    safe=True,
                ),
            ))

    def _check_spatial_shape(self, sp: Any, report: CheckReport) -> None:
        if not self._spatial_shape:
            return
        src_shape = getattr(sp, "spatial_shape", None)
        if src_shape is None:
            report.add(CheckFinding(
                code="RUNTIME.SPATIAL_SHAPE_UNKNOWN",
                severity=CheckSeverity.UNKNOWN,
                message="Source spatial shape is unknown.",
                evidence=[EvidenceRecord(
                    field="spatial_shape",
                    state=EvidenceState.missing,
                    observed_source="SourceProfile",
                )],
            ))
            return
        if tuple(src_shape) != tuple(self._spatial_shape):
            report.add(CheckFinding(
                code="RUNTIME.SPATIAL_SHAPE_MISMATCH",
                severity=CheckSeverity.WARN,
                message=(
                    f"Source spatial shape {src_shape} differs from required {self._spatial_shape}. "
                    "A resample_spatial or pad_or_crop transform will be needed."
                ),
                expected=self._spatial_shape,
                observed=src_shape,
                evidence=[EvidenceRecord(
                    field="spatial_shape",
                    state=EvidenceState.contradicted,
                    claimed_value=self._spatial_shape,
                    observed_value=src_shape,
                    claimed_source="model contract",
                    observed_source="SourceProfile",
                )],
            ))

    def _check_dtype(self, sp: Any, report: CheckReport) -> None:
        if not self._dtype:
            return
        src_dtype = getattr(sp, "dtype", None)
        if src_dtype is None:
            return
        if src_dtype != self._dtype:
            report.add(CheckFinding(
                code="RUNTIME.DTYPE_MISMATCH",
                severity=CheckSeverity.INFO,
                message=(
                    f"Source dtype '{src_dtype}' differs from required '{self._dtype}'. "
                    "A cast_dtype transform will be applied."
                ),
                expected=self._dtype,
                observed=src_dtype,
            ))

    def _check_memory_estimate(self, sp: Any, report: CheckReport) -> None:
        n_ch = getattr(sp, "n_channels", None) or 0
        srate = getattr(sp, "sampling_rate_hz", None) or 0.0
        dur = getattr(sp, "duration_s", None)
        spatial = getattr(sp, "spatial_shape", None)

        if spatial and len(spatial) >= 3:
            from functools import reduce
            from operator import mul
            n_vox = reduce(mul, spatial, 1)
            n_vol = getattr(sp, "n_volumes", 1) or 1
            mem_bytes = n_vox * n_vol * 4  # float32
        elif n_ch and srate and dur:
            mem_bytes = int(n_ch * srate * dur * 4)
        else:
            return

        report.record_evidence(EvidenceRecord(
            field="estimated_memory_bytes",
            state=EvidenceState.inferred,
            observed_value=mem_bytes,
            observed_source="SourceProfile",
        ))

        if mem_bytes > _BLOCK_MEMORY_BYTES:
            report.add(CheckFinding(
                code="RUNTIME.MEMORY_TOO_LARGE",
                severity=CheckSeverity.WARN,
                message=(
                    f"Estimated data size is {mem_bytes / 1024**3:.1f} GiB. "
                    "Streaming or chunked loading is strongly recommended."
                ),
                observed=mem_bytes,
                suggested_fix=SuggestedFix(
                    description="Use source.stream() instead of source.read_batch().",
                    safe=True,
                ),
            ))
        elif mem_bytes > _WARN_MEMORY_BYTES:
            report.add(CheckFinding(
                code="RUNTIME.MEMORY_LARGE",
                severity=CheckSeverity.INFO,
                message=f"Estimated data size is {mem_bytes / 1024**3:.1f} GiB; consider streaming.",
                observed=mem_bytes,
            ))

    def _check_transforms(self, report: CheckReport) -> None:
        unsafe_transforms = {"resample_spatial", "reorient", "pad_or_crop"}
        warn_transforms = {"normalize", "rescale_intensity"}

        for t in self._transforms:
            if t in unsafe_transforms:
                report.add(CheckFinding(
                    code="RUNTIME.POTENTIALLY_UNSAFE_TRANSFORM",
                    severity=CheckSeverity.INFO,
                    message=(
                        f"Transform '{t}' will modify spatial geometry. "
                        "Verify that coordinate frame and affine are preserved in the output."
                    ),
                    observed=t,
                ))
            if t in warn_transforms:
                report.add(CheckFinding(
                    code="RUNTIME.NORMALIZATION_TRANSFORM",
                    severity=CheckSeverity.INFO,
                    message=(
                        f"Transform '{t}' alters signal scale. "
                        "Ensure fit scope is train-split-only to prevent leakage."
                    ),
                    observed=t,
                ))
