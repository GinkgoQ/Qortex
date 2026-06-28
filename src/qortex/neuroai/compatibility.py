"""Compatibility Engine — checks whether a source can satisfy a model.

This is the core decision engine of the NeuroAI runtime.  It compares the
``SourceProfile`` against the model's ``InputContract`` on every dimension
that matters for inference correctness:

  - modality match
  - channel count / channel names
  - sampling frequency
  - spatial shape + voxel spacing
  - dtype
  - axis convention
  - required metadata presence
  - memory budget estimate

For each mismatch it determines whether a preprocessing transform can bridge
the gap and whether that transform is allowed by the ``PreprocessSpec``.

Output is a ``CompatibilityReport`` with a ``status`` of:
  - ``compatible``                  — source satisfies model as-is
  - ``compatible_with_transforms``  — satisfiable after allowed transforms
  - ``uncertain``                   — some dimensions are unknown
  - ``incompatible``                — cannot be satisfied even with transforms
"""

from __future__ import annotations

import logging
from typing import Any

from qortex.neuroai.contracts import (
    CompatibilityReport,
    CompatibilityStatus,
    EvidenceStatus,
    InputContract,
    ModelProfile,
    SourceProfile,
    TransformDescriptor,
    TransformKind,
    WarningItem,
)
from qortex.neuroai.spec import PreprocessSpec

log = logging.getLogger(__name__)

# Memory headroom multiplier: model weights + activations
_MEMORY_MULTIPLIER = 3.5

# Spatial coordinate frame identifiers (handled by _check_coordinate_frame)
_SPATIAL_FRAMES = {"LPS", "RAS", "LAS", "SPATIAL_ZYX", "SPATIAL_XYZ"}


class CompatibilityEngine:
    """Check source-model compatibility and determine required transforms.

    Usage::

        engine = CompatibilityEngine()
        report = engine.check(source_profile, model_profile, preprocess_spec)
        if report.is_runnable:
            plan = PreprocessPlanner().build_plan(source_profile, model_profile, preprocess_spec)
    """

    def check(
        self,
        source: SourceProfile,
        model: ModelProfile,
        preprocess: PreprocessSpec | None = None,
    ) -> CompatibilityReport:
        """Run all compatibility checks.

        Parameters
        ----------
        source:
            SourceProfile from a probed data source.
        model:
            ModelProfile from an inspected model.
        preprocess:
            Which transforms are allowed.  ``None`` = auto mode (all allowed).

        Returns
        -------
        CompatibilityReport
            Structured report with status, required transforms, blockers, and warnings.
        """
        if preprocess is None:
            preprocess = PreprocessSpec(mode="auto")

        contract = model.input_contract
        if contract is None:
            return CompatibilityReport(
                status=CompatibilityStatus.uncertain,
                source_id=source.source_id,
                model_id=model.model_id,
                unknowns=["model.input_contract is None — cannot verify compatibility"],
                warnings=[WarningItem(
                    code="NO_INPUT_CONTRACT",
                    message="Model has no declared input contract. Compatibility is uncertain.",
                    severity="warning",
                    suggestion="Inspect the model card or supply an explicit InputContract.",
                )],
            )

        transforms: list[TransformDescriptor] = []
        blockers: list[WarningItem] = []
        warnings: list[WarningItem] = []
        unknowns: list[str] = []
        evidence: list[dict[str, Any]] = []

        # ── Modality check ────────────────────────────────────────────────────
        self._check_modality(source, contract, blockers, warnings, evidence)

        # ── Channel check (signal sources) ───────────────────────────────────
        ch_match = self._check_channels(
            source, contract, preprocess, transforms, blockers, warnings, unknowns, evidence
        )

        # ── Sampling rate check ───────────────────────────────────────────────
        sr_match = self._check_sampling_rate(
            source, contract, preprocess, transforms, warnings, unknowns, evidence
        )

        # ── Spatial shape check (volume sources) ─────────────────────────────
        shape_match = self._check_spatial_shape(
            source, contract, preprocess, transforms, blockers, warnings, unknowns, evidence
        )

        # ── Dtype check ───────────────────────────────────────────────────────
        dtype_match = self._check_dtype(
            source, contract, transforms, warnings, evidence
        )

        # ── Axis convention check ─────────────────────────────────────────────
        axis_match = self._check_axis_convention(
            source, contract, transforms, warnings, evidence
        )

        # ── Voxel spacing check (volume sources) ──────────────────────────────
        self._check_voxel_spacing(source, contract, warnings, unknowns, evidence)

        # ── Coordinate frame check ─────────────────────────────────────────────
        self._check_coordinate_frame(source, contract, transforms, warnings, evidence)

        # ── fMRI timebase / TR check ───────────────────────────────────────────
        self._check_fmri_timebase(source, contract, warnings, unknowns, evidence)

        # ── Memory estimate ───────────────────────────────────────────────────
        mem_mb = self._estimate_memory(source, model)
        if mem_mb > 8192:
            warnings.append(WarningItem(
                code="HIGH_MEMORY_ESTIMATE",
                message=f"Estimated runtime memory: {mem_mb:.0f} MB. "
                        "Ensure sufficient GPU/RAM is available.",
                severity="warning",
            ))

        # ── Required metadata ─────────────────────────────────────────────────
        for meta_key in contract.required_metadata:
            if meta_key not in (source.extra or {}):
                unknowns.append(f"Required metadata {meta_key!r} not present in source")

        # ── Blocker check from model warnings ─────────────────────────────────
        for w in model.warnings:
            if w.severity == "error":
                blockers.append(w)

        # ── Status determination ──────────────────────────────────────────────
        if blockers:
            status = CompatibilityStatus.incompatible
        elif unknowns:
            status = CompatibilityStatus.uncertain
        elif transforms:
            status = CompatibilityStatus.compatible_with_transforms
        else:
            status = CompatibilityStatus.compatible

        return CompatibilityReport(
            status=status,
            source_id=source.source_id,
            model_id=model.model_id,
            required_transforms=transforms,
            blockers=blockers,
            warnings=warnings,
            unknowns=unknowns,
            evidence=evidence,
            channel_match=ch_match,
            sampling_rate_match=sr_match,
            spatial_shape_match=shape_match,
            dtype_match=dtype_match,
            axis_convention_match=axis_match,
            memory_estimate_mb=mem_mb,
        )

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_modality(
        self,
        source: SourceProfile,
        contract: InputContract,
        blockers: list,
        warnings: list,
        evidence: list,
    ) -> None:
        src_mod = _contract_value(source.modality).lower()
        req_mod = _contract_value(contract.modality).lower()

        if not req_mod or req_mod == "unknown":
            evidence.append({"check": "modality", "status": "unknown"})
            return

        _ALIASES: dict[str, set[str]] = {
            "eeg": {"eeg", "bdf", "edf"},
            "meg": {"meg", "fif"},
            "mri": {"mri", "anat", "nifti"},
            "fmri": {"fmri", "bold", "func"},
        }

        def _matches(a: str, b: str) -> bool:
            if a == b:
                return True
            for canonical, aliases in _ALIASES.items():
                if a in aliases and b in aliases:
                    return True
            return False

        if not _matches(src_mod, req_mod):
            blockers.append(WarningItem(
                code="MODALITY_MISMATCH",
                message=f"Model expects modality {req_mod!r} but source provides {src_mod!r}.",
                severity="error",
            ))
            evidence.append({"check": "modality", "status": "blocked",
                             "source": src_mod, "required": req_mod})
        else:
            evidence.append({"check": "modality", "status": "ok",
                             "source": src_mod, "required": req_mod})

    def _check_channels(
        self,
        source: SourceProfile,
        contract: InputContract,
        preprocess: PreprocessSpec,
        transforms: list,
        blockers: list,
        warnings: list,
        unknowns: list,
        evidence: list,
    ) -> EvidenceStatus:
        src_n = source.n_channels
        req_n = contract.n_channels
        req_names = contract.required_channels

        if req_n is None and not req_names:
            unknowns.append("model.n_channels is not specified")
            return EvidenceStatus.unknown

        if src_n is None:
            unknowns.append("source.n_channels is unknown")
            return EvidenceStatus.unknown

        # Exact channel name matching
        if req_names and source.channel_names:
            missing = [ch for ch in req_names if ch not in source.channel_names]
            if missing:
                if preprocess.allows("channel_map"):
                    transforms.append(TransformDescriptor(
                        kind=TransformKind.channel_map,
                        required_by="input_contract.required_channels",
                        params={"missing_channels": missing},
                        reversible=False,
                        evidence_status=EvidenceStatus.inferred,
                    ))
                    warnings.append(WarningItem(
                        code="CHANNEL_MAP_REQUIRED",
                        message=f"Source is missing {len(missing)} required channels: "
                                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                                "Channel mapping will be applied.",
                        severity="warning",
                        evidence={"missing": missing[:10]},
                        suggestion="Ensure source has all required channels or provide a channel map.",
                    ))
                else:
                    blockers.append(WarningItem(
                        code="MISSING_CHANNELS",
                        message=f"Model requires channels {req_names} but source has "
                                f"{source.channel_names}. "
                                "Transform 'channel_map' is not allowed.",
                        severity="error",
                    ))
                    return EvidenceStatus.missing

            # Channel count mismatch after name check
            extra = [ch for ch in source.channel_names if ch not in req_names]
            if extra and preprocess.allows("channel_select"):
                transforms.append(TransformDescriptor(
                    kind=TransformKind.channel_select,
                    required_by="input_contract.required_channels",
                    params={"keep": req_names},
                    reversible=True,
                ))
            return EvidenceStatus.confirmed

        # Numeric count check only
        if req_n is not None and src_n != req_n:
            if src_n > req_n and preprocess.allows("channel_select"):
                transforms.append(TransformDescriptor(
                    kind=TransformKind.channel_select,
                    required_by="input_contract.n_channels",
                    params={"target_n": req_n},
                    reversible=True,
                ))
                warnings.append(WarningItem(
                    code="CHANNEL_SELECT_REQUIRED",
                    message=f"Source has {src_n} channels; model expects {req_n}. "
                            f"First {req_n} channels will be selected.",
                    severity="warning",
                ))
            elif src_n < req_n:
                blockers.append(WarningItem(
                    code="INSUFFICIENT_CHANNELS",
                    message=f"Model expects {req_n} channels but source only provides {src_n}. "
                            "Cannot pad channels without explicit mapping.",
                    severity="error",
                ))
                return EvidenceStatus.missing
            elif src_n > req_n and not preprocess.allows("channel_select"):
                blockers.append(WarningItem(
                    code="CHANNEL_COUNT_MISMATCH",
                    message=f"Source has {src_n} channels but model requires {req_n} and "
                            "channel_select is not allowed.",
                    severity="error",
                ))
                return EvidenceStatus.missing
        else:
            # src_n == req_n: exact match — record as confirmed
            evidence.append({"check": "channels", "status": "ok",
                             "source_n": src_n, "required_n": req_n})

        return EvidenceStatus.confirmed

    def _check_sampling_rate(
        self,
        source: SourceProfile,
        contract: InputContract,
        preprocess: PreprocessSpec,
        transforms: list,
        warnings: list,
        unknowns: list,
        evidence: list,
    ) -> EvidenceStatus:
        src_sr = source.sampling_rate_hz
        req_sr = contract.sampling_rate_hz
        src_mod = _contract_value(source.modality).lower()
        req_mod = _contract_value(contract.modality).lower()
        temporal_context = (
            src_mod in {"eeg", "meg", "ieeg", "fnirs", "timeseries", "fmri", "dwi", "lsl", "xdf"}
            or req_mod in {"eeg", "meg", "ieeg", "fnirs", "timeseries", "fmri", "dwi"}
            or contract.window_duration_s is not None
        )

        if req_sr is None:
            if temporal_context:
                unknowns.append("model.sampling_rate_hz is not specified")
                return EvidenceStatus.unknown
            return EvidenceStatus.confirmed
        if src_sr is None:
            if temporal_context:
                unknowns.append("source.sampling_rate_hz is unknown")
                return EvidenceStatus.unknown
            return EvidenceStatus.confirmed

        if abs(src_sr - req_sr) / max(req_sr, 1) > 0.01:  # > 1% difference
            if preprocess.allows("resample"):
                transforms.append(TransformDescriptor(
                    kind=TransformKind.resample,
                    required_by="input_contract.sampling_rate_hz",
                    params={"from_hz": src_sr, "to_hz": req_sr},
                    reversible=False,
                    irreversible_reason="Resampling introduces interpolation artifacts",
                ))
                warnings.append(WarningItem(
                    code="RESAMPLE_REQUIRED",
                    message=f"Source sampling rate {src_sr:.1f} Hz ≠ model requirement "
                            f"{req_sr:.1f} Hz. Resampling will be applied (irreversible).",
                    severity="warning",
                    evidence={"source_hz": src_sr, "required_hz": req_sr},
                ))
                return EvidenceStatus.confirmed
            else:
                blockers.append(WarningItem(
                    code="SAMPLING_RATE_MISMATCH",
                    message=f"Source at {src_sr:.1f} Hz; model expects {req_sr:.1f} Hz. "
                            "Resampling is not allowed by preprocessing spec.",
                    severity="error",
                    suggestion="Add 'resample' to preprocessing.allow in the pipeline YAML, "
                               "or choose a model trained at the source sampling rate.",
                ))
                return EvidenceStatus.missing

        evidence.append({"check": "sampling_rate", "status": "ok",
                         "source_hz": src_sr, "required_hz": req_sr})
        return EvidenceStatus.confirmed

    def _check_spatial_shape(
        self,
        source: SourceProfile,
        contract: InputContract,
        preprocess: PreprocessSpec,
        transforms: list,
        blockers: list,
        warnings: list,
        unknowns: list,
        evidence: list,
    ) -> EvidenceStatus:
        req_shape = contract.spatial_shape
        src_shape = source.spatial_shape

        if req_shape is None:
            return EvidenceStatus.unknown
        if src_shape is None:
            unknowns.append("source.spatial_shape is unknown")
            return EvidenceStatus.unknown

        if src_shape != req_shape:
            if preprocess.allows("pad_or_crop"):
                transforms.append(TransformDescriptor(
                    kind=TransformKind.pad_or_crop,
                    required_by="input_contract.spatial_shape",
                    params={"from_shape": list(src_shape), "to_shape": list(req_shape)},
                    reversible=False,
                    irreversible_reason="Cropping discards spatial information",
                ))
                warnings.append(WarningItem(
                    code="SPATIAL_RESHAPE_REQUIRED",
                    message=f"Source shape {src_shape} ≠ model requirement {req_shape}. "
                            "Pad/crop will be applied.",
                    severity="warning",
                ))
                return EvidenceStatus.confirmed
            elif preprocess.allows("resample_spatial"):
                transforms.append(TransformDescriptor(
                    kind=TransformKind.resample_spatial,
                    required_by="input_contract.spatial_shape",
                    params={"from_shape": list(src_shape), "to_shape": list(req_shape)},
                    reversible=False,
                ))
                return EvidenceStatus.confirmed
            else:
                blockers.append(WarningItem(
                    code="SPATIAL_SHAPE_MISMATCH",
                    message=f"Source spatial shape {src_shape} ≠ model requirement {req_shape}. "
                            "No allowed transform to bridge the gap.",
                    severity="error",
                ))
                return EvidenceStatus.missing

        evidence.append({"check": "spatial_shape", "status": "ok"})
        return EvidenceStatus.confirmed

    def _check_dtype(
        self,
        source: SourceProfile,
        contract: InputContract,
        transforms: list,
        warnings: list,
        evidence: list,
    ) -> EvidenceStatus:
        src_dtype = _contract_value(source.dtype or "float64")
        req_dtype = _contract_value(contract.dtype or "float32")

        if src_dtype != req_dtype:
            transforms.append(TransformDescriptor(
                kind=TransformKind.cast_dtype,
                required_by="input_contract.dtype",
                params={"from": src_dtype, "to": req_dtype},
                reversible=False,
                irreversible_reason="Narrowing cast may lose precision",
            ))
            if "float16" in req_dtype:
                warnings.append(WarningItem(
                    code="FP16_CAST",
                    message=f"Casting {src_dtype} → {req_dtype}. FP16 reduces precision.",
                    severity="info",
                ))
        else:
            evidence.append({"check": "dtype", "status": "ok", "dtype": src_dtype})

        return EvidenceStatus.confirmed

    def _check_axis_convention(
        self,
        source: SourceProfile,
        contract: InputContract,
        transforms: list,
        warnings: list,
        evidence: list,
    ) -> EvidenceStatus:
        src_conv = source.axis_convention
        req_conv = contract.axis_convention

        if req_conv is None or src_conv is None:
            return EvidenceStatus.unknown

        src_str = _contract_value(src_conv)
        req_str = _contract_value(req_conv)

        if src_str != req_str:
            # Most common case: source is channels_time, model wants batch_channels_time
            if "batch" in req_str and "batch" not in src_str:
                transforms.append(TransformDescriptor(
                    kind=TransformKind.add_batch_dim,
                    required_by="input_contract.axis_convention",
                    params={"from": src_str, "to": req_str},
                    reversible=True,
                ))
                evidence.append({"check": "axis_convention", "status": "transform_required",
                                 "source": src_str, "required": req_str,
                                 "transform": "add_batch_dim"})
            elif src_str.upper() in _SPATIAL_FRAMES and req_str.upper() in _SPATIAL_FRAMES:
                # LPS↔RAS (or LAS/spatial_zyx/spatial_xyz) — _check_coordinate_frame handles
                # the reorient transform and emits a more specific warning.  Skip the generic
                # AXIS_CONVENTION_MISMATCH so users don't see two overlapping messages.
                evidence.append({"check": "axis_convention", "status": "transform_required",
                                 "source": src_str, "required": req_str,
                                 "transform": "reorient"})
            else:
                warnings.append(WarningItem(
                    code="AXIS_CONVENTION_MISMATCH",
                    message=f"Source axis convention {src_str!r} ≠ model requirement {req_str!r}. "
                            "Automatic transposition may be applied.",
                    severity="warning",
                    suggestion="Verify that the source and model share the same axis layout, "
                               "or add an explicit transpose transform.",
                ))
                evidence.append({"check": "axis_convention", "status": "mismatch",
                                 "source": src_str, "required": req_str})
        else:
            evidence.append({"check": "axis_convention", "status": "ok",
                             "source": src_str, "required": req_str})

        return EvidenceStatus.confirmed

    def _check_voxel_spacing(
        self,
        source: SourceProfile,
        contract: InputContract,
        warnings: list,
        unknowns: list,
        evidence: list,
    ) -> None:
        req_vox = contract.voxel_sizes_mm
        src_vox = source.voxel_sizes_mm

        if req_vox is None:
            return  # model does not declare spacing requirements
        if src_vox is None:
            unknowns.append("source.voxel_sizes_mm is unknown — cannot verify spacing match")
            return

        # Check element-wise within 10% tolerance
        if len(req_vox) != len(src_vox):
            warnings.append(WarningItem(
                code="VOXEL_SPACING_DIM_MISMATCH",
                message=f"Source voxel size has {len(src_vox)} dims but model expects "
                        f"{len(req_vox)}. Spacing check skipped.",
                severity="warning",
            ))
            return

        mismatched = [
            (i, float(src_vox[i]), float(req_vox[i]))
            for i in range(len(req_vox))
            if abs(float(src_vox[i]) - float(req_vox[i])) / max(float(req_vox[i]), 1e-6) > 0.10
        ]
        if mismatched:
            details = ", ".join(f"dim{i}: {s:.2f}mm vs {r:.2f}mm" for i, s, r in mismatched)
            warnings.append(WarningItem(
                code="VOXEL_SPACING_MISMATCH",
                message=f"Voxel spacing mismatch (>10%): {details}. "
                        "Consider using resample_spatial to match the model's expected resolution.",
                severity="warning",
                evidence={"source_mm": list(src_vox), "required_mm": list(req_vox)},
                suggestion="Add 'resample_spatial' to preprocessing.allow in the pipeline YAML.",
            ))
        else:
            evidence.append({"check": "voxel_spacing", "status": "ok",
                             "source_mm": list(src_vox), "required_mm": list(req_vox)})

    def _check_coordinate_frame(
        self,
        source: SourceProfile,
        contract: InputContract,
        transforms: list,
        warnings: list,
        evidence: list,
    ) -> None:
        src_conv = source.axis_convention
        req_conv = contract.axis_convention

        if req_conv is None or src_conv is None:
            return

        src_str = _contract_value(src_conv)
        req_str = _contract_value(req_conv)

        # Spatial coordinate frame mismatch (LPS vs RAS)
        _LPS = {"LPS", "spatial_zyx"}
        _RAS = {"RAS", "spatial_xyz"}

        src_is_lps = src_str.upper() in _LPS or "lps" in src_str.lower()
        req_is_ras = req_str.upper() in _RAS or req_str.upper() == "RAS"
        req_is_lps = req_str.upper() in _LPS or req_str.upper() == "LPS"

        if src_is_lps and req_is_ras:
            transforms.append(TransformDescriptor(
                kind=TransformKind.reorient,
                required_by="input_contract.axis_convention",
                params={"from": "LPS", "to": "RAS"},
                reversible=True,
                evidence_status=EvidenceStatus.inferred,
            ))
            warnings.append(WarningItem(
                code="COORDINATE_FRAME_MISMATCH",
                message="Source uses LPS (DICOM) convention; model expects RAS (NIfTI). "
                        "A reorient transform will flip the coordinate frame.",
                severity="warning",
                evidence={"source": src_str, "required": req_str},
                suggestion="Verify the model was trained on LPS or RAS volumes.",
            ))
        elif not src_is_lps and req_is_lps:
            warnings.append(WarningItem(
                code="COORDINATE_FRAME_MISMATCH",
                message=f"Source coordinate frame {src_str!r} ≠ model requirement {req_str!r}. "
                        "Manual reorientation may be required.",
                severity="warning",
            ))

    def _check_fmri_timebase(
        self,
        source: SourceProfile,
        contract: InputContract,
        warnings: list,
        unknowns: list,
        evidence: list,
    ) -> None:
        """Warn when source TR does not match the model's expected TR."""
        src_mod = _contract_value(source.modality).lower()
        if src_mod not in ("fmri", "bold", "func"):
            return  # only applies to fMRI

        src_tr = source.tr_s
        req_tr = getattr(contract, "tr_s", None)

        if req_tr is None:
            unknowns.append("model.tr_s (repetition time) is not specified for fMRI source")
            return
        if src_tr is None:
            unknowns.append("source.tr_s (repetition time) is unknown — cannot verify fMRI TR")
            return

        if abs(src_tr - req_tr) / max(req_tr, 1e-6) > 0.05:  # >5% mismatch
            warnings.append(WarningItem(
                code="FMRI_TR_MISMATCH",
                message=f"Source TR={src_tr:.3f}s but model expects TR={req_tr:.3f}s. "
                        "Timing-dependent features (HRF, connectivity) will be incorrect.",
                severity="warning",
                evidence={"source_tr_s": src_tr, "required_tr_s": req_tr},
                suggestion="Resample the fMRI volume or choose a model trained on "
                           f"TR={src_tr:.2f}s data.",
            ))
        else:
            evidence.append({"check": "fmri_tr", "status": "ok",
                             "source_tr_s": src_tr, "required_tr_s": req_tr})

    def _estimate_memory(self, source: SourceProfile, model: ModelProfile) -> float:
        """Rough memory estimate in MB."""
        model_mb = model.estimated_memory_mb or 500.0  # default 500 MB if unknown
        batch_bytes = 0
        if source.n_channels and source.sampling_rate_hz:
            win_s = 2.0  # assume 2s window
            n_t = int(source.sampling_rate_hz * win_s)
            batch_bytes = source.n_channels * n_t * 4  # float32
        elif source.spatial_shape:
            batch_bytes = 1
            for d in source.spatial_shape:
                batch_bytes *= d
            batch_bytes *= 4  # float32
        batch_mb = batch_bytes / 1e6
        return model_mb * _MEMORY_MULTIPLIER + batch_mb


def _contract_value(value: Any) -> str:
    """Return the semantic scalar value for enum/string contract fields."""
    if value is None:
        return ""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)
