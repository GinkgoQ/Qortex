"""Offline Braindecode model config extractor.

Turns an already-loaded HF Hub config.json-shaped dict into Qortex
contract fields. Pure function — no network access, no HF Hub download.
Missing fields are left unknown, never guessed, per
docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md section
11.2.

Mirrors the inline config-reading logic already present in
src/qortex/neuroai/models/braindecode.py's BrainDecodeAdapter — this is a
standalone, independently-testable version of that same extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    OutputContract,
)


@dataclass
class ExtractedBraindecodeContract:
    model_id: str
    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None


def _extract_input_contract(config: dict) -> InputContract | None:
    n_channels = config.get("n_chans", config.get("n_channels"))
    n_times = config.get("n_times")
    sfreq = config.get("sfreq")
    input_window_seconds = config.get("input_window_seconds")

    if n_channels is None and n_times is None and sfreq is None and input_window_seconds is None:
        return None

    window_duration_s = None
    if n_times is not None and sfreq:
        window_duration_s = n_times / sfreq
    elif input_window_seconds is not None:
        # Some Braindecode configs express window length directly in
        # seconds rather than as a sample count -- use it as-is when
        # n_times/sfreq aren't both present.
        window_duration_s = input_window_seconds

    confirmed = n_channels is not None and bool(sfreq) and n_times is not None
    return InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        n_channels=n_channels,
        sampling_rate_hz=sfreq,
        window_duration_s=window_duration_s,
        evidence_status=EvidenceStatus.confirmed if confirmed else EvidenceStatus.inferred,
    )


def _extract_output_contract(config: dict) -> OutputContract | None:
    n_outputs = config.get("n_outputs")
    if n_outputs is None:
        return None
    id2label = config.get("id2label") or {}
    classes = [id2label[k] for k in sorted(id2label, key=lambda x: int(x))] if id2label else []
    return OutputContract(
        output_type="classification",
        n_classes=n_outputs,
        classes=classes,
        produces_probabilities=False,
    )


def extract_braindecode_contract(model_id: str, config: dict) -> ExtractedBraindecodeContract:
    return ExtractedBraindecodeContract(
        model_id=model_id,
        input_contract=_extract_input_contract(config),
        output_contract=_extract_output_contract(config),
    )


__all__ = ["ExtractedBraindecodeContract", "extract_braindecode_contract"]
