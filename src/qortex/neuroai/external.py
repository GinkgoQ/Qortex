"""External NeuroAI runner integration.

Qortex does not reimplement mature segmentation engines such as nnU-Net or
TotalSegmentator.  This module provides a typed, validated subprocess boundary:
build a command, run it, capture provenance, verify outputs, and return paths
that can be passed to NeuroAI output adapters or showcase rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from qortex.core.exceptions import ModelAdapterError, QortexError

ExternalSegmentationEngine = Literal[
    "totalsegmentator", "nnunet", "synthseg", "synthstrip", "hdbet",
    "fastsurfer", "tractseg",
]

_LOG_TAIL_CHARS = 8_000


@dataclass(frozen=True)
class ExternalSegmentationRequest:
    """Request for a file-based external segmentation run."""

    engine: ExternalSegmentationEngine
    image_path: str | Path
    output_path: str | Path
    task: str | None = None
    model_folder: str | Path | None = None
    dataset_id: int | None = None
    configuration: str | None = None
    trainer: str | None = None
    plans: str | None = None
    folds: tuple[int | str, ...] = ("all",)
    subject_id: str | None = None  # required only for engine="fastsurfer"
    device: str | None = None
    timeout_s: float | None = None
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalSegmentationResult:
    """Result and provenance from an external segmentation process."""

    engine: ExternalSegmentationEngine
    command: tuple[str, ...]
    image_path: Path
    output_path: Path
    returncode: int
    elapsed_s: float
    started_at: str
    finished_at: str
    stdout: str
    stderr: str
    metadata_path: Path
    executable_path: str | None = None
    executable_sha256: str | None = None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    output_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.returncode == 0 and self.output_path.exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "command": list(self.command),
            "image_path": str(self.image_path),
            "output_path": str(self.output_path),
            "returncode": self.returncode,
            "elapsed_s": self.elapsed_s,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "output_summary": self.output_summary,
            "metadata_path": str(self.metadata_path),
            "executable_path": self.executable_path,
            "executable_sha256": self.executable_sha256,
            "success": self.success,
        }


class ExternalSegmentationError(QortexError):
    """Raised when an external segmentation engine cannot be run safely."""

    default_code = "neuroai.external_segmentation_error"


def run_external_segmentation(request: ExternalSegmentationRequest) -> ExternalSegmentationResult:
    """Run a supported external segmentation engine and verify its output."""

    image_path = Path(request.image_path).expanduser().resolve()
    output_path = Path(request.output_path).expanduser().resolve()
    _validate_external_request(request, image_path, output_path)
    command = _build_external_command(request, image_path, output_path)
    zoo_entry = _lookup_external_zoo_entry(request.engine)
    if zoo_entry is not None:
        try:
            from qortex.neuroai.models.security import check_executable_allowlist
            check_executable_allowlist(zoo_entry, command[0])
        except ModelAdapterError as exc:
            raise ExternalSegmentationError(str(exc)) from exc

    # Real executable identity, hashed at the moment of execution -- this is
    # the concrete foundation for detecting a mutated/swapped binary between
    # a compiled plan and its later execution (the allowlist only checked
    # the resolved basename; this records what actually ran).
    executable_path = command[0]
    executable_sha256 = _file_facts(Path(executable_path))["sha256"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.update(request.env)
    if request.engine == "nnunet" and request.model_folder is not None:
        env["nnUNet_results"] = str(Path(request.model_folder).expanduser().resolve())

    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=request.timeout_s,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        raise ExternalSegmentationError(
            f"{request.engine} timed out after {float(exc.timeout):.1f}s",
            context={
                "command": command,
                "timeout_s": float(exc.timeout),
                "stdout": stdout[-4000:],
                "stderr": stderr[-4000:],
            },
            suggestion="Increase --timeout-s for large models, use a smaller task/device setting, or run the external engine directly to pre-download required weights.",
            retriable=True,
        ) from exc
    elapsed = time.perf_counter() - t0
    finished = datetime.now(timezone.utc)

    result = ExternalSegmentationResult(
        engine=request.engine,
        command=tuple(command),
        image_path=image_path,
        output_path=output_path,
        returncode=int(completed.returncode),
        elapsed_s=float(elapsed),
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        stdout=_tail_text(completed.stdout, _LOG_TAIL_CHARS),
        stderr=_tail_text(completed.stderr, _LOG_TAIL_CHARS),
        metadata_path=_metadata_path_for_output(output_path),
        executable_path=executable_path,
        executable_sha256=executable_sha256,
        stdout_bytes=len(completed.stdout.encode("utf-8", errors="replace")),
        stderr_bytes=len(completed.stderr.encode("utf-8", errors="replace")),
        stdout_truncated=len(completed.stdout) > _LOG_TAIL_CHARS,
        stderr_truncated=len(completed.stderr) > _LOG_TAIL_CHARS,
        output_summary=_summarize_external_output(output_path),
    )
    result.metadata_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    if zoo_entry is not None:
        _write_zoo_provenance(result.metadata_path, zoo_entry, image_path, output_path)

    if completed.returncode != 0:
        raise ExternalSegmentationError(
            f"{request.engine} failed with exit code {completed.returncode}",
            context={
                "command": command,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
            suggestion="Check that the external tool is installed, its weights are available, and the input modality matches the selected task.",
        )
    if not output_path.exists():
        raise ExternalSegmentationError(
            f"{request.engine} finished without writing the expected output: {output_path}",
            context={"command": command, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]},
        )
    return result


def build_external_segmentation_command(request: ExternalSegmentationRequest) -> tuple[str, ...]:
    """Return the command that would be executed for a request."""

    image_path = Path(request.image_path).expanduser().resolve()
    output_path = Path(request.output_path).expanduser().resolve()
    _validate_external_request(request, image_path, output_path, check_image_exists=False)
    return tuple(_build_external_command(request, image_path, output_path))


def available_external_segmentation_engines() -> dict[str, bool]:
    """Report which supported external segmentation CLIs are on PATH."""

    return {
        "totalsegmentator": shutil.which("TotalSegmentator") is not None,
        "nnunet": shutil.which("nnUNetv2_predict") is not None,
        "synthseg": shutil.which("mri_synthseg") is not None,
        "synthstrip": shutil.which("mri_synthstrip") is not None,
        "hdbet": shutil.which("hd-bet") is not None,
        "fastsurfer": shutil.which("run_fastsurfer.sh") is not None,
        "tractseg": shutil.which("TractSeg") is not None,
    }


_SUPPORTED_ENGINES = (
    "totalsegmentator", "nnunet", "synthseg", "synthstrip", "hdbet",
    "fastsurfer", "tractseg",
)


def _validate_external_request(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
    *,
    check_image_exists: bool = True,
) -> None:
    if request.engine not in _SUPPORTED_ENGINES:
        raise ExternalSegmentationError(f"Unsupported external segmentation engine: {request.engine!r}")
    if check_image_exists and not image_path.exists():
        raise ExternalSegmentationError(f"Input image does not exist: {image_path}")
    if request.engine == "nnunet":
        missing = []
        if request.dataset_id is None:
            missing.append("dataset_id")
        if request.configuration is None:
            missing.append("configuration")
        if missing:
            raise ExternalSegmentationError(
                f"nnU-Net request is missing required fields: {', '.join(missing)}"
            )
    if request.engine == "fastsurfer" and request.subject_id is None:
        raise ExternalSegmentationError(
            "FastSurfer request is missing required field: subject_id"
        )


def _build_external_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    builders = {
        "totalsegmentator": _build_totalsegmentator_command,
        "nnunet": _build_nnunet_command,
        "synthseg": _build_synthseg_command,
        "synthstrip": _build_synthstrip_command,
        "hdbet": _build_hdbet_command,
        "fastsurfer": _build_fastsurfer_command,
        "tractseg": _build_tractseg_command,
    }
    return builders[request.engine](request, image_path, output_path)


def _build_totalsegmentator_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("TotalSegmentator")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    if request.task:
        command.extend(["--task", request.task])
    if request.device:
        command.extend(["--device", request.device])
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_nnunet_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("nnUNetv2_predict")
    command = [
        executable,
        "-i",
        str(image_path.parent),
        "-o",
        str(output_path.parent),
        "-d",
        str(request.dataset_id),
        "-c",
        str(request.configuration),
        "-f",
        *[str(fold) for fold in request.folds],
    ]
    if request.trainer:
        command.extend(["-tr", request.trainer])
    if request.plans:
        command.extend(["-p", request.plans])
    if request.device:
        command.extend(["-device", request.device])
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_synthseg_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("mri_synthseg")
    command = [executable, "--i", str(image_path), "--o", str(output_path)]
    if request.device == "cpu":
        command.append("--cpu")
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_synthstrip_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("mri_synthstrip")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    if request.device and request.device != "cpu":
        command.append("--gpu")
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_hdbet_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("hd-bet")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    if request.device:
        command.extend(["-device", request.device])
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_fastsurfer_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    # FastSurfer's CLI shape differs from the others: it writes into a
    # subjects-directory layout keyed by subject_id, not a single output
    # file/dir. subject_id is validated as required in
    # _validate_external_request before this builder ever runs.
    executable = _require_executable("run_fastsurfer.sh")
    command = [
        executable,
        "--t1", str(image_path),
        "--sid", str(request.subject_id),
        "--sd", str(output_path),
    ]
    if request.device:
        command.extend(["--device", request.device])
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_tractseg_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("TractSeg")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _require_executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise ExternalSegmentationError(
            f"Required executable is not on PATH: {name}",
            suggestion=f"Install {name} and confirm it is available in the active environment.",
        )
    return resolved


def _clean_extra_args(args: Sequence[str]) -> list[str]:
    cleaned = []
    for arg in args:
        value = str(arg)
        if "\x00" in value:
            raise ExternalSegmentationError("External command arguments must not contain NUL bytes")
        cleaned.append(value)
    return cleaned


def _metadata_path_for_output(output_path: Path) -> Path:
    if output_path.exists() and output_path.is_dir():
        return output_path / "qortex_external_segmentation.json"
    suffix = output_path.suffix or ".out"
    return output_path.with_suffix(suffix + ".qortex.json")


def _tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _summarize_external_output(output_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": output_path.exists(),
        "is_dir": output_path.is_dir(),
        "file_count": 0,
        "nifti_count": 0,
        "nonempty_nifti_count": 0,
        "empty_nifti_count": 0,
        "nonempty_nifti_examples": [],
        "empty_nifti_examples": [],
        "warnings": [],
    }
    if not output_path.exists():
        summary["warnings"].append("Expected output path does not exist.")
        return summary

    files = (
        sorted(p for p in output_path.rglob("*") if p.is_file())
        if output_path.is_dir()
        else [output_path]
    )
    summary["file_count"] = len(files)
    nifti_files = [p for p in files if _is_nifti_path(p)]
    summary["nifti_count"] = len(nifti_files)
    if not nifti_files:
        return summary

    try:
        import nibabel as nib
        import numpy as np
    except ImportError:
        summary["warnings"].append("nibabel/numpy unavailable; NIfTI mask content was not inspected.")
        return summary

    for path in nifti_files:
        rel = str(path.relative_to(output_path)) if output_path.is_dir() else path.name
        try:
            image = nib.load(str(path))
            data = np.asanyarray(image.dataobj)
            nonzero = int(np.count_nonzero(data))
        except Exception as exc:
            summary["warnings"].append(f"Could not inspect {rel}: {exc}")
            continue
        item = {
            "path": rel,
            "shape": [int(dim) for dim in data.shape],
            "dtype": str(data.dtype),
            "nonzero_voxels": nonzero,
        }
        if nonzero:
            summary["nonempty_nifti_count"] += 1
            if len(summary["nonempty_nifti_examples"]) < 10:
                summary["nonempty_nifti_examples"].append(item)
        else:
            summary["empty_nifti_count"] += 1
            if len(summary["empty_nifti_examples"]) < 10:
                summary["empty_nifti_examples"].append(item)
    if summary["nifti_count"] and summary["nonempty_nifti_count"] == 0:
        summary["warnings"].append("All generated NIfTI outputs are empty.")
    elif summary["nifti_count"] and summary["empty_nifti_count"] / summary["nifti_count"] >= 0.8:
        summary["warnings"].append("Most generated NIfTI outputs are empty; verify task/input modality fit.")
    return summary


def _is_nifti_path(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _lookup_external_zoo_entry(engine: str):
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401
    from qortex.neuroai.models.zoo.registry import lookup

    return lookup(f"external.{engine}")


def _file_facts(path: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": None,
        "sha256": None,
    }
    if path.is_file():
        facts["size_bytes"] = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        facts["sha256"] = digest.hexdigest()
    return facts


def _write_zoo_provenance(metadata_path: Path, zoo_entry: Any, image_path: Path, output_path: Path) -> None:
    provenance_path = metadata_path.with_name(metadata_path.name + ".model_zoo_entry.json")
    payload = {
        "zoo_entry_id": zoo_entry.id,
        "provider": zoo_entry.provider,
        "source_url": zoo_entry.source_url,
        "license_evidence_status": zoo_entry.license.evidence_status.value,
        "security_executable_names": list(zoo_entry.security.executable_names),
        "geometry_ledger": {
            "scope": "file_level",
            "note": "Records existence, size, and sha256 only; NIfTI header geometry is not parsed.",
            "input": _file_facts(image_path),
            "output": _file_facts(output_path),
        },
    }
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ExternalSegmentationEngine",
    "ExternalSegmentationError",
    "ExternalSegmentationRequest",
    "ExternalSegmentationResult",
    "available_external_segmentation_engines",
    "build_external_segmentation_command",
    "run_external_segmentation",
]
