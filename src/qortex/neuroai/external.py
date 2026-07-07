"""External NeuroAI runner integration.

Qortex does not reimplement mature segmentation engines such as nnU-Net or
TotalSegmentator.  This module provides a typed, validated subprocess boundary:
build a command, run it, capture provenance, verify outputs, and return paths
that can be passed to NeuroAI output adapters or showcase rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Literal, Sequence

from qortex.core.exceptions import QortexError

ExternalSegmentationEngine = Literal["totalsegmentator", "nnunet"]


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
            "metadata_path": str(self.metadata_path),
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.update(request.env)
    if request.engine == "nnunet" and request.model_folder is not None:
        env["nnUNet_results"] = str(Path(request.model_folder).expanduser().resolve())

    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=request.timeout_s,
        env=env,
        check=False,
    )
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
        stdout=completed.stdout,
        stderr=completed.stderr,
        metadata_path=_metadata_path_for_output(output_path),
    )
    result.metadata_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    if completed.returncode != 0:
        raise ExternalSegmentationError(
            f"{request.engine} failed with exit code {completed.returncode}",
            context={"command": command, "stderr": completed.stderr[-4000:]},
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
    }


def _validate_external_request(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
    *,
    check_image_exists: bool = True,
) -> None:
    if request.engine not in ("totalsegmentator", "nnunet"):
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


def _build_external_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    if request.engine == "totalsegmentator":
        return _build_totalsegmentator_command(request, image_path, output_path)
    return _build_nnunet_command(request, image_path, output_path)


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


__all__ = [
    "ExternalSegmentationEngine",
    "ExternalSegmentationError",
    "ExternalSegmentationRequest",
    "ExternalSegmentationResult",
    "available_external_segmentation_engines",
    "build_external_segmentation_command",
    "run_external_segmentation",
]
