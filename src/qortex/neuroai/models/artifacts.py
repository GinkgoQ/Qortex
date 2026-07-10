"""Model artifact resolution and explicit download helpers."""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

from qortex.core.exceptions import ModelAdapterError


class ModelArtifactError(ModelAdapterError):
    """Raised when a model artifact is missing, incomplete, or invalid."""


ArtifactState = Literal[
    "installed_false",
    "configured_false",
    "load_failed",
    "bundle_missing",
    "weights_missing",
    "ready",
]


@dataclass(frozen=True)
class ArtifactStatus:
    model_id: str
    installed: bool
    ready: bool
    state: ArtifactState
    message: str
    path: str | None = None
    weight_path: str | None = None


MEDSAM_ZENODO_URL = "https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1"
MEDSAM_MIN_BYTES = 1_000_000
VISTA3D_REPO_ID = "MONAI/VISTA3D-HF"
VISTA3D_MIN_BYTES = 1_000_000
_VISTA_WEIGHT_PATTERNS = (
    "vista3d_pretrained_model/model.safetensors",
    "vista3d_pretrained_model/*.safetensors",
    "models/model.pt",
    "models/model.pth",
    "models/*.pt",
    "models/*.pth",
    "model.pt",
    "model.pth",
    "model.safetensors",
    "*.pt",
    "*.pth",
    "*.safetensors",
)


def qortex_model_cache_dir() -> Path:
    override = os.environ.get("QORTEX_MODEL_CACHE_DIR") or os.environ.get("QORTEX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "qortex" / "models"


def resolve_medsam_checkpoint(checkpoint: str | Path | None = None) -> Path:
    candidates = _medsam_checkpoint_candidates(checkpoint)
    last_error: Exception | None = None
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_file() or path.stat().st_size < MEDSAM_MIN_BYTES:
            continue
        try:
            _validate_torch_checkpoint(path)
        except Exception as exc:
            last_error = exc
            continue
        return path.resolve()
    if last_error is not None:
        raise ModelArtifactError(f"No valid MedSAM checkpoint found; last invalid checkpoint error: {last_error}")
    raise ModelArtifactError(
        "MedSAM is installed, but no valid medsam_vit_b.pth checkpoint was found. "
        "Set QORTEX_MEDSAM_CHECKPOINT or place medsam_vit_b.pth under "
        f"{qortex_model_cache_dir() / 'medsam'}."
    )


def download_medsam_checkpoint(
    *,
    target: str | Path | None = None,
    url: str = MEDSAM_ZENODO_URL,
) -> Path:
    path = Path(target).expanduser() if target else qortex_model_cache_dir() / "medsam" / "medsam_vit_b.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    expected = _remote_content_length(url)
    if path.exists() and expected is not None and path.stat().st_size != expected:
        path.unlink()
    if path.exists():
        return resolve_medsam_checkpoint(path)
    _download_with_resume(url, tmp, expected_size=expected)
    tmp.replace(path)
    return resolve_medsam_checkpoint(path)


def medsam_artifact_status(checkpoint: str | Path | None = None) -> ArtifactStatus:
    if find_spec("segment_anything") is None:
        return ArtifactStatus(
            model_id="foundation.medsam",
            installed=False,
            ready=False,
            state="installed_false",
            message="MedSAM dependencies are not installed: segment-anything is missing.",
        )
    try:
        path = resolve_medsam_checkpoint(checkpoint)
    except ModelArtifactError as exc:
        return ArtifactStatus(
            model_id="foundation.medsam",
            installed=True,
            ready=False,
            state="configured_false",
            message=str(exc),
        )
    return ArtifactStatus(
        model_id="foundation.medsam",
        installed=True,
        ready=True,
        state="ready",
        message="MedSAM checkpoint is valid.",
        path=str(path),
        weight_path=str(path),
    )


def resolve_vista3d_bundle(bundle_dir: str | Path | None = None) -> Path:
    candidates = _vista3d_bundle_candidates(bundle_dir)
    for candidate in candidates:
        path = candidate.expanduser()
        if not path.is_dir():
            continue
        resolve_vista3d_weights(path)
        _validate_vista3d_bundle_files(path)
        return path.resolve()
    raise ModelArtifactError(
        "No VISTA3D bundle directory was configured. Set QORTEX_VISTA3D_BUNDLE "
        f"or download the bundle under {qortex_model_cache_dir() / 'vista3d'}."
    )


def resolve_vista3d_weights(bundle_dir: str | Path) -> Path:
    bundle_path = Path(bundle_dir).expanduser().resolve()
    if not bundle_path.is_dir():
        raise ModelArtifactError(f"VISTA3D bundle directory does not exist: {bundle_path}")
    candidates: list[Path] = []
    for pattern in _VISTA_WEIGHT_PATTERNS:
        candidates.extend(bundle_path.glob(pattern))
    valid = sorted(
        {
            candidate.resolve()
            for candidate in candidates
            if candidate.is_file() and candidate.stat().st_size >= VISTA3D_MIN_BYTES
        },
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not valid:
        raise ModelArtifactError(
            "The VISTA3D bundle is incomplete: configs are present but model weights are absent. "
            f"Bundle path: {bundle_path}. Re-download the complete Hugging Face snapshot."
        )
    return valid[0]


def download_vista3d_bundle(
    *,
    repo_id: str = VISTA3D_REPO_ID,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelArtifactError(
            "VISTA3D artifact download requires huggingface_hub. "
            "Install with: pip install 'qortex[hf]' or pip install huggingface-hub."
        ) from exc
    target = Path(cache_dir).expanduser() if cache_dir else qortex_model_cache_dir() / "vista3d-hf"
    target.parent.mkdir(parents=True, exist_ok=True)
    root = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        token=_resolve_hf_token(token),
        local_files_only=False,
    )
    bundle_path = Path(root).resolve()
    resolve_vista3d_weights(bundle_path)
    return bundle_path


def vista3d_artifact_status(bundle_dir: str | Path | None = None) -> ArtifactStatus:
    if find_spec("monai") is None:
        return ArtifactStatus(
            model_id="monai.vista3d",
            installed=False,
            ready=False,
            state="installed_false",
            message="MONAI or VISTA3D dependencies are not installed.",
        )
    candidates = _vista3d_bundle_candidates(bundle_dir)
    existing = [path.expanduser().resolve() for path in candidates if path.expanduser().is_dir()]
    if not existing:
        return ArtifactStatus(
            model_id="monai.vista3d",
            installed=True,
            ready=False,
            state="bundle_missing",
            message="No VISTA3D bundle directory was configured.",
        )
    bundle_path = existing[0]
    try:
        weights = resolve_vista3d_weights(bundle_path)
        _validate_vista3d_bundle_files(bundle_path)
    except ModelArtifactError as exc:
        return ArtifactStatus(
            model_id="monai.vista3d",
            installed=True,
            ready=False,
            state="weights_missing",
            message=str(exc),
            path=str(bundle_path),
        )
    return ArtifactStatus(
        model_id="monai.vista3d",
        installed=True,
        ready=True,
        state="ready",
        message="VISTA3D bundle, checkpoint, and inference configuration are valid.",
        path=str(bundle_path),
        weight_path=str(weights),
    )


def _medsam_checkpoint_candidates(checkpoint: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if checkpoint is not None:
        candidates.append(Path(checkpoint))
    env_path = os.environ.get("QORTEX_MEDSAM_CHECKPOINT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        qortex_model_cache_dir() / "medsam" / "medsam_vit_b.pth",
        Path.home() / ".cache" / "qortex" / "medsam" / "medsam_vit_b.pth",
        Path.home() / ".cache" / "medsam" / "medsam_vit_b.pth",
    ])
    return candidates


def _vista3d_bundle_candidates(bundle_dir: str | Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if bundle_dir is not None:
        candidates.append(Path(bundle_dir))
    env_path = os.environ.get("QORTEX_VISTA3D_BUNDLE")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        qortex_model_cache_dir() / "vista3d-hf",
        qortex_model_cache_dir() / "vista3d",
        Path.home() / ".cache" / "qortex" / "vista3d",
        Path.home() / ".cache" / "torch" / "hub" / "bundle" / "vista3d",
    ])
    return candidates


def _validate_vista3d_bundle_files(bundle_path: Path) -> None:
    hf_config = bundle_path / "vista3d_pretrained_model" / "config.json"
    helper = bundle_path / "hugging_face_pipeline.py"
    monai_config = bundle_path / "configs" / "inference.json"
    if hf_config.is_file() and helper.is_file():
        return
    if monai_config.is_file():
        return
    raise ModelArtifactError(
        "VISTA3D inference configuration was not found. Expected either "
        f"{hf_config} with {helper}, or {monai_config}."
    )


def _validate_torch_checkpoint(path: Path) -> None:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise ModelArtifactError(f"Invalid torch checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise ModelArtifactError(f"Unexpected torch checkpoint format: {path}")


def _resolve_hf_token(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for key in ("QORTEX_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            if key.strip() in {"QORTEX_HF_TOKEN", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
                value = value.strip().strip("\"'")
                if value:
                    return value
    return None


def _remote_content_length(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.headers.get("content-length")
            return int(raw) if raw else None
    except Exception:
        return None


def _download_with_resume(url: str, path: Path, *, expected_size: int | None) -> None:
    max_attempts = 8
    last_size = -1
    for _attempt in range(max_attempts):
        existing = path.stat().st_size if path.exists() else 0
        if expected_size is not None and existing == expected_size:
            return
        if existing == last_size and existing > 0:
            pass
        last_size = existing
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        request = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing else "wb"
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if existing and getattr(response, "status", None) == 200:
                    mode = "wb"
                with path.open(mode) as fh:
                    shutil.copyfileobj(response, fh, length=1 << 20)
        except Exception as exc:
            if _attempt == max_attempts - 1:
                raise ModelArtifactError(f"Download failed for {url}: {exc}") from exc
            continue
    observed = path.stat().st_size if path.exists() else 0
    if expected_size is not None and observed != expected_size:
        raise ModelArtifactError(
            f"Download incomplete for {url}: expected {expected_size} bytes, got {observed}."
        )
