"""Real pretrained-model validation against a pinned public BraTS case."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from qortex.neuroai.models.cache import CacheEntry, ModelCache
from qortex.neuroai.models.monai import MONAIBundleAdapter
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

MODEL_ID = "monai.brats_mri_segmentation"
MODEL_REPO_ID = "MONAI/brats_mri_segmentation"
MODEL_REVISION = "370f7f9d062745fbac445e7fe6d6616d35df04ec"
MODEL_SOURCE_URL = f"https://huggingface.co/{MODEL_REPO_ID}"

DATASET_REPO_ID = "MedOtter/brats2023-gli-dataset"
DATASET_REVISION = "b032d353a3e80911a5f850bc54e6fb575298a354"
DATASET_SOURCE_URL = f"https://huggingface.co/datasets/{DATASET_REPO_ID}"
DATASET_HOMEPAGE = "https://www.synapse.org/#!Synapse:syn51156910"
DATASET_LICENSE = "CC-BY-4.0"
DEFAULT_CASE_ID = "BraTS-GLI-00000-000"

_MODEL_CHANNEL_TO_DATASET_KEY = {
    "T1c": "t1c",
    "T1": "t1n",
    "T2": "t2w",
    "FLAIR": "t2f",
}
_INFERENCE_LOCK = threading.Lock()


def _default_model_bundle() -> Path:
    cache_root = Path(os.environ.get("QORTEX_CACHE_DIR", Path.home() / ".qortex" / "model_cache"))
    return cache_root / "bundles" / "brats_mri_segmentation"


def _default_dataset_root() -> Path:
    return Path.home() / ".qortex" / "datasets" / "brats2023-gli"


def _default_run_root() -> Path:
    return Path.home() / ".qortex" / "runs" / "brats-validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_model_bundle(bundle_dir: Path) -> Path:
    checkpoint = bundle_dir / "models" / "model.pt"
    if checkpoint.is_file():
        return bundle_dir
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "Public BraTS validation requires huggingface-hub. "
            "Install qortex with the 'hf' optional dependency."
        ) from None
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        revision=MODEL_REVISION,
        local_dir=bundle_dir,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Downloaded MONAI bundle has no checkpoint at {checkpoint}")
    return bundle_dir


def _dataset_record(case_id: str) -> tuple[dict[str, Any], list[str]]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise ImportError(
            "Public BraTS validation requires huggingface-hub. "
            "Install qortex with the 'hf' optional dependency."
        ) from None

    datalist = Path(hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename="train.jsonl",
        repo_type="dataset",
        revision=DATASET_REVISION,
    ))
    record = None
    with datalist.open(encoding="utf-8") as stream:
        for line in stream:
            candidate = json.loads(line)
            if candidate.get("patient_id") == case_id:
                record = candidate
                break
    if record is None:
        raise KeyError(f"Case {case_id!r} is not present in the pinned public dataset revision")
    repo_files = HfApi().list_repo_files(
        DATASET_REPO_ID,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    return record, repo_files


def _resolve_declared_repo_path(declared: str, repo_files: list[str]) -> str:
    if declared in repo_files:
        return declared
    filename = PurePosixPath(declared).name
    matches = [path for path in repo_files if PurePosixPath(path).name == filename]
    if len(matches) != 1:
        raise ValueError(
            f"Dataset manifest path {declared!r} does not resolve uniquely in the pinned revision"
        )
    return matches[0]


def _download_public_case(
    case_id: str,
    dataset_root: Path,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[list[Path], Path, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    record, repo_files = _dataset_record(case_id)
    modalities = record.get("modalities")
    if not isinstance(modalities, dict):
        raise ValueError(f"Public dataset record {case_id!r} has no modalities mapping")

    model_order = ["T1c", "T1", "T2", "FLAIR"]
    declared = [modalities.get(_MODEL_CHANNEL_TO_DATASET_KEY[channel]) for channel in model_order]
    if not all(isinstance(path, str) and path for path in declared):
        raise ValueError(f"Public dataset record {case_id!r} lacks the four required MRI modalities")
    mask_declared = record.get("mask")
    if not isinstance(mask_declared, str) or not mask_declared:
        raise ValueError(f"Public dataset record {case_id!r} has no ground-truth mask")

    resolved = [_resolve_declared_repo_path(path, repo_files) for path in [*declared, mask_declared]]
    local_paths: list[Path] = []
    total = len(resolved) + 4
    for index, repo_path in enumerate(resolved, start=1):
        local_paths.append(Path(hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=repo_path,
            repo_type="dataset",
            revision=DATASET_REVISION,
            local_dir=dataset_root,
        )))
        if on_progress:
            on_progress(index, total)

    provenance = {
        "repo_id": DATASET_REPO_ID,
        "revision": DATASET_REVISION,
        "source_url": DATASET_SOURCE_URL,
        "original_homepage": DATASET_HOMEPAGE,
        "license": DATASET_LICENSE,
        "case_id": case_id,
        "official_split": record.get("official_split"),
        "declared_paths": [*declared, mask_declared],
        "resolved_repo_paths": resolved,
    }
    return local_paths[:4], local_paths[4], provenance


def _validate_geometry(image_paths: list[Path], mask_path: Path) -> tuple[Any, tuple[int, ...]]:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "Public BraTS validation requires nibabel. Install qortex with the 'mri' extra."
        ) from None

    references = [nib.load(str(path)) for path in [*image_paths, mask_path]]
    shape = references[0].shape
    affine = references[0].affine
    for path, image in zip([*image_paths, mask_path], references):
        if image.shape != shape:
            raise ValueError(f"Geometry mismatch: {path} has shape {image.shape}, expected {shape}")
        if not np.allclose(image.affine, affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"Affine mismatch between the public case volumes: {path}")
    if len(shape) != 3:
        raise ValueError(f"BraTS validation expects aligned 3D volumes, got shape {shape}")
    return references[0], shape


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(target.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _runtime_environment(torch: Any, resolved_device: str) -> dict[str, Any]:
    packages = {}
    for distribution in ("qortex", "monai", "torch", "numpy", "nibabel"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": None,
    }
    if resolved_device == "cuda" and torch.cuda.is_available():
        cuda["device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda": cuda,
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "process_executable": str(Path(sys.executable).resolve()),
    }


def run_public_brats_validation(
    *,
    case_id: str = DEFAULT_CASE_ID,
    device: str = "auto",
    model_bundle: Path | str | None = None,
    dataset_root: Path | str | None = None,
    run_root: Path | str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    execution_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run pinned MONAI weights against one real, public BraTS 2023 case."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be auto, cpu, or cuda; got {device!r}")
    if not case_id:
        raise ValueError("case_id must not be empty")

    bundle_dir = _download_model_bundle(Path(model_bundle) if model_bundle else _default_model_bundle())
    checkpoint = bundle_dir / "models" / "model.pt"
    image_paths, mask_path, dataset_provenance = _download_public_case(
        case_id,
        Path(dataset_root) if dataset_root else _default_dataset_root(),
        on_progress,
    )
    reference_image, input_shape = _validate_geometry(image_paths, mask_path)
    if on_progress:
        on_progress(6, 9)

    try:
        import monai
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "Public BraTS validation requires MONAI and nibabel. "
            "Install qortex with the 'monai' and 'mri' extras."
        ) from exc

    parser = monai.bundle.ConfigParser()
    parser.read_config(str(bundle_dir / "configs" / "inference.json"))
    preprocessing = parser.get_parsed_content("preprocessing")
    preprocess_started = time.perf_counter()
    prepared = preprocessing({"image": [str(path) for path in image_paths]})
    input_tensor = prepared["image"]
    preprocess_seconds = time.perf_counter() - preprocess_started
    if tuple(input_tensor.shape) != (4, *input_shape):
        raise ValueError(
            f"Bundle preprocessing produced shape {tuple(input_tensor.shape)}, "
            f"expected {(4, *input_shape)}"
        )

    adapter = MONAIBundleAdapter(ModelSpec(
        provider="monai",
        id=str(bundle_dir),
        extra={"required_transforms": [{
            "kind": "normalize_intensity",
            "nonzero": True,
            "channel_wise": True,
        }]},
    ))
    runtime = RuntimeSpec(device=device, fp16=True)
    with _INFERENCE_LOCK:
        adapter.load(runtime)
        import torch

        resolved_device = adapter._device
        if resolved_device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        inference_started = time.perf_counter()
        output = adapter.predict(input_tensor)
        if resolved_device == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_started
        peak_memory = {
            "allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "measurement": "torch CUDA peak counters reset immediately before synchronized inference",
        } if resolved_device == "cuda" else {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "measurement": "Per-run CPU peak memory is not measured by this execution path.",
        }
        environment = _runtime_environment(torch, resolved_device)
        adapter.unload()
    if on_progress:
        on_progress(7, 9)

    regions = np.asarray(output.mask, dtype=np.uint8)
    if regions.shape != (3, *input_shape):
        raise ValueError(f"BraTS model returned mask shape {regions.shape}, expected {(3, *input_shape)}")
    truth = np.asanyarray(nib.load(str(mask_path)).dataobj)
    target_regions = np.stack((np.isin(truth, [1, 3]), truth > 0, truth == 3))
    region_names = ("tumor_core", "whole_tumor", "enhancing_tumor")
    metrics = {
        name: {
            "dice": _dice(pred.astype(bool), target),
            "predicted_voxels": int(pred.sum()),
            "target_voxels": int(target.sum()),
        }
        for name, pred, target in zip(region_names, regions, target_regions)
    }
    consistency = {
        "tumor_core_outside_whole_tumor_voxels": int(np.logical_and(regions[0], ~regions[1].astype(bool)).sum()),
        "enhancing_tumor_outside_tumor_core_voxels": int(np.logical_and(regions[2], ~regions[0].astype(bool)).sum()),
    }

    categorical = np.zeros(input_shape, dtype=np.uint8)
    categorical[regions[1].astype(bool)] = 2
    categorical[regions[0].astype(bool)] = 1
    categorical[regions[2].astype(bool)] = 3

    runs = Path(run_root) if run_root else _default_run_root()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    final_run_dir = runs / run_id
    run_dir = runs / f".{run_id}.tmp"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(image_paths[0], run_dir / "input_t1c.nii.gz")
    shutil.copy2(mask_path, run_dir / "ground_truth.nii.gz")
    nib.save(nib.Nifti1Image(categorical, reference_image.affine, reference_image.header), run_dir / "prediction.nii.gz")
    nib.save(
        nib.Nifti1Image(np.moveaxis(regions, 0, -1), reference_image.affine),
        run_dir / "prediction_regions.nii.gz",
    )
    if on_progress:
        on_progress(8, 9)

    model_hash = _sha256(checkpoint)
    ModelCache().record(CacheEntry(
        model_id=MODEL_ID,
        provider="monai",
        local_path=str(bundle_dir.resolve()),
        size_bytes=checkpoint.stat().st_size,
        sha256=model_hash,
        downloaded_at=datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat(),
        source_url=MODEL_SOURCE_URL,
    ))
    result = {
        "run_id": run_id,
        "status": "completed",
        "research_use_only": True,
        "execution_profile": execution_profile,
        "model": {
            "id": MODEL_ID,
            "repo_id": MODEL_REPO_ID,
            "revision": MODEL_REVISION,
            "bundle_version": adapter._metadata.get("version"),
            "checkpoint_sha256": model_hash,
            "license": "Apache-2.0",
            "source_url": MODEL_SOURCE_URL,
        },
        "dataset": dataset_provenance,
        "input": {
            "channel_order": ["T1c", "T1", "T2", "FLAIR"],
            "shape": [4, *input_shape],
            "voxel_spacing_mm": list(reference_image.header.get_zooms()[:3]),
            "files": [
                {"path": str(path.resolve()), "sha256": _sha256(path)}
                for path in [*image_paths, mask_path]
            ],
        },
        "runtime": {
            "device": resolved_device,
            "mixed_precision": resolved_device == "cuda",
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "roi_size": list(adapter._inference_settings["roi_size"]),
            "overlap": adapter._inference_settings["overlap"],
            "threshold": adapter._inference_settings["threshold"],
            "peak_memory": peak_memory,
        },
        "reproducibility": {
            "seed": None,
            "seed_evidence": "No seed is configured; this run performs pretrained inference without training.",
            "precision": "mixed float16/float32" if resolved_device == "cuda" else "float32",
            "model_config_sha256": {
                name: _sha256(bundle_dir / "configs" / name)
                for name in ("inference.json", "metadata.json")
                if (bundle_dir / "configs" / name).is_file()
            },
            "environment": environment,
        },
        "metrics": metrics,
        "region_consistency": consistency,
        "artifacts": {
            "input": "input_t1c.nii.gz",
            "ground_truth": "ground_truth.nii.gz",
            "prediction": "prediction.nii.gz",
            "prediction_regions": "prediction_regions.nii.gz",
            "provenance": "result.json",
        },
    }
    result = json.loads(json.dumps(result, default=_json_default))
    result["artifact_inventory"] = {
        name: {
            "path": filename,
            "size_bytes": (run_dir / filename).stat().st_size,
            "sha256": _sha256(run_dir / filename),
        }
        for name, filename in result["artifacts"].items()
        if name != "provenance"
    }
    result["artifact_inventory_evidence"] = (
        "Every binary run artifact is hashed after writing. result.json is the provenance record "
        "and cannot contain a non-recursive hash of itself."
    )
    _write_json(run_dir / "result.json", result)
    run_dir.replace(final_run_dir)
    if on_progress:
        on_progress(9, 9)
    return result


def load_public_brats_run(run_id: str, *, run_root: Path | str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else _default_run_root()).resolve()
    result_path = (root / run_id / "result.json").resolve()
    if root not in result_path.parents or not result_path.is_file():
        raise FileNotFoundError(f"No public BraTS validation run {run_id!r}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run result {result_path} is not a JSON object")
    return payload


def public_brats_artifact_path(run_id: str, artifact: str) -> Path:
    result = load_public_brats_run(run_id)
    filename = result.get("artifacts", {}).get(artifact)
    if not isinstance(filename, str):
        raise KeyError(f"Run {run_id!r} has no artifact {artifact!r}")
    root = _default_run_root().resolve()
    path = (root / run_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Artifact {artifact!r} is missing for run {run_id!r}")
    return path


__all__ = [
    "DEFAULT_CASE_ID",
    "load_public_brats_run",
    "public_brats_artifact_path",
    "run_public_brats_validation",
]
