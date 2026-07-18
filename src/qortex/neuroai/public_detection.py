"""Artifact-backed object-detection validation on a pinned public COCO image."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import threading
import time
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import numpy as np

from qortex.neuroai.models.cache import CacheEntry, ModelCache
from qortex.neuroai.models.torchvision_adapter import TorchvisionAdapter
from qortex.neuroai.showcase import Detection, DetectionShowcaseInput, render_detection_showcase
from qortex.neuroai.spec import ModelSpec, RuntimeSpec

MODEL_ID = "torchvision/fasterrcnn_resnet50_fpn_v2"
MODEL_NAME = "fasterrcnn_resnet50_fpn_v2"
MODEL_WEIGHTS = "COCO_V1"
MODEL_SOURCE_URL = "https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html"

DATASET_ID = "coco-2017-val"
DATASET_HOMEPAGE = "https://cocodataset.org/#download"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
DEFAULT_IMAGE_ID = 397133
DEFAULT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000397133.jpg"
DEFAULT_IMAGE_SHA256 = "09e1d25c75f7879bdaa69c327fece5cabacd53939c8c2ef9e87f1c97a2e478c4"
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5

_INFERENCE_LOCK = threading.Lock()


def _default_dataset_root() -> Path:
    return Path.home() / ".qortex" / "datasets" / "coco2017"


def _default_run_root() -> Path:
    return Path.home() / ".qortex" / "runs" / "detection-validation"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_download(url: str, destination: Path, expected_sha256: str) -> Path:
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "Qortex public validation"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target, length=1 << 20)
    actual = _sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Downloaded artifact hash mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(destination)
    return destination


def _load_coco_sample(
    image_id: int,
    dataset_root: Path,
    on_progress: Callable[[int, int], None] | None,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if image_id != DEFAULT_IMAGE_ID:
        raise ValueError(
            f"Only the pinned COCO validation image {DEFAULT_IMAGE_ID} is supported; got {image_id}"
        )
    archive_path = _ensure_download(
        ANNOTATIONS_URL,
        dataset_root / Path(urlparse(ANNOTATIONS_URL).path).name,
        ANNOTATIONS_SHA256,
    )
    if on_progress:
        on_progress(1, 7)
    annotations_path = dataset_root / "instances_val2017.json"
    if not annotations_path.is_file():
        temporary = annotations_path.with_suffix(".json.tmp")
        with zipfile.ZipFile(archive_path) as archive:
            try:
                with archive.open(ANNOTATIONS_MEMBER) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
            except KeyError:
                raise ValueError(f"Pinned COCO archive has no {ANNOTATIONS_MEMBER}") from None
        temporary.replace(annotations_path)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_record = next((item for item in payload.get("images", []) if item.get("id") == image_id), None)
    if not isinstance(image_record, dict):
        raise ValueError(f"COCO annotations contain no image id {image_id}")
    image_url = image_record.get("coco_url")
    if image_url != DEFAULT_IMAGE_URL:
        raise ValueError(f"Pinned image URL changed in COCO annotations: {image_url!r}")
    image_path = _ensure_download(image_url, dataset_root / image_record["file_name"], DEFAULT_IMAGE_SHA256)
    if on_progress:
        on_progress(2, 7)
    annotations = [
        item for item in payload.get("annotations", [])
        if item.get("image_id") == image_id and not item.get("iscrowd")
    ]
    categories = [item for item in payload.get("categories", []) if isinstance(item, dict)]
    licenses = [item for item in payload.get("licenses", []) if isinstance(item, dict)]
    image_license = next((item for item in licenses if item.get("id") == image_record.get("license")), None)
    return image_path, image_record, annotations, categories, {
        "id": DATASET_ID,
        "split": "val2017",
        "image_id": image_id,
        "source_url": image_url,
        "homepage": DATASET_HOMEPAGE,
        "annotation_url": ANNOTATIONS_URL,
        "annotation_archive_sha256": ANNOTATIONS_SHA256,
        "annotation_json_sha256": _sha256(annotations_path),
        "image_license": image_license,
    }


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, width, height = [float(value) for value in box]
    return [x, y, x + width, y + height]


def _iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_single_image(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Greedily match score-ranked detections to unmatched same-class boxes."""
    unmatched = set(range(len(ground_truth)))
    evaluated: list[dict[str, Any]] = []
    matched_ious: list[float] = []
    for prediction in sorted(predictions, key=lambda item: item["score"], reverse=True):
        candidates = [
            (index, _iou(prediction["bbox_xyxy"], target["bbox_xyxy"]))
            for index, target in enumerate(ground_truth)
            if index in unmatched and target["category_id"] == prediction["category_id"]
        ]
        best_index, best_iou = max(candidates, key=lambda item: item[1], default=(None, 0.0))
        matched = best_index is not None and best_iou >= iou_threshold
        if matched:
            unmatched.remove(best_index)
            matched_ious.append(best_iou)
        evaluated.append({
            **prediction,
            "match": "true_positive" if matched else "false_positive",
            "matched_annotation_id": ground_truth[best_index]["annotation_id"] if matched else None,
            "matched_iou": best_iou if matched else None,
        })
    true_positives = len(matched_ious)
    false_positives = len(evaluated) - true_positives
    false_negatives = len(unmatched)
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    metrics = {
        "metric_scope": "single pinned COCO val2017 image; not dataset mAP",
        "matching": "score-ranked greedy one-to-one matching by identical COCO category id",
        "iou_threshold": iou_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": true_positives / precision_denominator if precision_denominator else 0.0,
        "recall": true_positives / recall_denominator if recall_denominator else 0.0,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else None,
        "ground_truth_objects": len(ground_truth),
        "evaluated_predictions": len(evaluated),
    }
    return metrics, evaluated


def _runtime_environment(torch: Any, resolved_device: str) -> dict[str, Any]:
    packages = {}
    for distribution in ("qortex", "torch", "torchvision", "numpy", "pillow", "matplotlib"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda": {
            "available": bool(torch.cuda.is_available()),
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device())
            if resolved_device == "cuda" and torch.cuda.is_available() else None,
        },
        "process_executable": str(Path(sys.executable).resolve()),
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_public_detection_validation(
    *,
    image_id: int = DEFAULT_IMAGE_ID,
    device: str = "auto",
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    dataset_root: Path | str | None = None,
    run_root: Path | str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    execution_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run pinned pretrained Torchvision weights against a real COCO validation image."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be auto, cpu, or cuda; got {device!r}")
    if not 0.0 < score_threshold < 1.0:
        raise ValueError("score_threshold must be in (0, 1)")
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")

    image_path, image_record, raw_truth, categories, dataset = _load_coco_sample(
        image_id,
        Path(dataset_root) if dataset_root else _default_dataset_root(),
        on_progress,
    )
    category_names = {int(item["id"]): str(item["name"]) for item in categories}
    ground_truth = [{
        "annotation_id": int(item["id"]),
        "category_id": int(item["category_id"]),
        "class_name": category_names[int(item["category_id"])],
        "bbox_xyxy": _xywh_to_xyxy(item["bbox"]),
        "area": float(item["area"]),
    } for item in raw_truth]

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Public detection validation requires torch, torchvision, and Pillow") from exc

    adapter = TorchvisionAdapter(ModelSpec(
        provider="torchvision",
        id=MODEL_NAME,
        task="detection",
        extra={"pretrained": True, "weights": MODEL_WEIGHTS},
    ))
    with _INFERENCE_LOCK:
        adapter.load(RuntimeSpec(device=device))
        resolved_device = adapter._device
        weights = adapter.weights
        preprocess_started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            input_tensor = weights.transforms()(image)
            image_array = np.asarray(image)
        preprocess_seconds = time.perf_counter() - preprocess_started
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
            "measurement": "CUDA counters reset immediately before synchronized inference",
        } if resolved_device == "cuda" else {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "measurement": "Per-run CPU peak memory is not measured by this execution path.",
        }
        environment = _runtime_environment(torch, resolved_device)
        weights_url = weights.url
        weights_name = weights.name
        weights_metrics = weights.meta.get("_metrics", {})
        adapter.unload()
    if on_progress:
        on_progress(4, 7)

    predictions = [{
        "bbox_xyxy": [float(value) for value in box],
        "score": float(score),
        "category_id": int(label),
        "class_name": class_name,
    } for box, score, label, class_name in zip(
        output.metadata["boxes"],
        output.metadata["scores"],
        output.metadata["labels"],
        output.metadata["class_names"],
    ) if score >= score_threshold]
    metrics, evaluated_predictions = evaluate_single_image(
        predictions, ground_truth, iou_threshold=iou_threshold,
    )

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(weights_url).path).name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Torchvision did not persist downloaded weights at {checkpoint}")
    checkpoint_sha256 = _sha256(checkpoint)
    ModelCache().record(CacheEntry(
        model_id=MODEL_ID,
        provider="torchvision",
        local_path=str(checkpoint.resolve()),
        size_bytes=checkpoint.stat().st_size,
        sha256=checkpoint_sha256,
        downloaded_at=datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat(),
        source_url=weights_url,
    ))

    run_id = f"det-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    runs = Path(run_root) if run_root else _default_run_root()
    final_run_dir = runs / run_id
    run_dir = runs / f".{run_id}.tmp"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(image_path, run_dir / image_path.name)
    _write_json(run_dir / "ground-truth.json", ground_truth)
    showcase = render_detection_showcase(DetectionShowcaseInput(
        image=image_array,
        detections=[Detection(
            bbox=tuple(item["bbox_xyxy"]),
            class_name=item["class_name"],
            confidence=item["score"],
        ) for item in evaluated_predictions],
        output_dir=run_dir,
        case_id=f"COCO val2017 image {image_id}",
        model_id=f"{MODEL_ID}:{weights_name}",
        source_id=dataset["source_url"],
        threshold=score_threshold,
        metadata={"iou_threshold": iou_threshold, "metrics": metrics},
    ))
    if on_progress:
        on_progress(6, 7)

    result = {
        "run_id": run_id,
        "status": "completed",
        "research_use_only": True,
        "execution_profile": execution_profile,
        "model": {
            "id": MODEL_ID,
            "weights": weights_name,
            "weights_url": weights_url,
            "checkpoint_sha256": checkpoint_sha256,
            "license": {
                "name": "BSD-3-Clause",
                "url": "https://github.com/pytorch/vision/blob/main/LICENSE",
                "evidence_status": "inferred",
                "scope": "Torchvision source code; the weights metadata does not state a separate weights license.",
            },
            "source_url": MODEL_SOURCE_URL,
            "published_metrics": weights_metrics,
        },
        "dataset": dataset,
        "input": {
            "width": int(image_record["width"]),
            "height": int(image_record["height"]),
            "file_name": image_record["file_name"],
            "sha256": DEFAULT_IMAGE_SHA256,
            "ground_truth_objects": len(ground_truth),
        },
        "runtime": {
            "device": resolved_device,
            "precision": "float32",
            "preprocessing": repr(weights.transforms()),
            "preprocess_seconds": preprocess_seconds,
            "inference_seconds": inference_seconds,
            "score_threshold": score_threshold,
            "peak_memory": peak_memory,
        },
        "reproducibility": {
            "seed": None,
            "seed_evidence": (
                "No seed is configured because this is pretrained evaluation-mode inference without training. "
                "The runtime does not claim bitwise determinism across devices or library builds."
            ),
            "environment": environment,
        },
        "metrics": metrics,
        "predictions": evaluated_predictions,
        "artifacts": {
            "input": image_path.name,
            "board": showcase.board.name,
            "detections": showcase.detections_json.name,
            "ground_truth": "ground-truth.json",
            "showcase_manifest": showcase.manifest.name,
            "provenance": "result.json",
        },
    }
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
        "Every binary or source-data artifact is hashed after writing. result.json is excluded to avoid a recursive self-hash."
    )
    _write_json(run_dir / "result.json", result)
    run_dir.replace(final_run_dir)
    if on_progress:
        on_progress(7, 7)
    return result


def load_public_detection_run(run_id: str, *, run_root: Path | str | None = None) -> dict[str, Any]:
    root = (Path(run_root) if run_root else _default_run_root()).resolve()
    result_path = (root / run_id / "result.json").resolve()
    if root not in result_path.parents or not result_path.is_file():
        raise FileNotFoundError(f"No public detection validation run {run_id!r}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run result {result_path} is not a JSON object")
    return payload


def public_detection_artifact_path(run_id: str, artifact: str) -> Path:
    result = load_public_detection_run(run_id)
    filename = result.get("artifacts", {}).get(artifact)
    if not isinstance(filename, str):
        raise KeyError(f"Run {run_id!r} has no artifact {artifact!r}")
    root = _default_run_root().resolve()
    path = (root / run_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Artifact {artifact!r} is missing for run {run_id!r}")
    return path


__all__ = [
    "DEFAULT_IMAGE_ID",
    "evaluate_single_image",
    "load_public_detection_run",
    "public_detection_artifact_path",
    "run_public_detection_validation",
]
