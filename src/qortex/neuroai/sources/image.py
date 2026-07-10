"""Image and video source adapter.

Reads image files (.png, .jpg, .tif, etc.) and video files (.mp4, .avi, .mov)
using PIL/Pillow for images and OpenCV for video.  All imports are deferred.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import numpy as np

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    QortexVolume,
    SourceProfile,
)
from qortex.neuroai.sources._base import SourceAdapter, QortexData
from qortex.neuroai.spec import SourceSpec, WindowSpec

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


class ImageVideoAdapter(SourceAdapter):
    """Source adapter for image files and video files.

    Parameters
    ----------
    spec:
        ``SourceSpec`` with ``type="image"`` or ``type="video"`` and
        ``path=...`` pointing to a file or directory of images.
    window_spec:
        For video: number of frames per window = ``duration_s * fps``.
    """

    def __init__(
        self,
        spec: SourceSpec,
        *,
        window_spec: WindowSpec | None = None,
    ) -> None:
        if not spec.path:
            raise ValueError("ImageVideoAdapter requires spec.path")
        self._path = Path(spec.path).expanduser().resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Image/video path not found: {self._path}")
        self._spec = spec
        self._window_spec = window_spec
        self._is_video = self._path.suffix.lower() in _VIDEO_EXTS
        self._is_dir = self._path.is_dir()

    # ── SourceAdapter interface ───────────────────────────────────────────────

    def probe(self) -> SourceProfile:
        if self._is_video:
            return self._probe_video()
        else:
            return self._probe_image()

    def read_batch(self) -> list[QortexData]:
        if self._is_video:
            return [self._load_video()]
        elif self._is_dir:
            return [self._load_image(f) for f in self._list_image_files()]
        else:
            return [self._load_image(self._path)]

    def stream(self) -> Iterator[QortexData]:
        if self._is_video:
            yield from self._stream_video()
        elif self._is_dir:
            for f in self._list_image_files():
                yield self._load_image(f)
        else:
            yield self._load_image(self._path)

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _probe_image(self) -> SourceProfile:
        PIL = _require_pil()
        target = self._path if not self._is_dir else next(iter(self._list_image_files()), None)
        if target is None:
            raise FileNotFoundError(f"No image files found in {self._path}")
        with PIL.Image.open(target) as img:
            w, h = img.size
            mode = img.mode
        n_channels = {"L": 1, "RGB": 3, "RGBA": 4}.get(mode, 3)
        file_list = self._list_image_files() if self._is_dir else [self._path]
        n_images = len(file_list)

        return SourceProfile(
            source_id=f"image:{self._path.name}",
            source_type="image",
            modality="image",
            n_channels=n_channels,
            sampling_rate_hz=None,
            spatial_shape=(n_images, h, w, n_channels),
            dtype="uint8",
            axis_convention=AxisConvention.spatial_zyx,
            path=str(self._path),
            extra={"n_images": n_images, "mode": mode},
            evidence={
                "spatial_shape": EvidenceStatus.confirmed,
                "n_channels": EvidenceStatus.confirmed,
            },
        )

    def _load_image(self, path: Path) -> QortexVolume:
        PIL = _require_pil()
        with PIL.Image.open(path) as img:
            arr = np.array(img, dtype=np.uint8)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]  # [H, W, 1]
        return QortexVolume(
            data=arr,
            shape=arr.shape,
            axes=["h", "w", "c"],
            dtype="uint8",
            units="pixel_intensity",
            affine=None,
            voxel_sizes_mm=None,
            coordinate_frame=None,
            source_provenance={"source_type": "image", "path": str(path)},
        )

    def _list_image_files(self) -> list[Path]:
        files = sorted(
            f for f in self._path.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        return files

    # ── Video helpers ─────────────────────────────────────────────────────────

    def _probe_video(self) -> SourceProfile:
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(self._path))
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        return SourceProfile(
            source_id=f"video:{self._path.name}",
            source_type="video",
            modality="video",
            n_channels=3,
            sampling_rate_hz=fps,
            spatial_shape=(n_frames, height, width, 3),
            dtype="uint8",
            axis_convention=AxisConvention.spatial_zyx,
            path=str(self._path),
            extra={"fps": fps, "n_frames": n_frames, "duration_s": n_frames / fps},
            evidence={
                "spatial_shape": EvidenceStatus.confirmed,
                "sampling_rate": EvidenceStatus.confirmed,
            },
        )

    def _load_video(self) -> QortexVolume:
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(self._path))
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        arr = np.stack(frames, axis=0) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)
        return QortexVolume(
            data=arr,
            shape=arr.shape,
            axes=["n", "h", "w", "c"],
            dtype="uint8",
            units="pixel_intensity",
            affine=None,
            voxel_sizes_mm=None,
            coordinate_frame=None,
            source_provenance={"source_type": "video", "path": str(self._path)},
        )

    def _stream_video(self) -> Iterator[QortexVolume]:
        cv2 = _require_cv2()
        fps = self._probe_video().sampling_rate_hz or 25.0
        win_frames = max(1, int((self._window_spec.duration_s if self._window_spec else 1.0) * fps))

        cap = cv2.VideoCapture(str(self._path))
        batch: list[np.ndarray] = []
        window_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(batch) >= win_frames:
                    arr = np.stack(batch, axis=0)
                    yield QortexVolume(
                        data=arr,
                        shape=arr.shape,
                        axes=["n", "h", "w", "c"],
                        dtype="uint8",
                        units="pixel_intensity",
                        affine=None,
                        voxel_sizes_mm=None,
                        coordinate_frame=None,
                        source_provenance={
                            "source_type": "video",
                            "path": str(self._path),
                            "window_index": window_idx,
                        },
                    )
                    window_idx += 1
                    batch = []
        finally:
            cap.release()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise ImportError(
            "Image support requires Pillow. "
            "Install with: pip install 'qortex[visual]' or pip install Pillow"
        )


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "Video support requires opencv-python. "
            "Install with: pip install 'qortex[visual]' or pip install opencv-python"
        )
