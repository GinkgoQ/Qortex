"""Anatomical MRI loader — NiBabel with RAS canonicalization and header validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qortex.core.entities import FileRecord, ImageRecord, SampleRecord
from qortex.core.exceptions import LoadError
from qortex.parse._mne_utils import load_json_sidecar

log = logging.getLogger(__name__)

_MRI_EXTENSIONS = frozenset({".nii", ".nii.gz", ".mgz", ".mgh"})


def _require_nibabel():
    try:
        import nibabel
        return nibabel
    except ImportError:
        raise ImportError(
            "MRI loading requires NiBabel: pip install 'qortex[mri]'"
        )


class MRILoader:
    modality = "mri"
    supported_extensions = _MRI_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype == "anat"
            and file.extension in self.supported_extensions
            and not file.is_dir
        )

    # ── inspect ───────────────────────────────────────────────────────────

    def inspect(self, file: FileRecord, local_path: Path) -> dict[str, Any]:
        nib = _require_nibabel()
        try:
            img = nib.load(str(local_path))
            hdr = img.header
            zooms = hdr.get_zooms()
            affine = img.affine
            affine_det = float(abs(np.linalg.det(affine)))

            # Orientation string (e.g. RAS, LAS)
            try:
                ornt = nib.orientations.aff2axcodes(affine)
                orientation = "".join(ornt)
            except Exception:
                orientation = "unknown"

            sidecar = load_json_sidecar(local_path)
            return {
                "shape": list(img.shape),
                "voxel_size_mm": [round(float(v), 4) for v in zooms[:3]],
                "orientation": orientation,
                "affine_det": round(affine_det, 6),
                "dtype": str(img.get_data_dtype()),
                "size_bytes": local_path.stat().st_size,
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "manufacturer": sidecar.get("Manufacturer"),
                "sequence_name": sidecar.get("SequenceName"),
                "flip_angle": sidecar.get("FlipAngle"),
                "echo_time": sidecar.get("EchoTime"),
                "repetition_time": sidecar.get("RepetitionTime"),
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect MRI {local_path}: {exc}"
            ) from exc

    # ── load / lazy_load ──────────────────────────────────────────────────

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Load MRI fully into memory, optionally reorienting to RAS."""
        nib = _require_nibabel()
        dtype = kwargs.pop("dtype", None)
        canonical = kwargs.pop("canonical", True)
        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
            data = img.get_fdata(dtype=dtype or np.float32)
            img = nib.Nifti1Image(data, img.affine, img.header)
        except Exception as exc:
            raise LoadError(
                f"Cannot load MRI {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Return a proxy NiBabel image — data read on first array access."""
        nib = _require_nibabel()
        canonical = kwargs.pop("canonical", True)
        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
        except Exception as exc:
            raise LoadError(
                f"Cannot lazy-load MRI {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def _to_record(self, file: FileRecord, img, local_path: Path) -> ImageRecord:
        nib = _require_nibabel()
        hdr = img.header
        zooms = hdr.get_zooms()
        try:
            ornt = nib.orientations.aff2axcodes(img.affine)
            orientation = "".join(ornt)
        except Exception:
            orientation = "unknown"

        sidecar = load_json_sidecar(local_path)
        return ImageRecord(
            file=file,
            img=img,
            shape=tuple(img.shape),
            voxel_size=tuple(round(float(v), 4) for v in zooms[:3]),
            affine=img.affine,
            metadata={
                "orientation": orientation,
                "affine_det": float(abs(np.linalg.det(img.affine))),
                "dtype": str(img.get_data_dtype()),
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "manufacturer": sidecar.get("Manufacturer"),
                "echo_time": sidecar.get("EchoTime"),
                "repetition_time": sidecar.get("RepetitionTime"),
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: ImageRecord, dtype=None, **kwargs) -> np.ndarray:
        return record.img.get_fdata(dtype=dtype or np.float32)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: ImageRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord containing the full 3D volume (x, y, z) as float32."""
        data = self.to_numpy(record, dtype=np.float32)
        ents = record.file.entities
        yield SampleRecord(
            data=data,
            modality=self.modality,
            subject=ents.subject,
            session=ents.session,
            task=ents.task,
            run=ents.run,
            provenance={
                "source": record.file.path,
                "shape": list(record.shape),
                "voxel_size_mm": list(record.voxel_size),
                "orientation": record.metadata.get("orientation"),
                "magnetic_field_strength": record.metadata.get("magnetic_field_strength"),
            },
        )
