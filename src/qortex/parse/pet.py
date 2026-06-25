"""PET loader — NiBabel with frame timing, SUV normalization, and decay correction detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from qortex.core.entities import FileRecord, ImageRecord, SampleRecord
from qortex.core.exceptions import LoadError
from qortex.parse._mne_utils import load_json_sidecar
from qortex.parse.mri import _require_nibabel

log = logging.getLogger(__name__)

_PET_EXTENSIONS = frozenset({".nii", ".nii.gz"})


class PETLoader:
    modality = "pet"
    supported_extensions = _PET_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype == "pet"
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
            shape = img.shape
            n_frames = int(shape[3]) if len(shape) > 3 else 1

            sidecar = load_json_sidecar(local_path)
            frame_starts = sidecar.get("FrameTimesStart")   # list of floats [s]
            frame_durations = sidecar.get("FrameDuration")  # list of floats [s] or scalar

            total_scan_time = None
            if frame_starts and frame_durations:
                try:
                    last_start = max(frame_starts)
                    last_dur = (
                        frame_durations[-1] if isinstance(frame_durations, list)
                        else frame_durations
                    )
                    total_scan_time = round(float(last_start) + float(last_dur), 3)
                except Exception:
                    pass

            return {
                "shape": list(shape),
                "n_frames": n_frames,
                "voxel_size_mm": [round(float(v), 4) for v in zooms[:3]],
                "total_scan_time_s": total_scan_time,
                "frame_times_start": frame_starts,
                "frame_durations": frame_durations,
                "tracer_name": sidecar.get("TracerName"),
                "tracer_radionuclide": sidecar.get("TracerRadionuclide"),
                "decay_correction": sidecar.get("DecayCorrectionFactor"),
                "reconstruction_method": sidecar.get("ReconMethodName"),
                "scanner_manufacturer": sidecar.get("Manufacturer"),
                "body_weight_kg": sidecar.get("BodyWeight"),
                "injected_activity_bq": sidecar.get("InjectedRadioactivity"),
                "suv_normalization_possible": all(
                    sidecar.get(k) is not None
                    for k in ("BodyWeight", "InjectedRadioactivity")
                ),
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect PET {local_path}: {exc}"
            ) from exc

    # ── load / lazy_load ──────────────────────────────────────────────────

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Load PET fully into memory.

        Parameters
        ----------
        canonical : bool, default True
            Reorient to RAS canonical orientation.
        suv : bool, default False
            Convert raw counts to SUVbw (Standardised Uptake Value) using
            body weight from the JSON sidecar.  Raises if metadata missing.
        """
        nib = _require_nibabel()
        canonical = kwargs.pop("canonical", True)
        compute_suv = kwargs.pop("suv", False)
        dtype = kwargs.pop("dtype", np.float32)

        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
            data = img.get_fdata(dtype=dtype)
            img = nib.Nifti1Image(data, img.affine, img.header)
        except Exception as exc:
            raise LoadError(
                f"Cannot load PET {local_path}: {exc}"
            ) from exc

        record = self._to_record(file, img, local_path)

        if compute_suv:
            record = _apply_suv(record)

        return record

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Proxy image — frame data read on demand."""
        nib = _require_nibabel()
        canonical = kwargs.pop("canonical", True)
        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
        except Exception as exc:
            raise LoadError(
                f"Cannot lazy-load PET {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def _to_record(self, file: FileRecord, img, local_path: Path) -> ImageRecord:
        nib = _require_nibabel()
        hdr = img.header
        zooms = hdr.get_zooms()
        shape = img.shape
        n_vols = int(shape[3]) if len(shape) > 3 else None

        sidecar = load_json_sidecar(local_path)
        frame_starts = sidecar.get("FrameTimesStart")
        frame_durations = sidecar.get("FrameDuration")

        # Normalise frame_durations to list
        if isinstance(frame_durations, (int, float)):
            frame_durations = [float(frame_durations)] * (n_vols or 1)

        try:
            ornt = nib.orientations.aff2axcodes(img.affine)
            orientation = "".join(ornt)
        except Exception:
            orientation = "unknown"

        body_weight_kg = sidecar.get("BodyWeight")
        injected_bq = sidecar.get("InjectedRadioactivity")

        return ImageRecord(
            file=file,
            img=img,
            shape=tuple(shape),
            voxel_size=tuple(round(float(v), 4) for v in zooms[:3]),
            affine=img.affine,
            n_volumes=n_vols,
            metadata={
                "orientation": orientation,
                "tracer_name": sidecar.get("TracerName"),
                "tracer_radionuclide": sidecar.get("TracerRadionuclide"),
                "frame_times_start": frame_starts,    # list[float] in seconds
                "frame_durations": frame_durations,   # list[float] in seconds
                "decay_correction_factor": sidecar.get("DecayCorrectionFactor"),
                "reconstruction_method": sidecar.get("ReconMethodName"),
                "body_weight_kg": body_weight_kg,
                "injected_radioactivity_bq": injected_bq,
                "suv_scale_factor": _suv_scale(body_weight_kg, injected_bq),
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "manufacturer": sidecar.get("Manufacturer"),
                "units": sidecar.get("Units", "Bq/mL"),
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: ImageRecord, dtype=None, **kwargs) -> np.ndarray:
        """Return 4D array (x, y, z, n_frames) or 3D (x, y, z) as float32."""
        return record.img.get_fdata(dtype=dtype or np.float32)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: ImageRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord per PET frame with timing and tracer metadata.

        Frame onset and duration are taken from FrameTimesStart / FrameDuration
        in the JSON sidecar (BIDS PET BEP009), not from the NIfTI header.
        """
        data = self.to_numpy(record, dtype=np.float32)
        ents = record.file.entities
        frame_starts = record.metadata.get("frame_times_start") or []
        frame_durations = record.metadata.get("frame_durations") or []
        suv_factor = record.metadata.get("suv_scale_factor")

        if data.ndim == 4:
            n_frames = data.shape[3]
            for i in range(n_frames):
                frame = data[:, :, :, i]

                onset = float(frame_starts[i]) if i < len(frame_starts) else None
                dur = float(frame_durations[i]) if i < len(frame_durations) else None

                yield SampleRecord(
                    data=frame,
                    modality=self.modality,
                    subject=ents.subject,
                    session=ents.session,
                    run=ents.run,
                    onset=onset,
                    duration=dur,
                    provenance={
                        "source": record.file.path,
                        "frame_index": i,
                        "n_frames": n_frames,
                        "frame_onset_s": onset,
                        "frame_duration_s": dur,
                        "tracer_name": record.metadata.get("tracer_name"),
                        "tracer_radionuclide": record.metadata.get("tracer_radionuclide"),
                        "units": record.metadata.get("units"),
                        "suv_scale_factor": suv_factor,
                        "voxel_size_mm": list(record.voxel_size),
                        "decay_correction_factor": record.metadata.get("decay_correction_factor"),
                    },
                )
        else:
            yield SampleRecord(
                data=data,
                modality=self.modality,
                subject=ents.subject,
                session=ents.session,
                run=ents.run,
                onset=float(frame_starts[0]) if frame_starts else None,
                duration=float(frame_durations[0]) if frame_durations else None,
                provenance={
                    "source": record.file.path,
                    "tracer_name": record.metadata.get("tracer_name"),
                    "units": record.metadata.get("units"),
                    "voxel_size_mm": list(record.voxel_size),
                },
            )


# ── SUV helpers ───────────────────────────────────────────────────────────────

def _suv_scale(body_weight_kg, injected_bq) -> float | None:
    """Compute SUVbw scale factor = body_weight_kg * 1000 / injected_bq.

    Multiply raw PET (Bq/mL) by this factor to get SUVbw (g/mL).
    Returns None if required metadata is absent.
    """
    if body_weight_kg is None or injected_bq is None:
        return None
    try:
        return float(body_weight_kg) * 1000.0 / float(injected_bq)
    except (TypeError, ZeroDivisionError):
        return None


def _apply_suv(record: ImageRecord) -> ImageRecord:
    """Multiply the image data by the SUVbw scale factor in-place."""
    factor = record.metadata.get("suv_scale_factor")
    if factor is None:
        raise ValueError(
            "Cannot compute SUV: BodyWeight and/or InjectedRadioactivity missing from sidecar."
        )
    import nibabel as nib
    data = record.img.get_fdata(dtype=np.float32) * float(factor)
    suv_img = nib.Nifti1Image(data, record.img.affine, record.img.header)
    record.img = suv_img
    record.metadata = dict(record.metadata)
    record.metadata["units"] = "SUVbw"
    return record
