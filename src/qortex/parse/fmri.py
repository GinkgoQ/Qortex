"""fMRI loader — NiBabel with TR from JSON sidecar, confound support, and brain masking."""

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

_FMRI_EXTENSIONS = frozenset({".nii", ".nii.gz"})

# Suffixes that indicate a 4D BOLD-family NIfTI acquisition.
# "events" and "physio" are .tsv files handled by BehaviorLoader — not here.
_FMRI_SUFFIXES = frozenset({"bold", "cbv", "phase", "sbref"})


class FMRILoader:
    modality = "fmri"
    supported_extensions = _FMRI_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype in {"func", "perf"}
            and file.extension in self.supported_extensions
            and file.suffix in _FMRI_SUFFIXES
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
            n_vols = int(shape[3]) if len(shape) > 3 else 1

            # TR must come from JSON sidecar — NIfTI header is unreliable for TR
            sidecar = load_json_sidecar(local_path)
            tr = sidecar.get("RepetitionTime")  # seconds
            if tr is None and len(zooms) > 3 and float(zooms[3]) > 0:
                tr = float(zooms[3])  # fallback: header pixdim[4]

            return {
                "shape": list(shape),
                "n_volumes": n_vols,
                "tr_s": tr,
                "voxel_size_mm": [round(float(v), 4) for v in zooms[:3]],
                "total_acquisition_time_s": round(n_vols * tr, 3) if tr else None,
                "task_name": sidecar.get("TaskName"),
                "slice_timing_available": "SliceTiming" in sidecar,
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "manufacturer": sidecar.get("Manufacturer"),
                "echo_time": sidecar.get("EchoTime"),
                "flip_angle": sidecar.get("FlipAngle"),
                "phase_encoding_direction": sidecar.get("PhaseEncodingDirection"),
                "confounds_available": _find_confound_file(local_path) is not None,
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect fMRI {local_path}: {exc}"
            ) from exc

    # ── load / lazy_load ──────────────────────────────────────────────────

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Load 4D fMRI fully into memory.

        Parameters
        ----------
        canonical : bool, default True
            Reorient to RAS canonical orientation.
        dtype : np.dtype, default np.float32
            Cast data array to this dtype on load.
        """
        nib = _require_nibabel()
        canonical = kwargs.pop("canonical", True)
        dtype = kwargs.pop("dtype", np.float32)
        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
            data = img.get_fdata(dtype=dtype)
            img = nib.Nifti1Image(data, img.affine, img.header)
        except Exception as exc:
            raise LoadError(
                f"Cannot load fMRI {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Return a proxy NiBabel image — volumes are read on demand via `dataobj`."""
        nib = _require_nibabel()
        canonical = kwargs.pop("canonical", True)
        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
        except Exception as exc:
            raise LoadError(
                f"Cannot lazy-load fMRI {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def _to_record(self, file: FileRecord, img, local_path: Path) -> ImageRecord:
        nib = _require_nibabel()
        hdr = img.header
        zooms = hdr.get_zooms()
        shape = img.shape
        n_vols = int(shape[3]) if len(shape) > 3 else None

        sidecar = load_json_sidecar(local_path)
        tr = sidecar.get("RepetitionTime")
        if tr is None and len(zooms) > 3 and float(zooms[3]) > 0:
            tr = float(zooms[3])

        try:
            ornt = nib.orientations.aff2axcodes(img.affine)
            orientation = "".join(ornt)
        except Exception:
            orientation = "unknown"

        confound_file = _find_confound_file(local_path)

        return ImageRecord(
            file=file,
            img=img,
            shape=tuple(shape),
            voxel_size=tuple(round(float(v), 4) for v in zooms[:3]),
            affine=img.affine,
            tr=tr,
            n_volumes=n_vols,
            metadata={
                "orientation": orientation,
                "task_name": sidecar.get("TaskName"),
                "slice_timing": sidecar.get("SliceTiming"),
                "phase_encoding_direction": sidecar.get("PhaseEncodingDirection"),
                "echo_time": sidecar.get("EchoTime"),
                "flip_angle": sidecar.get("FlipAngle"),
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "confound_file": str(confound_file) if confound_file else None,
                "total_acquisition_time_s": round(n_vols * tr, 3) if (n_vols and tr) else None,
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: ImageRecord, dtype=None, **kwargs) -> np.ndarray:
        """Return the 4D array (x, y, z, t) or 3D (x, y, z) for sbref."""
        return record.img.get_fdata(dtype=dtype or np.float32)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: ImageRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord per fMRI volume (x, y, z) with TR timing.

        For sbref / single-volume files, yields one sample for the whole image.
        Confound metadata (if available) is attached to provenance but not loaded
        here — use load_confounds() separately for denoising.
        """
        data = self.to_numpy(record, dtype=np.float32)
        ents = record.file.entities
        tr = record.tr or 0.0
        confound_file = record.metadata.get("confound_file")

        if data.ndim == 4:
            n_vols = data.shape[3]
            for vol_idx in range(n_vols):
                vol = data[:, :, :, vol_idx]
                yield SampleRecord(
                    data=vol,
                    modality=self.modality,
                    subject=ents.subject,
                    session=ents.session,
                    task=ents.task,
                    run=ents.run,
                    onset=round(vol_idx * tr, 6),
                    duration=tr,
                    provenance={
                        "source": record.file.path,
                        "volume_index": vol_idx,
                        "n_volumes": n_vols,
                        "tr_s": tr,
                        "shape_xyz": list(vol.shape),
                        "voxel_size_mm": list(record.voxel_size),
                        "confound_file": confound_file,
                        "task_name": record.metadata.get("task_name"),
                    },
                )
        else:
            yield SampleRecord(
                data=data,
                modality=self.modality,
                subject=ents.subject,
                session=ents.session,
                task=ents.task,
                run=ents.run,
                onset=0.0,
                provenance={
                    "source": record.file.path,
                    "shape": list(data.shape),
                    "voxel_size_mm": list(record.voxel_size),
                    "task_name": record.metadata.get("task_name"),
                },
            )

    # ── Confound helpers ──────────────────────────────────────────────────

    def load_confounds(self, record: ImageRecord):
        """Load the fMRIPrep/MRIQC confound timeseries if available.

        Returns a Polars DataFrame (n_volumes × n_confounds) or None.
        """
        confound_path = record.metadata.get("confound_file")
        if confound_path is None:
            return None
        path = Path(confound_path)
        if not path.exists():
            return None
        try:
            import polars as pl
            return pl.read_csv(str(path), separator="\t", null_values=["n/a", "N/A", ""])
        except Exception as exc:
            log.warning("Could not load confounds from %s: %s", confound_path, exc)
            return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _find_confound_file(local_path: Path) -> Path | None:
    """Locate the fMRIPrep confound timeseries TSV for a BOLD file.

    Looks for *_desc-confounds_timeseries.tsv (fMRIPrep v21+) and the legacy
    *_confounds.tsv pattern in the same directory or derivatives.
    """
    stem = local_path.name
    # Strip compound extension
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    directory = local_path.parent
    candidates = [
        directory / f"{stem}_desc-confounds_timeseries.tsv",
        directory / f"{stem}_confounds.tsv",
    ]
    # Also check derivatives/fmriprep relative to the BIDS root
    for p in candidates:
        if p.exists():
            return p

    # Walk up to find derivatives/ sibling
    try:
        from qortex.parse._mne_utils import find_bids_root
        bids_root = find_bids_root(local_path)
        if bids_root:
            # fMRIPrep derivatives pattern
            ents_part = local_path.relative_to(bids_root)
            deriv_candidate = bids_root / "derivatives" / "fmriprep" / ents_part.parent / f"{stem}_desc-confounds_timeseries.tsv"
            if deriv_candidate.exists():
                return deriv_candidate
    except Exception:
        pass

    return None
