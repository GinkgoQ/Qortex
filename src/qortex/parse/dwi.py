"""DWI loader — NiBabel + bvals/bvecs + optional dipy gradient table."""

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

_DWI_EXTENSIONS = frozenset({".nii", ".nii.gz"})


class DWILoader:
    modality = "dwi"
    supported_extensions = _DWI_EXTENSIONS

    def can_load(self, file: FileRecord) -> bool:
        return (
            file.datatype == "dwi"
            and file.extension in self.supported_extensions
            and file.suffix in {"dwi", None}
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
            n_dirs = int(shape[3]) if len(shape) > 3 else 1

            bvals, bvecs = _load_gradient_table(local_path)
            sidecar = load_json_sidecar(local_path)

            b_values_unique = sorted(set(int(round(b / 50) * 50) for b in bvals)) if bvals is not None else []
            n_b0 = int(np.sum(bvals < 50)) if bvals is not None else None

            return {
                "shape": list(shape),
                "n_directions": n_dirs,
                "voxel_size_mm": [round(float(v), 4) for v in zooms[:3]],
                "b_values": b_values_unique,
                "n_b0_volumes": n_b0,
                "bvec_available": bvecs is not None,
                "bval_available": bvals is not None,
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "manufacturer": sidecar.get("Manufacturer"),
                "parallel_imaging_factor": sidecar.get("ParallelReductionFactorInPlane"),
                "echo_time": sidecar.get("EchoTime"),
                "total_readout_time": sidecar.get("TotalReadoutTime"),
            }
        except Exception as exc:
            raise LoadError(
                f"Cannot inspect DWI {local_path}: {exc}"
            ) from exc

    # ── load / lazy_load ──────────────────────────────────────────────────

    def load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Load DWI fully, including gradient table (bvals, bvecs)."""
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
                f"Cannot load DWI {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def lazy_load(self, file: FileRecord, local_path: Path, **kwargs) -> ImageRecord:
        """Proxy image — gradient table loaded eagerly (small), volume data deferred."""
        nib = _require_nibabel()
        canonical = kwargs.pop("canonical", True)
        try:
            img = nib.load(str(local_path))
            if canonical:
                img = nib.as_closest_canonical(img)
        except Exception as exc:
            raise LoadError(
                f"Cannot lazy-load DWI {local_path}: {exc}"
            ) from exc
        return self._to_record(file, img, local_path)

    def _to_record(self, file: FileRecord, img, local_path: Path) -> ImageRecord:
        nib = _require_nibabel()
        hdr = img.header
        zooms = hdr.get_zooms()
        shape = img.shape
        n_vols = int(shape[3]) if len(shape) > 3 else None

        bvals, bvecs = _load_gradient_table(local_path)
        sidecar = load_json_sidecar(local_path)

        try:
            ornt = nib.orientations.aff2axcodes(img.affine)
            orientation = "".join(ornt)
        except Exception:
            orientation = "unknown"

        # Build a lightweight gradient table if dipy is available
        gtab_meta = _build_gtab_meta(bvals, bvecs) if (bvals is not None and bvecs is not None) else {}

        return ImageRecord(
            file=file,
            img=img,
            shape=tuple(shape),
            voxel_size=tuple(round(float(v), 4) for v in zooms[:3]),
            affine=img.affine,
            n_volumes=n_vols,
            metadata={
                "bvals": bvals.tolist() if bvals is not None else None,
                "bvecs": bvecs.tolist() if bvecs is not None else None,
                "orientation": orientation,
                "n_b0_volumes": gtab_meta.get("n_b0"),
                "b0_threshold": gtab_meta.get("b0_threshold", 50),
                "b_values_unique": gtab_meta.get("b_values_unique"),
                "dipy_gtab": gtab_meta.get("gtab"),  # dipy GradientTable or None
                "magnetic_field_strength": sidecar.get("MagneticFieldStrength"),
                "manufacturer": sidecar.get("Manufacturer"),
                "total_readout_time": sidecar.get("TotalReadoutTime"),
                "phase_encoding_direction": sidecar.get("PhaseEncodingDirection"),
            },
        )

    # ── to_numpy ──────────────────────────────────────────────────────────

    def to_numpy(self, record: ImageRecord, dtype=None, **kwargs) -> np.ndarray:
        """Return 4D array (x, y, z, n_directions) as float32."""
        return record.img.get_fdata(dtype=dtype or np.float32)

    # ── to_sample_records ─────────────────────────────────────────────────

    def to_sample_records(self, record: ImageRecord, **kwargs) -> Iterator[SampleRecord]:
        """Yield one SampleRecord per DWI direction/volume.

        Each sample's provenance carries the b-value and b-vector for that
        direction, enabling downstream shell selection or b0 extraction.
        """
        data = self.to_numpy(record, dtype=np.float32)
        ents = record.file.entities
        bvals = record.metadata.get("bvals")
        bvecs = record.metadata.get("bvecs")

        if data.ndim == 4:
            n_vols = data.shape[3]
            for i in range(n_vols):
                vol = data[:, :, :, i]
                bval = float(bvals[i]) if bvals and i < len(bvals) else None
                bvec = list(bvecs[i]) if bvecs and i < len(bvecs) else None
                yield SampleRecord(
                    data=vol,
                    modality=self.modality,
                    subject=ents.subject,
                    session=ents.session,
                    run=ents.run,
                    provenance={
                        "source": record.file.path,
                        "direction_index": i,
                        "n_directions": n_vols,
                        "bval": bval,
                        "bvec": bvec,
                        "is_b0": bval is not None and bval < 50,
                        "voxel_size_mm": list(record.voxel_size),
                        "magnetic_field_strength": record.metadata.get("magnetic_field_strength"),
                    },
                )
        else:
            bval = float(bvals[0]) if bvals else None
            bvec = list(bvecs[0]) if bvecs else None
            yield SampleRecord(
                data=data,
                modality=self.modality,
                subject=ents.subject,
                session=ents.session,
                run=ents.run,
                provenance={
                    "source": record.file.path,
                    "bval": bval,
                    "bvec": bvec,
                    "voxel_size_mm": list(record.voxel_size),
                },
            )


# ── Gradient table helpers ────────────────────────────────────────────────────

def _load_gradient_table(
    nii_path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load bvals and bvecs from BIDS-adjacent .bval/.bvec files.

    BIDS mandates that for a file `sub-01_dwi.nii.gz`, the gradient files
    are `sub-01_dwi.bval` and `sub-01_dwi.bvec` in the same directory.
    """
    stem = nii_path.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    bval_path = nii_path.parent / f"{stem}.bval"
    bvec_path = nii_path.parent / f"{stem}.bvec"

    bvals: np.ndarray | None = None
    bvecs: np.ndarray | None = None

    if bval_path.exists():
        try:
            bvals = np.loadtxt(str(bval_path))
            if bvals.ndim == 0:
                bvals = bvals.reshape(1)
        except Exception as exc:
            log.warning("Could not load bvals from %s: %s", bval_path, exc)

    if bvec_path.exists():
        try:
            raw_bvecs = np.loadtxt(str(bvec_path))
            # BIDS bvec format: (3, n_directions) → transpose to (n_directions, 3)
            if raw_bvecs.ndim == 2 and raw_bvecs.shape[0] == 3:
                bvecs = raw_bvecs.T
            elif raw_bvecs.ndim == 2 and raw_bvecs.shape[1] == 3:
                bvecs = raw_bvecs
            elif raw_bvecs.ndim == 1 and bvals is not None:
                bvecs = raw_bvecs.reshape(len(bvals), 3)
        except Exception as exc:
            log.warning("Could not load bvecs from %s: %s", bvec_path, exc)

    return bvals, bvecs


def _build_gtab_meta(bvals: np.ndarray, bvecs: np.ndarray) -> dict:
    """Build gradient table metadata; use dipy if available."""
    b0_threshold = 50
    n_b0 = int(np.sum(bvals < b0_threshold))
    b_values_unique = sorted(set(int(round(b / 50) * 50) for b in bvals))

    meta: dict = {
        "n_b0": n_b0,
        "b0_threshold": b0_threshold,
        "b_values_unique": b_values_unique,
        "gtab": None,
    }

    try:
        from dipy.core.gradients import gradient_table
        gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)
        meta["gtab"] = gtab
    except ImportError:
        pass
    except Exception as exc:
        log.debug("dipy gradient_table construction failed: %s", exc)

    return meta
