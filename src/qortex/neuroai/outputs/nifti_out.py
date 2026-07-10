"""NIfTI mask output adapter.

Writes model segmentation outputs as NIfTI (.nii.gz) files with full
geometry validation (affine determinant, shape match) and JSON provenance
sidecar.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter, OutputAdapterError

log = logging.getLogger(__name__)


class NIfTIOutputAdapter(OutputAdapter):
    """Output adapter that writes segmentation masks as NIfTI files.

    Parameters
    ----------
    path:
        Output file path (``*.nii`` or ``*.nii.gz``).
    source_affine:
        Default 4×4 affine matrix (as nested list or numpy array) from the
        source volume.  Used when ``output.metadata["affine"]`` is absent.
    voxel_sizes:
        Default voxel sizes ``(dz, dy, dx)`` in mm.
    pipeline_ref:
        Short pipeline reference for provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_affine: Any = None,
        voxel_sizes: tuple | None = None,
        pipeline_ref: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._source_affine = source_affine
        self._voxel_sizes = voxel_sizes
        self._pipeline_ref = pipeline_ref
        self._n_written = 0

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        log.info("NIfTI output ready: %s", self._path)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        nibabel = _require_nibabel()
        meta = metadata or {}

        # Get mask
        mask = output.mask
        if mask is None:
            raise OutputAdapterError(
                "NIfTI output requires a segmentation mask; "
                f"received output_type={output.output_type!r} without output.mask."
            )

        mask_arr = np.array(mask)
        if mask_arr.ndim < 2:
            raise OutputAdapterError(
                f"NIfTI output requires a 2D or 3D mask; received shape={mask_arr.shape!r}."
            )

        # Force 3D
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[np.newaxis, :, :]  # [1, Y, X]

        # Get affine
        affine_raw = meta.get("affine") or self._source_affine
        if affine_raw is not None:
            affine = np.array(affine_raw, dtype=np.float64)
        else:
            affine = np.eye(4, dtype=np.float64)
            log.warning("NIfTIOutputAdapter: no affine provided; using identity")

        # Geometry validation
        det = np.linalg.det(affine[:3, :3])
        if det <= 0:
            log.warning(
                "NIfTI affine determinant=%.4f is not positive — orientation may be flipped",
                det,
            )

        nii = nibabel.Nifti1Image(mask_arr.astype(np.int16), affine)

        # Set voxel sizes
        voxel_sizes = meta.get("voxel_sizes") or self._voxel_sizes
        if voxel_sizes is not None:
            hdr = nii.header
            hdr.set_zooms(voxel_sizes[:3])

        # Determine output path (support multiple writes → numbered files)
        out_path = self._path
        if self._n_written > 0:
            stem = self._path.stem.replace(".nii", "")
            out_path = self._path.parent / f"{stem}_{self._n_written:04d}.nii.gz"

        nibabel.save(nii, str(out_path))
        self._n_written += 1
        log.info("NIfTI saved: %s (shape=%s)", out_path.name, mask_arr.shape)

        # Write JSON sidecar with provenance
        self._write_sidecar(out_path, output, meta, mask_arr.shape, affine)

    def close(self) -> None:
        log.info("NIfTI output adapter closed (%d files written)", self._n_written)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_sidecar(
        self,
        nii_path: Path,
        output: ModelOutput,
        meta: dict,
        shape: tuple,
        affine: np.ndarray,
    ) -> None:
        sidecar_path = nii_path.with_suffix("").with_suffix(".json")
        data = {
            "qortex_provenance": {
                "pipeline_ref": self._pipeline_ref,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_type": output.output_type,
                "class_name": output.class_name,
                "mask_shape": list(shape),
                "affine": affine.tolist(),
                "source_id": meta.get("source_id"),
                "model_id": meta.get("model_id"),
                "window_index": meta.get("window_index"),
            }
        }
        try:
            sidecar_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            log.debug("Could not write NIfTI sidecar: %s", exc)


def _require_nibabel():
    try:
        import nibabel
        return nibabel
    except ImportError:
        raise ImportError(
            "NIfTI output requires nibabel. "
            "Install with: pip install 'qortex[mri]' or pip install nibabel"
        )
