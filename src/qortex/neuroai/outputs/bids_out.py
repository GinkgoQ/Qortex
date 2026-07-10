"""BIDS derivative output adapter.

Writes model outputs into a BIDS-compatible derivatives directory with
proper dataset_description.json, BIDS entity-based file naming, and
JSON sidecars for every prediction.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapter

log = logging.getLogger(__name__)

_BIDS_VERSION = "1.8.0"
_TASK_SUFFIX_MAP = {
    "classification": "pred",
    "segmentation": "mask",
    "detection": "det",
    "embedding": "emb",
    "regression": "pred",
    "unknown": "pred",
}


class BIDSDerivativeOutputAdapter(OutputAdapter):
    """Output adapter that writes model outputs as BIDS derivatives.

    Parameters
    ----------
    output_dir:
        Root derivatives directory (e.g., ``derivatives/``).
    pipeline_name:
        Name of the pipeline (used as derivative dataset name).
    pipeline_ref:
        Short pipeline reference hash.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        pipeline_name: str = "qortex",
        pipeline_ref: str | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._pipeline_name = pipeline_name
        self._pipeline_ref = pipeline_ref
        self._base = self._output_dir / pipeline_name
        self._n_written = 0
        self._provenance_records: list[dict] = []

    @property
    def n_written(self) -> int:
        return self._n_written

    def open(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        self._write_dataset_description()
        log.info("BIDS derivative output ready: %s", self._base)

    def write(self, output: ModelOutput, metadata: dict[str, Any] | None = None) -> None:
        meta = metadata or {}
        entities = meta.get("bids_entities", {})

        subject = _normalise_bids_entity(entities.get("subject") or meta.get("subject"), "sub", "unknown")
        session = _normalise_bids_entity(entities.get("session") or meta.get("session"), "ses", None)
        task = entities.get("task") or meta.get("task")
        run = entities.get("run") or meta.get("run")

        # Build subject directory
        sub_dir = self._base / f"sub-{subject}"
        if session:
            sub_dir = sub_dir / f"ses-{session}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # Determine suffix
        suffix = _TASK_SUFFIX_MAP.get(output.output_type, "pred")

        # Build filename
        parts = [f"sub-{subject}"]
        if session:
            parts.append(f"ses-{session}")
        if task:
            parts.append(f"task-{task}")
        if run:
            parts.append(f"run-{run}")
        if self._n_written > 0:
            parts.append(f"idx-{self._n_written:04d}")
        filename = "_".join(parts) + f"_{suffix}"

        # Write prediction JSON
        pred_path = sub_dir / f"{filename}.json"
        pred_data = {
            "output_type": output.output_type,
            "class_name": output.class_name,
            "class_index": output.class_index,
            "probabilities": output.probabilities,
            "regression_value": output.regression_value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_ref": self._pipeline_ref,
            "source_id": meta.get("source_id"),
            "model_id": meta.get("model_id"),
            "bids_entities": entities,
        }
        pred_path.write_text(json.dumps(pred_data, indent=2), encoding="utf-8")

        # Write NIfTI mask for segmentation
        if output.output_type == "segmentation" and output.mask is not None:
            self._write_nifti_mask(sub_dir, filename, output, meta)

        self._provenance_records.append({
            "file": str(pred_path.relative_to(self._base)),
            "output_type": output.output_type,
            "subject": subject,
        })
        self._n_written += 1
        log.debug("BIDS derivative written: %s", pred_path.name)

    def close(self) -> None:
        self._write_provenance_summary()
        log.info("BIDS derivative adapter closed (%d files)", self._n_written)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_dataset_description(self) -> None:
        desc_path = self._base / "dataset_description.json"
        if desc_path.exists():
            return
        try:
            qortex_version = _get_qortex_version()
        except Exception:
            qortex_version = "unknown"
        data = {
            "Name": f"Qortex Derivatives — {self._pipeline_name}",
            "BIDSVersion": _BIDS_VERSION,
            "GeneratedBy": [
                {
                    "Name": "Qortex",
                    "Version": qortex_version,
                    "CodeURL": "https://github.com/GinkgoQ/Qortex",
                }
            ],
            "PipelineDescription": {
                "Name": self._pipeline_name,
                "Ref": self._pipeline_ref,
            },
        }
        desc_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _write_nifti_mask(
        self,
        out_dir: Path,
        filename: str,
        output: ModelOutput,
        meta: dict,
    ) -> None:
        try:
            import nibabel as nib
            import numpy as np
            mask = np.array(output.mask)
            if mask.ndim == 2:
                mask = mask[np.newaxis, :, :]
            affine = np.array(meta.get("affine", np.eye(4)))
            nii = nib.Nifti1Image(mask.astype(np.int16), affine)
            nii_path = out_dir / f"{filename}.nii.gz"
            nib.save(nii, str(nii_path))
        except ImportError:
            log.debug("nibabel not available; NIfTI mask not written")
        except Exception as exc:
            log.warning("Failed to write NIfTI mask: %s", exc)

    def _write_provenance_summary(self) -> None:
        prov_path = self._base / "provenance.json"
        data = {
            "pipeline_ref": self._pipeline_ref,
            "n_outputs": self._n_written,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "records": self._provenance_records,
        }
        prov_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _get_qortex_version() -> str:
    try:
        from importlib.metadata import version
        return version("qortex")
    except Exception:
        return "unknown"


def _normalise_bids_entity(value: Any, prefix: str, default: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip()
    marker = f"{prefix}-"
    if text.startswith(marker):
        return text[len(marker):]
    return text
