"""MONAI PersistentDataset / CacheDataset JSON datalist exporter.

Generates the standard MONAI datalist JSON consumed by::

    from monai.data import DataLoader, Dataset, load_decathlon_datalist

    datalist = load_decathlon_datalist("dataset.json", data_list_key="training")
    ds = Dataset(data=datalist, transform=transforms)

Capabilities
------------
* Auto-discovers NIfTI files in a local BIDS tree by datatype + suffix.
* Supports **multi-channel** inputs: specify multiple suffixes
  (e.g. ``{"image": ["T1w", "T2w"]}``) and each sample gets
  ``{"image": ["path/T1w.nii.gz", "path/T2w.nii.gz"], ...}``.
* Label extraction from participants.tsv, sessions.tsv, or scans.tsv.
* Leakage-safe subject-level train / val / test splits (deterministic
  given the same seed, using sorted subject list + modular assignment).
* Writes a ``dataset.json`` in Medical Decathlon format plus a Qortex
  ``manifest.json`` sidecar with full provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_COMPOUND_EXTENSIONS = frozenset({
    ".nii.gz", ".tar.gz", ".nii", ".mgz", ".mgh",
    ".edf", ".fif", ".bdf", ".set", ".mef",
})


def _strip_ext(filename: str) -> str:
    for ext in sorted(_COMPOUND_EXTENSIONS, key=len, reverse=True):
        if filename.endswith(ext):
            return filename[: -len(ext)]
    base, _, _ = filename.rpartition(".")
    return base or filename


def _resolve_extension(suffix: str, extension: str | None) -> str:
    if extension:
        return extension if extension.startswith(".") else f".{extension}"
    return ".nii.gz"


@dataclass
class MONAIDataset:
    """In-memory representation of a MONAI datalist before serialisation."""

    dataset_name: str
    dataset_description: str
    modality: str
    label_classes: dict[str, int]
    training: list[dict[str, Any]] = field(default_factory=list)
    validation: list[dict[str, Any]] = field(default_factory=list)
    test: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_training(self) -> int:
        return len(self.training)

    @property
    def n_validation(self) -> int:
        return len(self.validation)

    @property
    def n_test(self) -> int:
        return len(self.test)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.dataset_name,
            "description": self.dataset_description,
            "modality": {"0": self.modality},
            "labels": {str(v): k for k, v in self.label_classes.items()},
            "training": self.training,
            "validation": self.validation,
            "test": self.test,
            "numTraining": self.n_training,
            "numValidation": self.n_validation,
            "numTest": self.n_test,
        }

    def save(self, path: Path, *, indent: int = 2) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return path


class MONAIExporter:
    """Export a locally-downloaded BIDS dataset as a MONAI datalist JSON.

    Parameters
    ----------
    bids_root:
        Root of the BIDS tree (contains ``sub-*`` folders).  This is the
        ``target_dir / dataset_id / snapshot`` path produced by
        ``Dataset.download()``.
    """

    def __init__(self, bids_root: Path) -> None:
        self.bids_root = Path(bids_root).expanduser().resolve()
        if not self.bids_root.is_dir():
            raise FileNotFoundError(f"BIDS root not found: {self.bids_root}")

    # ── Public API ────────────────────────────────────────────────────────

    def export(
        self,
        output_dir: Path,
        *,
        # Image specification
        datatype: str = "anat",
        suffix: str | list[str] = "T1w",
        extension: str = ".nii.gz",
        include_derivatives_from: str | None = None,
        # Label specification
        label_source: str = "participants",
        label_column: str | None = None,
        label_map: dict[str, int] | None = None,
        negative_label_value: int = 0,
        # Segmentation mask (optional)
        seg_suffix: str | None = None,
        seg_datatype: str | None = None,
        # Split
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        # Metadata
        dataset_name: str | None = None,
        dataset_description: str | None = None,
        modality_name: str = "MRI",
        # Output control
        use_absolute_paths: bool = True,
        indent: int = 2,
    ) -> Path:
        """Build and write the MONAI datalist JSON.

        Parameters
        ----------
        output_dir:
            Directory where ``dataset.json`` and ``manifest.json`` are written.
        datatype:
            BIDS datatype sub-folder (e.g. ``"anat"``, ``"func"``).
        suffix:
            BIDS suffix string (e.g. ``"T1w"``), or list of suffixes for
            multi-channel input (e.g. ``["T1w", "T2w"]``).
        extension:
            NIfTI extension to look for.
        include_derivatives_from:
            When set, look for the image in a derivative pipeline directory
            instead of the raw BIDS tree (e.g. ``"fmriprep"``).
        label_source:
            ``"participants"`` | ``"sessions"`` | ``"scans"`` | ``"constant"``.
        label_column:
            Column in the TSV to use as label.  None → label field omitted.
        label_map:
            Explicit string-to-int mapping.  When None, integer labels are
            assigned alphabetically.
        negative_label_value:
            Integer for rows where the label_column value is missing / NaN.
        seg_suffix:
            Optional segmentation mask suffix (e.g. ``"dseg"``).  When set,
            each sample gets a ``"label"`` field pointing to the mask path
            (in addition to / instead of the classification label).
        train_frac / val_frac:
            Subject-level split fractions.  test_frac = 1 - train - val.
        seed:
            Deterministic shuffle seed.
        use_absolute_paths:
            When True (default), paths in the JSON are absolute.  When False,
            paths are BIDS-relative strings (useful for portable datasets).

        Returns
        -------
        Path
            Path to the written ``dataset.json``.
        """
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        suffixes = [suffix] if isinstance(suffix, str) else list(suffix)
        multi_channel = len(suffixes) > 1

        # ── Discover subjects ─────────────────────────────────────────────
        all_subjects = sorted({
            p.name for p in self.bids_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        })
        if not all_subjects:
            raise FileNotFoundError(
                f"No sub-* directories found in {self.bids_root}"
            )
        log.info("Found %d subjects in %s", len(all_subjects), self.bids_root)

        # ── Load labels ───────────────────────────────────────────────────
        label_lookup: dict[str, Any] = {}
        label_classes: dict[str, int] = {}
        if label_column and label_source != "constant":
            label_lookup, label_classes = self._load_labels(
                label_source=label_source,
                label_column=label_column,
                label_map=label_map,
            )

        # ── Build samples ─────────────────────────────────────────────────
        samples: list[dict[str, Any]] = []
        missing_images: list[str] = []

        for subject in all_subjects:
            image_paths = self._find_images(
                subject=subject,
                datatype=datatype,
                suffixes=suffixes,
                extension=extension,
                derivatives_pipeline=include_derivatives_from,
            )
            if not image_paths:
                missing_images.append(subject)
                log.debug("No image found for %s — skipping", subject)
                continue

            sample: dict[str, Any] = {}

            if multi_channel:
                sample["image"] = [
                    str(p) if use_absolute_paths else self._rel(p)
                    for p in image_paths
                ]
            else:
                p = image_paths[0]
                sample["image"] = str(p) if use_absolute_paths else self._rel(p)

            # Segmentation mask
            if seg_suffix:
                seg_path = self._find_single(
                    subject=subject,
                    datatype=seg_datatype or datatype,
                    suffix=seg_suffix,
                    extension=extension,
                    derivatives_pipeline=include_derivatives_from,
                )
                if seg_path:
                    sample["label"] = str(seg_path) if use_absolute_paths else self._rel(seg_path)
                else:
                    log.debug("No segmentation mask for %s — label path omitted", subject)

            # Classification label (integer)
            if label_column and not seg_suffix:
                raw_label = label_lookup.get(subject)
                if raw_label is None:
                    sample["label"] = negative_label_value
                else:
                    sample["label"] = label_classes.get(str(raw_label), negative_label_value)

            sample["subject_id"] = subject
            samples.append(sample)

        if missing_images:
            log.warning(
                "%d/%d subjects had no matching image for suffix %s — excluded",
                len(missing_images), len(all_subjects), suffixes,
            )

        if not samples:
            raise RuntimeError(
                f"No valid samples found. Check datatype={datatype!r} suffix={suffixes!r} "
                f"exist under {self.bids_root}."
            )

        # ── Split ─────────────────────────────────────────────────────────
        training, validation, test = self._split_subjects(
            samples=samples,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
        )

        # ── Assemble and write ────────────────────────────────────────────
        ds_name = dataset_name or f"Qortex — {self.bids_root.name}"
        ds_desc = dataset_description or (
            f"Auto-generated by Qortex from {self.bids_root}. "
            f"Suffix: {suffixes}. Label: {label_column}."
        )
        monai_ds = MONAIDataset(
            dataset_name=ds_name,
            dataset_description=ds_desc,
            modality=modality_name,
            label_classes=label_classes,
            training=training,
            validation=validation,
            test=test,
        )
        out_path = output_dir / "dataset.json"
        monai_ds.save(out_path, indent=indent)
        log.info(
            "Wrote MONAI dataset.json — %d training, %d val, %d test samples",
            len(training), len(validation), len(test),
        )

        self._write_manifest(
            output_dir=output_dir,
            monai_ds=monai_ds,
            suffixes=suffixes,
            label_column=label_column,
            label_source=label_source,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
            missing_count=len(missing_images),
        )
        return out_path

    # ── Private helpers ───────────────────────────────────────────────────

    def _find_images(
        self,
        subject: str,
        datatype: str,
        suffixes: list[str],
        extension: str,
        derivatives_pipeline: str | None,
    ) -> list[Path]:
        """Return one Path per suffix (ordered) or empty list on any miss."""
        found: list[Path] = []
        for suffix in suffixes:
            p = self._find_single(subject, datatype, suffix, extension, derivatives_pipeline)
            if p is None:
                return []   # require all channels — fail-closed
            found.append(p)
        return found

    def _find_single(
        self,
        subject: str,
        datatype: str,
        suffix: str,
        extension: str,
        derivatives_pipeline: str | None,
    ) -> Path | None:
        ext = _resolve_extension(suffix, extension)
        search_root = (
            self.bids_root / "derivatives" / derivatives_pipeline
            if derivatives_pipeline
            else self.bids_root
        )
        # Direct datatype folder
        direct = search_root / subject / datatype
        if direct.is_dir():
            candidates = list(direct.glob(f"*_{suffix}{ext}"))
            if candidates:
                candidates.sort(key=lambda p: len(p.name))
                return candidates[0]
        # Session-level folder
        for ses_dir in sorted((search_root / subject).glob("ses-*")):
            dt_dir = ses_dir / datatype
            if dt_dir.is_dir():
                candidates = list(dt_dir.glob(f"*_{suffix}{ext}"))
                if candidates:
                    candidates.sort(key=lambda p: len(p.name))
                    return candidates[0]
        return None

    def _load_labels(
        self,
        label_source: str,
        label_column: str,
        label_map: dict[str, int] | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        """Parse TSV file and build {subject: raw_label} and {label: int} maps."""
        tsv_paths: dict[str, Path] = {
            "participants": self.bids_root / "participants.tsv",
            "sessions": None,   # not a single file — handled below
            "scans": None,
        }
        subject_col_map: dict[str, str] = {
            "participants": "participant_id",
            "sessions": "participant_id",
        }

        if label_source == "sessions":
            tsv_file = next(
                iter(sorted(self.bids_root.glob("sub-*/sessions.tsv"))), None
            )
        elif label_source == "scans":
            tsv_file = next(
                iter(sorted(self.bids_root.glob("sub-*/*/scans.tsv"))), None
            )
        else:
            tsv_file = tsv_paths.get(label_source)

        if tsv_file is None or not tsv_file.is_file():
            log.warning(
                "Label TSV not found for source=%r. Labels will be missing.", label_source
            )
            return {}, {}

        import csv
        rows: list[dict[str, str]] = []
        with open(tsv_file, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            rows = list(reader)

        subject_col = subject_col_map.get(label_source, "participant_id")
        if subject_col not in (rows[0] if rows else {}):
            # Fallback: first column
            subject_col = next(iter(rows[0])) if rows else subject_col

        if label_column not in (rows[0] if rows else {}):
            log.warning(
                "Column %r not found in %s. Available: %s",
                label_column, tsv_file.name,
                ", ".join(rows[0].keys()) if rows else "(empty)",
            )
            return {}, {}

        label_lookup: dict[str, Any] = {}
        for row in rows:
            sub_id = row.get(subject_col, "").strip()
            if not sub_id.startswith("sub-"):
                sub_id = f"sub-{sub_id}"
            val = row.get(label_column, "").strip()
            if val and val.lower() not in ("n/a", "na", "nan", ""):
                label_lookup[sub_id] = val

        # Build integer encoding
        if label_map:
            label_classes = dict(label_map)
        else:
            unique_values = sorted({str(v) for v in label_lookup.values()})
            label_classes = {v: i for i, v in enumerate(unique_values)}

        return label_lookup, label_classes

    def _split_subjects(
        self,
        samples: list[dict[str, Any]],
        train_frac: float,
        val_frac: float,
        seed: int,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Deterministic subject-level split. Subject order randomised by hash."""
        import hashlib
        n = len(samples)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)
        n_test = n - n_train - n_val

        # Sort by HMAC of (seed, subject_id) for reproducible, seed-sensitive ordering
        def sort_key(s: dict[str, Any]) -> str:
            raw = f"{seed}:{s.get('subject_id', s.get('image', ''))}"
            return hashlib.sha256(raw.encode()).hexdigest()

        shuffled = sorted(samples, key=sort_key)
        training = shuffled[:n_train]
        validation = shuffled[n_train: n_train + n_val]
        test = shuffled[n_train + n_val:]
        return training, validation, test

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.bids_root))
        except ValueError:
            return str(path)

    def _write_manifest(
        self,
        output_dir: Path,
        monai_ds: MONAIDataset,
        **kwargs: Any,
    ) -> None:
        meta = {
            "generated_by": "qortex.export.MONAIExporter",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bids_root": str(self.bids_root),
            "n_training": monai_ds.n_training,
            "n_validation": monai_ds.n_validation,
            "n_test": monai_ds.n_test,
            "label_classes": monai_ds.label_classes,
            **{k: str(v) if isinstance(v, Path) else v for k, v in kwargs.items()},
        }
        manifest_path = output_dir / "qortex_manifest.json"
        manifest_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.debug("Wrote Qortex manifest to %s", manifest_path)
