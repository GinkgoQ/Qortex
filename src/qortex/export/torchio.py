"""TorchIO SubjectsDataset JSON manifest exporter.

Generates a JSON file consumable directly by::

    import torchio as tio

    subjects = tio.SubjectsDataset.load_json("torchio_manifest.json")
    dataset  = tio.SubjectsDataset(subjects)

Each entry maps to one ``tio.Subject`` with named image fields.  The exporter
supports:

* Multiple scalar / label map image types per subject (T1w, T2w, FLAIR, etc.)
* Segmentation masks as ``LabelMap`` fields
* Scalar labels (for classification) stored as custom metadata attributes
* Session-aware subjects (each session becomes a separate Subject entry)
* Leakage-safe subject-level splits stored in the ``split`` attribute

JSON format::

    {
      "subjects": [
        {
          "name": "sub-01",
          "split": "training",
          "label": 1,
          "T1w": {"path": "sub-01/anat/sub-01_T1w.nii.gz", "type": "ScalarImage"},
          "T2w": {"path": "sub-01/anat/sub-01_T2w.nii.gz", "type": "ScalarImage"},
          "dseg": {"path": "sub-01/anat/sub-01_dseg.nii.gz", "type": "LabelMap"}
        },
        ...
      ],
      "meta": {...}
    }
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_TORCHIO_IMAGE_TYPES = frozenset({"ScalarImage", "LabelMap"})
_COMPOUND_EXTENSIONS = frozenset({".nii.gz", ".nii", ".mgz", ".mgh"})

_DEFAULT_MODALITIES: dict[str, str] = {
    "T1w": "ScalarImage",
    "T2w": "ScalarImage",
    "FLAIR": "ScalarImage",
    "T2star": "ScalarImage",
    "dwi": "ScalarImage",
    "bold": "ScalarImage",
    "dseg": "LabelMap",
    "mask": "LabelMap",
    "aseg": "LabelMap",
}


@dataclass
class TorchIOSubject:
    """In-memory representation of one TorchIO Subject."""

    name: str
    split: str | None = None
    label: int | None = None
    images: dict[str, dict[str, str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.split is not None:
            d["split"] = self.split
        if self.label is not None:
            d["label"] = self.label
        d.update(self.images)
        d.update(self.metadata)
        return d


class TorchIOExporter:
    """Export a locally-downloaded BIDS dataset as a TorchIO subjects JSON.

    Parameters
    ----------
    bids_root:
        Root of the downloaded BIDS tree.
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
        # Image specification: suffix → TorchIO image type
        modalities: dict[str, str] | None = None,
        datatype: str = "anat",
        extension: str = ".nii.gz",
        derivatives_pipeline: str | None = None,
        # Label
        label_column: str | None = None,
        label_source: str = "participants",
        label_map: dict[str, int] | None = None,
        negative_label_value: int = 0,
        # Sessions: when True, each session becomes a separate Subject
        per_session: bool = False,
        # Split
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        seed: int = 42,
        # Paths
        use_absolute_paths: bool = True,
        output_filename: str = "torchio_manifest.json",
        indent: int = 2,
    ) -> Path:
        """Build and write the TorchIO subjects manifest JSON.

        Parameters
        ----------
        modalities:
            Mapping from BIDS suffix to TorchIO image type, e.g.
            ``{"T1w": "ScalarImage", "dseg": "LabelMap"}``.
            Defaults to all standard anatomical modalities found on disk.
        datatype:
            BIDS datatype folder to scan.
        per_session:
            When True, each (subject, session) pair becomes a separate
            Subject rather than collapsing all sessions into one.
        use_absolute_paths:
            When True (default), image paths in the JSON are absolute.
        output_filename:
            Name of the output JSON file.

        Returns
        -------
        Path
            Path to the written manifest JSON.
        """
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        mods = modalities or self._infer_modalities(datatype, extension)
        if not mods:
            raise RuntimeError(
                f"No modalities specified and none auto-detected in {self.bids_root}. "
                "Pass a modalities dict explicitly."
            )

        # ── Discover subjects ─────────────────────────────────────────────
        all_subjects = sorted({
            p.name for p in self.bids_root.iterdir()
            if p.is_dir() and p.name.startswith("sub-")
        })
        if not all_subjects:
            raise FileNotFoundError(f"No sub-* directories in {self.bids_root}")

        # ── Labels ───────────────────────────────────────────────────────
        label_lookup: dict[str, Any] = {}
        label_classes: dict[str, int] = {}
        if label_column:
            label_lookup, label_classes = self._load_labels(label_source, label_column, label_map)

        # ── Build subjects ────────────────────────────────────────────────
        subjects: list[TorchIOSubject] = []
        missing: list[str] = []

        for subject in all_subjects:
            if per_session:
                ses_list = self._sessions(subject)
                if not ses_list:
                    ses_list = [None]
            else:
                ses_list = [None]

            for session in ses_list:
                images = self._collect_images(
                    subject=subject,
                    session=session,
                    datatype=datatype,
                    modalities=mods,
                    extension=extension,
                    derivatives_pipeline=derivatives_pipeline,
                    use_absolute_paths=use_absolute_paths,
                )
                if not images:
                    missing.append(subject if session is None else f"{subject}/{session}")
                    continue

                name = subject if session is None else f"{subject}_{session}"
                raw_label = label_lookup.get(subject)
                int_label: int | None = None
                if label_column and raw_label is not None:
                    int_label = label_classes.get(str(raw_label), negative_label_value)
                elif label_column:
                    int_label = negative_label_value

                tio_subject = TorchIOSubject(
                    name=name,
                    label=int_label,
                    images=images,
                    metadata={
                        "subject_id": subject,
                        **({"session_id": session} if session else {}),
                    },
                )
                subjects.append(tio_subject)

        if missing:
            log.warning(
                "%d subjects/sessions had no matching images — excluded: %s",
                len(missing), missing[:10],
            )

        if not subjects:
            raise RuntimeError(
                f"No valid subjects. Check modalities={mods!r}, datatype={datatype!r}."
            )

        # ── Split ─────────────────────────────────────────────────────────
        subjects = self._assign_splits(subjects, train_frac, val_frac, seed)

        # ── Write ─────────────────────────────────────────────────────────
        doc = {
            "meta": {
                "generated_by": "qortex.export.TorchIOExporter",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bids_root": str(self.bids_root),
                "modalities": mods,
                "datatype": datatype,
                "label_column": label_column,
                "label_classes": label_classes,
                "n_subjects": len(subjects),
                "splits": _count_splits(subjects),
            },
            "subjects": [s.to_dict() for s in subjects],
        }
        out_path = output_dir / output_filename
        out_path.write_text(json.dumps(doc, indent=indent), encoding="utf-8")
        log.info(
            "Wrote TorchIO manifest: %d subjects → %s",
            len(subjects), out_path,
        )
        return out_path

    # ── Helpers ───────────────────────────────────────────────────────────

    def _infer_modalities(self, datatype: str, extension: str) -> dict[str, str]:
        """Auto-detect which suffixes are present in the BIDS tree."""
        detected: dict[str, str] = {}
        for sub_dir in sorted(self.bids_root.glob("sub-*")):
            dt_dir = sub_dir / datatype
            if not dt_dir.is_dir():
                for ses_dir in sorted(sub_dir.glob("ses-*")):
                    dt_dir = ses_dir / datatype
                    if dt_dir.is_dir():
                        break
                else:
                    continue
            for p in dt_dir.glob(f"*{extension}"):
                stem = p.name
                for ext in sorted(_COMPOUND_EXTENSIONS, key=len, reverse=True):
                    if stem.endswith(ext):
                        stem = stem[: -len(ext)]
                        break
                suffix = stem.rsplit("_", 1)[-1] if "_" in stem else stem
                if suffix not in detected:
                    detected[suffix] = _DEFAULT_MODALITIES.get(suffix, "ScalarImage")
            if detected:
                break   # one subject is enough to infer structure
        return detected

    def _sessions(self, subject: str) -> list[str]:
        sub_dir = self.bids_root / subject
        return sorted({p.name for p in sub_dir.glob("ses-*") if p.is_dir()})

    def _find_file(
        self,
        subject: str,
        session: str | None,
        datatype: str,
        suffix: str,
        extension: str,
        derivatives_pipeline: str | None,
    ) -> Path | None:
        search_root = (
            self.bids_root / "derivatives" / derivatives_pipeline
            if derivatives_pipeline
            else self.bids_root
        )
        sub_root = search_root / subject
        if session:
            dt_dir = sub_root / session / datatype
        else:
            dt_dir = sub_root / datatype
        if dt_dir.is_dir():
            candidates = list(dt_dir.glob(f"*_{suffix}{extension}"))
            if candidates:
                return sorted(candidates, key=lambda p: len(p.name))[0]
        if not session:
            for ses_dir in sorted(sub_root.glob("ses-*")):
                dt_dir = ses_dir / datatype
                if dt_dir.is_dir():
                    candidates = list(dt_dir.glob(f"*_{suffix}{extension}"))
                    if candidates:
                        return sorted(candidates, key=lambda p: len(p.name))[0]
        return None

    def _collect_images(
        self,
        subject: str,
        session: str | None,
        datatype: str,
        modalities: dict[str, str],
        extension: str,
        derivatives_pipeline: str | None,
        use_absolute_paths: bool,
    ) -> dict[str, dict[str, str]]:
        images: dict[str, dict[str, str]] = {}
        for suffix, image_type in modalities.items():
            if image_type not in _TORCHIO_IMAGE_TYPES:
                raise ValueError(
                    f"Invalid TorchIO image type {image_type!r} for suffix {suffix!r}. "
                    f"Must be one of {sorted(_TORCHIO_IMAGE_TYPES)}."
                )
            p = self._find_file(
                subject=subject,
                session=session,
                datatype=datatype,
                suffix=suffix,
                extension=extension,
                derivatives_pipeline=derivatives_pipeline,
            )
            if p is not None:
                path_str = str(p) if use_absolute_paths else str(
                    p.relative_to(self.bids_root)
                )
                images[suffix] = {"path": path_str, "type": image_type}
        return images

    def _load_labels(
        self,
        label_source: str,
        label_column: str,
        label_map: dict[str, int] | None,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        if label_source == "participants":
            tsv = self.bids_root / "participants.tsv"
        else:
            tsv = next(iter(sorted(self.bids_root.glob(f"**/{label_source}.tsv"))), None)

        if tsv is None or not tsv.is_file():
            log.warning("Label source TSV not found: %r", label_source)
            return {}, {}

        import csv
        with open(tsv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))

        if not rows:
            return {}, {}

        sub_col = "participant_id" if "participant_id" in rows[0] else next(iter(rows[0]))
        if label_column not in rows[0]:
            log.warning("Column %r not in %s", label_column, tsv)
            return {}, {}

        label_lookup: dict[str, Any] = {}
        for row in rows:
            sub = row[sub_col].strip()
            if not sub.startswith("sub-"):
                sub = f"sub-{sub}"
            val = row.get(label_column, "").strip()
            if val and val.lower() not in ("n/a", "na", "nan", ""):
                label_lookup[sub] = val

        if label_map:
            label_classes = dict(label_map)
        else:
            unique = sorted({str(v) for v in label_lookup.values()})
            label_classes = {v: i for i, v in enumerate(unique)}
        return label_lookup, label_classes

    def _assign_splits(
        self,
        subjects: list[TorchIOSubject],
        train_frac: float,
        val_frac: float,
        seed: int,
    ) -> list[TorchIOSubject]:
        n = len(subjects)
        n_train = round(n * train_frac)
        n_val = round(n * val_frac)

        def sort_key(s: TorchIOSubject) -> str:
            return hashlib.sha256(f"{seed}:{s.name}".encode()).hexdigest()

        ordered = sorted(subjects, key=sort_key)
        for i, s in enumerate(ordered):
            if i < n_train:
                s.split = "training"
            elif i < n_train + n_val:
                s.split = "validation"
            else:
                s.split = "test"
        return ordered


def _count_splits(subjects: list[TorchIOSubject]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in subjects:
        sp = s.split or "unassigned"
        counts[sp] = counts.get(sp, 0) + 1
    return counts
