from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import qortex


DATASET_ID = os.environ.get("QORTEX_REAL_TEST_DATASET", "ds000001")
SNAPSHOT = os.environ.get("QORTEX_REAL_TEST_SNAPSHOT") or None
_SHARED_ROOT_ENV = "QORTEX_REAL_METADATA_ROOT"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def print_kv(title: str, values: dict[str, Any]) -> None:
    banner(title)
    width = max((len(str(key)) for key in values), default=0)
    for key, value in values.items():
        print(f"{str(key).ljust(width)} : {value}")


def print_rows(title: str, rows: list[dict[str, Any]], *, limit: int = 8) -> None:
    banner(title)
    if not rows:
        print("(no rows)")
        return
    rows = rows[:limit]
    columns = list(rows[0])
    widths = {
        column: min(
            90,
            max(len(column), *(len(_cell(row.get(column, ""))) for row in rows)),
        )
        for column in columns
    }
    print(" | ".join(column.ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(" | ".join(_cell(row.get(column, "")).ljust(widths[column])[: widths[column]] for column in columns))


def real_dataset() -> qortex.Dataset:
    return qortex.Dataset(DATASET_ID, snapshot=SNAPSHOT)


def real_manifest():
    ds = real_dataset()
    return ds, ds.manifest()


def real_metadata_root() -> tuple[tempfile.TemporaryDirectory[str], qortex.Dataset, Path]:
    shared_root = os.environ.get(_SHARED_ROOT_ENV)
    if shared_root:
        tmp = _NoCleanup()
        root = Path(shared_root).expanduser().resolve() / DATASET_ID
    else:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / DATASET_ID
    ds = real_dataset()
    if not _metadata_tree_ready(root):
        root.parent.mkdir(parents=True, exist_ok=True)
        result = ds.download_metadata(output_dir=root, max_size_gb=0.2)
        require(result.success, result.report())
        require(result.plan.n_files > 0, "metadata download plan was empty")
    else:
        ds._data_dir = root
    return tmp, ds, root


def primary_recording_with_events(manifest):
    from qortex.manifest.graph import ManifestGraph

    for recording in ManifestGraph(manifest).recordings():
        if recording.companions.events is not None and recording.primary.urls:
            return recording
    raise RuntimeError("No event-complete primary recording found in real OpenNeuro manifest")


def first_event_file(manifest):
    for file in manifest.files:
        if file.suffix == "events" and file.extension == ".tsv":
            return file
    raise RuntimeError("No events.tsv file found in real OpenNeuro manifest")


def first_metadata_table(manifest):
    for filename in ("participants.tsv", "sessions.tsv"):
        file = manifest.get_file(filename)
        if file is not None:
            return file
    for file in manifest.files:
        if file.extension in {".tsv", ".csv"}:
            return file
    raise RuntimeError("No metadata table found in real OpenNeuro manifest")


def downloaded_event_file(root: Path):
    for path in sorted(root.rglob("*_events.tsv")):
        return path
    raise RuntimeError("No downloaded real events.tsv file found")


def downloaded_manifest(ds: qortex.Dataset):
    return ds.manifest()


def split_subject_rows(train, val, test) -> list[dict[str, Any]]:
    rows = []
    for split, samples in (("train", train), ("val", val), ("test", test)):
        subjects = sorted({sample.subject for sample in samples if sample.subject})
        rows.append(
            {
                "split": split,
                "subjects": ", ".join(subjects),
                "n_subjects": len(subjects),
                "n_samples": len(samples),
            }
        )
    return rows


def artifact_dir(root: Path, name: str) -> Path:
    base = Path(os.environ.get("QORTEX_REAL_ARTIFACT_ROOT", ""))
    if not str(base):
        base = root.parent / "artifacts"
    path = base.expanduser().resolve() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cell(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _metadata_tree_ready(root: Path) -> bool:
    return (
        root.exists()
        and (root / "dataset_description.json").exists()
        and any(root.rglob("*_events.tsv"))
    )


class _NoCleanup:
    name = ""

    def cleanup(self) -> None:
        return None
