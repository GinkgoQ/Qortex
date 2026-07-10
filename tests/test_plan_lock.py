from __future__ import annotations

from pathlib import Path

from qortex.plan.lock import LockFile


def test_lock_file_exposes_dataset_identity(tmp_path: Path):
    lock_path = tmp_path / ".qortex" / "download.lock.yaml"
    lock_path.parent.mkdir()
    lock_path.write_text(
        "\n".join([
            "qortex_version: 0.1.0",
            "dataset_id: ds000001",
            "snapshot: 1.0.0",
            "files: {}",
            "",
        ]),
        encoding="utf-8",
    )

    lock = LockFile.load(lock_path)

    assert lock.dataset_id == "ds000001"
    assert lock.snapshot == "1.0.0"
