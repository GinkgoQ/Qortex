from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from qortex.console.annotation_store import list_annotations, load_annotation, save_annotation


def _payload(title: str = "Review") -> dict:
    return {
        "title": title,
        "layers": [
            {"id": "distances", "name": "Distances", "kind": "measurements", "visible": True, "item_ids": ["d1"]},
            {"id": "rois", "name": "ROIs", "kind": "rois", "visible": True, "item_ids": ["r1"]},
            {"id": "bookmarks", "name": "Bookmarks", "kind": "bookmarks", "visible": True, "item_ids": ["b1"]},
        ],
        "measurements": [
            {"id": "d1", "kind": "distance", "distance_mm": 12.5},
            {"id": "r1", "kind": "roi", "start_voxel": [1, 2, 3], "end_voxel": [4, 5, 6], "mean": 10.0, "std": 2.0, "voxel_count": 64, "volume_mm3": 128.0},
        ],
        "bookmarks": [{"id": "b1", "name": "Finding", "voxel": [3, 4, 5], "world_mm": [1.0, -2.0, 3.0], "frame": 0}],
        "viewport": {"crosshair_voxel": [3, 4, 5], "crosshair_world_mm": [1.0, -2.0, 3.0], "frame": 0, "layout": "grid", "cal_min": 0.0, "cal_max": 100.0},
    }


def _source() -> dict:
    return {"path": "sub-01/anat/sub-01_T1w.nii.gz", "checksum": "abc123", "size_bytes": 100}


def test_annotation_versions_and_lists_head(tmp_path: Path) -> None:
    created = save_annotation(
        dataset_id="ds000001", snapshot="1.0.0", source=_source(), payload=_payload(), store_root=tmp_path,
    )
    updated = save_annotation(
        dataset_id="ds000001", snapshot="1.0.0", source=_source(), payload=_payload("Updated"),
        annotation_id=created["annotation_id"], expected_revision=1, store_root=tmp_path,
    )

    assert created["revision"] == 1
    assert updated["revision"] == 2
    assert load_annotation("ds000001", "1.0.0", created["annotation_id"], revision=1, store_root=tmp_path)["title"] == "Review"
    assert load_annotation("ds000001", "1.0.0", created["annotation_id"], store_root=tmp_path)["title"] == "Updated"
    rows = list_annotations("ds000001", "1.0.0", source_path=_source()["path"], store_root=tmp_path)["annotations"]
    assert rows[0]["revision"] == 2
    assert rows[0]["measurement_count"] == 2


def test_annotation_update_uses_optimistic_revision(tmp_path: Path) -> None:
    created = save_annotation(
        dataset_id="ds000001", snapshot="1.0.0", source=_source(), payload=_payload(), store_root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="revision conflict"):
        save_annotation(
            dataset_id="ds000001", snapshot="1.0.0", source=_source(), payload=_payload(),
            annotation_id=created["annotation_id"], expected_revision=0, store_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("path", "sub-02/anat/sub-02_T1w.nii.gz"), ("checksum", "different"), ("size_bytes", 101)),
)
def test_annotation_source_is_immutable_across_revisions(tmp_path: Path, field: str, value: object) -> None:
    created = save_annotation(
        dataset_id="ds000001", snapshot="1.0.0", source=_source(), payload=_payload(), store_root=tmp_path,
    )
    changed = {**_source(), field: value}
    with pytest.raises(ValueError, match="source identity cannot change"):
        save_annotation(
            dataset_id="ds000001", snapshot="1.0.0", source=changed, payload=_payload(),
            annotation_id=created["annotation_id"], expected_revision=1, store_root=tmp_path,
        )


def test_annotation_import_rejects_unknown_layer_item() -> None:
    payload = _payload()
    payload["layers"][0]["item_ids"] = ["not-present"]
    with pytest.raises(ValidationError, match="unknown items"):
        save_annotation(
            dataset_id="ds000001", snapshot="1.0.0", source=_source(), payload=payload,
        )
