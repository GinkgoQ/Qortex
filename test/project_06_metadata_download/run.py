"""project_06_metadata_download

Downloads only metadata files (sidecar JSONs, TSVs, dataset_description.json)
for a real dataset, then checks that the local tree is BIDS-coherent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    banner, print_kv, real_metadata_root,
    require, require_gt, passed,
)


def main() -> None:
    banner("project_06: metadata-only download")

    ctx, ds, root = real_metadata_root()
    try:
        require(root.exists(), f"metadata root does not exist: {root}")
        require((root / "dataset_description.json").exists(), "dataset_description.json missing")

        downloaded = list(root.rglob("*"))
        files = [p for p in downloaded if p.is_file()]

        print_kv("download root", str(root))
        print_kv("total files downloaded", len(files))

        require_gt(len(files), 0, "downloaded file count")

        # BIDS-required root files
        desc = root / "dataset_description.json"
        require(desc.exists(), "dataset_description.json not downloaded")

        import json
        with open(desc) as fh:
            desc_data = json.load(fh)
        require("Name" in desc_data or "name" in desc_data, "dataset_description.json missing Name field")
        print_kv("dataset name", desc_data.get("Name") or desc_data.get("name"))
        print_kv("BIDS version", desc_data.get("BIDSVersion", "not specified"))

        # check for participants.tsv
        participants = root / "participants.tsv"
        if participants.exists():
            lines = participants.read_text().splitlines()
            print_kv("participants.tsv rows", len(lines) - 1)
            require(len(lines) >= 2, "participants.tsv has no data rows")

        # at least some sidecar JSON files
        json_files = list(root.rglob("*.json"))
        print_kv("JSON sidecar count", len(json_files))
        require(json_files, "no JSON sidecars in downloaded metadata")

        # events TSV files
        events_files = list(root.rglob("*_events.tsv"))
        print_kv("events.tsv count", len(events_files))

        if events_files:
            sample_ev = events_files[0]
            lines = sample_ev.read_text().splitlines()
            require(len(lines) >= 2, f"{sample_ev.name} has no data rows")
            require("onset" in lines[0], f"{sample_ev.name} header missing 'onset'")
            print_kv("sample events file", sample_ev.name)

    finally:
        ctx.cleanup()

    passed("project_06_metadata_download")


if __name__ == "__main__":
    main()
