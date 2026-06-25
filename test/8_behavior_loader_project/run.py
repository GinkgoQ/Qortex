from __future__ import annotations

import sys
from pathlib import Path

from qortex.parse.behavior import BehaviorLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import downloaded_event_file, print_kv, print_rows, real_metadata_root, require  # noqa: E402


def main() -> None:
    tmp, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        local_event = downloaded_event_file(root)
        event_file = manifest.get_file(local_event.relative_to(root).as_posix())
        require(event_file is not None, f"downloaded event file not found in manifest: {local_event}")

        loader = BehaviorLoader()
        inspection = loader.inspect(event_file, local_event)
        record = loader.load(event_file, local_event)
        samples = list(loader.to_sample_records(record))

        print_kv(
            "PROJECT 8: real BIDS events loader",
            {
                "dataset": ds.dataset_id,
                "event file": event_file.path,
                "can load": loader.can_load(event_file),
                "rows": inspection["n_rows"],
                "columns": ", ".join(inspection["columns"]),
                "label column": inspection["label_column"],
                "label preview": inspection["label_values_preview"],
                "samples emitted": len(samples),
            },
        )
        print_rows(
            "Real event samples",
            [
                {
                    "label": sample.label,
                    "label_name": sample.label_name,
                    "onset": sample.onset,
                    "duration": sample.duration,
                    "response_time": sample.provenance.get("response_time"),
                }
                for sample in samples[:12]
            ],
            limit=12,
        )

        require(loader.can_load(event_file), "BehaviorLoader refused a real events.tsv")
        require(inspection["n_rows"] > 0, "real event file has no rows")
        require(len(samples) == inspection["n_rows"], "events loader emitted wrong number of samples")
    finally:
        tmp.cleanup()

    print("RESULT: real behavior-loader project passed")


if __name__ == "__main__":
    main()
