from __future__ import annotations

import sys
from pathlib import Path

from qortex.convert.splits import SplitSpec, apply_split
from qortex.parse.behavior import BehaviorLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import print_kv, print_rows, real_metadata_root, require, split_subject_rows  # noqa: E402


def main() -> None:
    tmp, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        loader = BehaviorLoader()
        samples = []
        for local_event in sorted(root.rglob("*_events.tsv")):
            file = manifest.get_file(local_event.relative_to(root).as_posix())
            if file is None or not loader.can_load(file):
                continue
            samples.extend(loader.to_sample_records(loader.load(file, local_event)))

        train, val, test = apply_split(
            samples,
            SplitSpec(strategy="subject", train=0.7, val=0.15, test=0.15, seed=42),
        )

        print_kv(
            "PROJECT 9: real event samples and subject-safe split assignment",
            {
                "dataset": ds.dataset_id,
                "event samples": len(samples),
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "subjects": len({sample.subject for sample in samples if sample.subject}),
            },
        )
        print_rows(
            "Subject allocation by split",
            split_subject_rows(train, val, test),
        )
        print_rows(
            "Real split sample preview",
            [
                {
                    "subject": sample.subject,
                    "split": sample.split,
                    "task": sample.task,
                    "label_name": sample.label_name,
                    "onset": sample.onset,
                }
                for sample in samples[:16]
            ],
            limit=16,
        )

        train_subjects = {sample.subject for sample in train if sample.subject}
        val_subjects = {sample.subject for sample in val if sample.subject}
        test_subjects = {sample.subject for sample in test if sample.subject}
        require(samples, "no real event samples were loaded")
        require(train or val or test, "real split produced no partitions")
        require(train_subjects.isdisjoint(val_subjects), "subject leaked between train and val")
        require(train_subjects.isdisjoint(test_subjects), "subject leaked between train and test")
        require(val_subjects.isdisjoint(test_subjects), "subject leaked between val and test")
    finally:
        tmp.cleanup()

    print("RESULT: real split project passed")


if __name__ == "__main__":
    main()
