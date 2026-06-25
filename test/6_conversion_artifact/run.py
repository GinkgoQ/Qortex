from __future__ import annotations

import sys
from pathlib import Path

import qortex
from qortex.convert.pipeline import ConversionPipeline
from qortex.convert.splits import SplitSpec

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import artifact_dir, print_kv, print_rows, real_metadata_root, require  # noqa: E402


def main() -> None:
    tmp, ds, root = real_metadata_root()
    try:
        manifest = ds.manifest()
        out = artifact_dir(root, "project6_conversion")
        result = ConversionPipeline(
            manifest=manifest,
            data_dir=root,
            output_dir=out,
            output_format="parquet",
            split_spec=SplitSpec(strategy="subject", train=0.7, val=0.15, test=0.15),
            shard_size=100,
        ).run()
        artifact = qortex.Artifact.open(out)
        files = sorted(path.name for path in out.iterdir())

        print_kv(
            "PROJECT 6: real OpenNeuro event/table-to-Parquet artifact",
            {
                "dataset": ds.dataset_id,
                "snapshot": manifest.snapshot,
                "output format": result.output_format,
                "samples": result.n_samples,
                "subjects": result.n_subjects,
                "splits": result.splits,
                "artifact id": artifact.summary()["artifact_id"],
            },
        )
        print_rows("Real artifact directory", [{"file": name} for name in files], limit=20)
        print_rows("Real artifact summary", [artifact.summary()])

        require(result.n_samples > 0, "real conversion produced no samples")
        require((out / "artifact_manifest.json").exists(), "artifact manifest missing")
        require((out / "_SUCCESS").exists(), "success marker missing")
        require(any(path.suffix == ".parquet" for path in out.iterdir()), "no Parquet shard was written")
        require(artifact.summary()["dataset_id"] == ds.dataset_id, "Artifact.open returned wrong dataset")
    finally:
        tmp.cleanup()

    print("RESULT: real conversion project passed")


if __name__ == "__main__":
    main()
