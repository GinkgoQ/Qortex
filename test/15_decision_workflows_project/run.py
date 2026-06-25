from __future__ import annotations

import sys
from pathlib import Path

import qortex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import (  # noqa: E402
    artifact_dir,
    print_kv,
    real_dataset,
    real_manifest,
    real_metadata_root,
    require,
)


def main() -> None:
    ds, manifest = real_manifest()

    doctor = ds.doctor()
    print("\n=== PROJECT 15: doctor report on real OpenNeuro manifest ===")
    print(doctor.to_text())

    label_plan = ds.minimum(goal="label-check", output_dir=Path("real-decision-data"))
    print("\n=== Smallest real label-check download plan ===")
    print(label_plan.to_text())

    first_batch_plan = real_dataset().first_batch(limit=3)
    print("\n=== First-batch decision before local content exists ===")
    print(first_batch_plan.to_text())

    train_remote = ds.can_train()
    print("\n=== Training decision before local label confirmation ===")
    print(train_remote.to_text())

    tmp, metadata_ds, metadata_root = real_metadata_root()
    try:
        doctor_local = metadata_ds.doctor(local_path=metadata_root)
        train_local = metadata_ds.can_train(local_path=metadata_root)
        content = metadata_ds.content_status(metadata_root)

        print("\n=== Doctor report after real metadata download ===")
        print(doctor_local.to_text())

        print("\n=== Training decision after real metadata inspection ===")
        print(train_local.to_text())

        print("\n=== Local content status for metadata-only materialization ===")
        print(content.to_text())

        recipe_path = artifact_dir(metadata_root, "decision_workflow") / "qortex_recipe.json"
        recipe = qortex.Recipe(
            dataset_id=metadata_ds.dataset_id,
            snapshot=manifest.snapshot,
            modality=(manifest.summary.modalities[0] if manifest.summary.modalities else None),
            target="trial_type",
            split="subject",
            goal="first-batch",
            output_dir=str(recipe_path.parent / "downloads"),
        )
        qortex.write_recipe(recipe, recipe_path)
        loaded_recipe = qortex.read_recipe(recipe_path)
        print_kv(
            "Reusable workflow recipe",
            {
                "path": recipe_path,
                "dataset": loaded_recipe.dataset_id,
                "snapshot": loaded_recipe.snapshot,
                "modality": loaded_recipe.modality,
                "target": loaded_recipe.target,
                "split": loaded_recipe.split,
                "goal": loaded_recipe.goal,
                "output_dir": loaded_recipe.output_dir,
            },
        )

        require(doctor.status in {"possible", "uncertain"}, "doctor returned invalid status")
        require(label_plan.plan.n_files > 0, "label-check minimum plan is empty")
        require(first_batch_plan.required_plan is not None, "first-batch did not return a required plan")
        require(train_remote.label_status in {"candidate", "confirmed", "missing"}, "bad training label status")
        require(content.n_files > 0, "metadata content status saw no local files")
        require(loaded_recipe.dataset_id == metadata_ds.dataset_id, "recipe round-trip changed dataset id")
        require(qortex.leakage_check is not None, "public leakage_check export is missing")
    finally:
        tmp.cleanup()

    print("RESULT: real decision workflow project passed")


if __name__ == "__main__":
    main()
