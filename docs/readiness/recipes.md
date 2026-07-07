# Recipes

A Qortex recipe is a small JSON file that records a reproducible decision workflow: dataset ID, snapshot, modality, target, split unit, goal, and output directory. Use it when a download plan or first-batch check must be rerun exactly by another person or on another machine.

## Using a recipe

```python
from qortex import Dataset

ds = Dataset("ds000001", snapshot="1.0.0")
report = ds.minimum(goal="first-batch", target="trial_type", output_dir="data/ds000001")
print(report.to_text())
```

Or from the CLI:

```bash
qortex make-recipe ds000001 recipes/ds000001_first_batch.json \
  --snapshot 1.0.0 \
  --target trial_type \
  --goal first-batch \
  --output-dir data/ds000001

qortex run-recipe recipes/ds000001_first_batch.json
```

## What belongs in a recipe

Keep recipes operational rather than aspirational:

- `dataset_id` and `snapshot` pin the remote dataset state.
- `target` names the label column to check, such as `trial_type`.
- `modality` narrows the plan when a dataset has several signal types.
- `goal` chooses the evidence level: `metadata`, `label-check`, `first-batch`, or `validation`.
- `output_dir` fixes where downloaded files will land if `--download` is used.

## Creating and saving a recipe

```python
from qortex.decision import Recipe, write_recipe

recipe = Recipe(
    dataset_id="ds000001",
    snapshot="1.0.0",
    modality="fmri",
    target="trial_type",
    split="subject",
    goal="first-batch",
    output_dir="data/ds000001",
)
write_recipe(recipe, "recipes/ds000001_first_batch.json")
```

Load and use:

```bash
qortex run-recipe recipes/ds000001_first_batch.json
```

## Recipe output

A recipe run produces the same structured minimum-plan report as `ds.minimum()`. The recipe does not invent new thresholds; it freezes the inputs so the plan can be audited and repeated.




## Related

- [Doctor](doctor.md) — the underlying readiness check engine
- [Can train](can-train.md) — structured label-readiness report
