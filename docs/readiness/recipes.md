# Recipes

A recipe is a named set of readiness parameters tuned for a specific task type. Instead of specifying every check parameter manually, you pick a recipe and get sensible defaults for that modality and task combination.

## Using a recipe

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.doctor(recipe="eeg-classification")
```

Or from the CLI:

```bash
qortex make-recipe eeg-classification
qortex doctor ds004130 --recipe eeg-classification
```

## Available recipes

### eeg-classification

For supervised classification from EEG epochs:

- Requires `trial_type` or equivalent events column
- Minimum 20 subjects
- Minimum 2 classes, 30 samples per class
- Checks for `channels.tsv` and `coordsystem.json`
- Window: 1.0 s, no overlap (event-aligned)

### eeg-regression

For regression from EEG (e.g., predicting reaction time):

- Requires a numeric events column
- Minimum 20 subjects
- Checks continuous range of target column

### fmri-classification

For supervised classification from fMRI BOLD:

- Requires `events.tsv` with `trial_type`
- Minimum 15 subjects
- TR check: warns if TR > 2.5 s
- Checks for fMRIPrep confounds if present

### fmri-regression

For regression from fMRI (e.g., predicting behavioral scores from resting-state):

- Requires numeric column in participants.tsv
- Minimum 30 subjects (regression needs more data)
- Checks for resting-state BOLD (task=rest or no task)

### dwi

For DWI-based analysis (tractography, diffusion metrics):

- Requires `bval` and `bvec` files
- Checks number of gradient directions (warns if < 30)
- Checks for multiple b-value shells

### anat-segmentation

For anatomical segmentation tasks:

- Requires T1w or T2w
- No events required
- Minimum 30 subjects

## Creating and saving a recipe

```python
from qortex.readiness import Recipe

recipe = Recipe(
    name="my-custom-recipe",
    modality="eeg",
    target_col="condition",
    min_subjects=50,
    min_classes=3,
    min_per_class=20,
    window_s=2.0,
)
recipe.save("my_recipe.json")
```

Load and use:

```bash
qortex run-recipe my_recipe.json --dataset ds004130
```

## Recipe output

A recipe run produces the same structured report as `doctor()`, but the findings are evaluated against the recipe's thresholds rather than the defaults. This makes comparison across datasets using the same criteria straightforward.

## Related

- [Doctor](doctor.md) — the underlying readiness check engine
- [Can train](can-train.md) — binary version of the label check
