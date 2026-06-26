# Leakage Check

`ds.leakage_check()` verifies that no subject appears in more than one split of a converted artifact. Subject leakage — where the same person's data is in both training and test splits — invalidates generalization metrics.

## Python

```python
from qortex import Dataset, Artifact

ds = Dataset("ds004130")
art = Artifact.open("artifacts/ds004130_parquet/")

result = ds.leakage_check(art)
print(result.ok)          # True if no leakage
print(result.to_text())
```

Output with no leakage:

```
Leakage check: ok
  Splits: train (61 subjects), val (14 subjects), test (13 subjects)
  Overlap: none
```

Output with leakage:

```
Leakage check: FAILED
  sub-03 appears in train AND test
  sub-17 appears in val AND test
  Action: re-run convert with strategy="subject"
```

## CLI

```bash
qortex leakage-check ds004130 --artifact artifacts/ds004130_parquet/
```

## What is checked

**Subject-level overlap.** Each sample in the artifact has a `subject_id` metadata field. The check reads subject IDs from all splits and looks for IDs that appear in more than one.

**Session-level overlap (optional).** If the dataset has multiple sessions per subject, the check can also verify that no (subject, session) pair crosses split boundaries:

```python
result = ds.leakage_check(art, level="session")
```

**Temporal overlap (not checked).** Qortex does not check for overlapping time windows within a subject across splits. This is by design — temporal overlap within a subject is fine as long as subjects are assigned to splits at the subject level.

## When leakage occurs

Leakage in Qortex output means the conversion was run without `strategy="subject"` in the SplitSpec. This can happen if:

- You used `strategy="random"` or `strategy="time"` during conversion
- You manually merged artifacts from two separate conversion runs
- A custom split was applied after conversion that did not respect subject boundaries

Fix by re-running conversion with the default subject-level split:

```python
art = ds.convert(
    output_dir="artifacts/ds004130_fixed/",
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
)
```

## Limitations

- Leakage check reads the artifact's `subject_id` column. If your conversion did not preserve subject IDs in the metadata, the check cannot detect leakage.
- Phenotypic leakage (e.g., samples from the same scanner site in train and test) is not detected. This requires domain-specific knowledge about the dataset.
