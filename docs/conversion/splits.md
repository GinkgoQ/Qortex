# Splits

Qortex assigns subjects to train, val, and test splits before windowing. This ensures that all windows from a given subject land in exactly one split — preventing subject leakage between train and test.

## SplitSpec

```python
from qortex.convert import SplitSpec

split = SplitSpec(
    strategy="subject",      # always use subject-level assignment
    val_frac=0.15,           # 15% of subjects to validation
    test_frac=0.15,          # 15% of subjects to test
    stratify_by_label=True,  # balance label classes across splits
    seed=42,                 # random seed for reproducibility
)
```

Pass as a dict to `ds.convert()`:

```python
art = ds.convert(
    split=dict(strategy="subject", val_frac=0.15, test_frac=0.15),
    ...
)
```

## Strategy: subject

The only supported strategy. Subjects are randomly assigned to splits. All windows from a given subject go to the same split.

**Why this is the only strategy:** Random window-level splitting (without subject grouping) produces optimistic test metrics. A classifier can learn subject-specific features (head shape, electrode impedance, scanner noise) and generalize to the test set because the test set contains data from the same subjects. This is not what you want. Subject-level splitting enforces that the test set contains subjects the model has never seen.

## Val and test fractions

The train fraction is `1 - val_frac - test_frac`. With 88 subjects and default fractions:

- Train: 75 subjects
- Val: 13 subjects
- Test: 8 subjects (minimum for 15% of 88)

If the dataset has fewer than 6 subjects, val and test may have only 1 subject each.

## Stratification

When `stratify_by_label=True`, the split assignment tries to balance label class proportions across splits. This is important when classes are rare — without stratification, a minority class might end up entirely in train with no representation in test.

Stratification uses the subject's most common label class as the stratification key. It is an approximation, not a guarantee, because subjects rarely have perfectly balanced trial counts.

## Reproducibility

The same `seed` value with the same `val_frac` and `test_frac` always produces the same split assignment. The seed is recorded in `artifact_manifest.json`.

If you need to re-run conversion with the same split, pass the original seed:

```python
art = ds.convert(split=dict(strategy="subject", seed=42), ...)
```

## Reading split assignments

After conversion, see which subjects are in each split:

```python
art = Artifact.open("artifacts/ds004130/")
print(art.manifest.train_subjects)
print(art.manifest.val_subjects)
print(art.manifest.test_subjects)
```

## Leakage verification

After conversion, verify no subject appears in two splits:

```python
result = ds.leakage_check(art)
print(result.ok)  # True if no leakage
```

See [Leakage check](../readiness/leakage-check.md).
