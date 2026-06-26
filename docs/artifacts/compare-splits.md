# Compare Splits

`art.compare_splits()` checks whether the train, val, and test splits have similar distributions. Large differences suggest a data issue.

## Python

```python
from qortex import Artifact

art = Artifact.open("artifacts/ds004130/")
report = art.compare_splits()
report.show()
```

## What is compared

**Class distribution.** The fraction of each class in each split should be approximately equal (assuming stratified splitting). A large discrepancy indicates that stratification failed or the dataset has very few subjects with some classes.

**Mean signal amplitude.** The mean feature value per split should be similar. A split with systematically higher amplitude may contain subjects from a different scanner site or a different acquisition session.

**Feature correlation.** The top-10 most correlated feature pairs should have similar correlation strength across splits. Very different correlation structure between train and test is a red flag for distribution shift.

**Subject count.** Displayed per split.

## Output

```
Compare splits: artifacts/ds004130/
  Split sizes:  train=1,200  val=270  test=270
  Class balance:
    rest:       train=33.3%  val=33.3%  test=33.3%  ✓
    eyes-open:  train=33.2%  val=33.0%  test=32.6%  ✓
    task:       train=33.5%  val=33.7%  test=34.1%  ✓
  Signal amplitude:
    train: mean=0.42  val=0.41  test=0.44  ✓
  Status: ok — no distribution shift detected
```

If issues are found:

```
  WARNING: 'task' class is absent from val split
  Suggestion: re-run conversion with stratify_by_label=True
```

## Programmatic access

```python
for finding in report.findings:
    print(finding.severity, finding.message)

print(report.class_distribution)   # dict: split → class → fraction
print(report.amplitude_summary)    # dict: split → {"mean", "std"}
```

## Related

- [Splits](../conversion/splits.md) — how subjects are assigned to splits
- [Leakage check](../readiness/leakage-check.md) — verify no subject leakage
