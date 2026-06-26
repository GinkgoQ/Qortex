# Readiness

Readiness checks answer "is this dataset usable for training?" before you download anything expensive. Each check works on the manifest and small sidecar files — not on imaging data.

## Checks and what they answer

[**Doctor**](doctor.md) — Full structural readiness report. Covers subjects, modalities, companion files, label coverage, split feasibility, and total size. Start here.

[**Minimum**](minimum.md) — What is the smallest download that achieves a specific goal? Returns a file list with sizes.

[**Can train**](can-train.md) — Binary check: does this dataset have enough labeled samples to train a classifier? Takes a target column and minimum requirements.

[**First batch**](first-batch.md) — Return a minimal subset of subjects sufficient to run one complete pipeline pass end-to-end.

[**Label readiness**](label-readiness.md) — Per-subject label coverage, class counts, and missing subjects. Works against either manifest or local events.tsv files.

[**Content status**](content-status.md) — After download, verify that local files are complete and not Git LFS pointers.

[**Recipes**](recipes.md) — Predefined task-specific readiness recipes (e.g., "fmri-classification", "eeg-regression") that bundle the most common check parameters.

[**Leakage check**](leakage-check.md) — After conversion, verify that no subject appears in two splits.

## Reading readiness reports

All readiness methods return structured Pydantic objects with:

- `state` — a string enum (e.g., `"not_usable"`, `"manifest_only"`, `"download_ready"`)
- `findings` — list of `Finding` objects with severity, code, and message
- `next_action` — a string describing the recommended next step
- `to_text()` — human-readable summary
- `to_dict()` / `to_json()` — machine-readable output

```python
report = ds.doctor()
if report.state == "not_usable":
    print(report.to_text())
    for f in report.findings:
        if f.severity == "error":
            print(f"  ERROR: {f.message}")
```
