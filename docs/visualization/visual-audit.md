# Visual Audit

`VisualAuditReport` is a coverage check across all subjects and file types in a dataset. It answers which files are present, which are missing, and how many of each type exist.

## Running a visual audit

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
report = ds.visual_audit()
report.show()  # opens browser with HTML report
```

Or directly:

```python
from qortex.visualize._audit import run_visual_audit

report = run_visual_audit("data/ds004130/")
report.show()
```

## From the CLI

```bash
qortex visual-audit ds004130 --data-dir data/ds004130/ --output audit.html
```

Without `--data-dir`, runs in manifest-only mode (coverage matrix only, no thumbnails).

## Report contents

### Coverage matrix

A table with rows = subjects, columns = file suffixes (T1w, bold, events, channels, etc.). Each cell is green (present), red (missing), or gray (not applicable).

Access as a dict:

```python
matrix = report.coverage_matrix()
# {subject_id: {suffix: bool}}
print(matrix["sub-01"]["bold"])   # True
print(matrix["sub-03"]["events"]) # False
```

### Per-suffix counts

```python
for suffix, count in report.per_suffix_counts.items():
    print(f"{suffix}: {count}")
# bold: 88
# events: 86
# T1w: 88
# channels: 88
```

### Warning summary

```python
for warning in report.warning_summary():
    print(warning)
# MISSING: sub-23_task-rest_events.tsv (1 subject affected)
# MISSING: sub-47_task-rest_events.tsv (1 subject affected)
```

### Action items

`action_items()` returns the highest-priority issues, sorted by number of subjects affected:

```python
for item in report.action_items():
    print(item.severity, item.message)
```

### Missing expected files

`missing_expected_files()` returns files that exist for most subjects but not all:

```python
for path in report.missing_expected_files(threshold=0.9):
    # path is missing for > 10% of subjects
    print(path)
```

## Manifest mode (no download required)

```python
ds = Dataset("ds004130")
report = ds.visual_audit(mode="manifest")
report.show()
```

In manifest mode, coverage is based on the remote manifest. No files are read from disk. Thumbnails are not included.

## Exporting the report

```python
report.to_html("audit.html")     # full HTML with coverage matrix
report.to_json("audit.json")     # machine-readable JSON
report.to_markdown("audit.md")   # markdown table
```

The JSON export includes `visual_manifest_json()` — a nested dict of all files with their paths, sizes, entities, and coverage status.

## Interpreting the coverage matrix

A missing file (red cell) is not always a problem. In datasets where some subjects did not complete all tasks, gaps in the coverage matrix are expected. Use `missing_expected_files(threshold=0.8)` to focus on files that are present for at least 80% of subjects but absent for some — those are the unexpected gaps.

## Related

- [First visual audit](../getting-started/first-visual-audit.md) — quickstart walkthrough
- [fMRI QC](fmri-qc.md) — per-file quality metrics beyond coverage
