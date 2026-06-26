# First Visual Audit

A visual audit is a coverage check with thumbnails. It answers two questions:

1. Which subjects/sessions/tasks have each expected file type?
2. What do the center slices look like — any obvious acquisition failures?

## Requirements

```bash
pip install "qortex[visual-all]"
```

## Run against a local BIDS directory

If you have a dataset on disk at `data/ds004130/`:

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
report = ds.visual_audit()
report.show()  # opens browser
```

`show()` renders an HTML page with:

- A coverage matrix (subjects × suffixes, green/red cells)
- Per-suffix file counts
- Warning summary (missing files, unexpected file sizes)
- Thumbnails for a sample of each suffix

## Run from the manifest only (no download needed)

You can run a coverage-only audit against the remote manifest without downloading any data:

```python
ds = Dataset("ds004130")
report = ds.visual_audit(mode="manifest")
report.show()
```

In manifest mode, you get the coverage matrix and warnings but no thumbnails.

## CLI

```bash
qortex visual-audit ds004130 --data-dir data/ds004130/ --output audit_report.html
```

Without `--data-dir`, runs in manifest mode.

## Save the report

```python
report.to_html("audit_report.html")
report.to_json("audit_report.json")
report.to_markdown("audit_report.md")
```

## Interpreting the coverage matrix

Each cell is colored by whether a file was found for that subject/session combination:

- Green — file present
- Red — file expected (based on other subjects) but missing
- Gray — not applicable (subject did not have this session or task)

The `action_items()` method returns a list of the highest-priority missing files sorted by how many subjects are affected:

```python
for item in report.action_items():
    print(item)
# WARNING: sub-03_task-rest_events.tsv missing (affects 1/88 subjects)
# WARNING: sub-17_T1w.nii.gz missing (affects 1/88 subjects)
```

## Next steps

- [Visual audit reference](../visualization/visual-audit.md) — full VisualAuditReport API
- [fMRI QC](../visualization/fmri-qc.md) — per-file QC panels
- [Download](../download/index.md) — download the flagged missing files
