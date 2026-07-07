# Metadata-Only Download

A metadata-only download fetches everything except the large imaging files. The result is a local directory with JSON sidecars, TSV files, bval/bvec files, and root-level metadata — but no NIfTIs, EEG recordings, or DICOM files.

## Why metadata first

Before spending an hour downloading gigabytes of NIfTI data, a metadata-only pass answers:

- Are events.tsv files present for all subjects? (label readiness)
- What trial types are in the events files? (class distribution)
- What is the TR, number of volumes, and slice order? (preprocessing planning)
- How many EEG channels and what type? (channel configuration)
- What b-values appear in the DWI acquisition? (shell count)

This information costs seconds to fetch and requires almost no disk space.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")
ds.download_metadata(data_dir="data/ds004130/")
```

Files downloaded:

- `dataset_description.json`
- `participants.tsv`
- `README`
- `CHANGES`
- `*_bold.json`, `*_T1w.json`, `*_eeg.json`, etc. (all sidecar JSON)
- `*_events.tsv`, `*_channels.tsv`, `*_electrodes.tsv`, `*_coordsystem.json`
- `*.bval`, `*.bvec`
- `*_physio.json` (not the .tsv.gz)

Imaging files excluded:

- `*.nii.gz`, `*.nii`
- `*.set`, `*.edf`, `*.fif`, `*.cnt`
- `*.dcm` and DICOM directories
- `*.tgz`, `*.zip`

## CLI

```bash
qortex metadata ds004130 --download --output-dir data/ds004130/
```

## Check label coverage after metadata download

After a metadata download, the events.tsv files are present. Use `ds.label_landscape()` to see the label distribution across subjects:

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
print(ds.label_landscape())
```

Output:

```
trial_type  count  subjects_with_label
rest         88    88/88
eyes-open    86    86/88  ← 2 subjects missing
task         88    88/88
```

## Size estimate

For most datasets, a metadata-only download is 1–5 MB total, regardless of dataset size. The events.tsv files are the largest component.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-minimum-plan.png" alt="Horizontal bar chart of the ds000001 first-batch download plan and file sizes.">
  <figcaption>Real `minimum(goal='first-batch')` plan: metadata, sidecar, events, and one BOLD run.</figcaption>
</figure>

```python
plan = ds.minimum(goal='first-batch', output_dir=Path('data/ds000001'))
print(plan.to_text())
```

Result artifact: [ds000001-minimum-first-batch.txt](/Qortex/assets/results/ds000001-minimum-first-batch.txt)

<!-- qortex-evidence:end -->

## Related

- [Selective download](selective-download.md) — download specific imaging files after metadata review
- [Label readiness](../readiness/label-readiness.md) — check label coverage after metadata download
