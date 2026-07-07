# Selective Download

You rarely need an entire OpenNeuro dataset. Selective download lets you specify exactly which subjects, sessions, tasks, runs, and file types to transfer.

## Filter parameters

All filter parameters are accepted by both `ds.download()` and `ds.plan()`:

| Parameter | Type | Example |
|-----------|------|---------|
| `subjects` | `list[str]` | `["01", "02", "03"]` |
| `sessions` | `list[str]` | `["baseline", "followup"]` |
| `tasks` | `list[str]` | `["rest", "nback"]` |
| `runs` | `list[str]` | `["01", "02"]` |
| `suffixes` | `list[str]` | `["bold", "T1w"]` |
| `datatypes` | `list[str]` | `["func", "anat"]` |
| `include` | `list[str]` | glob patterns to include |
| `exclude` | `list[str]` | glob patterns to exclude |
| `max_size_gb` | `float` | skip files larger than this |

## Examples

Download resting-state BOLD from the first 10 subjects:

```python
ds.download(
    subjects=[f"{i:02d}" for i in range(1, 11)],
    tasks=["rest"],
    suffixes=["bold"],
    data_dir="data/ds004130/",
)
```

Download all EEG files including events:

```python
ds.download(
    datatypes=["eeg"],
    data_dir="data/ds004130/",
)
```

Download using glob patterns:

```python
ds.download(
    include=["sub-01/**", "sub-02/**"],
    exclude=["**/*_desc-confounds_*"],
    data_dir="data/ds004130/",
)
```

Skip files larger than 500 MB:

```python
ds.download(max_size_gb=0.5, data_dir="data/ds004130/")
```

## CLI

```bash
# Subjects 01–10, resting-state BOLD
qortex download ds004130 \
    --subjects 01 02 03 04 05 06 07 08 09 10 \
    --tasks rest \
    --suffixes bold \
    --output-dir data/ds004130/

# All EEG
qortex download ds004130 --modalities eeg --output-dir data/ds004130/

# Skip large files
qortex download ds004130 --modalities eeg --output-dir data/ds004130/ --dry-run
```

## Companion files

When you download a primary imaging file, its companions (sidecar JSON, events.tsv, bval/bvec) are included automatically. The companion inclusion follows the BIDS specification:

- BOLD → includes `*_bold.json`, `*_events.tsv`, optionally `*_desc-confounds_timeseries.tsv`
- DWI → includes `*.bval`, `*.bvec`, `*_dwi.json`
- EEG → includes `*_eeg.json`, `*_channels.tsv`, `*_coordsystem.json`

Root-level essential files (`participants.tsv`, `dataset_description.json`) are always included.

To suppress companion inclusion:

```python
ds.download(suffixes=["bold"], include_companions=False, data_dir="data/")
```

## Checking what will be downloaded before transferring

```python
plan = ds.plan(subjects=["01"], tasks=["rest"])
print(f"{len(plan.files)} files, {plan.size_gb:.2f} GB")
```








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

- [Plan](plan.md) — inspect the file list before downloading
- [Metadata only](metadata-only.md) — fetch only JSON, TSV, bval/bvec
