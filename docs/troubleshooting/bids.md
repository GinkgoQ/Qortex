# BIDS Troubleshooting

## BIDS entity parsing returns None

If `file.subject` or `file.task` is None for a file you expect to have those entities, the file path does not follow BIDS naming conventions.

Check the actual path:

```python
manifest = ds.manifest()
for f in manifest.files:
    if f.subject is None:
        print(f.path)
```

Files at the root of the dataset (`dataset_description.json`, `participants.tsv`, etc.) intentionally have `subject=None` — they are dataset-level files.

Files with unexpected paths (e.g., `extra_files/notes.txt`) will also have `subject=None`.

## Filtering returns empty list

```python
files = ds.files(subjects=["01"], suffixes=["bold"])
print(len(files))  # 0
```

Check the actual subject and suffix labels in the manifest:

```python
subjects = {f.subject for f in ds.manifest().files if f.subject}
suffixes = {f.suffix for f in ds.manifest().files if f.suffix}
print(subjects)  # might be {"sub-01"} instead of {"01"}
print(suffixes)
```

BIDS subject IDs do not include the `sub-` prefix. If you see `sub-01` in the subjects set, the paths are being parsed incorrectly. This indicates a non-standard dataset.

## BIDS validation fails

`ds.validate()` wraps the official BIDS Validator. If it fails:

1. Check the error output — each BIDS violation is listed with the file path and rule code
2. BIDS Validator requires the `validation` extra: `pip install "qortex[validation]"`
3. Common validation issues:
   - Missing `dataset_description.json` (required at root)
   - `participants.tsv` missing the `participant_id` column
   - JSON sidecar has `TaskName` but events.tsv is missing

## local-index skips files

`local-index` skips files it cannot parse as BIDS. To see which files are skipped:

```bash
qortex local-index data/ds004130/ --dataset-id ds004130 --verbose
```

## Related

- [BIDS entities](../dataset/bids-entities.md) — how Qortex parses entity labels
- [Local index](../download/local-index.md) — building a manifest from a local directory
