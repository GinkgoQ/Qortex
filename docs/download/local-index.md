# Local Index

If you already have a BIDS dataset on disk — from OpenNeuro CLI, DataLad, or your institution's storage — you can point Qortex at it without downloading anything.

`index_local()` scans the local directory and builds a manifest that Qortex uses for all subsequent operations.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="/scratch/datasets/ds004130/")
ds.index_local()

# Now all readiness and conversion commands work
report = ds.doctor()
art = ds.convert(output_dir="artifacts/")
```

## CLI

```bash
qortex local-index /scratch/datasets/ds004130/ --dataset-id ds004130
```

After indexing, pass `--local` to other commands to use the local manifest:

```bash
qortex inspect ds004130 --local
qortex doctor ds004130 --local
```

## What local indexing does

1. Walks the directory tree recursively
2. Parses BIDS entity labels from each file path
3. Computes file sizes from disk
4. Resolves companion relationships
5. Writes `.qortex/manifest.json` inside the data directory

The local manifest does not include CDN URLs. If you want to download missing files after indexing, Qortex fetches the remote manifest and cross-references with the local index.

## Checking for missing files after indexing

```python
ds.index_local()
report = ds.doctor()
# report.missing_files lists files in the remote manifest that are absent locally
```

Or use visual audit:

```python
report = ds.visual_audit(mode="local")
for item in report.missing_expected_files():
    print(item)
```

## Limitations

- Local indexing does not validate BIDS structure. It only parses entity labels from path names. Run `ds.validate()` for structural validation.
- Files that do not follow BIDS naming conventions are indexed but have None for most entity fields. They appear in the manifest with unknown datatype and suffix.
- Symlinks are followed during indexing. If the link target does not exist, the file is skipped.

## Related

- [Cache](cache.md) — where Qortex stores downloaded files
- [Validate](../readiness/doctor.md) — check the local BIDS structure after indexing
