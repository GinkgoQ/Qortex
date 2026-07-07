# Artifact Troubleshooting

## artifact_manifest.json not found

```
ArtifactError: artifact_manifest.json not found in artifacts/ds004130/
```

The directory may not be a valid Qortex artifact. Check:

1. Did the conversion complete successfully? If it was interrupted, the manifest is written last and may be absent.
2. Is the path pointing to the artifact root (not to a split subdirectory)?

```python
# Wrong
art = Artifact.open("artifacts/ds004130/train/")

# Correct
art = Artifact.open("artifacts/ds004130/")
```

## Feature count mismatch after re-conversion

If you re-run conversion with different parameters and the feature count changes, any model trained on the old artifact will not work with the new one. Always check:

```python
old_art = Artifact.open("artifacts/ds004130_old/")
new_art = Artifact.open("artifacts/ds004130_new/")
print(len(old_art.manifest.feature_names))  # e.g., 491520
print(len(new_art.manifest.feature_names))  # must match for model reuse
```

## Shard integrity check fails

```
IntegrityError: shard-000003.parquet: expected 128 rows, found 127
```

A shard was written incompletely. Re-run conversion with `overwrite=True`:

```python
art = ds.convert(output_dir="artifacts/ds004130/", overwrite=True, ...)
```

## sklearn() returns wrong shape

If `art.sklearn()` returns X with unexpected shape, check the window parameters:

```python
print(art.manifest.window.duration_s)     # 30.0 seconds
print(art.manifest.feature_names[:3])     # ['Fp1_0', 'Fp1_1', 'Fp1_2']
# n_features = n_channels × (duration_s × sampling_rate)
```

## PyTorch DataLoader: workers=0 required

On some systems, PyTorch DataLoader with `num_workers > 0` fails when reading Parquet files. This is a multiprocessing / file handle issue.

Set `num_workers=0` as a workaround:

```python
loader = DataLoader(train_ds, batch_size=32, num_workers=0)
```

For production training, use Zarr format which is more compatible with multiprocessing:

```python
art = ds.convert(format="zarr", ...)
```




## Related

- [Conversion pipeline](../conversion/pipeline.md) — re-run conversion
- [Artifact inspect](../artifacts/inspect.md) — check artifact structure
- [Compare splits](../artifacts/compare-splits.md) — distribution checks
