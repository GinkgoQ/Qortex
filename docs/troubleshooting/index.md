# Troubleshooting

This section covers the most common errors and how to resolve them.

[**Install**](install.md) — Dependency conflicts, missing extras, Python version issues.

[**OpenNeuro**](openneuro.md) — API errors, rate limiting, manifest fetch failures.

[**Downloads**](downloads.md) — CDN 403/410 errors, LFS pointers, incomplete files, resume failures.

[**BIDS**](bids.md) — Manifest parsing errors, missing entity labels, non-standard naming.

[**Labels**](labels.md) — Missing events.tsv, empty label columns, encoding issues.

[**Visualization**](visualization.md) — Missing extras, blank figures, memory errors.

[**Overlays**](overlays.md) — Shape mismatch, affine mismatch, resampling errors.

[**DICOM**](dicom.md) — Missing pydicom, compressed DICOM, multi-frame errors.

[**Artifacts**](artifacts.md) — Corrupted shards, missing manifest, feature count mismatch.

## Quick reference: common error codes

| Error | Location | Fix |
|-------|----------|-----|
| `NO_EVENTS` | doctor | events.tsv missing for all subjects |
| `EMPTY_EVENTS` | doctor | events.tsv exists but has no rows |
| `LFS_POINTER` | content-status | file is a Git LFS pointer, not real data |
| `AFFINE_MISMATCH` | overlay | background and overlay have different affines |
| `MISSING_EXTRA` | any | install the required extra (`pip install "qortex[mri]"`) |
| `HTTP_403` | download | CDN link expired — re-run to refresh |
| `BIDS_PARSE_ERROR` | index_local | file path does not follow BIDS naming |
| `SPLIT_TOO_SMALL` | convert | not enough subjects for val/test split |
