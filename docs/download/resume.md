# Resume

Downloads interrupted by network errors, timeouts, or Ctrl-C resume automatically. Qortex uses HTTP range requests to continue from where a transfer stopped.

## How it works

Qortex tracks download progress per file using a `.part` file alongside the destination path. When a download resumes:

1. The `.part` file is detected at the target location
2. Qortex issues a `Range: bytes={size}-` HTTP request for the remaining bytes
3. The partial content is appended to the `.part` file
4. On completion, the `.part` file is atomically moved to the final path

If the CDN does not support range requests (rare), Qortex falls back to restarting the file from the beginning.

## Resuming a stopped download

Simply re-run the same download command:

```python
ds.download(subjects=["01", "02"], data_dir="data/ds004130/")
# stopped at 60% of sub-02 EEG file

# Run again — picks up from where it stopped
ds.download(subjects=["01", "02"], data_dir="data/ds004130/")
```

Files already fully downloaded are skipped. The partial file resumes.

## Parallel downloads

Qortex uses a thread pool for concurrent file transfers. The default concurrency is 4 simultaneous downloads. Override with:

```python
ds.download(subjects=["01"], concurrency=8, data_dir="data/")
```

Or set in configuration:

```bash
qortex download ds004130 --output-dir data/ds004130
```

Higher concurrency does not always improve throughput — CDN rate limiting may kick in above 4–8 concurrent streams.

## Handling CDN errors

OpenNeuro CDN links expire after a few hours. If a download fails with HTTP 403 or 410, the link has expired. Re-running `ds.download()` fetches fresh links before retrying.

Qortex automatically refreshes expired CDN links during a resume — you do not need to restart from scratch.

## Integrity check after download

After all files complete, Qortex checks that each file's size on disk matches the expected size from the manifest. Files with size mismatches are logged as warnings:

```
WARNING: sub-03_task-rest_bold.nii.gz size mismatch: expected 450 MB, got 312 MB
```

To re-download size-mismatched files:

```python
ds.download(subjects=["03"], force=True, data_dir="data/")
```

`force=True` re-downloads even files that already exist at the correct size.

## Limitations

- Checksums are not verified after download. Qortex only checks file size. A file with the right size but corrupted content will pass.
- If the manifest changes between the initial download and a resume (e.g., a snapshot was updated), some CDN URLs may differ. Pin a snapshot to avoid this.
