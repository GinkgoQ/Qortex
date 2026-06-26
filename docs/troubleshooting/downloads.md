# Download Troubleshooting

## HTTP 403 or 410: link expired

OpenNeuro CDN links expire after a few hours. If you see:

```
DownloadError: HTTP 403 for sub-01_task-rest_bold.nii.gz
```

Re-run the download command. Qortex fetches fresh CDN URLs before each download session.

```python
ds.download(subjects=["01"], data_dir="data/")
```

## Files are Git LFS pointers

If a downloaded file is 134 bytes and contains `version https://git-lfs.github.com/spec/`, it is an LFS pointer, not real data. This happens when a dataset is cloned via DataLad or Git without running `git-annex get`.

Check with content-status:

```bash
qortex content-status ds004130 --data-dir data/ds004130/
```

Fix:
1. If you used DataLad: `datalad get data/ds004130/sub-01/`
2. If you used Qortex download and got pointers: re-run download with `force=True`

```python
ds.download(subjects=["01"], force=True, data_dir="data/ds004130/")
```

Some datasets on OpenNeuro have LFS pointer issues at the source. If Qortex downloads return pointers even with `force=True`, the CDN itself is serving pointers. Report this to OpenNeuro.

## Incomplete download (size mismatch)

After a download, content-status shows a file is smaller than expected:

```
INCOMPLETE: sub-05/eeg/sub-05_task-rest_eeg.set (312 MB / 450 MB expected)
```

The download was interrupted mid-file. Re-run to resume:

```python
ds.download(subjects=["05"], data_dir="data/ds004130/")
```

Qortex detects the `.part` file and continues from byte 312 MB.

## Disk full during download

If disk fills during a download, partial files are left in place. After freeing space, re-run the download — partial files resume automatically.

## Very slow downloads

OpenNeuro CDN throughput varies by region and time of day. For large datasets (> 50 GB), consider:

1. Increase concurrency: `qortex download --concurrency 8`
2. Run the download overnight when CDN load is lower
3. Use AWS `--region` if the CDN is in a specific S3 region near you

## Related

- [Resume](../download/resume.md) — how resume works
- [Cache](../download/cache.md) — where files are stored
