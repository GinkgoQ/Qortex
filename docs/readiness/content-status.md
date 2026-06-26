# Content Status

`ds.content_status()` checks the local files after download. It answers: are the files real data, or are some of them Git LFS pointers or incomplete transfers?

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130", data_dir="data/ds004130/")
status = ds.content_status()
print(status.to_text())
```

Output:

```
Content status: ok
  Files checked: 1,056
  Size on disk:  4.2 GB
  LFS pointers: 0
  Incomplete:    0
  Unreadable:    0
```

If problems are found:

```
Content status: issues_found
  Files checked: 1,056
  LFS pointers: 3
    sub-03/func/sub-03_task-rest_bold.nii.gz  (134 bytes — pointer)
    sub-07/func/sub-07_task-rest_bold.nii.gz  (134 bytes — pointer)
    sub-12/func/sub-12_task-rest_bold.nii.gz  (134 bytes — pointer)
  Incomplete: 1
    sub-05/eeg/sub-05_task-rest_eeg.set (312 MB / 450 MB expected)
```

## CLI

```bash
qortex content-status ds004130 --data-dir data/ds004130/
```

## What is checked

**LFS pointer detection.** A file is flagged as an LFS pointer if its size is between 100 and 200 bytes and its content starts with `version https://git-lfs.github.com/spec/`. These files look like real files to the OS but contain no imaging data.

**Incomplete transfers.** Files whose on-disk size is more than 5% smaller than the manifest-reported size. These may be partial downloads interrupted before completion.

**Unreadable files.** Files where a basic read (first 512 bytes) raises an IOError. Rare — usually indicates filesystem corruption.

**Missing files.** Files in the manifest that have no local counterpart. These are listed separately as "not downloaded."

## Fixing LFS pointers

LFS pointers appear when a dataset was downloaded through DataLad or Git annex without running `git-annex get` or `datalad get`. The solution depends on how the dataset was obtained:

```bash
# DataLad
datalad get sub-03/func/sub-03_task-rest_bold.nii.gz

# Git annex
git annex get sub-03/func/sub-03_task-rest_bold.nii.gz
```

Or re-download through Qortex:

```python
ds.download(subjects=["03", "07", "12"], force=True, data_dir="data/ds004130/")
```

## Related

- [Resume](../download/resume.md) — fix incomplete downloads
- [Doctor](doctor.md) — includes a content status check when data_dir is set
