# Visualize OpenNeuro

The `visualize-openneuro` CLI command renders center-slice thumbnails for an OpenNeuro dataset without downloading full files. It fetches only the bytes needed to render one slice from each NIfTI.

## How it works

NIfTI files on the OpenNeuro CDN support HTTP range requests. Qortex reads the NIfTI header (first 352 bytes) to get voxel dimensions and data offset, then fetches only the bytes corresponding to the center axial slice. For a 256×256×176 T1w at float32, that is roughly 256 KB per slice instead of 30 MB for the full volume.

For EEG and other non-NIfTI formats, full-file download is still required because the formats do not support range reads. The command skips non-NIfTI files unless `--include-all` is passed.

## CLI

```bash
qortex visualize-openneuro ds004130
qortex visualize-openneuro ds004130 --subjects 01 02 --suffixes T1w bold
qortex visualize-openneuro ds004130 --output figures/ --format png
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--subjects` | all | Subjects to include |
| `--suffixes` | T1w, bold, dwi | File suffixes to render |
| `--snapshot` | latest | Snapshot version |
| `--output` | . | Directory for output files |
| `--format` | html | `html` or `png` |
| `--include-all` | False | Include non-NIfTI files (downloads full file) |

## Python

```python
from qortex.visualize import visualize

results = visualize("openneuro://ds004130/sub-01/anat/sub-01_T1w.nii.gz")
results[0].show()
```

The `openneuro://` URI scheme triggers remote rendering. Pass a manifest FileRecord instead to avoid constructing the URI manually:

```python
from qortex import Dataset
from qortex.visualize import visualize

ds = Dataset("ds004130")
files = ds.files(subjects=["01"], suffixes=["T1w"])
results = [visualize(f) for f in files]
```

## Performance

A center-slice thumbnail fetch takes 1–3 seconds per file on a typical broadband connection. The limiting factor is CDN latency, not bandwidth. Batch requests are parallelized up to 8 concurrent fetches.

## Limitations

- Range reads only work for NIfTI. DICOM, EEG .set/.fif, and CIFTI files require full downloads.
- The NIfTI must have a valid header at byte 0. Some older neuroimaging software writes malformed NIfTI headers. These files fail with `InvalidNIfTIHeaderError` and are skipped.
- Range reads bypass Qortex's local cache. Each call fetches from CDN. If you plan to inspect many slices repeatedly, download the files first.
