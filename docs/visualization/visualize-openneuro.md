# Visualize OpenNeuro

Use `qortex visualize-openneuro` when you need to inspect one or a few OpenNeuro files without downloading the full dataset. The command selects matching manifest records, downloads only those files to the Qortex cache or an output directory, then renders a visual audit report.

For remote NIfTI slice work in Python, use `Dataset.stream_slice()`. It reads header metadata and the requested slice through Qortex streaming code instead of requiring a full BOLD file download.

## Real Example

This figure was generated from public OpenNeuro dataset `ds000001`, subject `01`, run `01`:

```python
from qortex import Dataset

ds = Dataset("ds000001")
info = ds.nifti_info("sub-01/func/sub-01_task-balloonanalogrisktask_run-01_bold.nii.gz")
slice_2d = ds.stream_slice(subject="01", modality="bold", run="01", time_index=0, axis=2)

print(info)
print(slice_2d.shape, slice_2d.dtype)
```

Observed output:

```text
4D fMRI 64×64×33×300 vox=3.12×3.12×4.00mm TR=2.000s
(64, 64) int16
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-bold-axial.png" alt="Axial BOLD slice streamed from OpenNeuro ds000001 sub-01 without downloading the full NIfTI file">
  <figcaption>Real BOLD center slice streamed from OpenNeuro. The generator script saved this PNG after calling `Dataset.stream_slice()`; the full 47 MB BOLD file was not downloaded for the Python example.</figcaption>
</figure>

## CLI

```bash
qortex visualize-openneuro ds000001 --subject 01 --suffix bold --datatype func --output ds000001-bold.html
qortex visualize-openneuro ds000001 --subject 01 --suffix T1w --open
qortex visualize-openneuro ds000001 --datatype func --suffix bold --output-dir data/ds000001-viz
```

Options:

| Option | Default | Description |
|---|---:|---|
| `--subject`, `-s` | any | Subject ID without `sub-`. |
| `--suffix` | any visual suffix | BIDS suffix such as `T1w`, `bold`, or `dwi`. |
| `--datatype`, `-d` | any | BIDS datatype folder such as `anat`, `func`, or `dwi`. |
| `--output`, `-o` | auto | HTML output path. |
| `--output-dir` | Qortex cache | Destination for the selected downloaded file. |
| `--mode`, `-m` | `auto` | Rendering mode: `auto`, `qc`, `thumbnail`, `interactive`, or `static`. |
| `--max-size-mb` | `500` | Skip matching files larger than this limit. |
| `--n-per-suffix` | `1` | Number of files to render per suffix. |
| `--open` | false | Open the HTML report in a browser. |

## When To Use Which Path

| Need | Use |
|---|---|
| One streamed slice for docs, notebooks, or lightweight QC | `Dataset.stream_slice()` |
| Header facts such as shape, voxel size, TR, and number of volumes | `Dataset.nifti_info(path)` |
| A browser report for selected OpenNeuro files | `qortex visualize-openneuro` |
| Repeated interactive inspection of the same large file | Download the file once, then use the local viewer |

## Reproduce The Figure

```bash
python scripts/generate_docs_examples.py
```

The script writes the PNG to `docs/assets/images/examples/ds000001-bold-axial.png` and the numeric metadata to `docs/assets/results/ds000001-example-results.json`.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-bold-axial.png" alt="Axial BOLD slice from OpenNeuro ds000001 subject 01 run 01.">
  <figcaption>Real BOLD axial slice streamed with `Dataset.stream_slice()` without downloading the full NIfTI file.</figcaption>
</figure>

```python
sl = ds.stream_slice(subject='01', modality='bold', run='01', time_index=0, axis=2)
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->
