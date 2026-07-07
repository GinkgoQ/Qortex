# First Visual Audit

A visual audit is a fast QC pass for local BIDS files. It answers:

1. Which visualizable files are present?
2. Which subjects or suffixes are missing?
3. Do the thumbnails reveal obvious geometry, intensity, overlay, or loading problems?

It reads one representative slice per NIfTI file for thumbnail mode, so it is safe to run before expensive conversion.

## Real Rendered Example

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-bold-axial.png" alt="Axial BOLD slice from OpenNeuro ds000001 subject 01 run 01.">
  <figcaption>OpenNeuro `ds000001` BOLD slice rendered through `Dataset.stream_slice()`. A local visual audit uses the same visual stack across many files and writes an HTML report.</figcaption>
</figure>

## Requirements

```bash
pip install "qortex[visual-all]"
```

## Run Against A Local BIDS Directory

```python
from qortex import Dataset

ds = Dataset("ds000001")
report = ds.visual_audit(
    output_dir="qc/ds000001",
    local_path="data/ds000001",
    suffixes=["bold"],
    max_files=8,
)
print(report.summary())
report.to_html("qc/ds000001/visual_audit.html")
```

The report includes:

- a coverage matrix
- per-suffix counts
- per-subject counts
- warning summaries
- rendered thumbnails
- action items for missing or failed files

## CLI

```bash
qortex visual-audit ds000001 \
  --local data/ds000001 \
  --output-dir qc/ds000001 \
  --suffixes bold \
  --max-files 8
```

Add `--open` to launch the generated HTML report in your browser.

## Manifest-Aware Use

When you pass `local_path`, Qortex compares local files to the OpenNeuro manifest. This lets the report distinguish a file that does not exist in the dataset from a file that should exist but has not been downloaded.

## Save Report Formats

```bash
qortex visual-audit ds000001 \
  --local data/ds000001 \
  --output-dir qc/ds000001 \
  --json \
  --markdown \
  --manifest-json
```

## Next Steps

- [Visual audit reference](../visualization/visual-audit.md)
- [fMRI QC](../visualization/fmri-qc.md)
- [Visualize OpenNeuro](../visualization/visualize-openneuro.md)
- [Download](../download/index.md)








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
