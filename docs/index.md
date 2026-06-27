<div class="on-stats">
  <div class="on-stats-top">
    <div class="on-stats-source">
      <span class="on-stats-label">OpenNeuro · openneuro.org</span>
      <span class="on-live-badge"><span class="on-live-dot"></span>Live</span>
    </div>
    <div class="on-stats-numbers">
      <div class="on-stat">
        <strong id="on-public-datasets">—</strong>
        <span>Public datasets</span>
      </div>
      <div class="on-sep"></div>
      <div class="on-stat">
        <strong id="on-participants">—</strong>
        <span>Participants</span>
      </div>
    </div>
  </div>
  <div class="on-modalities-list" id="on-modalities"></div>
</div>

<section class="tq-hero">
  <div class="tq-hero-copy">
    <div class="tq-eyebrow">OpenNeuro · BIDS · ML-ready neurodata</div>
    <h1>Qortex</h1>
    <p>
      Qortex is a Python library for moving from an OpenNeuro dataset to a first
      ML batch. It reads the remote manifest before any download, plans the
      minimum transfer for a concrete goal, verifies local completeness, runs
      visual QC, converts to six ML-ready formats, and exposes provenance at
      every step.
    </p>
    <p>
      The design separates inspection from download, and download from conversion.
      Each stage can be run independently, checked, and interrupted without losing
      work.
    </p>
    <div class="tq-actions">
      <a class="tq-button primary" href="getting-started/quickstart/">Quickstart</a>
      <a class="tq-button secondary" href="concepts/what-is-qortex/">Read the concepts</a>
      <a class="tq-button secondary" href="api/">API reference</a>
    </div>
  </div>
  <div class="tq-hero-visual" role="img" aria-label="Qortex workflow from OpenNeuro manifest inspection to ML artifact">
    <div class="tq-diagram-head">
      <span>OpenNeuro → ML pipeline</span>
      <code>inspect · plan · download · verify · convert · train</code>
    </div>
    <div class="tq-page-card">
      <div class="tq-page-meta">
        <span>manifest first</span>
        <span>no wasted bytes</span>
        <span>provenance kept</span>
      </div>
      <div class="tq-slot-address">
        <span>inspect</span>
        <code>ds = Dataset("ds004130")  # no download yet</code>
      </div>
      <div class="tq-region-row">
        <span>subjects · sessions</span>
        <span>suffixes · events</span>
      </div>
      <div class="tq-component-row">
        <b>dl</b>
        <span class="tq-payload">download(subjects=["01","02"], suffixes=["bold"])</span>
        <span>2 subj</span>
      </div>
      <div class="tq-component-row">
        <b>cv</b>
        <span class="tq-payload">convert(output_format="parquet", window=2.0)</span>
        <span>artifact</span>
      </div>
      <div class="tq-byte-formula">
        <span><b>readiness</b><code>doctor · minimum · can-train · first-batch</code></span>
        <span><b>visual QC</b><code>one center slice per NIfTI — no full load</code></span>
        <span><b>formats</b><code>parquet · zarr · hdf5 · webdataset · hf · tfrecord</code></span>
      </div>
    </div>
    <div class="tq-route-line">
      <span>inspect</span>
      <i></i>
      <span>download</span>
      <i></i>
      <span>train</span>
    </div>
    <p>Readiness checks run at every stage. The manifest answers subject counts, event coverage, and total size before any data transfer.</p>
  </div>
</section>

## What Qortex does

Qortex is not a download manager. It answers: "can I train on this dataset, and what exactly do I need to download to do it?"

It reads the OpenNeuro manifest before any file transfer. From the manifest alone it can tell you how many subjects have T1w files, which tasks have events.tsv companions, what the total size of a filtered subset would be, and whether supervised training is feasible on a named label column.

After download it runs BIDS validation, builds ML-format artifacts with subject-level train/val/test splits, and provides visual QC tools that read one center slice from each NIfTI without loading the full volume.

<div class="tq-flow">
  <div>
    <b>1. Inspect</b>
    <span>Read the remote manifest. Get subject counts, modalities, event coverage, and total size — no download.</span>
  </div>
  <div>
    <b>2. Download</b>
    <span>Filter by subject, task, modality, or suffix. Resume automatically on interruption.</span>
  </div>
  <div>
    <b>3. Convert</b>
    <span>Write Parquet, Zarr, HDF5, or WebDataset with subject-level splits and full provenance.</span>
  </div>
  <div>
    <b>4. Train</b>
    <span>Load directly into PyTorch, scikit-learn, Lightning, HuggingFace, or BrainDecode.</span>
  </div>
</div>

## Current Status

| Area                | Status                                                                               |
| ------------------- | ------------------------------------------------------------------------------------ |
| Manifest inspection | Implemented. Reads remote file tree, builds typed Manifest with BIDS entities.       |
| Selective download  | Implemented. Filter by subject, session, task, modality, suffix, and size.           |
| BIDS validation     | Implemented. Wraps official BIDS Validator with caching and normalized JSON output.  |
| Catalog search      | Implemented. Local DuckDB catalog with free-text and structured filters.             |
| Readiness checks    | Implemented. doctor, minimum, can-train, first-batch, leakage-check, content-status. |
| Conversion          | Implemented. Parquet, Zarr, HDF5, WebDataset, HuggingFace, TFRecord.                 |
| ML adapters         | Implemented. PyTorch, Lightning, scikit-learn, HuggingFace, BrainDecode, Dask.       |
| Visual QC — NIfTI   | Implemented. Interactive HTML viewer, ortho, lightbox, fMRI QC, DWI QC.              |
| Visual QC — EEG/MEG | Implemented. Butterfly, PSD, spectrogram, epoched via MNE.                           |
| Visual QC — DICOM   | Implemented. Series browser with PHI protection.                                     |
| Surface rendering   | Not implemented. GIFTI/CIFTI falls through to summary-only output.                   |

## Where to start

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="getting-started/install/">Install</a></h3>
    <p>Base install, optional extras, and what each extras group enables.</p>
  </div>
  <div class="tq-card">
    <h3><a href="getting-started/quickstart/">Quickstart</a></h3>
    <p>End-to-end example: inspect ds004130, download two subjects, convert to Parquet, train a classifier.</p>
  </div>
  <div class="tq-card">
    <h3><a href="concepts/readiness-first/">Readiness concepts</a></h3>
    <p>Why Qortex checks usability before downloading and what the five readiness states mean.</p>
  </div>
  <div class="tq-card">
    <h3><a href="visualization/">Visualization</a></h3>
    <p>Visual QC for NIfTI, DICOM, EEG, and converted artifacts without loading full volumes.</p>
  </div>
</div>

## Reader paths

| Reader       | Suggested path                                                                                                        |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| Researcher   | [Concepts](concepts/index.md), [Readiness](readiness/index.md), [Visualization](visualization/index.md).              |
| ML engineer  | [Quickstart](getting-started/quickstart.md), [Conversion](conversion/index.md), [Artifacts](artifacts/index.md).      |
| Data curator | [Dataset inspection](dataset/index.md), [Download](download/index.md), [Visual audit](visualization/visual-audit.md). |
| Systems user | [CLI reference](api/cli.md), [Troubleshooting](troubleshooting/index.md).                                             |
