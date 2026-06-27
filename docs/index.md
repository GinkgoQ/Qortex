<section class="tq-hero">
  <div class="tq-hero-strip">
    <div class="tq-strip-source">
      <span class="on-stats-label">OpenNeuro · openneuro.org</span>
      <span class="on-live-badge"><span class="on-live-dot"></span>Live</span>
    </div>
    <div class="tq-strip-numbers">
      <div class="tq-strip-stat">
        <strong id="on-public-datasets">—</strong>
        <span>datasets</span>
      </div>
      <div class="tq-strip-sep"></div>
      <div class="tq-strip-stat">
        <strong id="on-participants">—</strong>
        <span>participants</span>
      </div>
    </div>
    <div class="on-modalities-list" id="on-modalities"></div>
  </div>

  <div class="tq-hero-copy">
    <div class="tq-eyebrow">Python · OpenNeuro · BIDS</div>
    <h1>Qortex</h1>
    <ul class="tq-feature-list">
      <li><code>can_train("trial_type")</code> — manifest verdict before any download</li>
      <li><code>minimum("first-batch")</code> — exact files and byte count for a concrete ML goal</li>
      <li><code>doctor(recipe)</code> — 15-point BIDS readiness check: events, companions, label coverage, splits</li>
      <li>Selective download by subject · task · modality · suffix — resumable, size-limited</li>
      <li>Parquet · Zarr · HDF5 · WebDataset · HuggingFace · TFRecord — subject-level splits, leakage verified</li>
      <li>MRI · fMRI · DWI · PET · MEG · EEG · iEEG · fNIRS — uniform inspect / convert API</li>
    </ul>
    <div class="tq-actions">
      <a class="tq-button primary" href="getting-started/quickstart/">Quickstart</a>
      <a class="tq-button secondary" href="concepts/what-is-qortex/">Concepts</a>
      <a class="tq-button secondary" href="api/">API reference</a>
    </div>
  </div>

  <div class="tq-hero-visual" role="presentation">
    <div class="tq-code-header">
      <span class="tq-code-dot"></span><span class="tq-code-dot"></span><span class="tq-code-dot"></span>
      <span class="tq-code-filename">session.py</span>
    </div>
<pre class="tq-code-block"><code><span class="tq-ck">from</span> qortex <span class="tq-ck">import</span> Dataset

ds = Dataset(<span class="tq-cs">"ds004130"</span>)
<span class="tq-c"># 88 subjects · MRI + EEG · 156 files · 4.8 GB</span>
<span class="tq-c"># no download yet</span>

ds.can_train(<span class="tq-cs">"trial_type"</span>)
<span class="tq-c"># True  (2 classes · 240 windows · 86 subjects)</span>

ds.minimum(<span class="tq-cs">"first-batch"</span>)
<span class="tq-c"># 2 subjects · 4 files · 0.81 GB</span>

ds.download(subjects=[<span class="tq-cs">"01"</span>, <span class="tq-cs">"02"</span>], datatypes=[<span class="tq-cs">"func"</span>])
<span class="tq-c"># ████████████  100%  831 MB</span>

art = ds.convert(
    format=<span class="tq-cs">"parquet"</span>,
    window=<span class="tq-ck">dict</span>(mode=<span class="tq-cs">"event_aligned"</span>, tmin=<span class="tq-cn">-0.2</span>, tmax=<span class="tq-cn">0.8</span>),
    label_col=<span class="tq-cs">"trial_type"</span>,
)
<span class="tq-c"># train 180 · val 30 · test 30 samples</span>

X, y = art.sklearn(split=<span class="tq-cs">"train"</span>)
<span class="tq-c"># X: (180, 64, 256)   y: (180,)</span></code></pre>
  </div>
</section>

## Key interfaces

Three phases. Each is independent — inspect without downloading, download without converting, convert in any format.

**Before download — remote manifest only**

| Call | Returns |
|------|---------|
| `Dataset("ds_id")` | Manifest: subjects, files, entities — no data transferred |
| `ds.can_train(target_col)` | `True/False` + blocking reasons |
| `ds.minimum(goal)` | File list + byte count for `"first-batch"`, `"label-check"`, `"validation"` |
| `ds.label_landscape()` | Per-class trial counts and subject coverage across all events.tsv |
| `ds.doctor()` | Full readiness report — events, companions, sizes, split feasibility |
| `DatasetQuery().modality("eeg").has_events().min_subjects(30).fetch()` | Filtered catalog results |
| `client.search_datasets_rich(modality="MRI", sort_by="downloads")` | Live API results with engagement |

**After selective download**

| Call | Returns |
|------|---------|
| `ds.download(subjects, datatypes, suffixes, max_size_gb)` | Resumable selective transfer |
| `ds.doctor(recipe="eeg-classification")` | Modality-specific checks post-download |
| `ds.visual_audit(output_dir)` | Center-slice thumbnails — no full volume loaded |
| `ds.get_validation_issues(tag)` | BIDS validator errors and warnings |

**Convert to artifact**

| Call | Returns |
|------|---------|
| `ds.convert(format, window, label_col, split)` | Artifact with subject-level train/val/test split |
| `art.sklearn(split)` | `(X, y)` numpy arrays |
| `art.torch(split)` | PyTorch `Dataset` |
| `art.huggingface(split)` | HuggingFace `Dataset` |
| `art.braindecode(split)` | BrainDecode `BaseConcatDataset` |

## Implementation status

| Area | Status |
|------|--------|
| Manifest inspection | Remote file tree, typed Manifest, BIDS entities extracted |
| Selective download | Subject · session · task · modality · suffix · size filters; automatic resume |
| BIDS validation | Wraps official validator; cached, normalized JSON output |
| Catalog search | Local DuckDB + live API; `DatasetQuery` fluent builder with 10 filters |
| Readiness checks | `doctor` · `minimum` · `can-train` · `first-batch` · `leakage-check` · `content-status` |
| Conversion | Parquet · Zarr · HDF5 · WebDataset · HuggingFace · TFRecord |
| ML adapters | PyTorch · Lightning · scikit-learn · HuggingFace · BrainDecode · Dask |
| Visual QC — NIfTI | Ortho · lightbox · fMRI QC · DWI QC · PET overlay · one center slice per file |
| Visual QC — EEG/MEG | Butterfly · PSD · spectrogram · topomap · epoched previews via MNE |
| Visual QC — DICOM | Series browser with PHI protection |
| Surface rendering | Not implemented — GIFTI/CIFTI falls to summary-only |

## Where to start

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="getting-started/quickstart/">Quickstart</a></h3>
    <p>End-to-end: inspect ds004130, run readiness checks, download two subjects, convert to Parquet, load into PyTorch.</p>
  </div>
  <div class="tq-card">
    <h3><a href="concepts/readiness-first/">Readiness first</a></h3>
    <p>Why the manifest answers most questions before any file transfer, and how each readiness stage gates the next.</p>
  </div>
  <div class="tq-card">
    <h3><a href="modalities/">Modalities</a></h3>
    <p>MRI · fMRI · DWI · PET · MEG · EEG · iEEG · fNIRS — what each modality looks like in BIDS and what Qortex can do with it.</p>
  </div>
  <div class="tq-card">
    <h3><a href="api/cli/">CLI reference</a></h3>
    <p>All commands: <code>search</code> · <code>inspect</code> · <code>doctor</code> · <code>download</code> · <code>convert</code> · <code>visualize</code> · <code>can-train</code>.</p>
  </div>
</div>
