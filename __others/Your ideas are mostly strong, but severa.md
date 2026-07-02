Your ideas are mostly strong, but several are **too optimistic as written**. The right Qortex contribution is not “semantic intelligence” in the LLM sense. It should be **evidence-grounded consistency validation**: deterministic checks over files, headers, BIDS entities, sidecars, signal/image statistics, event tables, splits, provenance, and runtime observations.

Also, be careful with the claim “no existing tool does this.” Parts already exist: BIDS Validator validates BIDS compliance; MRIQC extracts no-reference image quality metrics from structural/functional MRI; fMRIPrep provides robust fMRI preprocessing and visual reports; BIDS defines detailed rules for events, fMRI timing, DWI gradients, coordinate systems, and units. The real gap is **not that nothing exists**. The gap is that no single lightweight library gives **goal-aware, ML-safety-oriented, cross-modal, evidence-propagating preflight checks** before visualization, conversion, training, and runtime. ([bids-validator.readthedocs.io][1])

# Overall verdict

| Idea                                     |                     Verdict |                                                    Implement? |
| ---------------------------------------- | --------------------------: | ------------------------------------------------------------: |
| Evidence-propagating semantic validation |          **Valid approach** |                                                       **Now** |
| Semantic `SourceProfile` probe           |          **Valid approach** |                                                       **Now** |
| Signal-statistics unit inference         |         **Partially valid** |                                         Later, heuristic only |
| Preprocessing state estimation           |         **Partially valid** |                                       Limited now, full later |
| Cross-subject cohort anomaly detection   |          **Valid approach** |                                                       **Now** |
| Confound graph auto-construction         |         **Partially valid** |                    Now as association graph, not causal graph |
| Signal-event coherence check             | **Partially valid / risky** |                                    Later, optional diagnostic |
| Frame propagation through transforms     |          **Valid approach** |                                                       **Now** |
| DWI gradient frame auto-detection        |         **Weak as written** | Avoid auto-detection; implement deterministic gradient checks |
| Preprocessing fingerprint compatibility  |          **Valid approach** |                                                       **Now** |
| Leakage-safe confound-balanced splitting |          **Valid approach** |                                                       **Now** |
| Signal budget / input coverage metric    |          **Valid approach** |                                                       **Now** |
| Memory-safe streaming estimator          |          **Valid approach** |                                                       **Now** |
| Unified `ReadinessScore`                 |         **Partially valid** |                                Later; report dimensions first |
| Runtime → probe feedback loop            |      **Valid but advanced** |                                                         Later |

The strongest immediate Qortex features are:

1. **Evidence-propagating `SourceProfile`**
2. **Header–sidecar consistency checks**
3. **Companion-file closure checks**
4. **Cohort anomaly detection**
5. **Leakage-safe splitting**
6. **Frame/axis/unit/timebase tracking**
7. **Preprocessing fingerprints**
8. **Goal-aware preflight reports**

The weakest or riskiest ideas are:

1. Automatic PET unit inference from value range.
2. Automatic DWI gradient frame detection as described.
3. fMRI event-signal coherence as a hard validator.
4. A single global readiness score if users treat it as truth.

---

# 1. Evidence-propagating semantic validation

**Verdict: Valid approach. Implement now.**

This is the core idea and it is strong.

## What it solves

BIDS Validator can tell you whether files follow the BIDS specification, but it does not prove that the dataset is safe for model training, conversion, or runtime inference. For example, BIDS defines `events.tsv` onset/duration rules, fMRI `RepetitionTime` / `VolumeTiming`, DWI `.bval` / `.bvec`, coordinate systems, and unit formatting, but Qortex can go further by checking whether the files, headers, sidecars, and downstream workflow assumptions agree. ([bids-specification.readthedocs.io][2])

## Inputs needed

```text
manifest files
BIDS entities
sidecar JSON
TSV files
raw headers
NIfTI headers
MNE-readable signal headers
DICOM metadata where available
event tables
channel tables
electrode/coordinate files
planned workflow goal
```

## Deterministic checks

```text
header value == sidecar value
sidecar applies to expected files through inheritance
events fit recording duration
channel count matches raw file
bvec/bval length matches DWI volumes
coordinate system declared when coordinates exist
units are valid BIDS/SI-style strings
split groups prevent leakage
```

## Output

```python
EvidenceItem(
    field="SamplingFrequency",
    claimed_value=512,
    observed_value=500,
    evidence_status="contradicted",
    source="raw_header",
    severity="blocker",
)
```

## Failure modes

It cannot prove biological correctness. It can only prove consistency, missingness, contradiction, plausibility, and risk.

## Qortex decision

This should be Qortex’s foundation. Rename it from “semantic validation” to:

```text
Evidence-grounded neurodata validation
```

That avoids implying vague LLM reasoning.

---

# 2. Semantic `SourceProfile`

**Verdict: Valid approach. Implement now.**

## What it solves

A `SourceProfile` should not merely say:

```python
sampling_rate_hz=512
```

It should say:

```python
sampling_rate_hz=512
evidence="confirmed_from_raw_header"
sidecar_value=512
events_compatible=True
```

## Inputs needed

```text
raw header
sidecar JSON
channels.tsv
events.tsv
participants.tsv
file path entities
companion files
```

## Checks

```text
read header without full data load
cross-check sidecar against header
check required companion files
check declared units
check channel count/name/type consistency
check duration against event onsets
```

## Output

```python
SourceProfile(
    sampling_rate_hz=512,
    evidence={
        "sampling_rate_hz": "confirmed",
        "channels": "confirmed",
        "events": "compatible",
        "units": "claimed_only",
    },
    warnings=[...],
)
```

## Where it fails

Some formats do not expose complete metadata. Some OpenNeuro datasets are metadata-only until files are downloaded. Live streams may be unknowable before connection.

## Qortex decision

Implement now. This is low-risk and high-value.

---

# 3. Signal-statistics-driven unit inference

**Verdict: Partially valid approach. Useful, but do not auto-correct.**

The idea is attractive, but it is dangerous if Qortex pretends it can infer true units from value ranges alone.

## What it solves

It targets real failures:

```text
EEG values stored in volts but interpreted as microvolts
PET values stored as SUV-like normalized values but labeled Bq/mL
fNIRS values already converted to optical density/HbO/HbR but treated as raw intensity
MRI files corrupted or constant-valued
```

BIDS does care about units and recommends SI/CMIXF-style representations, including `uV`/`µV` for microvolts and `Bq` for radioactivity; so unit validation is absolutely a Qortex concern. ([bids-specification.readthedocs.io][3])

## Deterministic checks that can work

For EEG:

```text
amplitude quantiles
median absolute deviation
peak-to-peak amplitude
unit declared in channels.tsv
raw file physical dimension if available
MNE channel calibration metadata
```

For MRI:

```text
min/max
percentiles
constant image detection
NaN/Inf detection
nonzero fraction
brain/background ratio
```

For PET:

```text
declared Units
tracer metadata
FrameTimesStart / FrameDuration
ImageDecayCorrected
ImageDecayCorrectionTime
SUV-related metadata if present
```

PET is especially metadata-dependent. BIDS PET requires `TimeZero`, `ScanStart`, `InjectionStart`, frame timing arrays, decay-correction flags, and reconstruction metadata; value range alone is not enough to infer whether data are Bq/mL, kBq/mL, SUV, or scanner-normalized values. ([bids-specification.readthedocs.io][4])

## Correct output

Not:

```python
units = "V"  # corrected automatically
```

Better:

```python
UnitPlausibilityCheck(
    claimed_unit="uV",
    observed_scale="volts_like",
    severity="warning",
    confidence=0.72,
    action="manual_confirmation_required",
)
```

## Where it may fail

- EEG amplitudes vary by preprocessing, referencing, amplifier scaling, and artifact contamination.
- PET intensities depend on tracer, reconstruction, decay correction, body weight, and normalization.
- fNIRS values depend on whether data are raw intensity, optical density, HbO/HbR, or filtered derivatives.
- MRI intensities are usually arbitrary and scanner/protocol dependent.

## Qortex decision

Implement later as **plausibility warnings**, not as validation truth. Valid for obvious scale errors and corruption detection; invalid as automatic unit correction.

---

# 4. Preprocessing state estimation without documentation

**Verdict: Partially valid approach. Implement only narrow checks now.**

## What it solves

Real OpenNeuro datasets often lack complete preprocessing history. You want to infer whether data may already be filtered, notch-filtered, normalized, cropped, defaced, resampled, or otherwise transformed.

## Strong deterministic checks

These are feasible now:

```text
DWI bvec norms near 1 or 0
DWI volume count == bval count == bvec columns
MRI all-zero/all-constant detection
MRI NaN/Inf detection
EEG flatline channel detection
EEG saturated channel detection
EEG line-noise power around 50/60 Hz
NIfTI shape/affine/voxel-size distribution
fMRI first-volume intensity discontinuity
```

DWI gradient checks are especially valid because BIDS explicitly requires `.bvec` to have 3 rows and N columns matching N volumes, with each vector unit norm or zero for b=0 volumes. ([bids-specification.readthedocs.io][5])

## Weaker checks

These are possible but should be marked uncertain:

```text
notch filter detection from PSD notch depth
high-pass cutoff estimation from PSD rolloff
dummy scan removal from first-volume intensity
fNIRS preprocessing state from amplitude distribution
PET SUV-vs-Bq/mL from value range
```

## Output

```python
EstimatedPreprocessingState(
    field="line_noise",
    estimate="notch_filter_likely",
    evidence="PSD shows narrow suppression at 50 Hz",
    confidence=0.68,
    severity="info",
)
```

## Where it fails

- A low 50/60 Hz peak does not prove notch filtering; it may reflect shielding, preprocessing, or low line noise.
- A high DC component does not prove DC offset was not removed; reference and montage matter.
- fMRI first volumes may vary because of motion, acquisition, or task effects, not only dummy scans.
- PSD-based filter estimation can be misleading for short/noisy recordings.

## Qortex decision

Implement a small `PreprocessingStateAudit` now with conservative labels:

```text
confirmed
likely
possible
unknown
contradicted
```

Do not use it as a blocker unless the evidence is hard, such as malformed DWI gradients or constant images.

---

# 5. Cross-subject statistical anomaly detection

**Verdict: Valid approach. Implement now.**

This is one of the best ideas.

## What it solves

It catches semantically suspicious but BIDS-valid datasets:

```text
one subject has different voxel size
one run has half the DWI directions
half the EEG files are 256 Hz and half are 512 Hz
one site has different scanner model
one fMRI run has different TR
one PET subject has abnormal frame durations
```

BIDS allows many acquisition parameters to vary across runs/datasets, so cohort-relative anomaly detection is a practical complement to BIDS validation. For example, fMRI timing metadata and DWI gradient schemes are explicitly structured, but the spec does not tell you whether one subject’s valid value is suspicious relative to the cohort. ([bids-specification.readthedocs.io][5])

## Inputs needed

```text
SourceProfile per file/run
subject/session/run entities
voxel size
shape
TR
sampling rate
channel count
bval shell counts
frame durations
scanner/site metadata
label distribution
```

## Algorithms

```text
exact uniqueness counts
mode/majority comparison
IQR outlier detection
MAD robust z-score
bimodality detection
categorical entropy
per-site grouped summaries
```

## Output

```python
CohortAnomaly(
    field="sampling_rate_hz",
    majority_value=512,
    outlier_subjects=["sub-17", "sub-21"],
    observed_values={"512": 47, "256": 2},
    severity="warning",
)
```

## Where it may fail

Some datasets intentionally combine protocols. Qortex should not call this “wrong”; it should call it “heterogeneous” and explain downstream risk.

## Qortex decision

Implement now. This is deterministic, generalizable, and valuable.

---

# 6. Confound graph auto-construction

**Verdict: Partially valid approach. Implement as association graph, not causal graph.**

The idea is valuable, but the proposed directed graph is too strong.

## What it solves

It detects ML risks like:

```text
diagnosis nearly equals site
scanner model nearly equals age group
field strength differs by label
TR differs by class
sex imbalance across splits
```

This is a real ML safety issue. Subject-level leakage and confounding are especially dangerous in neurodata, and EEG/MRI studies repeatedly show that subject-wise splitting and data partitioning affect reliability and can inflate performance. ([arXiv][6])

## Inputs needed

```text
participants.tsv
sessions.tsv
scans.tsv
SourceProfile fields
site/scanner fields if available
target label
split assignments
```

## Valid deterministic algorithms

```text
mutual information
Cramér's V for categorical-categorical
ANOVA / Kruskal-Wallis for numeric-vs-categorical
correlation for numeric-numeric
standardized mean difference
KL / Jensen-Shannon divergence across splits
chi-square expected-count checks
```

## Correct output

Not:

```text
site → scanner_model → diagnosis
```

That implies causality.

Better:

```python
ConfoundAssociation(
    target="diagnosis",
    variable="site",
    association_metric="normalized_mutual_information",
    score=0.89,
    severity="high",
    interpretation="diagnosis is strongly associated with site; classifier may learn site-specific acquisition instead of biology",
)
```

## Where it may fail

- Small sample sizes make association estimates unstable.
- Missing site/scanner metadata reduces usefulness.
- Continuous covariates need binning or robust tests.
- Association does not prove causality.

## Qortex decision

Implement now as `ConfoundAssociationGraph`, not `ConfoundGraph`. Avoid directed causal arrows unless the direction is explicitly from metadata ontology, such as `scanner_model` belongs to `site`.

---

# 7. Signal-event coherence check

**Verdict: Partially valid, but risky. Implement later as optional diagnostic, not as validation blocker.**

This is scientifically interesting but not robust enough as a generic OpenNeuro validator.

## What it solves

It tries to catch:

```text
events shifted by dummy scans
EEG trigger offset
wrong sampling rate
wrong event file attached to wrong run
events outside data range
task timing mismatch
```

The structural event checks are definitely valid because BIDS requires `onset` and `duration`; onset is measured from the first stored data point, negative onsets are allowed, and dummy-volume handling must be interpreted carefully. BIDS also recommends enough onset precision for the modality, for example millisecond precision for 1000 Hz EEG. ([bids-specification.readthedocs.io][2])

## Strong checks to implement now

```text
onset/duration numeric
duration >= 0
events sorted by onset
event end <= recording duration
trial_type documented
stim_file exists
event precision adequate for sampling rate
event file matches task/run/entities
```

## Weaker signal-coherence checks

EEG ERP coherence:

```text
event-locked average
baseline/noise estimate
condition-wise SNR
latency of max absolute response
estimated trigger offset
```

fMRI event-BOLD coherence:

```text
design matrix construction
simple GLM
HRF-lag scan
cross-correlation with mean BOLD or ROI signal
```

## Where it may fail

- Many EEG tasks do not produce strong ERPs.
- Some subjects have weak or absent task responses.
- Resting-state datasets have no event-locked response.
- fMRI task effects may be localized, not visible in global mean BOLD.
- Low R² does not prove event misalignment.
- HRF delays vary by region, age, pathology, acquisition, and preprocessing.
- Wrong noise model can create false alarms.

## Correct output

```python
EventCoherenceDiagnostic(
    status="low_confidence",
    estimated_offset_s=8.0,
    possible_causes=["dummy volume mismatch", "wrong event file", "weak task signal"],
    severity="warning",
    requires_manual_confirmation=True,
)
```

## Qortex decision

Do not make this a core validator now. Build structural event validation now; add signal-event coherence later as an **optional diagnostic**.

---

# 8. Frame propagation through every transform

**Verdict: Valid approach. Implement now.**

This is one of the strongest ideas.

## What it solves

It prevents silent axis/frame corruption:

```text
DICOM LPS treated as NIfTI RAS
NIfTI reoriented but affine not updated
segmentation mask saved in wrong frame
tensor axes changed but metadata not updated
custom transform flips axis silently
DWI bvecs not updated after image reorientation
```

BIDS explicitly states that coordinates require origin, axis interpretation, and units, and that device, DICOM/file-format, head, and NIfTI coordinate systems may differ. That makes explicit frame tracking directly aligned with the real BIDS problem. ([bids-specification.readthedocs.io][7])

## Inputs needed

```text
affine
qform/sform
NIfTI orientation
DICOM ImageOrientationPatient
DICOM ImagePositionPatient
coordinate system sidecars
electrode coordinate files
transform chain
```

## Algorithms

```text
orientation extraction
affine determinant
axis code conversion
frame label propagation
transform input/output contract validation
mask/source affine equality check
bvec update-required flag after reorientation
```

## Output

```python
FrameTrace(
    input_frame="LPS",
    transforms=[
        {"op": "reorient", "from": "LPS", "to": "RAS"},
        {"op": "resample_spatial", "frame": "RAS"},
        {"op": "to_tensor", "axes": ["batch", "channel", "z", "y", "x"]},
    ],
    status="confirmed",
)
```

## Where it may fail

If source metadata is absent or wrong, Qortex can only say `unknown` or `contradicted`.

## Qortex decision

Implement now. This should be part of every conversion and NeuroAI runtime artifact.

---

# 9. DWI gradient frame auto-detection

**Verdict: Weak as written. Implement deterministic gradient checks, but avoid auto-detection/auto-conversion.**

The problem is real. The proposed method is not reliable.

## What it tries to solve

It tries to detect whether b-vectors are expressed in image space or scanner space and prevent wrong tractography or diffusion modeling.

BIDS explicitly states that DWI b-vectors are interpreted with respect to NIfTI image axes and are not equivalent to DICOM scanner-coordinate conventions. It also has handedness-specific sign logic. ([bids-specification.readthedocs.io][5])

## Why the proposed algorithm is weak

The method says:

```text
R @ bvec would produce unit vectors
compare both hypotheses
```

But a proper rotation matrix preserves vector norm. If `b` is unit norm, `R @ b` is also unit norm. Norm comparison alone cannot distinguish scanner-space vs image-space b-vectors.

## Valid deterministic checks

```text
bvec shape is 3 x N
bval length == bvec columns == NIfTI volume count
bvec norms are 1 or 0
b0 volumes match bval == 0
affine determinant and handedness recorded
image reorientation operation marks bvecs as requiring update
gradient table differs across runs only when allowed
```

## Possible advanced checks

These are later-stage and not always reliable:

```text
fit tensor model and inspect principal directions
compare tract orientation plausibility
compare against DICOM gradient metadata when available
check whether bvecs were transformed by dcm2niix-like convention
```

## Output

```python
DWIGradientReport(
    status="valid_bids_gradient_table",
    frame="bids_image_axes",
    warnings=[
        "Image was reoriented after bvec extraction; bvec transform status unknown"
    ],
)
```

## Qortex decision

Avoid “auto-detect and auto-convert” as a default feature. Implement **DWI gradient integrity + transform-safety tracking** now. Add advanced inference later only as diagnostic.

---

# 10. Preprocessing fingerprint compatibility

**Verdict: Valid approach. Implement now.**

This is very strong and practical.

## What it solves

It prevents mixing artifacts that look similar but were created differently:

```text
different high-pass cutoff
different notch frequency
different resampling target
different spatial resolution
different normalization
different software versions
different channel selection
different registration template
```

## Inputs needed

```text
ordered transform list
parameters
library versions
source hashes
output schema
axis/frame/units
random seed
fit scope
```

## Algorithm

```text
canonical JSON
sort keys
serialize operation list
hash with SHA-256
compare fingerprints across datasets/artifacts
classify differences as compatible / warning / incompatible
```

## Output

```python
PreprocessingFingerprint(
    operations=[
        {"op": "highpass", "cutoff_hz": 1.0},
        {"op": "resample", "target_hz": 256},
        {"op": "notch", "freq_hz": 60},
    ],
    library_versions={"mne": "...", "numpy": "..."},
    hash="abc123",
)
```

## Where it may fail

It cannot verify that an external preprocessing method did what it claimed unless Qortex has logs, sidecars, or artifact provenance.

## Qortex decision

Implement now. This should be mandatory for `convert`, `neuroai.run`, and artifact writing.

---

# 11. Leakage-safe split with confound balance

**Verdict: Valid approach. Implement now, but keep the optimization honest.**

## What it solves

It prevents:

```text
same subject in train/test
same recording windows split across train/test
same longitudinal subject split across folds
same stimulus identity split across train/test
diagnosis perfectly confounded with site
scanner imbalance across splits
age/sex imbalance across splits
```

This is important because leakage and partitioning are real failure modes in EEG and brain MRI, especially with subject identity, repeated measures, longitudinal scans, and non-independent samples. ([arXiv][6])

## Inputs needed

```text
subject IDs
session IDs
run IDs
source file IDs
event/stimulus IDs
target labels
site/scanner/acquisition covariates
age/sex/group covariates
desired split ratios
minimum class counts
```

## Algorithms

```text
grouped split by subject/source_file/stimulus
stratified grouped split
integer optimization or greedy local search
KL/Jensen-Shannon divergence for categorical confounds
standardized mean difference for numeric confounds
residual imbalance report
```

## Output

```python
SplitPlan(
    split_unit="subject",
    leakage_groups=["subject", "source_file"],
    assignments={"sub-01": "train", "sub-02": "test"},
    residual_imbalance={
        "age_median_difference": 2.3,
        "site_js_divergence": 0.18,
    },
    status="usable_with_warnings",
)
```

## Where it may fail

- Small cohorts cannot be perfectly balanced.
- Rare labels may make valid splits impossible.
- Some confounds are unobserved.
- Multi-label tasks are harder.
- Split balance may conflict with leakage safety; leakage safety must win.

## Qortex decision

Implement now. Use “best achievable split,” not “perfectly balanced split.”

---

# 12. Signal budget / input coverage metric

**Verdict: Valid approach. Implement now.**

## What it solves

Binary compatibility is too crude. A model trained on 64 EEG channels may technically run on 32 channels after channel selection or mapping, but the user must know that input coverage is degraded.

## Inputs needed

```text
model input contract
source profile
channel list
sampling frequency
window duration
spatial shape
voxel size
required metadata
```

## Algorithms

```text
channel coverage = available_required_channels / required_channels
sampling coverage = source_sfreq / required_sfreq after resampling
spatial coverage = source_shape / required_shape
duration coverage = available_duration / required_window_duration
metadata coverage = present_required_metadata / required_metadata
```

## Output

```python
InputCoverageReport(
    overall=0.72,
    dimensions={
        "channels": 0.50,
        "sampling_rate": 1.00,
        "window_duration": 1.00,
        "metadata": 0.80,
    },
    warning="Model expects 64 channels; source provides 32 usable channels.",
)
```

## Where it may fail

Coverage is not equal to model performance. It only measures contract coverage, not biological validity.

## Qortex decision

Implement now. It is deterministic and useful.

---

# 13. Memory-safe streaming estimator

**Verdict: Valid approach. Implement now.**

## What it solves

It prevents runtime crashes:

```text
GPU OOM
CPU memory blowup
large NIfTI full-volume load
large DICOM series assembly
unbounded output buffers
too many windows buffered
```

## Inputs needed

```text
source shape
dtype
window size
batch size
model estimated memory
output type
available RAM/GPU memory if detectable
runtime backend
```

## Algorithms

```text
bytes = product(shape) * dtype_size
peak = source_window + model_weights + activations + output_buffer
safety multiplier
available memory query
batch-size recommendation
```

## Output

```python
MemoryEstimate(
    estimated_peak_mb=7200,
    available_mb=8192,
    status="high_risk",
    suggestions=["batch_size=1", "fp16=true", "stream windows"]
)
```

## Where it may fail

Activation memory depends on model internals. Unknown model architectures may need conservative multipliers.

## Qortex decision

Implement now. It is a practical guardrail.

---

# 14. Unified `ReadinessScore`

**Verdict: Partially valid. Good UX, dangerous if overtrusted. Implement later.**

## What it solves

It gives users an overview of dataset readiness:

```text
completeness
consistency
ML safety
quality
provenance
```

## Why it is risky

A single scalar can hide blockers. A dataset with good completeness but severe train/test leakage should not get a comfortable score like `0.72`.

## Better design

Use this first:

```python
ReadinessReport(
    status="BLOCKED",
    dimensions={
        "completeness": 0.92,
        "metadata_consistency": 0.71,
        "ml_safety": 0.00,
        "quality": 0.80,
        "provenance": 0.50,
    },
    blockers=[...],
    warnings=[...],
)
```

Then later add:

```python
overall_score
```

But only after the blocker logic is stable.

## Where it may fail

- Scoring weights are subjective.
- Different goals require different scores.
- Users may compare scores across modalities incorrectly.
- A score can create false confidence.

## Qortex decision

Implement dimension reports now. Add scalar score later as a convenience, not as the source of truth.

---

# 15. Runtime → probe feedback loop

**Verdict: Valid but advanced. Implement later.**

## What it solves

Some failures appear only during streaming:

```text
z-score creates NaNs
one channel becomes flat mid-run
live stream drops samples
output writer fails after 1000 records
memory grows over time
event trigger streak behaves unexpectedly
```

## Inputs needed

```text
runtime windows
transform outputs
model outputs
output writer results
latency profiler
source profile ID
pipeline hash
```

## Algorithms

```text
per-window NaN/Inf check
zero-variance check
latency drift
dropped-window counter
output-write failure counter
channel-level runtime anomaly log
```

## Output

```python
RuntimeQualityEvent(
    window_index=847,
    issue="zero_variance_channel",
    channel="Cz",
    stage="normalize",
    severity="warning",
)
```

## Where it may fail

Do not mutate the original `SourceProfile` as if the whole source was bad. Runtime observations are contextual: they depend on selected windows, transforms, and pipeline.

## Qortex decision

Implement later as `RuntimeQualityReport`. Link it to `SourceProfile`, but do not rewrite the source profile silently.

---

# What should be removed or rewritten from the proposal

## Rewrite “No existing tool does this”

Too strong and likely false in parts. MRIQC already extracts MRI quality metrics; fMRIPrep already produces robust preprocessing and visual reports; BIDS Validator already validates schema/structure; MNE and NiBabel expose detailed headers and loading behavior. Qortex’s contribution is not that each individual check is unprecedented; it is the **integration of deterministic cross-modal evidence checks into goal-aware ML preflight and artifact contracts**. ([mriqc.readthedocs.io][8])

Better wording:

```text
Existing tools validate structure, load individual files, compute modality-specific QC, or perform preprocessing. Qortex should connect these into a deterministic, evidence-propagating, workflow-aware validation layer for visualization, conversion, training, and NeuroAI runtime execution.
```

## Rewrite “semantic”

Use:

```text
evidence-grounded
contract-aware
metadata-content consistency
goal-aware preflight
cross-file semantic consistency
```

Avoid “semantic intelligence” if it sounds like LLM inference.

## Rewrite unit inference

Do not say:

```text
infer actual unit is V
```

Say:

```text
observed numeric scale is more consistent with volts than microvolts; manual confirmation required
```

## Rewrite event-signal coherence

Do not say:

```text
events probably misaligned
```

Say:

```text
event-signal coherence is unexpectedly low under this model; possible timing mismatch, weak task response, wrong ROI, or insufficient SNR
```

## Rewrite DWI gradient frame auto-detection

Do not claim the proposed norm-based method can distinguish frames. It cannot reliably do that because rotations preserve norm.

---

# Recommended Qortex implementation order

## Phase 1 — Deterministic, high-trust checks

Implement these first:

```text
EvidenceSourceProfile
HeaderSidecarConsistency
CompanionClosure
EventTableIntegrity
DWIGradientIntegrity
CoordinateFrameTrace
CohortAnomalyProfile
PreprocessingFingerprint
LeakageGuard
InputCoverageReport
MemoryEstimate
```

These are feasible on real OpenNeuro/BIDS datasets and mostly deterministic.

## Phase 2 — Plausibility heuristics

Implement as warnings only:

```text
UnitPlausibilityAudit
PreprocessingStateAudit
SignalQualityAudit
LineNoiseAudit
DummyVolumeSuspicion
```

These are useful but not definitive.

## Phase 3 — Advanced diagnostics

Implement later:

```text
EventSignalCoherence
ConfoundAssociationGraph
RuntimeQualityFeedback
ReadinessScore
ModelSuggestionBasedOnSourceProfile
```

These need careful wording and should not block workflows unless evidence is strong.

---

# Final judgment

The proposal is directionally excellent, but Qortex should be strict about what counts as a “valid approach.”

The valid core is:

> Qortex should deterministically validate whether a neuroscience dataset is internally consistent and safe for a specific downstream workflow, using explicit evidence from files, headers, metadata, events, units, coordinate frames, signal/image statistics, splits, and provenance.

The ideas that pass that standard strongly are:

```text
Evidence propagation
Semantic SourceProfile
Cohort anomaly detection
Frame propagation
Preprocessing fingerprint
Leakage-safe splitting
Input coverage metric
Memory estimator
Deterministic event/label integrity checks
DWI gradient integrity checks
```

The ideas that are useful but must be limited are:

```text
unit inference
preprocessing-state inference
confound graph
event-signal coherence
readiness score
runtime feedback
```

The idea I would **not** implement as written is:

```text
DWI gradient frame auto-detection by norm comparison
```

Qortex should implement deterministic integrity first, probabilistic/plausibility diagnostics second, and never allow uncertain inference to silently become a validation decision.

[1]: https://bids-validator.readthedocs.io/en/latest/ "The BIDS Validator — BIDS Validator  documentation"
[2]: https://bids-specification.readthedocs.io/en/stable/modality-agnostic-files/events.html "Events - Brain Imaging Data Structure 1.11.1"
[3]: https://bids-specification.readthedocs.io/en/stable/appendices/units.html "Units - Brain Imaging Data Structure 1.11.1"
[4]: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/positron-emission-tomography.html "Positron Emission Tomography - Brain Imaging Data Structure 1.11.1"
[5]: https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html "Magnetic Resonance Imaging - Brain Imaging Data Structure 1.11.1"
[6]: https://arxiv.org/abs/2505.13021?utm_source=chatgpt.com "The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject analysis: a preliminary study"
[7]: https://bids-specification.readthedocs.io/en/stable/appendices/coordinate-systems.html "Coordinate systems - Brain Imaging Data Structure 1.11.1"
[8]: https://mriqc.readthedocs.io/en/stable/ "Welcome to MRIQC’s documentation! — mriqc 24.0.0.dev118+gd5b13cb documentation"
