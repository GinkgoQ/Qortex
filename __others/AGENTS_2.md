# AGENTS.md

This file defines mandatory engineering rules for agents working on this repository.

This repository is not a demo, scaffold, tutorial, toy package, thin wrapper, or sample implementation. Treat it as a production-grade scientific/AI infrastructure library. Every change must strengthen the library as a reliable, extensible, testable, and technically meaningful system.

---

## Project Standard

All code must be written as durable library code, not as exploratory script code.

Agents must design and implement features as part of the existing system architecture. Do not add isolated utilities, disconnected modules, shallow wrappers, placeholder APIs, or “good enough” implementations that only work on clean examples.

A valid contribution must satisfy all of the following:

- it solves a concrete user or system problem;
- it fits the existing package boundaries;
- it uses explicit contracts, typed data structures, and stable interfaces;
- it handles realistic edge cases and failure modes;
- it reports uncertainty, missing evidence, and invalid states explicitly;
- it integrates with existing validation, provenance, CLI, artifact, logging, and error-handling patterns where applicable;
- it can be tested without relying on hidden assumptions.

Do not optimize for looking complete. Optimize for being correct, inspectable, and hard to misuse.

---

## Implementation Bar

A mature implementation in this repository means:

- public APIs are typed, documented, and stable enough to be used by downstream code;
- internal boundaries are explicit: parsing, probing, validation, planning, execution, conversion, visualization, and artifact writing must not be mixed casually;
- data contracts define what enters and exits each subsystem;
- errors use structured exception types or structured report items, not silent failure or generic strings;
- warnings include actionable context: path, field, expected value, observed value, evidence source, and suggested fix when possible;
- optional dependencies are isolated and imported lazily;
- runtime behavior is deterministic unless stochastic behavior is explicitly requested and recorded;
- outputs preserve provenance, schema, units, axes, coordinate frames, timebase, labels, split groups, and source references where relevant;
- partial success is reported as partial success, never as full success.

Reject code that is only a wrapper around another library unless this repository adds clear value through validation, compatibility checking, planning, provenance, safety checks, artifact contracts, or workflow integration.

---

## Architecture and Design Expectations

Before modifying code, agents must identify:

- the subsystem being changed;
- the contract boundary affected;
- the input and output types;
- the invariants that must remain true;
- the existing call path;
- the failure modes introduced or removed;
- the user-facing API or CLI impact;
- the validation path required to prove the change works.

Do not patch symptoms without tracing the boundary where the invalid state enters the system.

Design changes must account for:

- architecture and package boundaries;
- edge cases and malformed inputs;
- incomplete, inconsistent, noisy, or partially wrong real-world data;
- API design and backwards compatibility;
- performance and memory behavior;
- concurrency and streaming behavior where relevant;
- correctness of units, axes, coordinate frames, sampling rates, timestamps, labels, and provenance;
- maintainability and future extension;
- user workflows from discovery to validation, conversion, runtime execution, visualization, and artifact reuse.

If a feature cannot define its inputs, outputs, invariants, and failure states, do not implement it yet.

---

## Polyglot / Multi-Stack Policy

Python is the default implementation language.

Additional stacks such as Rust, C++, Go, Java, native extensions, system services, or external runtimes are acceptable only when there is a measured or clearly justified need that Python cannot satisfy cleanly.

Acceptable reasons include:

- high-throughput streaming;
- low-latency ring buffers;
- memory-safe binary parsing;
- CPU-bound numerical kernels;
- native file-format bindings;
- stable service boundaries;
- interoperability with existing production systems.

Unacceptable reasons include:

- novelty;
- premature optimization;
- rewriting stable Python code without measurement;
- adding complexity to make the project look more advanced;
- introducing a second stack without a stable Python-facing contract.

Any non-Python component must expose a small, typed, documented interface to Python. It must have a fallback, a clear build path, and failure behavior that does not break unrelated parts of the package.

---

## Depth Over Surface Features

Do not add features only to increase feature count.

A shallow feature lists, loads, wraps, converts, or visualizes data without validating whether the result is meaningful. A deep feature improves the system’s ability to reason over real data constraints and prevent invalid downstream work.

Prefer features that provide:

- metadata-header consistency checks;
- companion-file closure;
- coordinate-frame and axis tracking;
- sampling-rate and timebase validation;
- unit and scaling checks;
- modality-specific integrity checks;
- train/test leakage prevention;
- preprocessing fingerprints;
- artifact contracts;
- compatibility reports;
- deterministic preflight validation;
- provenance-preserving conversion;
- failure-aware runtime execution.

A feature is not complete until it defines how it fails.

---

## Contribution Requirement

Every contribution must create a real reason to use this library instead of directly calling lower-level tools.

Do not duplicate existing tools unless this repository adds a higher-level contract, validation layer, planner, compatibility engine, provenance model, artifact system, or user workflow that the lower-level tool does not provide by itself.

A valid contribution must answer:

- what problem it solves;
- why existing libraries are insufficient alone;
- what evidence it uses;
- what assumptions it makes;
- how it detects invalid inputs;
- what it returns when evidence is missing;
- how it integrates with the rest of the repository.

Avoid marketing language, inflated claims, vague novelty statements, and “AI-generated GitHub library” style wording. Use precise engineering language. Describe behavior, contracts, checks, and failure modes.

---

## Agent Behavior Rules

Agents must:

- inspect existing code before adding new code;
- preserve established architecture unless explicitly asked to redesign it;
- prefer typed models over anonymous dictionaries at subsystem boundaries;
- avoid broad `except Exception` blocks unless errors are converted into structured warnings or typed failures;
- avoid silent fallbacks that hide incorrect behavior;
- avoid hardcoded dataset-specific logic;
- avoid placeholder implementations, TODO-driven APIs, fake support, and mock behavior in production paths;
- avoid adding public APIs that are not backed by real behavior;
- avoid advertising CLI commands, extras, adapters, or formats that are not implemented;
- validate changes with targeted tests, examples, or deterministic checks when possible;
- state unverified behavior explicitly.

If a requested change would create a brittle, shallow, or misleading feature, narrow the scope and implement the smallest correct vertical slice instead.

Add this after the opening section.

````md
---

## Data Integrity and Check System Vision ✅

Qortex must treat data checks as a first-class system, not as helper scripts.

The goal is to let users verify whether a dataset is safe for a specific operation: inspection, visualization, conversion, normalization, standardization, model training, NeuroAI runtime execution, or artifact export.

Checks must be deterministic, evidence-based, and workflow-aware. They must not rely on LLM interpretation, embeddings, vague semantic guessing, or undocumented assumptions. A check is valid only when it can inspect explicit evidence: file paths, BIDS entities, headers, sidecars, TSV columns, shapes, units, sampling rates, timestamps, coordinate frames, channel names, labels, checksums, provenance, or measured signal/image statistics.

Qortex checks must answer practical questions:

- Can this dataset be loaded?
- Can this dataset be visualized without spatial or temporal corruption?
- Can this dataset be converted without losing provenance, units, axes, or labels?
- Can this dataset be normalized or standardized safely?
- Can this dataset be used for ML without leakage or invalid labels?
- Can this source satisfy a model input contract?
- Which issues are blockers, warnings, unknowns, or acceptable heterogeneity?
- What evidence supports each decision?

Do not implement checks as generic linting. Implement them as explicit validation units with typed inputs, typed outputs, evidence, severity, and suggested fixes.

---

## Check Types ✅

Qortex must support multiple check modes.

### Targeted Checks ✅

A targeted check validates one specific concern.

Examples:

```text
qortex check events ./dataset
qortex check units ./dataset
qortex check geometry ./dataset
qortex check leakage ./dataset --target trial_type --split-unit subject
qortex check dwi-gradients ./dataset
qortex check eeg-channels ./dataset
```
````

Use targeted checks when the user knows what they want to verify.

### Preflight Checks ✅

A preflight check validates a dataset against a downstream goal.

Examples:

```text
qortex preflight ./dataset --goal visualize --modality mri
qortex preflight ./dataset --goal convert --modality eeg
qortex preflight ./dataset --goal train --target diagnosis --split-unit subject
qortex preflight ./dataset --goal neuroai-run --pipeline pipeline.yaml
```

Preflight checks must combine multiple validation units and produce a single structured report.

A preflight report must include:

- status: `PASS`, `WARN`, `BLOCK`, or `UNKNOWN`;
- blockers;
- warnings;
- missing evidence;
- affected files;
- affected subjects/sessions/runs;
- recommended fixes;
- machine-readable output.

### Lazy / Fast Monitoring Checks ✅

Qortex may include a fast checker that runs automatically during common operations.

This checker must be lightweight and safe to disable.

Its purpose is to detect obvious risk while users work:

- missing companion files;
- suspicious metadata/header mismatch;
- missing `events.tsv`;
- missing `channels.tsv`;
- inconsistent sampling rates;
- inconsistent shapes or voxel sizes;
- wrong or unknown units;
- likely train/test leakage;
- partial local downloads;
- derivative/raw mixing;
- unsupported model-source combinations.

Lazy checks must not perform expensive full-dataset computation unless explicitly requested. They should inspect manifests, headers, sidecars, cached profiles, and small samples.

Lazy checks must never silently mutate data. They may emit structured hints, warnings, and recommended commands.

Example behavior:

```text
Warning: EEG files have mixed sampling rates: 256 Hz, 512 Hz.
Run: qortex check eeg-sampling ./dataset --explain
```

The lazy checker must be configurable:

```text
QORTEX_LAZY_CHECKS=off
QORTEX_LAZY_CHECKS=warn
QORTEX_LAZY_CHECKS=strict
```

---

## Evidence Model ✅

Every check must report how it reached its conclusion.

Use evidence states consistently:

```text
confirmed      directly read from a reliable source
inferred       derived from deterministic computation
claimed        declared by metadata but not independently verified
missing        required evidence is absent
contradicted   two evidence sources disagree
unknown        not knowable without loading more data or user input
blocked        validation cannot continue because prerequisite evidence is invalid
```

Example:

```text
field: SamplingFrequency
claimed_value: 512
observed_value: 500
claimed_source: sub-01_task-rest_eeg.json
observed_source: raw EDF header
status: contradicted
severity: BLOCK
```

Do not collapse evidence into boolean flags. A missing value, a contradicted value, and an unknown value are different states and must be represented differently.

---

## Required Check Domains ✅

Agents should prefer adding checks that fit one of these domains.

### Structure Checks ✅

Validate dataset layout and file relationships.

Required concerns:

- BIDS entity consistency;
- subject/session/run/task consistency;
- raw vs derivative separation;
- local copy completeness;
- duplicate or conflicting files;
- sidecar inheritance resolution;
- companion-file closure.

### Metadata/Header Checks ✅

Cross-check documentation against file headers.

Required concerns:

- NIfTI header vs JSON sidecar;
- raw signal header vs sidecar;
- DICOM metadata vs converted output where available;
- channel count and channel names;
- sampling frequency;
- TR, volume count, frame timing;
- units and scaling fields;
- coordinate system declarations.

### Timebase and Event Checks ✅

Validate temporal consistency.

Required concerns:

- event onset and duration validity;
- event range within recording duration;
- event precision relative to sampling rate;
- task/run/entity matching;
- trigger/event alignment where evidence exists;
- dummy-volume ambiguity;
- multimodal synchronization risk.

### Coordinate and Geometry Checks ✅

Validate spatial correctness.

Required concerns:

- NIfTI affine;
- qform/sform consistency;
- orientation code;
- DICOM LPS vs NIfTI RAS;
- voxel size distribution;
- image shape consistency;
- mask/source affine compatibility;
- electrode coordinate system and units;
- DWI bvec/bval integrity.

### Unit and Scaling Checks ✅

Validate declared and observed measurement units.

Required concerns:

- declared units are parseable and allowed;
- channel-level units are consistent;
- signal scale is plausible for the declared unit;
- image values contain no NaN/Inf;
- constant or near-constant images are flagged;
- PET/fNIRS/EEG scale checks are warnings unless independently confirmed.

### Label and Leakage Checks ✅

Validate ML safety.

Required concerns:

- target labels exist;
- labels are available for enough subjects;
- label source is explicit;
- labels are not derived from forbidden downstream outputs;
- train/val/test splits respect subject/session/source/stimulus groups;
- repeated windows/slices/volumes do not cross splits;
- site/scanner/acquisition variables are reported as possible confounds.

### Conversion Readiness Checks ✅

Validate whether data can be converted safely.

Required concerns:

- source files load;
- axes order is known;
- units are known or explicitly unknown;
- timebase is known;
- coordinate frame is known;
- labels are preserved;
- provenance can be written;
- output schema can represent the data without silent loss.

### Runtime Compatibility Checks ✅

Validate whether a source can satisfy a model or pipeline contract.

Required concerns:

- modality compatibility;
- channel coverage;
- sampling-rate compatibility;
- window-duration compatibility;
- spatial-shape compatibility;
- dtype compatibility;
- memory estimate;
- preprocessing plan;
- unsupported or unsafe transforms.

---

## Converter, Normalizer, and Standardizer Policy ✅

Converters, normalizers, and standardizers must not be implemented as blind transforms.

Every transform must declare:

- input contract;
- output contract;
- required evidence;
- changed fields;
- preserved fields;
- invalidated assumptions;
- parameters;
- reversibility;
- provenance record.

A converter must preserve or explicitly record:

- source file path;
- source checksum;
- subject/session/run/task entities;
- modality;
- axes;
- units;
- coordinate frame;
- sampling frequency or TR;
- event timebase;
- labels;
- split group;
- transform history;
- output schema.

A normalizer must define its fit scope:

```text
per-window
per-file
per-subject
per-session
train-split-only
whole-dataset
external-reference
```

Whole-dataset fitting is unsafe for ML unless explicitly requested and recorded as leakage-relevant.

A standardizer must not erase heterogeneity. It must report what was standardized and what remains different.

Example:

```text
sampling_rate: standardized to 256 Hz
channels: 32 common channels retained
units: converted V -> uV
reference: inconsistent across subjects; not standardized
site: unchanged
```

---

## Check Output Requirements ✅

All checks must return structured reports, not only printed text.

A report must be usable by:

- CLI;
- Python API;
- tests;
- artifact writer;
- dashboard or HTML renderer;
- downstream automation.

Minimum report structure:

```text
CheckReport
  name
  scope
  status
  checked_at
  inputs
  blockers
  warnings
  infos
  unknowns
  evidence
  affected_files
  affected_subjects
  suggested_fixes
```

Severity must be operational:

```text
PASS      no issue detected for this check
INFO      useful context, no action required
WARN      possible issue; user should inspect or decide
BLOCK     unsafe to continue for the requested goal
UNKNOWN   evidence is insufficient
```

Do not use vague severity labels such as `bad`, `maybe`, `problematic`, or `suspicious` without a defined meaning.

---

## Check Implementation Rules ✅

Agents implementing checks must:

- keep checks small and composable;
- separate data collection from decision logic;
- avoid loading full data unless the check explicitly requires it;
- cache expensive probe results;
- make check thresholds configurable;
- use modality-specific rules where generic rules are insufficient;
- return `UNKNOWN` instead of guessing;
- return `BLOCK` only when evidence is strong;
- include affected file paths and BIDS entities in every actionable finding;
- avoid modifying files unless the command is explicitly a repair or conversion command.

A check must not claim a dataset is valid globally. It may only claim validity for a defined scope and goal.

Correct:

```text
PASS: event timing is valid for sub-01_task-rest_run-01_eeg.edf
```

Incorrect:

```text
Dataset is clean.
```

Correct:

```text
WARN: observed EEG amplitude scale is more consistent with volts than microvolts; manual confirmation required.
```

Incorrect:

```text
Units fixed automatically.
```

---

## Repair and Auto-Fix Policy ✅

Qortex may suggest fixes before it performs fixes.

Automatic repair is allowed only when:

- the correction is deterministic;
- the original value is preserved;
- the repair is written to a new output or patch file;
- provenance records the change;
- the user explicitly requested repair.

Safe repairs:

- normalize unit spelling;
- add generated report files;
- write conversion artifacts;
- produce corrected sidecar suggestions;
- generate split files;
- generate channel maps.

Unsafe repairs unless explicitly confirmed:

- changing raw sidecars in place;
- modifying event timings;
- changing labels;
- rewriting coordinate systems;
- changing DWI gradients;
- overwriting source data;
- deleting files;
- silently resampling or normalizing data.

---

## Practical Novelty Requirement for Checks ✅

A Qortex check must provide value beyond calling a lower-level loader or validator.

Valid Qortex value includes:

- cross-file consistency;
- header-sidecar contradiction detection;
- modality-specific integrity validation;
- workflow-specific readiness;
- leakage prevention;
- provenance validation;
- compatibility with a model or output contract;
- actionable repair guidance;
- structured reports that can be reused by conversion, training, runtime, and artifacts.

Do not add a check that merely calls another tool and forwards its raw output unless Qortex adds interpretation, integration, evidence mapping, or workflow-specific severity.

## Classical Computational Neuroscience Layer ✅

Qortex may include a dedicated classical-methods layer for deterministic, non-LLM, non-AI-first computational neuroscience workflows.

Use the Python package name:

```text
qortex.neuroclassic
```

Use the CLI namespace:

```text
qortex neuro-classic ...
```

This layer must add scientifically grounded computational methods that improve data inspection, validation, quality control, conversion readiness, feature extraction, reproducibility, or workflow safety. It must not become a collection of decorative plots, loosely connected notebooks, copied examples, or fragile research demos.

The purpose of `qortex.neuroclassic` is not to replace mature domain libraries. It must integrate established methods into Qortex’s contracts, reports, provenance, artifact system, dataset manifests, and workflow checks.

---

### Scope of `qortex.neuroclassic` ✅

Accept methods from classical neuroscience, computational neuroscience, statistics, numerical analysis, and signal/image processing only when they solve a concrete Qortex workflow problem.

Valid method families include:

```text
signal quality metrics
spectral analysis
event-related analysis
time-frequency summaries
artifact and outlier detection
channel-level electrophysiology QC
basic fMRI design diagnostics
image intensity and geometry QC
diffusion gradient and tensor sanity checks
connectivity matrix construction
graph-theoretic summaries
cohort-level statistical summaries
confound association analysis
split-balance diagnostics
information-theoretic feature checks
reproducibility fingerprints
numerical stability checks
```

Reject methods that are:

```text
AI-first
LLM-dependent
paper-demo only
clinically interpretive
hard to validate
not tied to a user workflow
only useful on one curated dataset
duplicated from another library without Qortex-level integration
too slow for practical use without a clear execution mode
```

---

### Admission Standard for Classical Methods ✅

Do not add a classical method because it is known, publishable, or commonly used. Add it only if it passes all gates below.

A method must define:

- target modality;
- target workflow;
- required inputs;
- optional inputs;
- assumptions;
- invalid input states;
- numerical method;
- parameter defaults;
- threshold policy;
- runtime cost;
- output schema;
- provenance record;
- integration point in Qortex.

A method is acceptable only if it can answer at least one of these operational questions:

```text
Is this dataset safe to inspect?
Is this dataset safe to visualize?
Is this dataset safe to convert?
Is this dataset safe to normalize or standardize?
Is this dataset safe to train on?
Is this dataset internally consistent?
Is this subject/session/run an outlier?
Is this signal/image corrupted, incomplete, or suspicious?
Is this derived artifact compatible with another artifact?
Is this result reproducible from recorded inputs and parameters?
```

Do not implement methods whose result cannot be converted into a structured report, check finding, artifact, or workflow decision.

---

### Required API Design ✅

Classical methods must expose stable, typed APIs.

Do not expose raw arrays and anonymous dictionaries across public boundaries unless the method is explicitly low-level and internal.

Preferred objects:

```text
NeuroClassicSpec
NeuroClassicResult
NeuroClassicReport
MetricResult
CohortMetricReport
SignalQualityReport
ImageQualityReport
ConnectivityReport
GraphMetricReport
StatisticalDiagnosticReport
```

Every result must include:

```text
method_name
method_version
modality
scope
inputs
parameters
assumptions
metrics
warnings
blockers
unknowns
provenance
runtime
```

Every method must be callable through at least one clear workflow:

```python
result = ds.neuroclassic.run("eeg_psd_qc", modality="eeg")
report = ds.neuroclassic.profile("cohort_anomalies")
qc = ds.preflight(goal="train", checks=["signal_quality", "leakage", "confounds"])
```

CLI commands must be explicit:

```text
qortex neuro-classic eeg-psd ./dataset
qortex neuro-classic signal-qc ./dataset --modality eeg
qortex neuro-classic image-qc ./dataset --modality mri
qortex neuro-classic connectivity ./dataset --method correlation
qortex neuro-classic cohort-anomalies ./dataset
```

Do not add CLI commands before the underlying API and report model are implemented.

---

### Classical Signal Processing Policy ✅

Signal-processing features must operate on explicit signal contracts.

Required evidence:

```text
sampling frequency
channel names
channel types
units
reference
duration
event table when event-locked analysis is requested
bad-channel status when available
```

Valid first-class methods:

```text
flatline detection
NaN/Inf detection
saturation detection
peak-to-peak amplitude
robust variance
line-noise power at 50/60 Hz
bandpower by frequency band
Welch PSD summary
spectral slope estimate
channel correlation outliers
event-count and event-timing checks
epoch rejection summaries
```

Optional later methods:

```text
time-frequency summaries
ERP/ERF diagnostic summaries
coherence
phase-locking value
cross-correlation
autocorrelation decay
sample entropy
mutual information
```

Do not infer clinical or cognitive meaning from these metrics. Report numerical evidence only.

Correct:

```text
WARN: channel Fp1 has peak-to-peak amplitude 18.2x cohort median.
```

Incorrect:

```text
Subject has abnormal frontal brain activity.
```

---

### Neuroimaging Analysis Policy ✅

Neuroimaging methods must distinguish between QC, preprocessing diagnostics, and scientific analysis.

Valid Qortex use cases:

```text
image loadability
shape and voxel-size profiling
affine and orientation checks
qform/sform consistency
NaN/Inf detection
constant-image detection
foreground/background intensity summaries
mask/source alignment checks
volume count checks
TR consistency checks
basic fMRI design-matrix diagnostics
fieldmap linkage checks
DWI bval/bvec integrity checks
```

Do not implement full neuroimaging pipelines that compete with specialized preprocessing tools unless Qortex only wraps them as optional, provenance-tracked integrations.

Do not claim that Qortex performs complete fMRI preprocessing, DWI modeling, registration, segmentation, tractography, or statistical inference unless the implementation is complete, validated, documented, and explicitly scoped.

Allowed:

```text
Check whether a DWI file has matching bval/bvec volume counts.
Check whether an fMRI event design is temporally compatible with the BOLD series.
Check whether a mask affine matches its source image.
Compute simple no-reference image QC metrics.
```

Not allowed:

```text
Add a half-implemented tractography pipeline.
Add a generic GLM command with no design validation.
Add segmentation metrics without verifying geometry.
Add registration output without transform provenance.
```

---

### Connectivity and Graph Methods Policy ✅

Connectivity and graph features are allowed only when the construction path is explicit.

A connectivity feature must declare:

```text
input signal type
node definition
parcellation or channel set
time window
preprocessing assumptions
connectivity metric
frequency band if applicable
thresholding rule
matrix symmetry
edge weight meaning
missing-node behavior
```

Graph metrics must not be computed on ambiguous graphs.

Valid graph metrics:

```text
degree
strength
density
clustering coefficient
path length
modularity
centrality
connected components
community assignments
small-world summary with explicit null model
```

Do not compute graph metrics without reporting how the graph was built.

Correct:

```text
Connectivity matrix built from 64 EEG channels using Pearson correlation over 2-second windows in alpha band; absolute threshold = 0.5.
```

Incorrect:

```text
Brain network analysis completed.
```

Graph outputs must include the adjacency matrix, node metadata, construction parameters, and metric definitions.

---

### Statistical Modeling and Confound Diagnostics ✅

Statistical features must be used for validation, cohort profiling, and workflow safety unless a broader analysis API is explicitly designed.

Valid methods:

```text
descriptive statistics
robust outlier detection
missingness analysis
class imbalance reports
effect-size summaries
split-balance diagnostics
mutual information for confound association
Cramer's V for categorical association
standardized mean difference for numeric covariates
permutation tests where sample size permits
simple GLM design diagnostics
```

Do not present association as causation.

Correct:

```text
diagnosis is strongly associated with site; model evaluation may be confounded.
```

Incorrect:

```text
site causes diagnosis effect.
```

Every statistical report must include sample size, missingness, variable type, method, parameters, and limitations.

If sample size is too small, return `UNKNOWN` or `LOW_CONFIDENCE`; do not produce authoritative conclusions.

---

### Information Theory and Dynamical Systems Policy ✅

Information-theoretic and dynamical-system methods are allowed only with strict scope.

Acceptable uses:

```text
entropy as signal complexity summary
mutual information as dependency diagnostic
autocorrelation as temporal redundancy diagnostic
spectral entropy as QC signal descriptor
state-transition summaries for event streams
```

Reject methods that require unstable parameter choices, long recordings unavailable in typical datasets, or domain interpretation that Qortex cannot validate.

Do not add nonlinear dynamics metrics such as Lyapunov exponents, fractal dimensions, Hurst exponents, or recurrence quantification unless:

- input length requirements are enforced;
- sampling assumptions are explicit;
- parameter sensitivity is documented;
- failure modes are reported;
- benchmarks show stable behavior on realistic data.

These methods must be diagnostic summaries, not scientific conclusions.

---

### Optimization and Numerical Methods Policy ✅

Optimization methods are acceptable when they solve concrete engineering problems.

Valid use cases:

```text
leakage-safe split assignment
confound-balanced split optimization
memory-aware batching
window selection under size constraints
minimal valid download planning
artifact compatibility matching
cohort subset selection
```

Optimization outputs must report:

```text
objective function
constraints
solver or heuristic
random seed
optimality status
residual imbalance
unmet constraints
runtime
```

Do not hide heuristic failure behind a valid-looking result. If the optimizer cannot satisfy constraints, return a partial plan with explicit violations.

---

### Quality-Control Metrics Policy ✅

QC metrics are first-class features only when they are reproducible and actionable.

Each QC metric must define:

```text
name
modality
input contract
formula or algorithm
required metadata
default threshold
threshold source
interpretation boundary
failure conditions
recommended action
```

QC metrics must separate:

```text
hard failure
statistical outlier
plausibility warning
informational summary
unknown due to missing evidence
```

Do not collapse QC into a single opaque score. A score may summarize a report, but the report is the source of truth.

---

### Integration Requirements ✅

Classical methods must integrate with existing Qortex systems.

Required integration points:

```text
SourceProfile
CheckReport
PreflightReport
CompatibilityReport
PreprocessPlan
ArtifactContract
VisualAuditReport
ConversionContract
CohortProfile
LeakageGuard
```

A method that produces files must write provenance.

A method that consumes data must record:

```text
source files
source checksums
BIDS entities
metadata fields used
parameters
library versions
random seed when relevant
output schema
```

A method that changes data must produce a new artifact, not silently mutate source files.

---

### Dependency Policy ✅

Keep classical-method dependencies optional.

Do not add heavy scientific packages to the core import path.

Use extras:

```text
qortex[neuroclassic]
qortex[eeg]
qortex[mri]
qortex[dwi]
qortex[stats]
qortex[graph]
```

Import optional dependencies lazily inside the method that needs them.

If a dependency is missing, raise a structured error with the required extra name and the affected method.

Do not fail package import because a classical-method dependency is absent.

---

### Validation and Testing Requirements ✅

Every classical method must have deterministic tests.

Required tests:

```text
valid minimal input
missing required metadata
malformed input
edge-case numeric input
known expected output
provenance serialization
report severity behavior
```

For numerical methods, include tolerance-based tests.

For stochastic or optimization methods, set and record the seed.

For cohort methods, test heterogeneous cohorts, missing covariates, small sample size, and impossible constraints.

For signal/image methods, test NaN, Inf, constant data, wrong shape, wrong units, and empty arrays.

Do not merge a classical method that only works on one hand-picked example.

---

### Rejection Rules ✅

Reject a proposed classical method if it:

- cannot define its input contract;
- cannot define its output schema;
- cannot state its assumptions;
- cannot detect invalid input;
- cannot report uncertainty;
- cannot be tested deterministically;
- requires full data loading when a header-level check is sufficient;
- duplicates another library without adding Qortex-level workflow value;
- produces a scientific interpretation instead of a measurable result;
- hides threshold choices;
- makes clinical claims;
- adds heavy dependencies to core imports;
- introduces public API before behavior is complete.

If the method is useful but fragile, implement it as an optional diagnostic with `LOW_CONFIDENCE` or `UNKNOWN` states, not as a blocking validator.

---

### Implementation Rule ✅

Classical computational features must deepen Qortex’s ability to inspect, validate, standardize, convert, and reuse neuroscience data.

They must not turn the repository into a scattered toolbox.

Implement the smallest complete vertical slice:

```text
input contract
method implementation
structured report
failure handling
provenance
tests
CLI/API integration
documentation example
```

Do not implement the next method until the current method is complete across that slice.
