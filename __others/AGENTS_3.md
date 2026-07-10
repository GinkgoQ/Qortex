> ## Implementation status (updated 2026-07-07, round 3)
>
> This repo already had ~12,000 lines of working `visualize/`, `neuroai/`,
> and `eda/` code before this spec was written, so it was executed as a
> gap-fill against real code, not a from-scratch build, across three rounds.
>
> | # | Subsystem | Status | Where |
> |---|---|---|---|
> | 1 | Visualization design system (`qortex.viz`) | **Done, scoped.** Built `qortex.visualize.design/` (`theme.py`, `palettes.py`, `typography.py`, `components.py`) — one font stack (Lato/Roboto over DejaVu default), one semantic color language, shared metric-card/badge/table/title components — and migrated every matplotlib figure in this package onto it. Not a 13-module speculative package; only what every figure actually needed. | `src/qortex/visualize/design/` |
> | 2 | Dataset readiness figures | **Done.** Report-card figure: status badge, metric cards, trainability bar chart, blocker/next-step panels, built on the design system. | `src/qortex/eda/dataset_readiness.py` |
> | 3 | Participant metadata figures | **Done.** Parser + dirty-categorical-value detector + violin/box/strip figure. Caught and fixed a real bug (raw `"M,"` was silently merged into the `"M"` group before the fix). | `src/qortex/eda/participants.py` |
> | 4 | Neuroimaging preview, quality-scored slice selection | **Done.** `_best_axial_index()` scores candidate slices by brain-tissue coverage instead of geometric midpoint. Verified on real data: midpoint z=96 vs. selected z=111. | `src/qortex/visualize/volume.py`, `_audit.py` |
> | 5 | NeuroAI detection/segmentation overlays | **Done.** `render_detection_showcase()` — annotated image + confidence table + per-class legend + threshold/NMS caption. | `src/qortex/neuroai/showcase.py` |
> | 6 | Model registry, multi-backend | **Done.** torchvision + keras adapters (real inference verified, no silent weight downloads), `list_models()`/`get_model_card()`/`register_model()` on the existing curated registry, plus a redesigned model-zoo dashboard (pill backend badges, color-coded task table, ranked inference bars). | `src/qortex/neuroai/models/torchvision_adapter.py`, `keras_adapter.py`, `zoo.py`, `showcase.py` |
> | 7 | Dataset explorer / search dashboard | **Explicitly out of scope** — this is "Qortex Atlas," built and maintained separately; not duplicated here. | — |
> | 8 | Viewer / QC dashboard, FD/DVARS | **Done.** DVARS computed directly from consecutive BOLD frames; FD from real motion parameters (Power et al. formula) when present. Verified against synthetic motion spikes. | `src/qortex/visualize/volume.py` |
> | 9 | Connectivity / signal / reproducibility figures | **Done.** Heatmap + circular network graph + hub-degree panel; PSD mean±SEM band across conditions (real Welch PSD per trial, not a fabricated interval); pipeline DAG + environment + real SHA-256 artifact-hash table. | `src/qortex/visualize/connectivity.py`, `psd.py`, `reproducibility.py` |
>
> **Round 3 additions (design polish + broader ecosystem integration):**
> - Full design-system pass: every matplotlib figure above was rebuilt on `qortex.visualize.design` — real font stack, semantic color tokens, pill badges, zebra-striped tables. Caught and fixed real layout bugs along the way (clipped y-tick labels, a missing glyph, DAG boxes running off-canvas) by rendering and visually inspecting every figure, not just compiling.
> - **MNE integration** — `TimeSeriesViewer.topomap_mne()` uses `mne.viz.plot_topomap` (real spherical-spline interpolation + head outline) when a real `mne.Info`/montage is available, instead of the hand-rolled IDW grid. Verified against a synthetic 10-20 montage with an injected posterior pattern.
> - **Nilearn integration** — `visualize/nilearn_bridge.py`: `glass_brain_connectome_figure()` (verified with real published Power-2011 ROI coordinates, bundled with nilearn — no fabricated coordinates), `stat_map_figure()`/`glass_brain_stat_figure()` (verified against the real repo T1w with a labeled-synthetic stat blob).
> - **NiiVue embed** — `visualize/niivue_viewer.py` vendors the NiiVue WebGL2 bundle (`_vendor/niivue.umd.js`, BSD-2-Clause, `_vendor/NOTICE.md`) and inlines it into a self-contained HTML page (no CDN dependency). Verified by actually rendering it in headless Chrome via Playwright (not just generating HTML) — real WebGL2 multiplanar view of the repo's T1w, screenshot in `artifacts/qortex_gallery/niivue_viewer_screenshot.png`.
> - **`web/` — scoped Three.js/React-Three-Fiber app.** A new, separate, minimal frontend (Vite + React + TypeScript + `@react-three/fiber`), not a replacement for the Python static-figure system. One scene: an interactive 3D connectome rendered from a real JSON export of `neuroclassic.connectivity` output (`web/scripts/export_connectome.py`). `npm install`, `tsc -b`, and `npm run build` all verified clean; the built app was served and screenshotted via headless Chrome to confirm it actually renders (`artifacts/qortex_gallery/web_connectome_3d_screenshot.png`). D3.js and vtk.js were not added — no second view exists yet to justify them.
>
> **Not implemented, with reason:**
> - Test coverage for `visualize/` (~7,500 lines, still untested) — explicitly deferred by request ("code only for now").
> - PSD/spectrogram/connectivity figures via a live D3 dashboard — out of scope per the "Qortex Atlas is separate" clarification.
>
> **Verification performed:** `python -m compileall`, full `pytest` (197 passed; 3 pre-existing failures in `test_neuroclassic_advanced.py` from an unrelated `numpy.trapezoid` version issue, confirmed untouched by this work), real (non-mocked) execution of every new code path — actual torchvision/keras/MNE/Nilearn calls, real FD/DVARS detection of injected synthetic artifacts, headless-Chrome rendering of both the NiiVue HTML viewer and the Three.js app (not just "should render"), and visual inspection of every generated figure in `artifacts/qortex_gallery/`.

---

Absolute Mode.

You are a senior AI engineer, Python library architect, scientific-visualization engineer, neuroimaging software engineer, UI/data-product designer, and NeuroAI systems engineer.

You are working on the Qortex repository from zero history.

I will attach 4 reference images. Treat them as visual direction references, not exact screenshots to copy. Extract their design language, structure, density, panel composition, information hierarchy, typography discipline, metric-card logic, neuroimaging visualization style, NeuroAI output presentation style, and production-grade dashboard/report quality.

Your task is to upgrade Qortex so its generated figures, reports, visual previews, NeuroAI outputs, documentation examples, and visual workflows reach the level of those references or better.

Do not produce raw debug plots.
Do not produce notebook-default figures.
Do not produce shallow charts.
Do not produce arbitrary slices.
Do not produce fake NeuroAI outputs.
Do not add decorative complexity.
Do not claim completion until outputs are visually, functionally, and technically verified.

Use Ponytail thinking:

deeply understand → choose the smallest complete solution → implement real functionality → verify objectively → polish details → remove unnecessary complexity

Use this loop:

inspect → trace → diagnose → design → implement → verify → visually review → document → repeat

Keep reasoning internal. Output only implementation evidence, verification results, and final artifacts.

Primary objective:
Build a mature Qortex visualization and NeuroAI-output system that generates professional multi-panel scientific figures, neuroimaging previews, dataset dashboards, validation reports, model inference summaries, segmentation overlays, bounding-box overlays, quality-control panels, reproducibility panels, and export-ready report figures.

The final result must be a working codebase with:

- real figure-generation APIs
- consistent design system
- high-quality neuroimaging rendering
- metadata validation before plotting
- advanced statistical plotting where appropriate
- NeuroAI output visualization support
- model-output adapters where needed
- tests
- documentation
- runnable examples
- exported sample figures matching the attached visual quality

Do not stop at planning.
Do not only edit documentation.
Do not only add tests.
Do not make cosmetic changes.
Do not leave placeholders, fake examples, TODOs, mocks, or incomplete modules.

## Root-Cause Fix

The previous output failed because the figures looked like raw matplotlib/debug plots:

- poor layout
- oversized titles
- weak typography
- arbitrary colors
- dirty metadata plotted as valid categories
- poor BOLD slice selection
- no image-quality scoring
- no figure composition system
- no metadata hierarchy
- no NeuroAI model/output layer
- no visual acceptance criteria
- no visual review

Fix those root causes directly.

Every visualization must be treated as a first-class product artifact.

The target pipeline is:

validated data → quality-ranked inputs → designed figure specification → polished figure rendering → exported artifact → visual QA → tests/docs

## Reference Image Interpretation

Use the 4 attached images as the target standard.

Extract these design principles:

- multi-panel figure composition
- clean scientific dashboard structure
- card-based visual hierarchy
- compact but readable layout
- professional spacing and margins
- consistent typography scale
- controlled color palette
- semantic status colors
- metadata chips
- summary metric cards
- structured captions
- panel labels: a, b, c, d, ...
- dense information without clutter
- neuroimaging-specific overlays
- segmentation masks with legends
- bounding boxes with confidence labels
- QC timelines and outlier markers
- reproducibility/provenance panels
- dataset coverage heatmaps
- search/filter workspace panels
- model-backend summary panels
- export/report workflow panels

Do not copy them blindly.
Implement reusable Qortex components that can generate figures with this quality from real or synthetic test data.

## Required Subsystems

Implement or upgrade these subsystems.

### 1. Qortex Visualization Design System

Create or upgrade a dedicated visualization package.

Expected structure can be adjusted if the repository already has better conventions:

qortex.viz

- theme.py
- layout.py
- typography.py
- palettes.py
- specs.py
- export.py
- quality.py
- dataset.py
- metadata.py
- neuroimage.py
- neuroai.py
- qc.py
- reports.py

Implement:

- central theme object
- typography scale
- semantic color tokens
- modality-specific color tokens
- figure-size presets
- aspect-ratio presets
- panel-grid system
- card layout helpers
- metadata chip components
- metric card components
- status badge components
- legends
- captions
- footnotes
- export presets
- visual QA checks

Required presets:

- notebook
- report
- publication
- dashboard
- poster
- retina_png
- svg
- pdf

Required aspect ratios:

- 16:9 dashboard
- 4:3 scientific report
- 1:1 square panel
- 3:2 article figure
- 5:4 neuroimage panel
- full-width multi-panel composite

Typography rules:

- title hierarchy must be controlled
- no giant debug titles
- axis labels must be readable
- annotations must not overlap
- captions must be structured
- metadata must live in chips/cards, not random text inside plots

Color rules:

- use semantic status colors for success/warning/error/unknown
- use perceptually safe palettes
- use consistent NeuroAI class colors
- avoid arbitrary default colors
- avoid excessive saturation
- ensure background, grid, and text contrast are professional

Layout rules:

- use explicit layout management
- no label collisions
- no floating annotations
- no random text boxes
- no overlapping legends
- no cropped colorbars
- no broken margins
- every panel has a clear purpose

### 2. Dataset Readiness Figures

Replace crude bar charts with professional dataset-readiness report figures.

Support:

- dataset ID
- dataset name
- modality list
- subject count
- session count
- run/recording count
- label-ready count
- target variable
- split strategy
- label status
- training status
- required download size
- blockers
- recommended next actions
- data availability score
- metadata completeness score
- trainability decision

Expected output:

- report-card style figure
- metric cards
- semantic readiness status
- compact trainability chart
- blocker panel
- next-step panel
- clean caption
- no raw function-call title such as Dataset.can_train(...)

Required behavior:

- if label-ready count is zero, show the reason clearly
- if dataset is uncertain, show uncertainty as a first-class status
- if download is required, show size and reason
- if labels are missing, show which files/fields are missing

### 3. Participant Metadata Figures

Implement mature participant metadata visualization.

Support:

- age distribution
- sex/gender categorical fields
- group/category distributions
- invalid category detection
- missing value reporting
- participants.tsv parsing
- participants.json sidecar interpretation when available
- category normalization
- warnings for dirty values such as "M,"
- n per group
- summary statistics table
- invalid/missing category panel

Do not plot dirty raw categories as normal groups.

Expected output:

- violin + box + jitter/strip when sample size supports it
- box + points for small groups
- metric cards for total/valid/invalid entries
- summary table
- clear invalid-value warning
- professional title and caption
- no implementation wording like "sex field" in final title

### 4. Neuroimaging Preview Figures

Implement high-quality neuroimaging previews.

Do not plot arbitrary first slices.

Required pipeline:
discover candidates
→ classify modality
→ validate image/header/affine/shape
→ score image quality
→ choose best file
→ choose best volume
→ choose best slice
→ normalize intensity
→ crop background
→ render with clean metadata
→ export figure

Support:

- T1w preview
- T2w preview
- BOLD/fMRI preview
- dMRI preview if available
- mean BOLD volume preview
- best representative slice selection
- axial/coronal/sagittal views
- slice montage
- ROI overlay
- segmentation overlay
- atlas/parcellation overlay
- orientation labels
- voxel spacing metadata
- TR metadata
- shape metadata
- subject/session/task/run metadata
- intensity percentile clipping
- colorbar formatting
- scale bar if feasible
- viewer-ready data structures

Quality scoring should consider:

- finite-value ratio
- nonzero voxel ratio
- brain coverage
- dynamic range
- robust contrast
- central-slice structure
- background ratio
- shape sanity
- voxel-size sanity
- affine/orientation sanity
- mean-volume availability for BOLD
- outlier volume detection if feasible

BOLD output must use a mean volume or quality-selected representative volume unless the user explicitly asks for a raw single volume.

Expected output:

- clean 3-view preview
- montage preview
- QC metadata cards
- quality metric cards
- orientation labels
- correct colorbar
- no debug telemetry inside the image
- no low-quality arbitrary raw slices as showcase examples

### 5. NeuroAI Output Visualization

Implement visualization support for NeuroAI outputs.

Support:

- image classification summary
- top-k prediction bars
- confidence table
- object detection bounding boxes
- bounding-box labels and confidence
- segmentation masks
- segmentation overlays
- alpha blending
- multi-class segmentation legends
- ROI overlays
- heatmaps or saliency maps if feasible
- model inference summary
- latency/device/dtype metadata
- input preprocessing metadata
- model card summary
- backend summary
- uncertainty/confidence display
- batch inference summary
- exportable prediction report

Bounding-box requirements:

- class-specific colors
- confidence label on each box
- box coordinates available in table
- threshold shown
- NMS IoU shown if used
- no overlapping unreadable labels
- legend included

Segmentation requirements:

- original image panel
- prediction-label panel
- overlay panel
- optional 3D/rendered summary if feasible
- class legend
- alpha value shown
- Dice/IoU metrics shown when ground truth exists
- no fake metric values unless synthetic demo explicitly labels them as synthetic

Overlay requirements:

- robust transparency
- clear boundaries
- correct image alignment
- correct shape validation
- meaningful errors when mask/image dimensions mismatch

### 6. Model Registry and Backend Integration

Implement a real model-access layer if not already present.

Expected package can be adjusted to existing architecture:

qortex.models

- registry.py
- cards.py
- cache.py
- preprocessing.py
- inference.py
- backends/torchvision.py
- backends/keras.py
- backends/huggingface.py
- backends/monai.py

Required APIs:

- list_models()
- get_model_card()
- load_model()
- register_model()
- model.predict()
- model.embed() where appropriate
- model.segment() where appropriate
- model.detect() where appropriate
- model.preprocess()
- model.info()

Backend support:

- PyTorch / torchvision
- Keras applications
- Hugging Face models
- MONAI bundles where feasible

Rules:

- use small, fast default models for examples
- support explicit download permission
- support offline mode
- support safe cache directory
- support device selection
- support dtype/precision selection where feasible
- provide actionable errors when optional dependencies are missing
- never silently download large models without user consent
- never fake model inference
- if real inference is unavailable, provide a clearly labeled synthetic fixture path only for tests/examples

ModelCard fields:

- name
- backend
- task
- modality
- input shape
- output type
- pretrained status
- source
- license if available
- parameter count if available
- model size if available
- preprocessing requirements
- supported devices
- cache behavior
- download requirement
- citation/source URL if available

### 7. Search, Filtering, Dataset Explorer Figures

Implement non-overlapping figure examples for:

- dataset explorer overview
- active search/filter workspace
- subject-session-run coverage heatmap
- modality availability summary
- task distribution
- BIDS structure overview
- conversion/export workflow
- metadata validation report

Expected output:

- dense but clean dashboard panels
- filter chips
- sortable table mock/data-driven table
- heatmap with available/missing/not-expected states
- summary metric cards
- validation pass/warning/error cards
- file-tree structure
- export workflow steps
- output package cards

### 8. Viewer and QC Figures

Implement non-overlapping figure examples for:

- interactive viewer overview
- slice montage
- ROI browser
- fMRI QC timeline
- framewise displacement
- DVARS
- global signal
- outlier volume markers
- cache/streaming performance
- download/cache manager
- viewer annotation tools

Expected output:

- viewer-style multi-panel figure
- crosshair views
- slice montage grid
- ROI inspector
- histogram
- QC time-series plots
- outlier markers
- retained/deleted volume summary
- cache metrics
- annotation panel

### 9. Connectivity, Signal Analytics, and Reproducibility Figures

Implement non-overlapping figure examples for:

- ROI connectivity matrix
- connectome graph
- power spectral density
- spectrogram
- complexity metrics
- cohort comparison
- feature extraction summary
- experiment benchmarking
- pipeline DAG
- environment/provenance summary
- artifact hash table
- report/export actions

Expected output:

- connectivity heatmap with correct diverging scale
- network graph with threshold and legend
- PSD with uncertainty band
- spectrogram with colorbar
- cohort plots with statistical annotation if valid
- benchmark table
- reproducibility DAG
- software/hardware environment panel
- artifact hash/provenance table

## Implementation Rules

Before coding:

1. Inspect the repository structure.
2. Find existing visualization, plotting, NeuroAI, dataset, metadata, and model modules.
3. Find tests and docs related to these systems.
4. Identify what exists, what is broken, what is missing, and what is superficial.
5. Build a focused implementation plan for this subsystem only.

During coding:

1. Implement reusable components first.
2. Replace one-off plotting code with figure builders.
3. Validate data before plotting.
4. Add quality selection before neuroimage rendering.
5. Add model/output abstractions before demo outputs.
6. Add tests beside each new behavior.
7. Add docs immediately after implementation.
8. Generate sample figures.
9. Visually inspect generated figures.
10. Refine spacing, labels, colors, hierarchy, and exports.
11. Repeat until the figures look professional.

Do not move to broad unrelated modules.

## Design Quality Bar

Every final figure must satisfy:

- coherent information hierarchy
- professional typography
- consistent spacing
- consistent panel labels
- no raw debug titles
- no arbitrary colors
- no dirty metadata shown as valid categories
- no low-quality random image slices
- no unreadable labels
- no overlapping annotations
- no cropped legends/colorbars
- no excessive whitespace
- no visual clutter without purpose
- no fake metrics
- no undocumented assumptions
- no broken export
- no unsupported documentation claims

Each figure must answer:

- What is this figure showing?
- What data was used?
- What quality checks were applied?
- What should the user conclude?
- What action is recommended if something is wrong?

## Data and Demo Rules

Use real repository sample data where available.

If real data is missing:

- create synthetic fixtures only for tests/examples
- clearly label synthetic fixtures
- do not claim synthetic outputs are real dataset outputs
- keep synthetic data realistic enough to test layout, overlays, masks, bounding boxes, and metadata

For neuroimaging fixtures:

- generate small synthetic NIfTI-like arrays if needed
- include realistic affine/header metadata where possible
- include synthetic masks for segmentation overlay tests
- include synthetic detections for bounding-box rendering tests
- include synthetic QC time series for QC figure tests

## Testing Requirements

Add meaningful tests for:

- theme creation
- figure spec creation
- export to PNG
- export to SVG if supported
- no label collision where testable
- metadata normalization
- invalid category detection
- participants.tsv parsing
- participants.json sidecar interpretation
- dataset readiness summary
- neuroimage quality scoring
- best slice selection
- percentile clipping
- background crop
- segmentation overlay shape validation
- bounding-box rendering validation
- model registry listing
- model-card validation
- optional dependency missing behavior
- offline model-loading behavior
- cache directory safety
- generated sample figure commands

Do not rely only on import tests.

## Documentation Requirements

Update:

- README.md
- visualization docs
- dataset docs
- neuroimaging docs
- NeuroAI docs
- model registry docs
- tutorial docs
- example gallery docs
- CLI docs if figure generation is exposed through CLI
- API docs if public APIs are added

Documentation must include:

- public API examples
- figure gallery
- explanation of design system
- neuroimage quality-selection strategy
- metadata validation strategy
- model registry usage
- backend dependency notes
- offline/cache behavior
- export presets
- generated sample commands

Every documented feature must work.

## Required CLI or Script Entry Points

Add or update commands/scripts to generate sample figures.

Expected examples:

- generate dataset readiness figure
- generate participant metadata figure
- generate neuroimage preview figure
- generate NeuroAI detection overlay figure
- generate NeuroAI segmentation overlay figure
- generate dataset explorer dashboard figure
- generate viewer/QC dashboard figure
- generate connectivity/signal/reproducibility dashboard figure

Each command must write outputs to a predictable directory such as:

artifacts/qortex_gallery/

Generated files should include:

- dataset_readiness.png
- participant_metadata.png
- neuroimage_preview.png
- neuroai_detection_overlay.png
- neuroai_segmentation_overlay.png
- dataset_explorer_dashboard.png
- viewer_qc_dashboard.png
- analytics_reproducibility_dashboard.png

## Verification Commands

Run the strongest available verification.

At minimum:

- python -m pytest -q
- python -m compileall -q src tests
- import sweep for qortex modules
- sample figure generation command
- export file existence and nonzero-size checks
- git diff --check

If available:

- lint
- type check
- docs build
- visual snapshot tests

If a tool is unavailable, state it exactly and do not pretend it passed.

## Visual Review Protocol

After generating figures:

1. Open or inspect each generated image.
2. Check layout balance.
3. Check panel alignment.
4. Check title hierarchy.
5. Check typography.
6. Check legends.
7. Check colorbars.
8. Check text overlap.
9. Check subplot spacing.
10. Check image quality.
11. Check metadata correctness.
12. Check export clarity.
13. Fix issues.
14. Regenerate.
15. Reinspect.

Do not claim completion without visual review.

## Acceptance Criteria

The task is complete only when:

- Qortex has a reusable visualization design system.
- Dataset-readiness figures are professional.
- Participant metadata figures validate data before plotting.
- Neuroimaging preview figures select high-quality inputs/slices.
- NeuroAI detection overlays work.
- NeuroAI segmentation overlays work.
- Model registry/model-card layer exists where feasible.
- Optional model backends fail gracefully when unavailable.
- Multi-panel dashboards can be generated.
- Generated figures match the maturity of the attached references.
- Tests pass.
- Docs match implementation.
- Sample gallery files are generated.
- No fake, placeholder, or debug-only figure remains in completed paths.

## Final Response Format

Return:

1. Completion status:
   - full completion or partial completion with exact blockers

2. Files inspected:
   - grouped by subsystem

3. Files changed:
   - grouped by subsystem

4. Implemented visualization components:
   - theme
   - layout
   - dataset figures
   - metadata figures
   - neuroimage figures
   - NeuroAI figures
   - model registry

5. Generated sample figures:
   - file paths
   - short description of each

6. Verification:
   - commands run
   - pass/fail result
   - unavailable tools

7. Visual QA:
   - issues found
   - fixes applied
   - final status

8. Remaining gaps:
   - exact path
   - exact reason
   - only if any remain

Do not respond with only “Done.”
Do not hide incomplete work.
Do not claim full completion unless the acceptance criteria are satisfied.
