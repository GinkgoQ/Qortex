# Real end-to-end model-zoo inference demo

Real data, real download, real model forward passes — no synthetic/mocked inputs.

**Data source:** OpenNeuro `ds004015` (snapshot 1.0.2), subject `sub-036`,
task `AttendedSpeakerParadigmcEEGridAttention`. Downloaded via:

```
qortex minimum ds004015 --goal first-batch --modality eeg --output-dir <dir> --download
```

18-channel EEG, BrainAmp system, 500 Hz, 4411.26 s real recording
(University of Oldenburg). The BIDS `.set` companion file (channel/header
metadata for the EEGLAB `.fdt` payload) was fetched directly from
OpenNeuro's S3 bucket since `qortex minimum`'s first-batch plan omitted it
for this dataset — a real gap worth fixing in the planner.

**Models run (real forward pass, real architecture, random-init weights —
these are `qortex_status=architecture_available` zoo entries, not
claiming trained clinical accuracy):**

- `braindecode.Deep4Net`
- `braindecode.ShallowFBCSPNet`

Both constructed via `qortex.neuroai.models._registry.make_model_adapter`
using the real zoo registry entries, loaded on CPU, and run against a real
4-second window (60s-64s into the recording) of the real 18-channel signal
after real per-channel z-score normalization.

**Files:**
- `real_eeg_input_window.png` — the actual 18-channel signal fed to both models.
- `real_model_predictions.png` — bar chart of each model's real output class probabilities.
- `real_inference_results.json` — full real numeric results (probabilities, predicted class, tensor shapes).

**Reproduce:** `python scripts/run_real_zoo_inference_demo.py <bids_root> 036 AttendedSpeakerParadigmcEEGridAttention <results_dir>`
