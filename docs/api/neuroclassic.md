# Neuroclassic API

`qortex.neuroclassic` provides deterministic, non-LLM computational neuroscience methods. Every function returns a structured report — not raw arrays.

```bash
pip install "qortex[neuroclassic]"   # scipy
pip install "qortex[eeg]"            # + MNE for signal loading
pip install "qortex[mri]"            # + nibabel for image loading
```

---

## Base types

Every method returns a `NeuroClassicResult` (single file) or `NeuroClassicReport` (whole dataset). `MetricResult` carries individual scalar/vector metrics. `CohortMetricReport` holds cohort-level descriptive statistics and outlier detection.

::: qortex.neuroclassic.MethodConfidence
    options:
      show_source: false

::: qortex.neuroclassic.MetricResult
    options:
      show_source: false
      members:
        - to_dict

::: qortex.neuroclassic.NeuroClassicSpec
    options:
      show_source: false
      members:
        - to_dict

::: qortex.neuroclassic.NeuroClassicResult
    options:
      show_source: false
      members:
        - add_metric
        - to_dict

::: qortex.neuroclassic.NeuroClassicReport
    options:
      show_source: false
      members:
        - add_result
        - all_warnings
        - all_blockers
        - has_blockers
        - to_dict

::: qortex.neuroclassic.CohortMetricReport
    options:
      show_source: false
      members:
        - compute
        - to_dict

---

## Signal QC

::: qortex.neuroclassic.compute_signal_qc
    options:
      show_source: false

::: qortex.neuroclassic.run_signal_qc_on_dataset
    options:
      show_source: false

::: qortex.neuroclassic.SignalQualityReport
    options:
      show_source: false
      members:
        - to_result
        - to_dict

::: qortex.neuroclassic.ChannelQC
    options:
      show_source: false
      members:
        - to_dict

---

## Image QC

::: qortex.neuroclassic.compute_image_qc
    options:
      show_source: false

::: qortex.neuroclassic.run_image_qc_on_dataset
    options:
      show_source: false

::: qortex.neuroclassic.ImageQualityReport
    options:
      show_source: false
      members:
        - to_result
        - to_dict

---

## Connectivity

::: qortex.neuroclassic.compute_pearson_connectivity
    options:
      show_source: false

::: qortex.neuroclassic.compute_phase_locking_value_connectivity
    options:
      show_source: false

::: qortex.neuroclassic.compute_graph_metrics
    options:
      show_source: false

::: qortex.neuroclassic.ConnectivityMatrix
    options:
      show_source: false
      members:
        - n_nodes
        - to_dict

::: qortex.neuroclassic.ConnectivitySpec
    options:
      show_source: false
      members:
        - summary

::: qortex.neuroclassic.GraphMetricReport
    options:
      show_source: false
      members:
        - to_result
        - to_dict

---

## Statistical diagnostics

::: qortex.neuroclassic.compute_statistical_diagnostics
    options:
      show_source: false

::: qortex.neuroclassic.build_cohort_metric_report
    options:
      show_source: false

::: qortex.neuroclassic.StatisticalDiagnosticReport
    options:
      show_source: false
      members:
        - to_dict
        - to_result

::: qortex.neuroclassic.VariableSummary
    options:
      show_source: false
      members:
        - missing_fraction
        - to_dict

::: qortex.neuroclassic.ConfoundAssociation
    options:
      show_source: false
      members:
        - to_dict

::: qortex.neuroclassic.SplitBalanceSummary
    options:
      show_source: false
      members:
        - to_dict

---

## Feature extraction

::: qortex.neuroclassic.compute_epoch_feature_matrix
    options:
      show_source: false

::: qortex.neuroclassic.compute_bandpower_features
    options:
      show_source: false

::: qortex.neuroclassic.EpochFeatureReport
    options:
      show_source: false
      members:
        - shape
        - to_result
        - to_dict

::: qortex.neuroclassic.DEFAULT_EEG_BANDS
    options:
      show_source: false

::: qortex.neuroclassic.SLEEP_EEG_BANDS
    options:
      show_source: false

::: qortex.neuroclassic.SEIZURE_EEG_BANDS
    options:
      show_source: false

---

## Information theory

::: qortex.neuroclassic.compute_spectral_entropy
    options:
      show_source: false

::: qortex.neuroclassic.compute_autocorrelation_summary
    options:
      show_source: false

::: qortex.neuroclassic.compute_higuchi_fractal_dimension
    options:
      show_source: false

::: qortex.neuroclassic.SpectralEntropyReport
    options:
      show_source: false
      members:
        - to_result
        - to_dict

::: qortex.neuroclassic.ChannelSpectralEntropy
    options:
      show_source: false
      members:
        - to_dict

::: qortex.neuroclassic.AutocorrelationReport
    options:
      show_source: false
      members:
        - to_result
        - to_dict

::: qortex.neuroclassic.ChannelAutocorrelation
    options:
      show_source: false
      members:
        - to_dict

::: qortex.neuroclassic.HiguchiFractalDimensionReport
    options:
      show_source: false
      members:
        - to_result
        - to_dict

::: qortex.neuroclassic.ChannelHiguchiFractalDimension
    options:
      show_source: false
      members:
        - to_dict

---

## Spatial filters

::: qortex.neuroclassic.compute_common_spatial_patterns
    options:
      show_source: false

::: qortex.neuroclassic.CSPReport
    options:
      show_source: false
      members:
        - n_components
        - n_epochs
        - transform
        - to_result
        - to_dict

---

## Leakage-safe split optimisation

::: qortex.neuroclassic.assign_leakage_safe_splits
    options:
      show_source: false

::: qortex.neuroclassic.SplitConstraints
    options:
      show_source: false

::: qortex.neuroclassic.SplitAssignmentResult
    options:
      show_source: false
      members:
        - to_dict
