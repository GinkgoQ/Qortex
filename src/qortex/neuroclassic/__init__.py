"""qortex.neuroclassic — Classical computational neuroscience layer.

Deterministic, non-LLM methods for signal QC, image QC, connectivity,
statistical diagnostics, cohort profiling, information-theoretic diagnostics,
and leakage-safe split optimisation.

Methods are integrated into Qortex's contract, provenance, and report system.
Every method returns a structured report — not raw arrays.

CLI namespace: ``qortex neuro-classic <method> <dataset>``

Extras:
    pip install 'qortex[neuroclassic]'   # core classical methods (scipy)
    pip install 'qortex[eeg]'            # MNE-based signal loading
    pip install 'qortex[mri]'            # nibabel-based image loading

Usage::

    from qortex.neuroclassic import compute_signal_qc, compute_image_qc
    import numpy as np

    data = np.random.randn(64, 10000).astype(np.float32)
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, scope="sub-01_eeg")
    print(qc.n_flatline, qc.n_nan, qc.warnings)

    from qortex.neuroclassic import compute_spectral_entropy
    se = compute_spectral_entropy(data, sampling_frequency_hz=256.0)
    print(se.mean_entropy)  # bits

    from qortex.neuroclassic import assign_leakage_safe_splits, SplitConstraints
    result = assign_leakage_safe_splits(rows, constraints=SplitConstraints(group_columns=["site"]))
    print(result.optimality_status, result.assignments)
"""

from qortex.neuroclassic._base import (
    CohortMetricReport,
    MethodConfidence,
    MetricResult,
    NeuroClassicReport,
    NeuroClassicResult,
    NeuroClassicSpec,
)

from qortex.neuroclassic.signal_qc import (
    ChannelQC,
    SignalQualityReport,
    compute_signal_qc,
    run_signal_qc_on_dataset,
)

from qortex.neuroclassic.image_qc import (
    ImageQualityReport,
    compute_image_qc,
    run_image_qc_on_dataset,
)

from qortex.neuroclassic.connectivity import (
    ConnectivityMatrix,
    ConnectivitySpec,
    GraphMetricReport,
    compute_graph_metrics,
    compute_phase_locking_value_connectivity,
    compute_pearson_connectivity,
)

from qortex.neuroclassic.stats import (
    ConfoundAssociation,
    SplitBalanceSummary,
    StatisticalDiagnosticReport,
    VariableSummary,
    build_cohort_metric_report,
    compute_statistical_diagnostics,
)

from qortex.neuroclassic.features import (
    DEFAULT_EEG_BANDS,
    SEIZURE_EEG_BANDS,
    SLEEP_EEG_BANDS,
    EpochFeatureReport,
    compute_bandpower_features,
    compute_epoch_feature_matrix,
)

from qortex.neuroclassic.infoth import (
    AutocorrelationReport,
    ChannelAutocorrelation,
    ChannelHiguchiFractalDimension,
    ChannelSpectralEntropy,
    HiguchiFractalDimensionReport,
    SpectralEntropyReport,
    compute_autocorrelation_summary,
    compute_higuchi_fractal_dimension,
    compute_spectral_entropy,
)

from qortex.neuroclassic.spatial import (
    CSPReport,
    compute_common_spatial_patterns,
)

from qortex.neuroclassic.split_optimizer import (
    SplitAssignmentResult,
    SplitConstraints,
    assign_leakage_safe_splits,
)

__all__ = [
    # Base types
    "CohortMetricReport",
    "MethodConfidence",
    "MetricResult",
    "NeuroClassicReport",
    "NeuroClassicResult",
    "NeuroClassicSpec",
    # Signal QC
    "ChannelQC",
    "SignalQualityReport",
    "compute_signal_qc",
    "run_signal_qc_on_dataset",
    # Image QC
    "ImageQualityReport",
    "compute_image_qc",
    "run_image_qc_on_dataset",
    # Connectivity
    "ConnectivityMatrix",
    "ConnectivitySpec",
    "GraphMetricReport",
    "compute_graph_metrics",
    "compute_phase_locking_value_connectivity",
    "compute_pearson_connectivity",
    # Statistics
    "ConfoundAssociation",
    "SplitBalanceSummary",
    "StatisticalDiagnosticReport",
    "VariableSummary",
    "build_cohort_metric_report",
    "compute_statistical_diagnostics",
    # Feature extraction
    "DEFAULT_EEG_BANDS",
    "SEIZURE_EEG_BANDS",
    "SLEEP_EEG_BANDS",
    "EpochFeatureReport",
    "compute_bandpower_features",
    "compute_epoch_feature_matrix",
    # Information theory
    "AutocorrelationReport",
    "ChannelAutocorrelation",
    "ChannelHiguchiFractalDimension",
    "ChannelSpectralEntropy",
    "HiguchiFractalDimensionReport",
    "SpectralEntropyReport",
    "compute_autocorrelation_summary",
    "compute_higuchi_fractal_dimension",
    "compute_spectral_entropy",
    # Spatial filters
    "CSPReport",
    "compute_common_spatial_patterns",
    # Split optimisation
    "SplitAssignmentResult",
    "SplitConstraints",
    "assign_leakage_safe_splits",
]
