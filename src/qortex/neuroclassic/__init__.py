"""qortex.neuroclassic — Classical computational neuroscience layer.

Deterministic, non-LLM methods for signal QC, image QC, connectivity,
statistical diagnostics, and cohort profiling.

Methods are integrated into Qortex's contract, provenance, and report system.
Every method returns a structured report — not raw arrays.

CLI namespace: ``qortex neuro-classic <method> <dataset>``

Extras:
    pip install 'qortex[neuroclassic]'   # core classical methods
    pip install 'qortex[eeg]'            # MNE-based signal loading
    pip install 'qortex[mri]'            # nibabel-based image loading

Usage::

    from qortex.neuroclassic import compute_signal_qc, compute_image_qc
    import numpy as np

    data = np.random.randn(64, 10000).astype(np.float32)
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, scope="sub-01_eeg")
    print(qc.n_flatline, qc.n_nan, qc.warnings)

    from qortex.neuroclassic import run_signal_qc_on_dataset
    report = run_signal_qc_on_dataset("./dataset", modality="eeg")
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
    "compute_pearson_connectivity",
    # Stats
    "ConfoundAssociation",
    "SplitBalanceSummary",
    "StatisticalDiagnosticReport",
    "VariableSummary",
    "build_cohort_metric_report",
    "compute_statistical_diagnostics",
]
