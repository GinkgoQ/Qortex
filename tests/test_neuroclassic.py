"""Deterministic tests for the qortex.neuroclassic classical-methods layer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qortex.neuroclassic import (
    ConnectivityMatrix,
    MethodConfidence,
    NeuroClassicResult,
    compute_graph_metrics,
    compute_image_qc,
    compute_pearson_connectivity,
    compute_signal_qc,
    compute_statistical_diagnostics,
)
from qortex.neuroclassic.stats import build_cohort_metric_report


# ── Signal QC ─────────────────────────────────────────────────────────────────

def test_signal_qc_clean_data():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 2560)).astype(np.float32)
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, scope="test")
    assert qc.n_channels == 8
    assert qc.n_times == 2560
    assert qc.duration_s == pytest.approx(10.0)
    assert qc.n_nan == 0
    assert qc.n_inf == 0


def test_signal_qc_detects_flatline():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((4, 2560)).astype(np.float32)
    data[2, :] = 0.0  # flatline channel
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, compute_psd=False)
    assert qc.n_flatline >= 1
    assert qc.channel_qc[2].is_flatline


def test_signal_qc_detects_nan():
    rng = np.random.default_rng(2)
    data = rng.standard_normal((4, 1000)).astype(np.float32)
    data[1, 10:20] = np.nan
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, compute_psd=False)
    assert qc.n_nan >= 1
    assert qc.channel_qc[1].has_nan


def test_signal_qc_detects_inf_blocks():
    rng = np.random.default_rng(3)
    data = rng.standard_normal((4, 1000)).astype(np.float32)
    data[0, 5] = np.inf
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, compute_psd=False)
    assert qc.n_inf >= 1
    assert any("Inf" in b for b in qc.blockers)


def test_signal_qc_invalid_sfreq_raises():
    data = np.zeros((2, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_signal_qc(data, sampling_frequency_hz=0.0)


def test_signal_qc_wrong_ndim_raises():
    data = np.zeros((100,), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_signal_qc(data, sampling_frequency_hz=256.0)


def test_signal_qc_channel_name_mismatch_raises():
    data = np.zeros((4, 100), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_signal_qc(data, sampling_frequency_hz=256.0, channel_names=["a", "b"])


def test_signal_qc_empty_times():
    data = np.zeros((4, 0), dtype=np.float32)
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0)
    assert qc.confidence == MethodConfidence.UNKNOWN
    assert qc.blockers


def test_signal_qc_short_recording_low_confidence():
    rng = np.random.default_rng(4)
    data = rng.standard_normal((4, 100)).astype(np.float32)  # < 1 second at 256 Hz
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, compute_psd=False)
    assert qc.confidence == MethodConfidence.LOW_CONFIDENCE


def test_signal_qc_result_serializable():
    rng = np.random.default_rng(5)
    data = rng.standard_normal((4, 1000)).astype(np.float32)
    qc = compute_signal_qc(data, sampling_frequency_hz=256.0, compute_psd=False)
    result = qc.to_result()
    assert isinstance(result, NeuroClassicResult)
    json.dumps(result.to_dict())


# ── Image QC ──────────────────────────────────────────────────────────────────

def test_image_qc_clean_3d():
    rng = np.random.default_rng(10)
    data = rng.random((16, 16, 16)).astype(np.float32)
    qc = compute_image_qc(data, voxel_sizes_mm=(1.0, 1.0, 1.0), scope="t1")
    assert not qc.is_constant
    assert not qc.has_nan
    assert qc.image_std > 0


def test_image_qc_constant_blocks():
    data = np.full((8, 8, 8), 5.0, dtype=np.float32)
    qc = compute_image_qc(data, voxel_sizes_mm=(1.0, 1.0, 1.0))
    assert qc.is_constant
    assert qc.blockers
    assert qc.confidence == MethodConfidence.UNKNOWN


def test_image_qc_nan_warns():
    rng = np.random.default_rng(11)
    data = rng.random((8, 8, 8)).astype(np.float32)
    data[0, 0, 0] = np.nan
    qc = compute_image_qc(data, voxel_sizes_mm=(1.0, 1.0, 1.0))
    assert qc.has_nan
    assert any("NaN" in w for w in qc.warnings)


def test_image_qc_inf_blocks():
    rng = np.random.default_rng(12)
    data = rng.random((8, 8, 8)).astype(np.float32)
    data[0, 0, 0] = np.inf
    qc = compute_image_qc(data, voxel_sizes_mm=(1.0, 1.0, 1.0))
    assert qc.has_inf
    assert qc.blockers


def test_image_qc_4d_tsnr():
    rng = np.random.default_rng(13)
    data = rng.random((8, 8, 8, 20)).astype(np.float32) + 100.0
    qc = compute_image_qc(data, voxel_sizes_mm=(2.0, 2.0, 2.0))
    assert qc.n_volumes == 20
    assert qc.tsnr is not None


def test_image_qc_wrong_ndim_raises():
    data = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_image_qc(data)


def test_image_qc_empty():
    data = np.zeros((0,), dtype=np.float32).reshape(0, 0, 0)
    qc = compute_image_qc(data)
    assert qc.blockers


# ── Connectivity ──────────────────────────────────────────────────────────────

def test_pearson_connectivity_shape():
    rng = np.random.default_rng(20)
    data = rng.standard_normal((8, 1000)).astype(np.float32)
    conn = compute_pearson_connectivity(data, sampling_hz=256.0, time_window_s=2.0)
    assert conn.matrix.shape == (8, 8)
    assert conn.n_nodes == 8
    # diagonal correlations are 1
    assert np.allclose(np.diag(conn.matrix), 1.0, atol=1e-6)


def test_pearson_connectivity_symmetric():
    rng = np.random.default_rng(21)
    data = rng.standard_normal((6, 1000)).astype(np.float32)
    conn = compute_pearson_connectivity(data, sampling_hz=256.0)
    assert np.allclose(conn.matrix, conn.matrix.T, atol=1e-6)


def test_connectivity_spec_summary():
    rng = np.random.default_rng(22)
    data = rng.standard_normal((4, 600)).astype(np.float32)
    conn = compute_pearson_connectivity(
        data, sampling_hz=256.0, threshold=0.5,
        channel_names=["A", "B", "C", "D"],
    )
    summary = conn.spec.summary()
    assert "pearson" in summary
    assert "4" in summary


def test_graph_metrics_complete():
    # fully connected graph
    n = 5
    mat = np.ones((n, n))
    conn = ConnectivityMatrix(
        matrix=mat,
        node_labels=[f"n{i}" for i in range(n)],
        spec=compute_pearson_connectivity(
            np.random.default_rng(23).standard_normal((n, 100)).astype(np.float32),
            sampling_hz=100.0,
        ).spec,
    )
    gm = compute_graph_metrics(conn)
    assert gm.n_nodes == n
    assert gm.density == pytest.approx(1.0)
    assert gm.n_connected_components == 1
    assert gm.clustering_coefficient == pytest.approx(1.0)


def test_graph_metrics_disconnected():
    # two disconnected pairs
    mat = np.zeros((4, 4))
    mat[0, 1] = mat[1, 0] = 1.0
    mat[2, 3] = mat[3, 2] = 1.0
    conn = ConnectivityMatrix(
        matrix=mat,
        node_labels=["a", "b", "c", "d"],
        spec=compute_pearson_connectivity(
            np.random.default_rng(24).standard_normal((4, 100)).astype(np.float32),
            sampling_hz=100.0,
        ).spec,
    )
    gm = compute_graph_metrics(conn)
    assert gm.n_connected_components == 2
    assert gm.confidence == MethodConfidence.LOW_CONFIDENCE


# ── Statistics ────────────────────────────────────────────────────────────────

def _make_rows():
    return [
        {"participant_id": "sub-01", "diagnosis": "control", "site": "A", "age": "30"},
        {"participant_id": "sub-02", "diagnosis": "patient", "site": "B", "age": "40"},
        {"participant_id": "sub-03", "diagnosis": "control", "site": "A", "age": "35"},
        {"participant_id": "sub-04", "diagnosis": "patient", "site": "B", "age": "45"},
        {"participant_id": "sub-05", "diagnosis": "control", "site": "A", "age": "28"},
        {"participant_id": "sub-06", "diagnosis": "patient", "site": "B", "age": "50"},
    ]


def test_stats_variable_summaries():
    report = compute_statistical_diagnostics(_make_rows(), target="diagnosis")
    assert report.n_samples == 6
    assert "age" in report.variables
    assert report.variables["age"].dtype == "numeric"
    assert report.variables["diagnosis"].dtype == "categorical"


def test_stats_confound_association_detected():
    # site perfectly predicts diagnosis
    report = compute_statistical_diagnostics(
        _make_rows(), target="diagnosis", confound_columns=["site"],
    )
    assoc = [a for a in report.confound_associations if a.variable_b == "site"]
    assert assoc
    assert assoc[0].effect_size is not None
    assert assoc[0].effect_size > 0.5


def test_stats_class_imbalance():
    report = compute_statistical_diagnostics(_make_rows(), target="diagnosis")
    assert "control" in report.class_imbalance
    assert report.class_imbalance["control"] == pytest.approx(0.5)


def test_stats_small_n_low_confidence():
    rows = _make_rows()[:3]
    report = compute_statistical_diagnostics(rows, target="diagnosis")
    assert report.confidence == MethodConfidence.LOW_CONFIDENCE


def test_stats_empty_unknown():
    report = compute_statistical_diagnostics([], target="diagnosis")
    assert report.confidence == MethodConfidence.UNKNOWN


def test_stats_missingness():
    rows = [
        {"x": "1", "y": "a"},
        {"x": "n/a", "y": "b"},
        {"x": "3", "y": ""},
    ]
    report = compute_statistical_diagnostics(rows)
    assert report.n_missing == 2


def test_cohort_metric_report_outliers():
    results = []
    for i in range(10):
        r = NeuroClassicResult(
            method_name="x", method_version="0.1", modality="eeg",
            scope=f"sub-{i}", inputs={}, parameters={}, assumptions=[],
        )
        from qortex.neuroclassic import MetricResult
        val = 100.0 if i == 5 else 1.0  # one big outlier
        r.add_metric(MetricResult("n_flatline_channels", val))
        results.append(r)
    cr = build_cohort_metric_report(
        results, method_name="x", metric_name="n_flatline_channels", modality="eeg",
    )
    assert cr.n_subjects == 10
    assert 5 in cr.outlier_indices
