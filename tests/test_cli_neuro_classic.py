"""CLI coverage for neuro-classic commands."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

import qortex.neuroclassic as neuroclassic
from qortex.cli.app import app


def test_neuro_classic_connectivity_rejects_unknown_method(tmp_path: Path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["neuro-classic", "connectivity", str(tmp_path), "--method", "coherence"],
    )

    assert result.exit_code != 0
    assert "Unsupported connectivity method" in result.output


def test_neuro_classic_connectivity_honors_max_files(
    tmp_path: Path,
    monkeypatch,
):
    eeg_dir = tmp_path / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    for idx in range(3):
        (eeg_dir / f"sub-01_task-rest_run-{idx + 1:02d}_eeg.edf").write_bytes(b"edf")

    read_paths: list[str] = []

    class Info:
        ch_names = ["C3", "C4"]

        def __getitem__(self, key):
            if key == "sfreq":
                return 128.0
            raise KeyError(key)

    class Raw:
        info = Info()

        def get_data(self):
            return np.ones((2, 8), dtype=np.float32)

    class Connectivity:
        pass

    class GraphMetrics:
        n_nodes = 2
        n_edges = 1
        density = 1.0
        clustering_coefficient = 0.0

        def to_dict(self):
            return {"n_nodes": self.n_nodes, "n_edges": self.n_edges}

    def read_raw_edf(path, *, preload, verbose):
        read_paths.append(path)
        assert preload is True
        assert verbose is False
        return Raw()

    fake_mne = types.SimpleNamespace(
        io=types.SimpleNamespace(read_raw_edf=read_raw_edf)
    )
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setattr(
        neuroclassic,
        "compute_pearson_connectivity",
        lambda *args, **kwargs: Connectivity(),
    )
    monkeypatch.setattr(
        neuroclassic,
        "compute_graph_metrics",
        lambda *args, **kwargs: GraphMetrics(),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["neuro-classic", "connectivity", str(tmp_path), "--max-files", "2"],
    )

    assert result.exit_code == 0, result.output
    assert len(read_paths) == 2
    assert result.output.count("nodes=2") == 2
