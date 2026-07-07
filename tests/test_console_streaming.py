"""Console-layer streaming behaviour: the Viewer fires many small requests at
the same NIfTI file, so the API must (a) reuse one streamer per URL across
requests — otherwise every slice re-fetches the header and re-decodes the
volume — and (b) be able to attach a real intensity histogram for the
windowing panel without a second fetch.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("fastapi")

from qortex.console import api  # noqa: E402


def test_streamer_is_reused_per_url():
    api._STREAMER_CACHE.clear()
    url = "https://cdn.example.org/sub-01/anat/sub-01_T1w.nii.gz"
    a = api._streamer_for(url)
    b = api._streamer_for(url)
    assert a is b, "same URL must return the identical cached streamer instance"
    other = api._streamer_for(url + "?v=2")
    assert other is not a, "a different URL must get its own streamer"


def test_histogram_excludes_nonfinite_and_reports_stats():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    arr[0, 0] = np.nan
    h = api._intensity_histogram(arr, bins=16)
    assert h is not None
    assert len(h["counts"]) == 16
    assert len(h["bin_edges"]) == 17
    assert h["n_nonfinite"] == 1
    assert h["n_finite"] == 99
    assert h["min"] == 1.0 and h["max"] == 99.0


def test_histogram_none_for_degenerate_slices():
    assert api._intensity_histogram(np.zeros((4, 4), dtype=np.float32), 16) is None
    assert api._intensity_histogram(np.full((4, 4), np.nan), 16) is None


def test_streamlit_app_import_does_not_require_streamlit():
    app_module = importlib.import_module("qortex.console.app")
    assert callable(app_module.main)
