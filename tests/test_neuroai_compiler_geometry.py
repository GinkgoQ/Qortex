"""Real header-based geometry evidence tests for qortex.neuroai.compiler.profile_source.

Uses real NIfTI files (via nibabel) and real EEG files (via mne) constructed
in-memory/on-disk — no byte-string fakes with biomedical-looking filenames.
"""

from __future__ import annotations

import numpy as np
import pytest

from qortex.neuroai.compiler import profile_source


def test_profile_source_reads_real_nifti_header(tmp_path):
    nib = __import__("nibabel")
    data = np.zeros((10, 10, 10), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    path = tmp_path / "volume.nii.gz"
    nib.save(img, str(path))

    profile = profile_source(str(path))

    assert profile.spatial_shape == (10, 10, 10)
    assert profile.voxel_sizes_mm == (1.0, 1.0, 1.0)
    assert profile.orientation == "RAS"
    assert profile.n_channels is None


def test_profile_source_reads_real_eeg_fif_header(tmp_path):
    mne = __import__("mne")
    sfreq = 100.0
    n_channels = 4
    duration_s = 2.0
    n_times = int(sfreq * duration_s)
    data = np.zeros((n_channels, n_times))
    info = mne.create_info(
        ch_names=[f"ch{i}" for i in range(n_channels)],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path / "recording_raw.fif"
    raw.save(str(path), overwrite=True, verbose="ERROR")

    profile = profile_source(str(path))

    assert profile.n_channels == n_channels
    assert profile.sampling_rate_hz == sfreq
    assert profile.duration_s == pytest.approx(duration_s, abs=0.1)
    assert profile.spatial_shape is None


def test_profile_source_non_biomedical_file_leaves_header_fields_none(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("just some plain text, not a biomedical file")

    profile = profile_source(str(path))

    assert profile.spatial_shape is None
    assert profile.voxel_sizes_mm is None
    assert profile.orientation is None
    assert profile.n_channels is None
    assert profile.sampling_rate_hz is None
    assert profile.duration_s is None


def test_profile_source_corrupted_nifti_degrades_without_raising(tmp_path):
    path = tmp_path / "corrupted.nii.gz"
    path.write_bytes(b"not-actually-gzip-nifti-data")

    profile = profile_source(str(path))

    assert profile.spatial_shape is None
    assert profile.voxel_sizes_mm is None
    assert profile.orientation is None
    assert any("NIfTI header could not be read" in note for note in profile.notes)
