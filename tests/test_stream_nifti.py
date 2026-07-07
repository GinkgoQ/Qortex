"""Correctness tests for NiftiStreamer, focused on the axis=2 single-slice
fast path (_fetch_slice_contiguous).

Uses a real synthetic NIfTI-1 file written via nibabel — NiftiStreamer
supports local paths directly (byte-range reads become plain seek+read), so
this exercises the exact same code path a remote HTTP Range fetch would,
with no network and no mocking. Every voxel is filled with a value encoding
its own (x, y, z) index, so any mixed-up axis/stride/orientation bug in the
fast-path byte-math shows up as a wrong *value*, not just a wrong shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from qortex.stream.nifti import NiftiStreamer  # noqa: E402


def _index_encoded_volume(nx: int, ny: int, nz: int) -> np.ndarray:
    """A volume where voxel (x, y, z) == x*10000 + y*100 + z, so any axis
    confusion in slice extraction produces values that don't match the
    expected pattern instead of silently looking like a plausible slice."""
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return (x * 10000 + y * 100 + z).astype(np.int16)


@pytest.fixture
def synthetic_nii(tmp_path: Path) -> tuple[Path, np.ndarray]:
    nx, ny, nz = 12, 16, 20
    vol = _index_encoded_volume(nx, ny, nz)
    img = nib.Nifti1Image(vol, affine=np.eye(4))
    path = tmp_path / "synthetic.nii"
    nib.save(img, str(path))
    return path, vol


@pytest.fixture
def synthetic_nii_gz(tmp_path: Path) -> tuple[Path, np.ndarray]:
    nx, ny, nz = 8, 10, 6
    vol = _index_encoded_volume(nx, ny, nz)
    img = nib.Nifti1Image(vol, affine=np.eye(4))
    path = tmp_path / "synthetic.nii.gz"
    nib.save(img, str(path))
    return path, vol


class TestAxialFastPath:
    """axis=2 uses _fetch_slice_contiguous (one Range request) — the fix
    under test. Every case is checked against nibabel's own ground truth,
    not against the library's own slow path, so a bug shared by both would
    still be caught."""

    def test_matches_ground_truth_for_every_axial_index(self, synthetic_nii):
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        nz = vol.shape[2]
        for z in range(nz):
            slc = streamer.get_slice(axis=2, index=z)
            assert slc.shape == (vol.shape[0], vol.shape[1])
            np.testing.assert_array_equal(slc, vol[:, :, z])

    def test_fast_path_matches_whole_volume_fallback_exactly(self, synthetic_nii):
        """Force the slow (whole-volume) path via get_volume() and compare
        against get_slice()'s fast path — same file, same index, must agree
        byte-for-byte, proving the optimization didn't change behavior."""
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        full = streamer.get_volume(t=0)
        for z in (0, 5, vol.shape[2] - 1):
            fast = streamer.get_slice(axis=2, index=z)
            np.testing.assert_array_equal(fast, full[:, :, z])

    def test_fetches_only_one_slice_worth_of_bytes(self, synthetic_nii, monkeypatch):
        """The whole point of the fix: a single axial slice must not pull
        the entire volume over the wire. Assert the actual byte range
        requested is bounded by one slice's size, not the volume's."""
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path, cache_backend="memory")
        nx, ny, nz = vol.shape
        itemsize = vol.dtype.itemsize
        expected_slice_bytes = nx * ny * itemsize

        requested_lengths = []
        real_fetch = streamer._fetch_bytes

        def spy(start, length, label=""):
            requested_lengths.append((label, length))
            return real_fetch(start, length, label)

        monkeypatch.setattr(streamer, "_fetch_bytes", spy)
        streamer.get_slice(axis=2, index=nz // 2)

        data_fetches = [length for label, length in requested_lengths if label.startswith("slice-ax2")]
        assert data_fetches, "expected the fast path to be used for axis=2"
        assert data_fetches[0] == expected_slice_bytes
        assert data_fetches[0] < nx * ny * nz * itemsize  # strictly less than the whole volume


class TestNonAxialFallback:
    """axis=0/1 are not contiguous in Fortran storage — must still fall back
    to the whole-volume fetch and remain correct (just not optimized)."""

    def test_sagittal_matches_ground_truth(self, synthetic_nii):
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        x = 3
        slc = streamer.get_slice(axis=0, index=x)
        assert slc.shape == (vol.shape[1], vol.shape[2])
        np.testing.assert_array_equal(slc, vol[x, :, :])

    def test_coronal_matches_ground_truth(self, synthetic_nii):
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        y = 4
        slc = streamer.get_slice(axis=1, index=y)
        assert slc.shape == (vol.shape[0], vol.shape[2])
        np.testing.assert_array_equal(slc, vol[:, y, :])


class TestGzipUnaffected:
    """.nii.gz never takes the new fast path (gzip can't be randomly seeked
    into) — confirm axis=2 still works correctly there too, unchanged."""

    def test_axial_slice_from_gz(self, synthetic_nii_gz):
        path, vol = synthetic_nii_gz
        streamer = NiftiStreamer(path)
        z = vol.shape[2] // 2
        slc = streamer.get_slice(axis=2, index=z)
        np.testing.assert_array_equal(slc, vol[:, :, z])


class TestFourD:
    """4D (fMRI-shaped) volumes: the fast path must index into the correct
    timepoint's slice, not always t=0."""

    def test_axial_slice_at_nonzero_timepoint(self, tmp_path: Path):
        nx, ny, nz, nt = 6, 8, 5, 4
        base = _index_encoded_volume(nx, ny, nz).astype(np.int32)
        vol4d = np.stack([base + t * 1_000_000 for t in range(nt)], axis=-1).astype(np.int32)
        img = nib.Nifti1Image(vol4d, affine=np.eye(4))
        path = tmp_path / "synthetic4d.nii"
        nib.save(img, str(path))

        streamer = NiftiStreamer(path)
        for t in range(nt):
            z = nz // 2
            slc = streamer.get_slice(axis=2, index=z, t=t)
            np.testing.assert_array_equal(slc, vol4d[:, :, z, t])

    def test_negative_timepoint_rejected_for_slice_and_volume(self, tmp_path: Path):
        nx, ny, nz, nt = 4, 5, 6, 2
        base = _index_encoded_volume(nx, ny, nz).astype(np.int32)
        vol4d = np.stack([base + t * 1_000_000 for t in range(nt)], axis=-1)
        img = nib.Nifti1Image(vol4d, affine=np.eye(4))
        path = tmp_path / "synthetic4d.nii"
        nib.save(img, str(path))

        streamer = NiftiStreamer(path)
        with pytest.raises(IndexError, match=r"Volume index -1 out of range"):
            streamer.get_slice(axis=2, index=nz // 2, t=-1)
        with pytest.raises(IndexError, match=r"Volume index -1 out of range"):
            streamer.get_volume(t=-1)


class TestProjections:
    """Correctness of the MIP/MinIP/mean projection math the console API's
    /nifti-projection-data route relies on (get_volume() + numpy max/min/
    mean along an axis) — the route itself isn't unit-tested here (it's a
    thin composition of this, already-tested auto_window/presets, and live
    manifest/URL resolution exercised via curl against the real API in
    development), but the actual numeric computation is, and shape-per-axis
    must match /nifti-slice-data's convention exactly since the frontend
    renders both through the same pipeline."""

    def test_mip_matches_ground_truth_every_axis(self, synthetic_nii):
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        full = streamer.get_volume(t=0, dtype=np.float32)
        for axis in (0, 1, 2):
            np.testing.assert_array_equal(full.max(axis=axis), vol.max(axis=axis).astype(np.float32))

    def test_minip_matches_ground_truth(self, synthetic_nii):
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        full = streamer.get_volume(t=0, dtype=np.float32)
        np.testing.assert_array_equal(full.min(axis=1), vol.min(axis=1).astype(np.float32))

    def test_mean_projection_matches_ground_truth(self, synthetic_nii):
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        full = streamer.get_volume(t=0, dtype=np.float32)
        np.testing.assert_allclose(full.mean(axis=2), vol.mean(axis=2))

    def test_projection_shape_matches_slice_shape_convention(self, synthetic_nii):
        # /nifti-projection-data must produce the same (rows, cols) shape per
        # axis that /nifti-slice-data does, since the frontend renders a
        # projection through the exact same renderSliceToCanvas() pipeline.
        path, vol = synthetic_nii
        streamer = NiftiStreamer(path)
        full = streamer.get_volume(t=0, dtype=np.float32)
        for axis in (0, 1, 2):
            proj_shape = full.max(axis=axis).shape
            slice_shape = streamer.get_slice(axis=axis, index=0).shape
            assert proj_shape == slice_shape


class TestBigEndian:
    """Big-endian NIfTI files are valid per spec and produced by some older
    FSL/AFNI pipelines and PPC-era scanners. NIfTI carries no explicit
    endianness flag, so a hand-rolled reader must detect it from sizeof_hdr;
    before that detection existed the streamer read every field (and the
    voxel data) as little-endian, turning a 4×5×6 volume into 1024×1280×1536
    garbage. These pin down that a big-endian file is now read identically to
    its little-endian twin, both header and data."""

    @staticmethod
    def _save(tmp_path: Path, vol: np.ndarray, *, endian: str, gz: bool):
        # nibabel writes native byte order by default; force the opposite by
        # setting the header's dtype byte order explicitly.
        img = nib.Nifti1Image(vol, affine=np.eye(4))
        img.header.set_data_dtype(np.dtype(vol.dtype).newbyteorder(endian))
        ext = ".nii.gz" if gz else ".nii"
        path = tmp_path / f"be_test{ext}"
        nib.save(img, str(path))
        return path

    def test_bigendian_header_matches_littleendian(self, tmp_path: Path):
        vol = _index_encoded_volume(6, 7, 8)
        be = NiftiStreamer(self._save(tmp_path, vol, endian=">", gz=False))
        hdr = be.header()
        assert hdr.shape == (6, 7, 8)
        assert hdr.dtype.itemsize == 2
        assert hdr.vox_offset >= 348

    def test_bigendian_slice_matches_ground_truth(self, tmp_path: Path):
        vol = _index_encoded_volume(6, 7, 8)
        be = NiftiStreamer(self._save(tmp_path, vol, endian=">", gz=False))
        for z in range(vol.shape[2]):
            got = be.get_slice(axis=2, index=z)
            np.testing.assert_array_equal(got, vol[:, :, z])

    def test_bigendian_gz_slice_matches_ground_truth(self, tmp_path: Path):
        vol = _index_encoded_volume(5, 6, 7)
        be = NiftiStreamer(self._save(tmp_path, vol, endian=">", gz=True))
        got = be.get_slice(axis=2, index=3)
        np.testing.assert_array_equal(got, vol[:, :, 3])
