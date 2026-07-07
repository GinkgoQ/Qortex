"""Roundtrip coverage for concrete conversion format writers."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from qortex.convert.formats.hdf5 import HDF5Writer
from qortex.convert.formats.huggingface import HuggingFaceWriter
from qortex.convert.formats.webdataset import WebDatasetWriter
from qortex.core.entities import SampleRecord


def _samples() -> list[SampleRecord]:
    return [
        SampleRecord(
            data=np.full((2, 3), i, dtype=np.float32),
            label=i,
            label_name=f"class-{i}",
            subject=f"0{i + 1}",
            task="rest",
            modality="eeg",
            onset=float(i),
            duration=2.0,
            sfreq=256.0,
            split="train",
            provenance={"source": f"sub-0{i + 1}_task-rest_eeg.set"},
        )
        for i in range(2)
    ]


def test_hdf5_writer_roundtrip_uniform_signals(tmp_path: Path):
    h5py = pytest.importorskip("h5py")

    out = HDF5Writer().write(iter(_samples()), tmp_path / "hdf5", metadata={"dataset_id": "ds-test"})

    with h5py.File(out, "r") as f:
        assert f.attrs["dataset_id"] == "ds-test"
        np.testing.assert_array_equal(f["signals"][:], np.stack([s.data for s in _samples()]))
        meta = f["metadata"][:]
        assert meta.shape == (2,)
        assert meta[0]["subject"].decode() == "01"
        assert meta[1]["label"] == 1


def test_webdataset_writer_roundtrip_tar_shards(tmp_path: Path):
    out = WebDatasetWriter().write(iter(_samples()), tmp_path / "wds", shard_size=1)

    index = json.loads((out / "_index.json").read_text())
    assert index == {"n_shards": 2, "n_samples": 2}
    shards = sorted(out.glob("*.tar"))
    assert len(shards) == 2

    with tarfile.open(shards[0], "r") as tf:
        names = sorted(tf.getnames())
        assert names == ["00000_000000.json", "00000_000000.npy"]
        meta = json.loads(tf.extractfile("00000_000000.json").read().decode())
        assert meta["subject"] == "01"
        arr = np.load(io.BytesIO(tf.extractfile("00000_000000.npy").read()))
        np.testing.assert_array_equal(arr, _samples()[0].data)


def test_webdataset_writer_empty_index_is_honest(tmp_path: Path):
    out = WebDatasetWriter().write(iter(()), tmp_path / "empty")

    assert json.loads((out / "_index.json").read_text()) == {"n_shards": 0, "n_samples": 0}
    assert not list(out.glob("*.tar"))


def test_huggingface_writer_roundtrip(tmp_path: Path):
    datasets = pytest.importorskip("datasets")

    out = HuggingFaceWriter().write(iter(_samples()), tmp_path / "hf", metadata={"dataset_id": "ds-test"})
    ds = datasets.load_from_disk(str(out))

    assert len(ds) == 2
    assert ds[0]["subject"] == "01"
    assert ds[1]["label_name"] == "class-1"
    np.testing.assert_array_equal(np.asarray(ds[0]["signal"], dtype=np.float32), _samples()[0].data)
    assert json.loads((out / "qortex_metadata.json").read_text()) == {"dataset_id": "ds-test"}
