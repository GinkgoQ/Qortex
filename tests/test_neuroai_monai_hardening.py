from __future__ import annotations

import json
import zipfile

import pytest

from qortex.neuroai.models.monai import (
    MONAIBundleAdapter,
    _load_json,
    _safe_extract_zip,
)
from qortex.neuroai.spec import ModelSpec, RuntimeSpec


def test_load_json_rejects_malformed_config(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        _load_json(path)


def test_safe_extract_zip_blocks_path_traversal(tmp_path):
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    with pytest.raises(ValueError, match="unsafe path"):
        _safe_extract_zip(archive, tmp_path / "extract")


def test_monai_predict_does_not_fallback_to_full_volume_after_sliding_window_error(monkeypatch):
    import torch

    class _FailingInferers:
        @staticmethod
        def sliding_window_inference(*args, **kwargs):
            raise RuntimeError("sliding window failed")

    fake_monai = type("FakeMonai", (), {"inferers": _FailingInferers})
    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id="local"))
    adapter._model = torch.nn.Identity()
    adapter._device = "cpu"
    adapter._inference_settings = {
        "roi_size": (2, 2, 2),
        "sw_batch_size": 1,
        "overlap": 0.25,
        "argmax_axis": 1,
        "label_map": {},
    }
    monkeypatch.setattr("qortex.neuroai.models.monai._require_monai", lambda: fake_monai)

    with pytest.raises(RuntimeError, match="sliding window failed"):
        adapter.predict(torch.zeros((1, 1, 2, 2, 2)))


def test_monai_load_rejects_state_dict_mismatch(tmp_path, monkeypatch):
    import torch

    bundle = tmp_path / "bundle"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "models").mkdir()
    (bundle / "configs" / "metadata.json").write_text("{}", encoding="utf-8")
    (bundle / "configs" / "inference.json").write_text(
        json.dumps({"network_def": {"spatial_dims": 3, "in_channels": 1, "out_channels": 1}}),
        encoding="utf-8",
    )
    torch.save({"unexpected.weight": torch.ones(1)}, bundle / "models" / "model.pt")

    class _FakeParser(dict):
        def read_config(self, path):
            self["network_def"] = torch.nn.Conv3d(1, 1, kernel_size=1)

        def get_parsed_content(self, key):
            return self[key]

    fake_monai = type("FakeMonai", (), {"bundle": type("B", (), {"ConfigParser": _FakeParser})})
    monkeypatch.setattr("qortex.neuroai.models.monai._require_monai", lambda: fake_monai)
    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id=str(bundle)))

    with pytest.raises(RuntimeError, match="state_dict mismatch"):
        adapter.load(RuntimeSpec(device="cpu"))


def _fake_monai_with_conv3d(monkeypatch):
    import torch

    class _FakeParser(dict):
        def read_config(self, path):
            self["network_def"] = torch.nn.Conv3d(1, 1, kernel_size=1)

        def get_parsed_content(self, key):
            return self[key]

    fake_monai = type("FakeMonai", (), {"bundle": type("B", (), {"ConfigParser": _FakeParser})})
    monkeypatch.setattr("qortex.neuroai.models.monai._require_monai", lambda: fake_monai)


def _make_bundle_without_checkpoint(tmp_path):
    bundle = tmp_path / "bundle_no_ckpt"
    (bundle / "configs").mkdir(parents=True)
    (bundle / "models").mkdir()
    (bundle / "configs" / "metadata.json").write_text("{}", encoding="utf-8")
    (bundle / "configs" / "inference.json").write_text(
        json.dumps({"network_def": {"spatial_dims": 3, "in_channels": 1, "out_channels": 1}}),
        encoding="utf-8",
    )
    return bundle


def test_monai_load_refuses_random_init_when_checkpoint_missing(tmp_path, monkeypatch):
    bundle = _make_bundle_without_checkpoint(tmp_path)
    _fake_monai_with_conv3d(monkeypatch)
    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id=str(bundle)))

    with pytest.raises(RuntimeError, match="no models/model.pt checkpoint"):
        adapter.load(RuntimeSpec(device="cpu"))


def test_monai_load_allows_missing_checkpoint_only_with_explicit_opt_in(tmp_path, monkeypatch):
    bundle = _make_bundle_without_checkpoint(tmp_path)
    _fake_monai_with_conv3d(monkeypatch)
    adapter = MONAIBundleAdapter(
        ModelSpec(provider="monai", id=str(bundle), extra={"allow_missing_weights": True})
    )

    adapter.load(RuntimeSpec(device="cpu"))  # must not raise

    assert adapter._loaded is True


def test_generative_bundle_output_schema_reflects_registry_not_hardcoded_segmentation():
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401  (triggers zoo registration)

    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id="monai.mednist_gan"))

    schema = adapter.output_schema()

    assert schema.output_type == "image_generation"
    assert schema.produces_probabilities is False


def test_generative_bundle_predict_refuses_segmentation_style_inference():
    from qortex.core.exceptions import ModelAdapterError
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401

    adapter = MONAIBundleAdapter(ModelSpec(provider="monai", id="monai.mednist_gan"))
    adapter._model = object()  # simulate a loaded (but generative) bundle

    with pytest.raises(ModelAdapterError, match="generative"):
        adapter.predict("fake_batch")
