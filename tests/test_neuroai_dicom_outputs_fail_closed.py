from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.outputs._base import OutputAdapterError
from qortex.neuroai.outputs.dicom_seg_out import DICOMSEGOutputAdapter
from qortex.neuroai.outputs.dicom_sr_out import DICOMSROutputAdapter


def test_dicom_seg_geometry_mismatch_raises_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("qortex.neuroai.outputs.dicom_seg_out._require_highdicom", lambda: object())
    monkeypatch.setattr("qortex.neuroai.outputs.dicom_seg_out._require_pydicom", lambda: object())
    adapter = DICOMSEGOutputAdapter(tmp_path / "seg.dcm")
    adapter.open()
    adapter._source_datasets = [
        SimpleNamespace(Rows=4, Columns=4),
        SimpleNamespace(Rows=4, Columns=4),
    ]

    output = ModelOutput(
        output_type="segmentation",
        raw=None,
        mask=np.zeros((1, 4, 4), dtype=np.uint8),
    )

    with pytest.raises(OutputAdapterError, match="geometry"):
        adapter.write(output)

    assert adapter.n_written == 0
    assert not list(tmp_path.glob("*.npy"))


def test_dicom_seg_construction_failure_raises_without_npy_fallback(tmp_path, monkeypatch):
    class _FailingSeg:
        class seg:
            class SegmentAlgorithmTypes:
                AUTOMATIC = "AUTOMATIC"

            class SegmentationTypeValues:
                BINARY = "BINARY"

            class SegmentDescription:
                def __init__(self, *args, **kwargs):
                    pass

            class Segmentation:
                def __init__(self, *args, **kwargs):
                    raise TypeError("bad highdicom call")

        class sr:
            class coding:
                class codes:
                    class SCT:
                        MorphologicallyAbnormalStructure = "category"
                        Nodule = "type"

                    class DCM:
                        ArtificialIntelligence = "ai"

        class AlgorithmIdentificationSequence:
            def __init__(self, *args, **kwargs):
                pass

        @staticmethod
        def UID():
            return "1.2.3"

    monkeypatch.setattr("qortex.neuroai.outputs.dicom_seg_out._require_highdicom", lambda: _FailingSeg)
    monkeypatch.setattr("qortex.neuroai.outputs.dicom_seg_out._require_pydicom", lambda: object())
    adapter = DICOMSEGOutputAdapter(tmp_path / "seg.dcm")
    adapter.open()
    output = ModelOutput(
        output_type="segmentation",
        raw=None,
        mask=np.zeros((1, 4, 4), dtype=np.uint8),
    )

    with pytest.raises(OutputAdapterError, match="DICOM SEG creation failed"):
        adapter.write(output)

    assert adapter.n_written == 0
    assert not list(tmp_path.glob("*.npy"))


def test_dicom_sr_construction_failure_raises_without_json_fallback(tmp_path, monkeypatch):
    class _FailingSR:
        class sr:
            class coding:
                class codes:
                    class DCM:
                        Device = "device"

                    class LN:
                        CTUnspecifiedBodyRegion = "ct"

            class templates:
                class ObserverContext:
                    def __init__(self, *args, **kwargs):
                        raise TypeError("bad highdicom call")

                class DeviceObserverIdentifyingAttributes:
                    def __init__(self, *args, **kwargs):
                        pass

                class MeasurementReport:
                    def __init__(self, *args, **kwargs):
                        pass

                class ObservationContext:
                    def __init__(self, *args, **kwargs):
                        pass

            class EnhancedSR:
                def __init__(self, *args, **kwargs):
                    pass

        @staticmethod
        def UID():
            return "1.2.3"

    monkeypatch.setattr("qortex.neuroai.outputs.dicom_sr_out._require_highdicom", lambda: _FailingSR)
    adapter = DICOMSROutputAdapter(tmp_path / "sr.dcm")
    adapter.open()
    output = ModelOutput(output_type="classification", raw=None, class_name="x")

    with pytest.raises(OutputAdapterError, match="DICOM SR creation failed"):
        adapter.write(output)

    assert adapter.n_written == 0
    assert not list(tmp_path.glob("*.json"))
