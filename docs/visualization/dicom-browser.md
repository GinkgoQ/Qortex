# DICOM Browser

`browse_dicom()` renders a series-level overview of a DICOM directory. It groups files by series, extracts metadata from DICOM headers, and shows one thumbnail per series.

## Python

```python
from qortex.visualize import browse_dicom

result = browse_dicom("dicom/sub-01/")
result.show()
```

The output shows one thumbnail per series with:

- Series description
- Study date
- Modality (MR, CT, PET, etc.)
- Number of slices
- Voxel dimensions
- Center axial slice

## CLI

```bash
qortex dicom-browser dicom/sub-01/ --output dicom_overview.html
```

## What it groups on

DICOM files are grouped by `SeriesInstanceUID`. Within each series, files are sorted by `InstanceNumber`. The center slice is the file at index `n // 2`.

## Supported DICOM types

- Standard DICOM (.dcm, no extension)
- Enhanced DICOM (multi-frame)
- DICOM-SEG (segmentation objects)
- SR (structured reports — shows metadata only, no image)

## Requirements

```bash
pip install "qortex[dicom]"
```

The `dicom` extra pulls in pydicom. Without it, `browse_dicom()` raises an `ImportError`.

## Limitations

- DICOM-RT (radiation therapy dose and structure sets) is not supported.
- Compressed DICOM (JPEG or JPEG 2000 transfer syntax) requires the `pylibjpeg` package. Install it separately if needed.
- Multi-frame enhanced DICOM is supported but tested on a limited set of vendors.

## Converting DICOM to NIfTI

Qortex does not convert DICOM to NIfTI. For that, use dcm2niix before calling Qortex:

```bash
dcm2niix -o output/ -z y dicom/sub-01/
```

After conversion, use the standard NIfTI visualization tools.
