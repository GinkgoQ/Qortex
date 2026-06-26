# DICOM Troubleshooting

## ImportError: pydicom not found

```
ImportError: DICOM support requires pydicom.
Install with: pip install "qortex[dicom]"
```

```bash
pip install "qortex[dicom]"
```

## Compressed DICOM: pixel data cannot be read

```
AttributeError: 'PixelData' / NotImplementedError: Pixel Data with Transfer Syntax ...
```

The DICOM files use JPEG or JPEG 2000 compression. pydicom needs an additional decoder:

```bash
pip install pylibjpeg pylibjpeg-openjpeg
```

After installing, pydicom will automatically use the decoder when reading compressed files.

## Files not recognized as DICOM

`browse_dicom` walks the directory looking for files with valid DICOM headers. Files without a DICOM preamble are skipped. If your DICOM files have no extension (common for older scanners), they should still be recognized.

If files are being skipped unexpectedly, check:

```python
import pydicom
try:
    ds = pydicom.dcmread("path/to/file")
    print(ds.Modality)
except Exception as e:
    print(f"Not valid DICOM: {e}")
```

## Multi-frame DICOM shows only one slice

Enhanced DICOM stores all slices in a single file. `browse_dicom` handles this by counting `NumberOfFrames`. If it shows only one slice, the file's `NumberOfFrames` tag may be missing or incorrect.

## Segmentation objects (DICOM-SEG)

DICOM-SEG files are recognized and shown as metadata summaries. The thumbnail shows one label contour on the reference slice, not a rendering of the full segmentation.

For full DICOM-SEG rendering, use a dedicated viewer like 3D Slicer.
