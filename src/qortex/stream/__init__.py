"""Lazy remote streaming for neuroimaging and electrophysiology files.

Enables zero-download data access via HTTP byte-range requests. Streams only
the bytes needed for a given header, slice, or epoch — no multi-gigabyte
download required.

Usage::

    from qortex.stream import NiftiStreamer, EDFStreamer

    # Stream T1w header from OpenNeuro CDN (< 65 KB transferred)
    streamer = NiftiStreamer("https://cdn.openneuro.org/...T1w.nii.gz")
    hdr = streamer.header()
    print(hdr.shape, hdr.voxel_sizes_mm)

    # Extract one 2D axial slice (MNI z=0) without downloading the full volume
    arr = streamer.get_slice(axis=2, index=128)  # shape (256, 256)

    # For 4D fMRI: stream one 3D volume by time index
    vol = streamer.get_volume(t=50)              # shape (x, y, z)

    # EDF streaming: fetch 10 s epoch for specific channels
    edf = EDFStreamer("https://cdn.openneuro.org/.../sub-01_task-rest_eeg.edf")
    epoch = edf.get_epoch(tmin=10.0, tmax=20.0, channels=["Cz", "Pz", "Oz"])
"""

from qortex.stream.nifti import NiftiStreamer, NiftiStreamHeader
from qortex.stream.edf import EDFStreamer, EDFStreamHeader

__all__ = [
    "NiftiStreamer",
    "NiftiStreamHeader",
    "EDFStreamer",
    "EDFStreamHeader",
]
