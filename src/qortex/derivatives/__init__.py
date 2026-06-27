"""First-class BIDS derivative support.

Indexes the ``derivatives/`` directory, auto-detects pipeline names, links
derivative files back to their raw-data counterparts, and parses QC metrics
from MRIQC and fMRIPrep output directories.

Usage::

    from qortex.derivatives import DerivativeIndexer

    idx = DerivativeIndexer(bids_root=Path("~/.cache/qortex/datasets/ds000001/1.0.0"))
    print(idx.pipelines)              # ["fmriprep", "mriqc", "freesurfer"]
    print(idx.for_subject("sub-01")) # all derivative files for sub-01
    print(idx.for_raw("sub-01/anat/sub-01_T1w.nii.gz"))  # linked derivative files
    qc = idx.qc_table("mriqc")       # polars DataFrame with MRIQC group metrics
"""

from qortex.derivatives.indexer import (
    DerivativeIndexer,
    DerivativeFile,
    PipelineInfo,
    DerivativeIndex,
)

__all__ = [
    "DerivativeIndexer",
    "DerivativeFile",
    "PipelineInfo",
    "DerivativeIndex",
]
