"""QC-based subject filtering for ML-ready cohort construction.

Chains threshold rules against MRIQC group metrics, fMRIPrep confound
summaries, or any TSV/DataFrame with quality metrics. Returns a structured
QCMask that tells downstream code exactly which subjects/sessions/runs pass
all criteria, and why each excluded unit was dropped.

Usage::

    from qortex.qc import QCFilter
    from qortex.derivatives import DerivativeIndexer

    idx  = DerivativeIndexer(bids_root).index
    mask = (
        QCFilter(idx)
        .mriqc_T1w()                    # require group_T1w.tsv exists
        .require("snr_total", ">", 15)  # signal-to-noise
        .require("qi_1", "<", 0.03)     # Mortamet artifact index
        .require("cjv", "<", 0.55)      # coefficient of joint variation
        .fmriprep_bold()
        .require("fd_mean", "<", 0.5)   # framewise displacement
        .require("aor", "<", 0.15)      # AFNI outlier ratio
        .apply()
    )

    print(mask.summary())
    print(mask.passing_subjects)     # list[str] of sub-XX IDs
    print(mask.excluded_subjects)    # dict[str, list[str]] sub → reasons
    df = mask.to_dataframe()         # full per-subject verdict table
"""

from qortex.qc.filter import QCFilter, QCMask, QCRule, QCViolation

__all__ = ["QCFilter", "QCMask", "QCRule", "QCViolation"]
