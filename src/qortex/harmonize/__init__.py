"""Tensor harmonization — detect and report spatial incompatibilities across subjects.

Usage::

    from qortex.harmonize import HarmonizationReporter

    reporter = HarmonizationReporter()
    report = reporter.from_records(image_records)
    print(report.summary())
    report.to_json("harmonization.json")
    target = report.resampling_target()  # consensus TensorSpec for resampling
"""

from qortex.harmonize.reporter import (
    HarmonizationReporter,
    HarmonizationReport,
    TensorSpec,
    HarmonizationGroup,
    HarmonizationIssue,
    IssueSeverity,
)

__all__ = [
    "HarmonizationReporter",
    "HarmonizationReport",
    "TensorSpec",
    "HarmonizationGroup",
    "HarmonizationIssue",
    "IssueSeverity",
]
