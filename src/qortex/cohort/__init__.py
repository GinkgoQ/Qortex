"""Multi-dataset mega-cohort engine.

Builds harmonized, ML-ready subject pools that span multiple OpenNeuro
datasets — the neuroimaging equivalent of Hugging Face's ``load_dataset``
applied to cross-study cohort construction.

Usage::

    from qortex.cohort import CohortBuilder

    cohort = (
        CohortBuilder()
        .require_modality("mri", datatype="anat", suffix="T1w")
        .min_subjects_per_dataset(10)
        .age_range(18, 45)
        .sex("F")
        .scanner_field_strength(3.0, tolerance=0.25)
        .add_dataset("ds000001")
        .add_dataset("ds000171")
        .add_live_search("healthy controls T1w", min_subjects=20)
        .build()
    )

    print(cohort.summary())         # per-dataset stats table
    cohort.export_monai(Path("./monai"))
    cohort.export_torchio(Path("./torchio"))
    df = cohort.subject_table()     # full polars DataFrame
"""

from qortex.cohort.builder import (
    CohortBuilder,
    CohortManifest,
    CohortSubject,
    CohortDatasetEntry,
)
from qortex.cohort.federated import FederatedCohort, FederatedSubject

__all__ = [
    "CohortBuilder",
    "CohortManifest",
    "CohortSubject",
    "CohortDatasetEntry",
    "FederatedCohort",
    "FederatedSubject",
]
