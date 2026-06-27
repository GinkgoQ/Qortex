"""Automated Hugging Face-style dataset card generator.

Produces a Markdown ``README.md`` with YAML front matter, demographic
statistics tables, per-label distribution, per-dataset provenance, and an
ML readiness self-assessment — compatible with the Hugging Face Hub
dataset card specification.

Called internally by ``FederatedCohort.generate_dataset_card()``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qortex.cohort.federated import FederatedCohort

log = logging.getLogger(__name__)


_CARD_TEMPLATE = """\
---
{yaml_front_matter}
---

# {cohort_name}

{description}

## Dataset Summary

| Property | Value |
|----------|-------|
| Cohort name | `{cohort_name}` |
| Total subjects | {n_subjects} |
| Datasets | {n_datasets} |
| Generated at | {generated_at} |
| ML task | {ml_task} |
| License | {license} |

## Label Distribution

{label_table}

## Demographic Summary

{demographic_table}

## Train / Validation / Test Splits

{split_table}

## Dataset Provenance

{provenance_section}

## Harmonization Applied

{harmonization_section}

## Data Leakage Mitigation

{leakage_section}

## ML Readiness Assessment

{readiness_section}

## Citation

If you use this cohort in your research, please cite the original OpenNeuro datasets:

{citations}

---
*Generated automatically by [Qortex](https://github.com/GinkgoQ/qortex)*
"""

_YAML_TEMPLATE = """\
annotations_creators:
  - found
language_creators:
  - found
language:
  - en
license: {license}
multilinguality:
  - monolingual
size_categories:
{size_category}
source_datasets:
{source_datasets}
task_categories:
{task_categories}
task_ids:
  - {ml_task}
tags:
  - neuroimaging
  - bids
  - openneuro
  - {modality_tag}
  - medical"""


class DatasetCardGenerator:
    """Generate a Hugging Face-style Markdown dataset card for a FederatedCohort.

    Parameters
    ----------
    cohort:
        A built ``FederatedCohort`` instance.
    """

    def __init__(self, cohort: "FederatedCohort") -> None:
        self._cohort = cohort

    def generate(
        self,
        output_dir: Path,
        *,
        model_type: str = "classification",
        task_categories: list[str] | None = None,
        license: str = "CC-BY-4.0",
    ) -> Path:
        """Write ``README.md`` to ``output_dir``.

        Parameters
        ----------
        output_dir:
            Directory to write the README.md into.
        model_type:
            Primary ML task type for YAML front matter, e.g.
            ``"classification"``, ``"segmentation"``, ``"regression"``.
        task_categories:
            Hugging Face task category strings.
        license:
            SPDX license identifier.

        Returns
        -------
        Path
            Path to the written ``README.md``.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        card_path = output_dir / "README.md"
        content = self._render(
            model_type=model_type,
            task_categories=task_categories or ["medical-imaging"],
            license=license,
        )
        card_path.write_text(content, encoding="utf-8")
        log.info("Dataset card written to %s", card_path)

        # Also write a JSON metadata sidecar for programmatic consumption
        meta_path = output_dir / "cohort_meta.json"
        meta = self._json_meta(model_type=model_type, license=license)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return card_path

    # ── Rendering ──────────────────────────────────────────────────────────

    def _render(
        self,
        model_type: str,
        task_categories: list[str],
        license: str,
    ) -> str:
        c = self._cohort
        subjects = c.subjects
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        label_dist = c._label_distribution()
        split_counts = c._split_counts()
        modalities = sorted({m for s in subjects for m in s.modalities})
        modality_tag = modalities[0] if modalities else "neuroimaging"

        size_cat = _size_category(c.n_subjects)

        yaml_fm = _YAML_TEMPLATE.format(
            license=license,
            size_category=f"  - {size_cat}",
            source_datasets="\n".join(f"  - {ds}" for ds in c.dataset_ids),
            task_categories="\n".join(f"  - {t}" for t in task_categories),
            ml_task=model_type,
            modality_tag=modality_tag,
        )

        description = _make_description(c)
        label_table = _label_markdown_table(label_dist)
        demo_table = _demographics_markdown_table(subjects)
        split_table = _split_markdown_table(split_counts, label_dist, subjects)
        provenance = _provenance_section(c)
        harmonization = _harmonization_section(c)
        leakage = _leakage_section(c)
        readiness = _readiness_section(c, model_type)
        citations = _citations_section(c.dataset_ids)

        return _CARD_TEMPLATE.format(
            yaml_front_matter=yaml_fm,
            cohort_name=c.name,
            description=description,
            n_subjects=c.n_subjects,
            n_datasets=len(c.dataset_ids),
            generated_at=now_str,
            ml_task=model_type,
            license=license,
            label_table=label_table,
            demographic_table=demo_table,
            split_table=split_table,
            provenance_section=provenance,
            harmonization_section=harmonization,
            leakage_section=leakage,
            readiness_section=readiness,
            citations=citations,
        )

    def _json_meta(self, *, model_type: str, license: str) -> dict[str, Any]:
        c = self._cohort
        subjects = c.subjects
        age_vals = [s.age for s in subjects if s.age is not None]
        sex_dist = dict(Counter(s.sex or "unknown" for s in subjects))
        return {
            "cohort_name": c.name,
            "n_subjects": c.n_subjects,
            "n_datasets": len(c.dataset_ids),
            "dataset_ids": c.dataset_ids,
            "label_distribution": c._label_distribution(),
            "split_counts": c._split_counts(),
            "age_stats": {
                "min": min(age_vals) if age_vals else None,
                "max": max(age_vals) if age_vals else None,
                "mean": sum(age_vals) / len(age_vals) if age_vals else None,
                "n_known": len(age_vals),
            },
            "sex_distribution": sex_dist,
            "modalities": sorted({m for s in subjects for m in s.modalities}),
            "ml_task": model_type,
            "license": license,
            "leakage_check_applied": c._do_leakage_check,
            "balance_applied": c._balance_col is not None,
            "balance_column": c._balance_col,
            "harmonize_rules": c._harmonize_rules,
        }


# ── Section builders ──────────────────────────────────────────────────────────

def _make_description(c: "FederatedCohort") -> str:
    from collections import Counter
    subjects = c.subjects
    label_dist = c._label_distribution()
    age_vals = [s.age for s in subjects if s.age is not None]
    modalities = sorted({m for s in subjects for m in s.modalities})
    label_summary = ", ".join(f"{k}: {v}" for k, v in sorted(label_dist.items()))
    age_range = (
        f"Ages range from {min(age_vals):.0f} to {max(age_vals):.0f} "
        f"(mean {sum(age_vals)/len(age_vals):.1f})."
        if age_vals else ""
    )
    return (
        f"A harmonized multi-site neuroimaging cohort assembled from "
        f"{len(c.dataset_ids)} OpenNeuro dataset(s): "
        f"{', '.join(c.dataset_ids)}. "
        f"The cohort comprises {c.n_subjects} subjects with the following "
        f"label distribution: {label_summary}. "
        f"{age_range} "
        f"Available modalities: {', '.join(modalities) or 'unspecified'}."
    )


def _label_markdown_table(label_dist: dict[str, int]) -> str:
    if not label_dist:
        return "_No label information available._"
    total = sum(label_dist.values())
    rows = [
        f"| {label} | {count} | {100*count/total:.1f}% |"
        for label, count in sorted(label_dist.items(), key=lambda x: -x[1])
    ]
    header = "| Label | Count | % |\n|-------|-------|---|"
    return header + "\n" + "\n".join(rows)


def _demographics_markdown_table(subjects: list[Any]) -> str:
    from collections import Counter
    if not subjects:
        return "_No subjects._"

    age_vals = [s.age for s in subjects if s.age is not None]
    sex_counts = Counter(s.sex or "unknown" for s in subjects)
    field_counts = Counter(
        f"{s.field_strength_T:.1f}T" if s.field_strength_T else "unknown"
        for s in subjects
    )
    site_counts = Counter(s.site or "unknown" for s in subjects)
    top_sites = sorted(site_counts.items(), key=lambda x: -x[1])[:5]
    site_str = "; ".join(f"{k}: {v}" for k, v in top_sites)

    age_str = (
        f"min={min(age_vals):.0f}, max={max(age_vals):.0f}, "
        f"mean={sum(age_vals)/len(age_vals):.1f}, n={len(age_vals)}"
        if age_vals else "unknown"
    )
    sex_str = "; ".join(f"{k}: {v}" for k, v in sorted(sex_counts.items()))
    field_str = "; ".join(f"{k}: {v}" for k, v in sorted(field_counts.items(), key=lambda x: -x[1])[:4])

    rows = [
        f"| Age | {age_str} |",
        f"| Sex | {sex_str} |",
        f"| Field strength | {field_str} |",
        f"| Sites | {site_str} |",
        f"| Total subjects | {len(subjects)} |",
    ]
    header = "| Attribute | Value |\n|-----------|-------|"
    return header + "\n" + "\n".join(rows)


def _split_markdown_table(
    split_counts: dict[str, int],
    label_dist: dict[str, int],
    subjects: list[Any],
) -> str:
    from collections import Counter
    if not split_counts:
        return "_No split information._"
    total = sum(split_counts.values())
    rows = []
    for split in ("train", "val", "test"):
        n = split_counts.get(split, 0)
        pct = 100 * n / total if total else 0.0
        split_subs = [s for s in subjects if s.split == split]
        label_in_split = Counter(
            s.harmonized_label or "unknown" for s in split_subs
        )
        label_str = "; ".join(f"{k}:{v}" for k, v in sorted(label_in_split.items()))
        rows.append(f"| {split} | {n} | {pct:.1f}% | {label_str} |")
    header = "| Split | Subjects | % | Label breakdown |\n|-------|----------|---|----------------|"
    return header + "\n" + "\n".join(rows)


def _provenance_section(c: "FederatedCohort") -> str:
    from collections import Counter
    subjects = c.subjects
    ds_counts = Counter(s.dataset_id for s in subjects)
    snap_map: dict[str, str] = {}
    for s in subjects:
        snap_map[s.dataset_id] = s.snapshot

    lines = ["| Dataset ID | Snapshot | Subjects |", "|------------|----------|---------|"]
    for ds_id in sorted(ds_counts.keys()):
        snap = snap_map.get(ds_id, "?")
        lines.append(f"| `{ds_id}` | `{snap}` | {ds_counts[ds_id]} |")
    return "\n".join(lines)


def _harmonization_section(c: "FederatedCohort") -> str:
    if not c._harmonize_rules:
        return "_No harmonization rules applied._"
    lines = []
    for col, mapping in c._harmonize_rules.items():
        lines.append(f"\n**Column: `{col}`**")
        lines.append("| Original Value | Harmonized To |")
        lines.append("|----------------|---------------|")
        for raw, canonical in sorted(mapping.items()):
            lines.append(f"| `{raw}` | `{canonical}` |")
    return "\n".join(lines)


def _leakage_section(c: "FederatedCohort") -> str:
    if not c._do_leakage_check:
        return (
            "> ⚠️  **No leakage detection was applied.** "
            "Subjects appearing in multiple datasets may be present in both "
            "training and test splits."
        )
    return (
        f"Cryptographic leakage detection was applied using the "
        f"`{c._leakage_method}` method. Subjects are fingerprinted by "
        f"(age bucket, sex, field strength, diagnosis) — any subject whose "
        f"fingerprint appears across multiple datasets has their duplicates "
        f"removed. Subjects are assigned to splits deterministically via "
        f"SHA-256 keyed shuffle, ensuring all data for one subject goes to "
        f"exactly one split."
    )


def _readiness_section(c: "FederatedCohort", model_type: str) -> str:
    issues: list[str] = []
    checks: list[tuple[bool, str]] = []

    subjects = c.subjects
    n = c.n_subjects
    label_dist = c._label_distribution()
    n_labels = len([k for k in label_dist if k != "unknown"])
    n_labeled = sum(v for k, v in label_dist.items() if k != "unknown")
    n_unknown = label_dist.get("unknown", 0)

    checks.append((n >= 20, f"Minimum subjects (20): {n}"))
    checks.append((n_labels >= 2, f"At least 2 classes: {n_labels}"))
    checks.append((n_unknown / max(n, 1) < 0.2, f"Label coverage > 80%: {100*n_labeled/max(n,1):.0f}%"))
    checks.append((c._do_leakage_check, "Leakage detection applied"))
    checks.append((c._balance_col is not None, "Demographic balancing applied"))

    if "train" in c._split_counts():
        train_n = c._split_counts().get("train", 0)
        checks.append((train_n >= 10, f"Train split has ≥ 10 subjects: {train_n}"))

    # Imbalance check
    if label_dist and n_labels > 1:
        vals = [v for k, v in label_dist.items() if k != "unknown"]
        ratio = max(vals) / max(min(vals), 1)
        checks.append((ratio < 5.0, f"Class imbalance ratio < 5:1 (actual {ratio:.1f}:1)"))

    passed = sum(1 for ok, _ in checks if ok)
    grade = "A" if passed == len(checks) else "B" if passed >= len(checks) * 0.8 else "C" if passed >= len(checks) * 0.6 else "D"

    lines = [
        f"**ML Readiness Grade: {grade}** ({passed}/{len(checks)} checks passed)\n",
        "| Check | Status |",
        "|-------|--------|",
    ]
    for ok, desc in checks:
        status = "✅ Pass" if ok else "❌ Fail"
        lines.append(f"| {desc} | {status} |")

    if grade in ("C", "D"):
        lines.append("\n> **Recommendation:** Address the failing checks before training. "
                     "Consider adding more subjects, applying label harmonization, or "
                     "enabling demographic balancing.")

    return "\n".join(lines)


def _citations_section(dataset_ids: list[str]) -> str:
    lines = []
    for ds_id in sorted(dataset_ids):
        lines.append(
            f"- [{ds_id}](https://openneuro.org/datasets/{ds_id}) — "
            f"OpenNeuro dataset {ds_id}"
        )
    return "\n".join(lines) if lines else "_No datasets._"


def _size_category(n: int) -> str:
    if n < 100:
        return "n<1K"
    elif n < 1000:
        return "1K<n<10K"
    elif n < 10_000:
        return "10K<n<100K"
    else:
        return "100K<n<1M"
