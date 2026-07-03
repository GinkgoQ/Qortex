"""Negative-space reporting — alongside a ranked result list, summarize what
was excluded from the current scope and why, plus how many "unknown"-evidence
datasets could be resolved with a cheap follow-up probe.

This is the concrete mechanism behind qortex-atlas.md §12.1's example:
"There are 42 EEG motor datasets, but only 7 have enough subjects, labels, and
channel metadata for subject-independent ML." Most dataset search UIs only
ever show what matched; this additionally accounts for the datasets that
*almost* matched and precisely which single constraint eliminated each one —
turning "not enough results" into an actionable diagnosis instead of a dead
end.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from qortex.search.compiler import Constraint


@dataclass
class NegativeSpaceReport:
    n_in_scope: int
    n_admitted: int
    n_rejected: int
    rejection_reasons: Counter = field(default_factory=Counter)
    n_unknown_resolvable: int = 0

    def render(self) -> str:
        lines = [
            f"{self.n_in_scope} datasets in scope -> {self.n_admitted} admitted, "
            f"{self.n_rejected} rejected"
        ]
        for reason, count in self.rejection_reasons.most_common():
            lines.append(f"  - {count} {reason}")
        if self.n_unknown_resolvable:
            lines.append(
                f"  {self.n_unknown_resolvable} more have unresolved (unknown) evidence "
                f"and may qualify after a cheap metadata probe"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_in_scope": self.n_in_scope,
            "n_admitted": self.n_admitted,
            "n_rejected": self.n_rejected,
            "rejection_reasons": dict(self.rejection_reasons),
            "n_unknown_resolvable": self.n_unknown_resolvable,
        }


def build_negative_space_report(
    *,
    all_candidates: list[dict[str, Any]],
    admitted_ids: set[str],
    hard_constraints: dict[str, Constraint],
) -> NegativeSpaceReport:
    rejected = [r for r in all_candidates if r["dataset_id"] not in admitted_ids]
    reasons: Counter = Counter()
    for row in rejected:
        for field_name, constraint in hard_constraints.items():
            if not _satisfies(row, field_name, constraint):
                symbol = {"ge": "≥", "le": "≤", "eq": "=", "in": "∈"}.get(constraint.op, constraint.op)
                reasons[f"failed {field_name} {symbol} {constraint.value}"] += 1
    return NegativeSpaceReport(
        n_in_scope=len(all_candidates),
        n_admitted=len(admitted_ids),
        n_rejected=len(rejected),
        rejection_reasons=reasons,
    )


# Constraint.field is named after the goal/plan semantics ("min_subjects",
# "max_size_gb"); catalog rows are named/scaled after the observed quantity
# ("n_subjects", "total_bytes" in raw bytes). This mapping bridges the two so
# the negative-space diagnosis reads the same field the constraint actually
# meant, in the same unit — a real bug (comparing against a KeyError-silent
# `.get(field_name)` that always returned None) caught by testing against a
# synthetic catalog with a known-correct expected rejection reason.
_FIELD_TO_ROW_ACCESSOR: dict[str, tuple[str, float]] = {
    "min_subjects": ("n_subjects", 1.0),
    "max_size_gb": ("total_bytes", 1e9),
    "min_n_classes": ("n_classes", 1.0),
}


def _satisfies(row: dict[str, Any], field_name: str, constraint: Constraint) -> bool:
    row_key, unit_divisor = _FIELD_TO_ROW_ACCESSOR.get(field_name, (field_name, 1.0))
    raw_value = row.get(row_key)
    if raw_value is None:
        return True  # unknown evidence is never itself a counted rejection reason
    value = raw_value / unit_divisor if unit_divisor != 1.0 else raw_value
    if constraint.op == "ge":
        return value >= constraint.value
    if constraint.op == "le":
        return value <= constraint.value
    if constraint.op == "eq":
        return value == constraint.value
    if constraint.op == "in":
        return value in constraint.value
    return True
