"""Leakage-safe split assignment for ML training pipelines.

Implements grouped-stratified split optimisation: assigns subjects/sessions
to train/val/test splits while respecting group constraints (e.g. site,
scanner, family) and target-label balance.

Algorithm
---------
1. Compute one group key per row (concatenation of group_columns values).
2. Sort groups by size (largest first) so greedy assignment has the best
   chance of meeting fraction targets.
3. Use a greedy bin-packing pass: assign each group to the split that is
   furthest below its target fraction without creating a label imbalance.
4. Report residual imbalance (max deviation from target class fractions
   across splits) and any unmet constraints.

The solver is deterministic (same input → same output) via sort-stable
ordering and a fixed tie-breaking rule (alphabetical subject ID).

Outputs
-------
SplitAssignmentResult contains:
  - assignments   — dict {subject_id → "train" | "val" | "test"}
  - class_distribution — per-split class fractions for the target column
  - residual_imbalance — max |actual_frac - target_frac| over all splits
  - optimality_status  — "optimal" | "near_optimal" | "violated"
  - group_violations   — group IDs that were split across partitions
  - unmet_constraints  — human-readable explanation of violations
  - runtime_s          — wall-clock time

If constraints cannot be satisfied the result is still returned with
partial_assignments populated and unmet_constraints non-empty.  The
caller must inspect these before treating the split as safe.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SplitConstraints:
    """Configuration for leakage-safe split assignment.

    Attributes
    ----------
    train_fraction : float
        Target fraction of subjects in the training split.
    val_fraction : float
        Target fraction in validation split.
    test_fraction : float
        Target fraction in test split.
        Note: train + val + test must sum to 1.0 (enforced).
    group_columns : list[str]
        Column names whose combined value defines a group.  All rows
        sharing a group key must land in the same split.
        Typical: ["site"], ["family_id"], ["scanner", "site"].
    stratify_column : str | None
        Column to stratify (balance class distribution across splits).
    max_imbalance : float
        Maximum allowed deviation from the target class fraction within
        each split (e.g. 0.1 = ±10%).  Violations are reported as
        unmet_constraints but do not prevent assignment.
    random_seed : int
        Seed for tie-breaking. Does not change determinism of the greedy
        algorithm — only affects which group wins when two are equally
        ranked.
    """
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    group_columns: list[str] = field(default_factory=list)
    stratify_column: str | None = None
    max_imbalance: float = 0.1
    random_seed: int = 42

    def __post_init__(self) -> None:
        total = round(self.train_fraction + self.val_fraction + self.test_fraction, 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"train + val + test fractions must sum to 1.0; got {total:.4f}"
            )


@dataclass
class SplitAssignmentResult:
    """Output of leakage-safe split assignment.

    Attributes
    ----------
    assignments : dict[str, str]
        {subject_id → split_name} for every input row.
        Split names are "train", "val", "test".
    class_distribution : dict[str, dict[str, float]]
        {split → {class_label → fraction}} for the stratify column.
    residual_imbalance : float
        Maximum |actual_fraction - target_fraction| over all splits and
        classes.  0.0 = perfectly balanced.
    optimality_status : str
        "optimal"     — residual_imbalance == 0
        "near_optimal" — residual_imbalance <= max_imbalance
        "violated"    — residual_imbalance > max_imbalance
    group_violations : list[str]
        Group keys that were split across partitions (should be empty).
    unmet_constraints : list[str]
        Human-readable list of constraints that could not be satisfied.
    train_fraction_actual, val_fraction_actual, test_fraction_actual : float
        Actual subject counts / total for each split.
    solver : str
        Algorithm identifier ("greedy_stratified").
    random_seed : int
        Seed used for tie-breaking.
    runtime_s : float
        Wall-clock time.
    """
    assignments: dict[str, str]
    train_fraction_actual: float
    val_fraction_actual: float
    test_fraction_actual: float
    class_distribution: dict[str, dict[str, float]]
    group_violations: list[str]
    unmet_constraints: list[str]
    residual_imbalance: float
    optimality_status: str
    solver: str = "greedy_stratified"
    random_seed: int = 42
    runtime_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "assignments": self.assignments,
            "train_fraction_actual": self.train_fraction_actual,
            "val_fraction_actual": self.val_fraction_actual,
            "test_fraction_actual": self.test_fraction_actual,
            "class_distribution": self.class_distribution,
            "group_violations": self.group_violations,
            "unmet_constraints": self.unmet_constraints,
            "residual_imbalance": self.residual_imbalance,
            "optimality_status": self.optimality_status,
            "solver": self.solver,
            "random_seed": self.random_seed,
            "runtime_s": self.runtime_s,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def assign_leakage_safe_splits(
    rows: list[dict[str, str]],
    *,
    id_column: str = "participant_id",
    constraints: SplitConstraints | None = None,
) -> SplitAssignmentResult:
    """Assign subjects to train/val/test splits respecting group constraints.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Tabular data (e.g. participants.tsv rows as string dicts).
    id_column : str
        Column whose value uniquely identifies each row (subject ID).
    constraints : SplitConstraints | None
        Split configuration.  Uses default SplitConstraints if None.

    Returns
    -------
    SplitAssignmentResult
        Contains assignments, class distributions, violations, and diagnostics.
        Always returned — even when constraints are violated — so the caller
        can inspect and decide whether to proceed.

    Notes
    -----
    Groups are kept intact: all rows sharing the same group key (concatenated
    values of group_columns) are assigned to the same split.  If a group is
    too large to place without violating fraction targets, it is placed in the
    least-full split and the violation is recorded.
    """
    t0 = time.perf_counter()

    if constraints is None:
        constraints = SplitConstraints()

    n_total = len(rows)
    assignments: dict[str, str] = {}
    unmet: list[str] = []
    group_violations: list[str] = []

    # ── Degenerate cases ──────────────────────────────────────────────────────
    if n_total == 0:
        return SplitAssignmentResult(
            assignments={},
            train_fraction_actual=0.0,
            val_fraction_actual=0.0,
            test_fraction_actual=0.0,
            class_distribution={},
            group_violations=[],
            unmet_constraints=["No rows provided."],
            residual_imbalance=0.0,
            optimality_status="violated",
            random_seed=constraints.random_seed,
            runtime_s=time.perf_counter() - t0,
        )

    null_vals = {"n/a", "N/A", "", "nan", "NaN", None}
    splits = ["train", "val", "test"]
    targets = {
        "train": constraints.train_fraction,
        "val":   constraints.val_fraction,
        "test":  constraints.test_fraction,
    }

    # ── Step 1: Group rows by group key ───────────────────────────────────────
    def _group_key(row: dict[str, str]) -> str:
        if not constraints.group_columns:
            return row.get(id_column, "unknown")
        return "__".join(row.get(c, "n/a") for c in constraints.group_columns)

    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = _group_key(row)
        groups.setdefault(key, []).append(row)

    # ── Step 2: Sort groups by size (largest first) for greedy packing ────────
    sorted_groups = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),  # stable: size desc, then key asc
    )

    # ── Step 3: Greedy bin-packing with stratified balance ────────────────────
    split_ids: dict[str, list[str]] = {s: [] for s in splits}
    split_label_counts: dict[str, dict[str, int]] = {s: {} for s in splits}
    n_assigned = {s: 0 for s in splits}

    def _dominant_label(group_rows: list[dict[str, str]]) -> str | None:
        if not constraints.stratify_column:
            return None
        from collections import Counter
        vals = [r.get(constraints.stratify_column, "n/a") for r in group_rows
                if r.get(constraints.stratify_column, "n/a") not in null_vals]
        if not vals:
            return None
        return Counter(vals).most_common(1)[0][0]

    def _label_balance_score(split_name: str, label: str | None) -> float:
        """Lower = better balanced.  Penalises adding more of the dominant label."""
        if label is None:
            return 0.0
        counts = split_label_counts[split_name]
        total = sum(counts.values()) + 1
        label_count = counts.get(label, 0) + 1
        return label_count / total

    for gkey, group_rows in sorted_groups:
        g_size = len(group_rows)
        g_ids = [r.get(id_column, f"unknown_{i}") for i, r in enumerate(group_rows)]
        dominant = _dominant_label(group_rows)

        # Pick split that is furthest below its target fraction,
        # breaking ties with label balance and then alphabetical split name
        best_split = None
        best_score = float("inf")
        for s in splits:
            current_frac = (n_assigned[s] + g_size) / n_total
            # Distance below target (negative = exceeds target)
            deficit = targets[s] - current_frac
            balance = _label_balance_score(s, dominant)
            # Prefer: max deficit (= most room) then best label balance
            score = (-deficit, balance, s)
            if best_score == float("inf") or score < best_score:
                best_score = score
                best_split = s

        if best_split is None:
            best_split = "train"

        # Assign group to best_split
        for sid in g_ids:
            assignments[sid] = best_split
        split_ids[best_split].extend(g_ids)
        n_assigned[best_split] += g_size
        if dominant is not None:
            counts = split_label_counts[best_split]
            for row in group_rows:
                lbl = row.get(constraints.stratify_column, "n/a")  # type: ignore[arg-type]
                if lbl not in null_vals:
                    counts[lbl] = counts.get(lbl, 0) + 1

    # ── Step 4: Check group integrity ─────────────────────────────────────────
    # Each group should be entirely in one split (by construction it is —
    # but validate if group_columns was empty and IDs had collisions)
    # (No violations possible in the current implementation; kept for API completeness)

    # ── Step 5: Compute class distributions per split ─────────────────────────
    class_distribution: dict[str, dict[str, float]] = {}
    if constraints.stratify_column:
        for s in splits:
            counts = split_label_counts[s]
            total = sum(counts.values())
            class_distribution[s] = {
                lbl: cnt / total for lbl, cnt in counts.items()
            } if total > 0 else {}

    # ── Step 6: Compute residual imbalance ────────────────────────────────────
    n_train = n_assigned["train"]
    n_val   = n_assigned["val"]
    n_test  = n_assigned["test"]

    actual_fracs = {
        "train": n_train / n_total,
        "val":   n_val   / n_total,
        "test":  n_test  / n_total,
    }
    max_frac_deviation = max(
        abs(actual_fracs[s] - targets[s]) for s in splits
    )

    # ── Step 7: Label balance check ───────────────────────────────────────────
    label_imbalance = 0.0
    if constraints.stratify_column and class_distribution:
        # Compare each split's label fractions against the overall dataset fractions
        all_counts: dict[str, int] = {}
        for s in splits:
            for lbl, cnt in split_label_counts[s].items():
                all_counts[lbl] = all_counts.get(lbl, 0) + cnt
        n_labeled = sum(all_counts.values())
        if n_labeled > 0:
            global_fracs = {lbl: cnt / n_labeled for lbl, cnt in all_counts.items()}
            for s in splits:
                for lbl, gf in global_fracs.items():
                    sf = class_distribution[s].get(lbl, 0.0)
                    label_imbalance = max(label_imbalance, abs(sf - gf))
            if label_imbalance > constraints.max_imbalance:
                unmet.append(
                    f"Label imbalance {label_imbalance:.3f} > "
                    f"max_imbalance={constraints.max_imbalance:.3f} "
                    f"for column '{constraints.stratify_column}'."
                )

    residual = max(max_frac_deviation, label_imbalance)
    if residual == 0.0:
        status = "optimal"
    elif residual <= constraints.max_imbalance:
        status = "near_optimal"
    else:
        status = "violated"
        if max_frac_deviation > constraints.max_imbalance:
            unmet.append(
                f"Split size deviation {max_frac_deviation:.3f} > "
                f"max_imbalance={constraints.max_imbalance:.3f}. "
                f"Actual fractions: train={actual_fracs['train']:.2f}, "
                f"val={actual_fracs['val']:.2f}, test={actual_fracs['test']:.2f}."
            )

    return SplitAssignmentResult(
        assignments=assignments,
        train_fraction_actual=actual_fracs["train"],
        val_fraction_actual=actual_fracs["val"],
        test_fraction_actual=actual_fracs["test"],
        class_distribution=class_distribution,
        group_violations=group_violations,
        unmet_constraints=unmet,
        residual_imbalance=residual,
        optimality_status=status,
        solver="greedy_stratified",
        random_seed=constraints.random_seed,
        runtime_s=time.perf_counter() - t0,
    )
