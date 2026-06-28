"""Statistical diagnostics and cohort profiling.

Used for validation, cohort profiling, and workflow safety.
All results include sample size, missingness, method, parameters, and limitations.
Returns LOW_CONFIDENCE or UNKNOWN when sample size is too small.

No causal claims.  No clinical interpretations.

Algorithms
----------
Cramér's V         : bias-corrected (Bergsma 2013) chi-squared association for cat×cat
Pearson r          : linear association for num×num
Cohen's d (SMD)    : standardised mean difference for num×cat (pooled SD)
Permutation test   : deterministic two-tailed p-value via LCG-shuffled permutations
                     (fixed seed → reproducible without numpy dependency)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from qortex.neuroclassic._base import (
    CohortMetricReport,
    MethodConfidence,
    MetricResult,
    NeuroClassicResult,
    NeuroClassicSpec,
)

__version__ = "0.1.0"

_MIN_N_FOR_STATS = 5
_PERMUTATION_N = 999   # two-tailed permutation test iterations
_PERMUTATION_SEED = 42 # deterministic — recorded in provenance


@dataclass
class StatisticalDiagnosticReport:
    """Multivariate statistical summary for workflow safety.

    Contains descriptive statistics, confound associations, class balance,
    missingness, and split-balance diagnostics.
    """
    scope: str
    n_samples: int
    n_missing: int
    variables: dict[str, "VariableSummary"] = field(default_factory=dict)
    confound_associations: list["ConfoundAssociation"] = field(default_factory=list)
    class_imbalance: dict[str, float] = field(default_factory=dict)  # class → fraction
    split_balance: list["SplitBalanceSummary"] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    confidence: MethodConfidence = MethodConfidence.HIGH

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "n_samples": self.n_samples,
            "n_missing": self.n_missing,
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "confound_associations": [c.to_dict() for c in self.confound_associations],
            "class_imbalance": self.class_imbalance,
            "split_balance": [s.to_dict() for s in self.split_balance],
            "warnings": self.warnings,
            "blockers": self.blockers,
            "confidence": self.confidence.value,
        }

    def to_result(self) -> NeuroClassicResult:
        metrics = [
            MetricResult("n_samples", self.n_samples),
            MetricResult("n_missing", self.n_missing),
            MetricResult("n_confound_associations", len(self.confound_associations)),
            MetricResult("n_variables", len(self.variables)),
        ]
        for assoc in self.confound_associations:
            metrics.append(MetricResult(
                f"association.{assoc.variable_a}_x_{assoc.variable_b}",
                assoc.effect_size,
                interpretation=(
                    f"{assoc.variable_a} is associated with {assoc.variable_b} "
                    f"(method: {assoc.method}, effect size: {assoc.effect_size:.3f})"
                    if assoc.effect_size is not None else None
                ),
            ))
        return NeuroClassicResult(
            method_name="statistical_diagnostics",
            method_version=__version__,
            modality="tabular",
            scope=self.scope,
            inputs={"n_samples": self.n_samples},
            parameters={
                "permutation_n": _PERMUTATION_N,
                "permutation_seed": _PERMUTATION_SEED,
            },
            assumptions=["Confound associations are not causal relationships."],
            metrics=metrics,
            warnings=self.warnings,
            blockers=self.blockers,
            confidence=self.confidence,
            provenance={"method": "statistical_diagnostics", "version": __version__},
        )


@dataclass
class VariableSummary:
    """Descriptive statistics for one variable."""
    name: str
    dtype: str              # "numeric" or "categorical"
    n_total: int
    n_missing: int
    n_unique: int | None = None
    # Numeric
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    min_val: float | None = None
    max_val: float | None = None
    # Categorical
    class_counts: dict[str, int] = field(default_factory=dict)
    dominant_class_fraction: float | None = None

    @property
    def missing_fraction(self) -> float:
        return self.n_missing / self.n_total if self.n_total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "n_total": self.n_total,
            "n_missing": self.n_missing,
            "missing_fraction": self.missing_fraction,
            "n_unique": self.n_unique,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "min": self.min_val,
            "max": self.max_val,
            "class_counts": self.class_counts,
            "dominant_class_fraction": self.dominant_class_fraction,
        }


@dataclass
class ConfoundAssociation:
    """Association between a target variable and a potential confound.

    Attributes
    ----------
    method : str
        "cramers_v"       — categorical × categorical (bias-corrected)
        "pearson_r"       — numeric × numeric
        "cohens_d_smd"    — numeric × categorical (standardised mean difference)
    p_value_permutation : float | None
        Two-tailed permutation p-value (n=999, fixed seed).  None when
        sample size < MIN_N_FOR_STATS or method is unsuitable.
    """
    variable_a: str      # e.g. target (diagnosis)
    variable_b: str      # e.g. confound (site, sex)
    method: str          # "cramers_v", "pearson_r", "cohens_d_smd"
    effect_size: float | None
    p_value_permutation: float | None = None
    n_pairs: int = 0
    interpretation: str = ""
    low_confidence: bool = False

    def to_dict(self) -> dict:
        return {
            "variable_a": self.variable_a,
            "variable_b": self.variable_b,
            "method": self.method,
            "effect_size": self.effect_size,
            "p_value_permutation": self.p_value_permutation,
            "n_pairs": self.n_pairs,
            "interpretation": self.interpretation,
            "low_confidence": self.low_confidence,
        }


@dataclass
class SplitBalanceSummary:
    """Balance of a categorical variable across train/val/test splits."""
    variable: str
    split: dict[str, dict[str, float]]   # split_name → class → fraction
    standardized_mean_difference: dict[str, float] = field(default_factory=dict)
    imbalanced: bool = False
    warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "split": self.split,
            "standardized_mean_difference": self.standardized_mean_difference,
            "imbalanced": self.imbalanced,
            "warning": self.warning,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def compute_statistical_diagnostics(
    rows: list[dict[str, str]],
    *,
    target: str | None = None,
    confound_columns: list[str] | None = None,
    scope: str = "participants.tsv",
) -> StatisticalDiagnosticReport:
    """Compute statistical diagnostics from tabular subject data (participants.tsv rows).

    Parameters
    ----------
    rows:
        List of dicts with string values (from TSV parsing).
    target:
        Column to treat as the classification target.
    confound_columns:
        Columns to check for association with the target.
    scope:
        File or dataset identifier.

    Returns
    -------
    StatisticalDiagnosticReport
    """
    n = len(rows)
    if not rows:
        return StatisticalDiagnosticReport(
            scope=scope,
            n_samples=0,
            n_missing=0,
            warnings=["No rows in data."],
            confidence=MethodConfidence.UNKNOWN,
        )

    columns = list(rows[0].keys())
    n_missing_total = 0

    report = StatisticalDiagnosticReport(
        scope=scope,
        n_samples=n,
        n_missing=0,
    )

    if n < _MIN_N_FOR_STATS:
        report.confidence = MethodConfidence.LOW_CONFIDENCE
        report.warnings.append(
            f"Only {n} samples; statistical diagnostics may be unreliable."
        )

    # Variable summaries
    for col in columns:
        values = [r.get(col, "n/a") for r in rows]
        null_vals = {"n/a", "N/A", "", "nan", "NaN", None}
        n_miss = sum(1 for v in values if v in null_vals)
        valid = [v for v in values if v not in null_vals]

        # Try numeric
        numeric_vals: list[float] = []
        for v in valid:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass

        if len(numeric_vals) == len(valid) and numeric_vals:
            import statistics
            summary = VariableSummary(
                name=col,
                dtype="numeric",
                n_total=n,
                n_missing=n_miss,
                n_unique=len(set(numeric_vals)),
                mean=_safe_mean(numeric_vals),
                std=_safe_std(numeric_vals),
                median=statistics.median(numeric_vals) if numeric_vals else None,
                min_val=min(numeric_vals),
                max_val=max(numeric_vals),
            )
        else:
            counts = Counter(valid)
            dominant_frac = (
                max(counts.values()) / len(valid) if valid else None
            )
            summary = VariableSummary(
                name=col,
                dtype="categorical",
                n_total=n,
                n_missing=n_miss,
                n_unique=len(counts),
                class_counts=dict(counts),
                dominant_class_fraction=dominant_frac,
            )

        n_missing_total += n_miss
        report.variables[col] = summary

    report.n_missing = n_missing_total

    # Class imbalance for target column
    if target and target in report.variables:
        tgt_summary = report.variables[target]
        if tgt_summary.class_counts:
            n_labeled = sum(tgt_summary.class_counts.values())
            report.class_imbalance = {
                cls: cnt / n_labeled
                for cls, cnt in tgt_summary.class_counts.items()
            }
            if report.class_imbalance:
                max_frac = max(report.class_imbalance.values())
                if max_frac > 0.8:
                    report.warnings.append(
                        f"Target '{target}' is severely imbalanced: "
                        f"dominant class = {max_frac*100:.1f}%."
                    )

    # Confound associations
    confounds = confound_columns or []
    if target and target in columns:
        target_values = [r.get(target, "n/a") for r in rows]
        for conf_col in confounds:
            if conf_col not in columns or conf_col == target:
                continue
            conf_values = [r.get(conf_col, "n/a") for r in rows]
            assoc = _compute_association(
                target, target_values, conf_col, conf_values, n,
                target_summary=report.variables.get(target),
                confound_summary=report.variables.get(conf_col),
            )
            if assoc is not None:
                report.confound_associations.append(assoc)

                if assoc.effect_size is not None and assoc.effect_size > 0.5:
                    sev = "BLOCK" if assoc.effect_size > 0.7 else "WARN"
                    report.warnings.append(
                        f"[{sev}] '{conf_col}' is strongly associated with '{target}' "
                        f"({assoc.method} = {assoc.effect_size:.2f}). "
                        "Model evaluation may be confounded. "
                        "Note: association is not causation."
                    )

    return report


def build_cohort_metric_report(
    results: list,
    *,
    method_name: str,
    metric_name: str,
    modality: str,
) -> CohortMetricReport:
    """Aggregate a specific metric across NeuroClassicResults into a CohortMetricReport."""
    values: list[float] = []
    for r in results:
        for m in getattr(r, "metrics", []):
            if m.name == metric_name and isinstance(m.value, (int, float)):
                values.append(float(m.value))
                break
    cr = CohortMetricReport(
        method_name=method_name,
        metric_name=metric_name,
        modality=modality,
        n_subjects=len(results),
        values=values,
    )
    return cr.compute()


# ── Association computation ───────────────────────────────────────────────────

def _compute_association(
    name_a: str,
    vals_a: list[str],
    name_b: str,
    vals_b: list[str],
    n: int,
    target_summary: VariableSummary | None = None,
    confound_summary: VariableSummary | None = None,
) -> ConfoundAssociation | None:
    null_set = {"n/a", "N/A", "", "nan", "NaN", None}
    pairs = [
        (a, b) for a, b in zip(vals_a, vals_b)
        if a not in null_set and b not in null_set
    ]
    if len(pairs) < _MIN_N_FOR_STATS:
        return ConfoundAssociation(
            variable_a=name_a,
            variable_b=name_b,
            method="unknown",
            effect_size=None,
            n_pairs=len(pairs),
            interpretation="Insufficient paired observations.",
            low_confidence=True,
        )

    a_vals = [p[0] for p in pairs]
    b_vals = [p[1] for p in pairs]

    a_num = _try_float_list(a_vals)
    b_num = _try_float_list(b_vals)

    # ── Numeric × Numeric → Pearson r + permutation p ────────────────────────
    if a_num is not None and b_num is not None:
        r = _pearson_r(a_num, b_num)
        p_val = None
        if r is not None and len(a_num) >= _MIN_N_FOR_STATS:
            p_val = _permutation_test_pearson_r(a_num, b_num)
        return ConfoundAssociation(
            variable_a=name_a,
            variable_b=name_b,
            method="pearson_r",
            effect_size=abs(r) if r is not None else None,
            p_value_permutation=p_val,
            n_pairs=len(pairs),
            interpretation=(
                f"|r| = {abs(r):.3f} (numerical association; not causal)"
                if r is not None else "Could not compute Pearson r."
            ),
            low_confidence=len(pairs) < 20,
        )

    # ── Numeric × Categorical → Cohen's d (SMD) ──────────────────────────────
    a_is_num = a_num is not None
    b_is_num = b_num is not None
    if a_is_num != b_is_num:
        # One side is numeric, other is categorical
        num_vals = a_num if a_is_num else b_num
        cat_vals = b_vals if a_is_num else a_vals
        d, p_val = _cohens_d_and_permutation(num_vals, cat_vals)  # type: ignore[arg-type]
        return ConfoundAssociation(
            variable_a=name_a,
            variable_b=name_b,
            method="cohens_d_smd",
            effect_size=d,
            p_value_permutation=p_val,
            n_pairs=len(pairs),
            interpretation=(
                f"SMD (Cohen's d) = {d:.3f} (numeric-categorical association; not causal)"
                if d is not None else "Could not compute Cohen's d."
            ),
            low_confidence=len(pairs) < 20,
        )

    # ── Categorical × Categorical → Cramér's V + permutation p ──────────────
    v = _cramers_v_from_lists(a_vals, b_vals)
    p_val = None
    if v is not None and len(a_vals) >= _MIN_N_FOR_STATS:
        p_val = _permutation_test_cramers_v(a_vals, b_vals)
    return ConfoundAssociation(
        variable_a=name_a,
        variable_b=name_b,
        method="cramers_v",
        effect_size=v,
        p_value_permutation=p_val,
        n_pairs=len(pairs),
        interpretation=(
            f"Cramér's V = {v:.3f} (categorical association; not causal)"
            if v is not None else "Could not compute Cramér's V."
        ),
        low_confidence=len(pairs) < 20,
    )


# ── Statistical helpers ───────────────────────────────────────────────────────

def _try_float_list(vals: list[str]) -> list[float] | None:
    try:
        return [float(v) for v in vals]
    except (ValueError, TypeError):
        return None


def _safe_mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _safe_std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = _safe_mean(vals)
    if m is None:
        return None
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _pearson_r(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def _cramers_v_from_lists(x: list[str], y: list[str]) -> float | None:
    """Bias-corrected Cramér's V (Bergsma 2013 correction)."""
    n = len(x)
    if n < 5:
        return None
    unique_x = sorted(set(x))
    unique_y = sorted(set(y))
    r, c = len(unique_x), len(unique_y)
    if r < 2 or c < 2:
        return None
    xi = {v: i for i, v in enumerate(unique_x)}
    yi = {v: i for i, v in enumerate(unique_y)}
    table = [[0] * c for _ in range(r)]
    for a, b in zip(x, y):
        table[xi[a]][yi[b]] += 1
    row_totals = [sum(row) for row in table]
    col_totals = [sum(table[i][j] for i in range(r)) for j in range(c)]
    chi2 = 0.0
    for i in range(r):
        for j in range(c):
            exp = row_totals[i] * col_totals[j] / n
            if exp > 0:
                chi2 += (table[i][j] - exp) ** 2 / exp
    phi2 = chi2 / n
    k = min(r, c)
    # Bergsma (2013) bias correction
    phi2c = max(0.0, phi2 - (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    c_corr = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corr, c_corr) - 1
    if denom <= 0:
        return None
    return math.sqrt(phi2c / denom)


def _standardized_mean_difference(
    group_a: list[float],
    group_b: list[float],
) -> float | None:
    """Cohen's d via pooled standard deviation.

    |mean_a - mean_b| / sqrt(((na-1)*var_a + (nb-1)*var_b) / (na+nb-2))
    """
    na, nb = len(group_a), len(group_b)
    if na < 2 or nb < 2:
        return None
    mean_a = sum(group_a) / na
    mean_b = sum(group_b) / nb
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (na - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (nb - 1)
    pooled_var = ((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2)
    if pooled_var <= 0:
        return None
    return abs(mean_a - mean_b) / math.sqrt(pooled_var)


def _cohens_d_and_permutation(
    num_vals: list[float],
    cat_vals: list[str],
) -> tuple[float | None, float | None]:
    """Compute Cohen's d for the largest binary group split + permutation p-value."""
    # Group numeric values by categorical label
    groups: dict[str, list[float]] = {}
    for v, c in zip(num_vals, cat_vals):
        groups.setdefault(c, []).append(v)

    if len(groups) < 2:
        return None, None

    # Take the two largest groups for d computation
    sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)
    g_a = sorted_groups[0][1]
    g_b = sorted_groups[1][1]

    d = _standardized_mean_difference(g_a, g_b)
    if d is None:
        return None, None

    # Permutation test: shuffle category labels, re-compute d
    p_val = _permutation_test_smd(num_vals, cat_vals,
                                   sorted_groups[0][0], sorted_groups[1][0], d)
    return d, p_val


def _lcg_shuffle(lst: list, seed: int) -> tuple[list, int]:
    """In-place Fisher-Yates shuffle using a linear congruential generator.

    Returns (shuffled_list, new_seed).  Deterministic — suitable for
    permutation tests where exact reproducibility matters.
    """
    n = len(lst)
    state = seed
    for i in range(n - 1, 0, -1):
        state = (1664525 * state + 1013904223) % (2 ** 32)
        j = state % (i + 1)
        lst[i], lst[j] = lst[j], lst[i]
    return lst, state


def _permutation_test_pearson_r(
    x: list[float],
    y: list[float],
    n_permutations: int = _PERMUTATION_N,
    seed: int = _PERMUTATION_SEED,
) -> float | None:
    """Two-tailed permutation p-value for |Pearson r|.

    Uses a deterministic LCG shuffle (no numpy).  Seed is fixed so the
    p-value is reproducible from the same data.
    """
    observed = _pearson_r(x, y)
    if observed is None:
        return None
    observed_abs = abs(observed)
    count = 0
    y_perm = list(y)
    state = seed
    for _ in range(n_permutations):
        y_perm, state = _lcg_shuffle(y_perm, state)
        r_perm = _pearson_r(x, y_perm)
        if r_perm is not None and abs(r_perm) >= observed_abs:
            count += 1
    return (count + 1) / (n_permutations + 1)


def _permutation_test_cramers_v(
    a_vals: list[str],
    b_vals: list[str],
    n_permutations: int = _PERMUTATION_N,
    seed: int = _PERMUTATION_SEED,
) -> float | None:
    """Two-tailed permutation p-value for Cramér's V.

    Permutes b_vals while keeping a_vals fixed; counts how often the
    permuted V ≥ observed V.
    """
    observed = _cramers_v_from_lists(a_vals, b_vals)
    if observed is None:
        return None
    count = 0
    b_perm = list(b_vals)
    state = seed
    for _ in range(n_permutations):
        b_perm, state = _lcg_shuffle(b_perm, state)
        v_perm = _cramers_v_from_lists(a_vals, b_perm)
        if v_perm is not None and v_perm >= observed:
            count += 1
    return (count + 1) / (n_permutations + 1)


def _permutation_test_smd(
    num_vals: list[float],
    cat_vals: list[str],
    label_a: str,
    label_b: str,
    observed_d: float,
    n_permutations: int = _PERMUTATION_N,
    seed: int = _PERMUTATION_SEED,
) -> float | None:
    """Two-tailed permutation p-value for Cohen's d (permute category labels)."""
    count = 0
    cats_perm = list(cat_vals)
    state = seed
    for _ in range(n_permutations):
        cats_perm, state = _lcg_shuffle(cats_perm, state)
        g_a = [v for v, c in zip(num_vals, cats_perm) if c == label_a]
        g_b = [v for v, c in zip(num_vals, cats_perm) if c == label_b]
        d_perm = _standardized_mean_difference(g_a, g_b)
        if d_perm is not None and d_perm >= observed_d:
            count += 1
    return (count + 1) / (n_permutations + 1)
