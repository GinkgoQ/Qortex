"""Explicit two-cohort comparisons for real participant-table variables."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

_MISSING = {"", "n/a", "na", "nan", "unknown"}


def _missing(value: Any) -> bool:
    return value is None or str(value).strip().lower() in _MISSING


def _clean_category(value: Any) -> str:
    return str(value).strip().strip(",;|").strip()


def _bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=p_values.__getitem__)
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = count - reverse_rank + 1
        running = min(running, p_values[index] * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _summary(values: np.ndarray) -> dict[str, Any]:
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else None,
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _numeric_comparison(
    column: str,
    labels: tuple[str, str],
    rows: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    alpha: float,
) -> dict[str, Any]:
    from scipy import stats

    arrays = []
    missing = []
    invalid: list[list[dict[str, Any]]] = []
    for group_rows in rows:
        values: list[float] = []
        group_invalid: list[dict[str, Any]] = []
        group_missing = 0
        for index, row in enumerate(group_rows, start=1):
            raw = row.get(column)
            if _missing(raw):
                group_missing += 1
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                group_invalid.append({"row": index, "value": raw})
                continue
            if not math.isfinite(value):
                group_invalid.append({"row": index, "value": raw})
                continue
            values.append(value)
        arrays.append(np.asarray(values, dtype=np.float64))
        missing.append(group_missing)
        invalid.append(group_invalid)

    left, right = arrays
    base = {
        "column": column,
        "kind": "numeric",
        "groups": {
            label: {
                "total_rows": len(group_rows),
                "missing": group_missing,
                "invalid": group_invalid,
                "summary": _summary(values) if values.size else None,
            }
            for label, group_rows, group_missing, group_invalid, values
            in zip(labels, rows, missing, invalid, arrays)
        },
    }
    if left.size < 2 or right.size < 2:
        return {
            **base,
            "status": "insufficient_data",
            "reason": "Each cohort requires at least two finite observations for variance-based inference.",
            "primary_test": None,
            "sensitivity_test": None,
        }

    variance_left = float(np.var(left, ddof=1))
    variance_right = float(np.var(right, ddof=1))
    standard_error_sq = variance_left / left.size + variance_right / right.size
    difference = float(np.mean(left) - np.mean(right))
    if standard_error_sq == 0:
        return {
            **base,
            "status": "not_estimable",
            "reason": "Welch inference is undefined because both cohorts have zero within-cohort variance.",
            "primary_test": None,
            "sensitivity_test": None,
        }
    if standard_error_sq > 0:
        numerator = standard_error_sq ** 2
        denominator = (
            (variance_left / left.size) ** 2 / (left.size - 1)
            + (variance_right / right.size) ** 2 / (right.size - 1)
        )
        degrees_freedom = numerator / denominator if denominator else None
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_freedom)) if degrees_freedom else None
        margin = critical * math.sqrt(standard_error_sq) if critical is not None else None
    welch = stats.ttest_ind(left, right, equal_var=False, alternative="two-sided")
    if not math.isfinite(float(welch.pvalue)):
        return {
            **base,
            "status": "not_estimable",
            "reason": "Welch inference is undefined for these within-cohort variance and sample values.",
            "primary_test": None,
            "sensitivity_test": None,
        }
    pooled_variance = (
        ((left.size - 1) * variance_left + (right.size - 1) * variance_right)
        / (left.size + right.size - 2)
    )
    pooled_std = math.sqrt(pooled_variance) if pooled_variance > 0 else 0.0
    correction = 1.0 - 3.0 / (4.0 * (left.size + right.size) - 9.0)
    hedges_g = correction * difference / pooled_std if pooled_std else None

    mann_whitney = stats.mannwhitneyu(left, right, alternative="two-sided", method="auto")
    rank_biserial = 2.0 * float(mann_whitney.statistic) / (left.size * right.size) - 1.0
    return {
        **base,
        "status": "completed",
        "estimand": f"mean({labels[0]}) - mean({labels[1]})",
        "primary_test": {
            "method": "Welch independent two-sample t-test",
            "alternative": "two-sided",
            "statistic": float(welch.statistic),
            "degrees_freedom": float(degrees_freedom) if degrees_freedom is not None else None,
            "p_value_raw": float(welch.pvalue),
            "mean_difference": difference,
            "confidence_level": 1.0 - alpha,
            "confidence_interval": [difference - margin, difference + margin] if margin is not None else None,
            "effect_size": {"name": "Hedges g", "value": hedges_g, "direction": f"positive means {labels[0]} is larger"},
        },
        "sensitivity_test": {
            "method": "Mann-Whitney U",
            "alternative": "two-sided",
            "statistic": float(mann_whitney.statistic),
            "p_value_raw": float(mann_whitney.pvalue),
            "effect_size": {"name": "rank-biserial correlation", "value": rank_biserial},
        },
    }


def _categorical_comparison(
    column: str,
    labels: tuple[str, str],
    rows: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> dict[str, Any]:
    from scipy import stats

    counts = []
    missing = []
    invalid = []
    for group_rows in rows:
        values = []
        group_invalid = []
        group_missing = 0
        for index, row in enumerate(group_rows, start=1):
            raw = row.get(column)
            if _missing(raw):
                group_missing += 1
                continue
            exact = str(raw).strip()
            if exact != _clean_category(raw):
                group_invalid.append({"row": index, "value": raw})
                continue
            values.append(exact)
        counts.append(Counter(values))
        missing.append(group_missing)
        invalid.append(group_invalid)
    categories = sorted(set(counts[0]) | set(counts[1]))
    table = np.asarray([[group.get(category, 0) for category in categories] for group in counts], dtype=np.int64)
    base = {
        "column": column,
        "kind": "categorical",
        "category_validation": "Exact already-clean values only; values requiring whitespace or trailing punctuation cleanup are reported invalid and excluded.",
        "categories": categories,
        "groups": {
            label: {"total_rows": len(group_rows), "missing": group_missing, "invalid": group_invalid, "counts": dict(group_counts)}
            for label, group_rows, group_missing, group_invalid, group_counts in zip(labels, rows, missing, invalid, counts)
        },
    }
    if len(categories) < 2 or np.any(table.sum(axis=1) == 0):
        return {
            **base,
            "status": "insufficient_data",
            "reason": "Both cohorts need observations across at least two combined categories.",
            "primary_test": None,
        }
    chi2, chi_p, _, expected = stats.chi2_contingency(table, correction=False)
    total = int(table.sum())
    cramers_v = math.sqrt(float(chi2) / total) if total else None
    if table.shape == (2, 2):
        fisher = stats.fisher_exact(table, alternative="two-sided")
        primary = {
            "method": "Fisher exact test",
            "alternative": "two-sided",
            "statistic": float(fisher.statistic),
            "p_value_raw": float(fisher.pvalue),
        }
    else:
        primary = {
            "method": "Pearson chi-square test",
            "alternative": "distribution differs by cohort",
            "statistic": float(chi2),
            "p_value_raw": float(chi_p),
        }
    primary["effect_size"] = {"name": "Cramer's V", "value": cramers_v}
    return {
        **base,
        "status": "completed",
        "contingency_table": table.tolist(),
        "expected_counts": np.asarray(expected, dtype=float).tolist(),
        "primary_test": primary,
        "sensitivity_test": {
            "method": "Pearson chi-square test",
            "statistic": float(chi2),
            "p_value_raw": float(chi_p),
        } if table.shape == (2, 2) else None,
    }


def compare_participant_cohorts(
    cohorts: dict[str, list[dict[str, Any]]],
    *,
    variables: list[dict[str, str]],
    alpha: float = 0.05,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare explicit numeric/categorical variables between two dataset-defined cohorts."""
    if len(cohorts) != 2:
        raise ValueError("Exactly two dataset-defined cohorts are required")
    if not variables:
        raise ValueError("At least one comparison variable is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    labels = tuple(cohorts)  # insertion order is the declared comparison direction
    rows = (cohorts[labels[0]], cohorts[labels[1]])
    results = []
    for variable in variables:
        column = variable.get("column")
        kind = variable.get("kind")
        if not column:
            raise ValueError("Every variable requires a non-empty column")
        if kind == "numeric":
            results.append(_numeric_comparison(column, labels, rows, alpha))
        elif kind == "categorical":
            results.append(_categorical_comparison(column, labels, rows))
        else:
            raise ValueError(f"Variable {column!r} kind must be numeric or categorical")

    completed = [result for result in results if result["status"] == "completed"]
    raw = [result["primary_test"]["p_value_raw"] for result in completed]
    for result, adjusted in zip(completed, _bh_adjust(raw)):
        result["primary_test"]["p_value_bh"] = adjusted
        result["primary_test"]["reject_at_alpha"] = adjusted < alpha
    return {
        "status": "completed",
        "group_definition": {
            "type": "dataset_membership",
            "groups": list(labels),
            "direction": f"{labels[0]} minus {labels[1]}",
            "warning": "Dataset membership also encodes study, acquisition, recruitment, and site differences; no causal group effect is claimed.",
        },
        "missingness_policy": "Per-variable complete-case analysis within each dataset; missing and invalid values are counted and excluded.",
        "test_policy": "Welch two-sample test for declared numeric variables; Fisher exact for 2x2 categorical tables; Pearson chi-square otherwise.",
        "multiplicity_policy": "Benjamini-Hochberg correction across completed primary variable tests in this report.",
        "alpha": alpha,
        "variables": results,
        "sources": sources or [],
    }


__all__ = ["compare_participant_cohorts"]
