from __future__ import annotations

import pytest

from qortex.neuroclassic.cohort_comparison import compare_participant_cohorts


def test_numeric_comparison_reports_effect_ci_missingness_and_sensitivity() -> None:
    cohorts = {
        "dataset-a": [{"age": value} for value in (20, 22, 24, 26, None, "invalid")],
        "dataset-b": [{"age": value} for value in (30, 32, 34, 36, 38)],
    }

    report = compare_participant_cohorts(
        cohorts, variables=[{"column": "age", "kind": "numeric"}], alpha=0.05,
    )

    result = report["variables"][0]
    assert report["group_definition"]["direction"] == "dataset-a minus dataset-b"
    assert result["status"] == "completed"
    assert result["groups"]["dataset-a"]["summary"]["n"] == 4
    assert result["groups"]["dataset-a"]["missing"] == 1
    assert result["groups"]["dataset-a"]["invalid"] == [{"row": 6, "value": "invalid"}]
    assert result["primary_test"]["method"] == "Welch independent two-sample t-test"
    assert result["primary_test"]["mean_difference"] == pytest.approx(-11.0)
    assert result["primary_test"]["confidence_interval"][1] < 0
    assert result["primary_test"]["effect_size"]["value"] < 0
    assert result["sensitivity_test"]["method"] == "Mann-Whitney U"
    assert result["primary_test"]["p_value_bh"] == result["primary_test"]["p_value_raw"]
    assert result["primary_test"]["reject_at_alpha"] is True


def test_categorical_comparison_uses_fisher_for_two_by_two_table() -> None:
    cohorts = {
        "dataset-a": [{"sex": value} for value in ["F"] * 8 + ["M"] * 2 + ["M,"]],
        "dataset-b": [{"sex": value} for value in ["F"] * 2 + ["M"] * 8 + [None]],
    }

    report = compare_participant_cohorts(
        cohorts, variables=[{"column": "sex", "kind": "categorical"}],
    )

    result = report["variables"][0]
    assert result["categories"] == ["F", "M"]
    assert result["contingency_table"] == [[8, 2], [2, 8]]
    assert result["groups"]["dataset-a"]["invalid"] == [{"row": 11, "value": "M,"}]
    assert result["groups"]["dataset-b"]["missing"] == 1
    assert result["primary_test"]["method"] == "Fisher exact test"
    assert result["primary_test"]["effect_size"]["name"] == "Cramer's V"
    assert result["sensitivity_test"]["method"] == "Pearson chi-square test"


def test_bh_adjustment_covers_all_completed_primary_variables() -> None:
    cohorts = {
        "a": [{"x": value, "y": value * 2} for value in (1, 2, 3, 4, 5)],
        "b": [{"x": value, "y": value * 2} for value in (4, 5, 6, 7, 8)],
    }

    report = compare_participant_cohorts(
        cohorts,
        variables=[{"column": "x", "kind": "numeric"}, {"column": "y", "kind": "numeric"}],
    )

    assert all("p_value_bh" in result["primary_test"] for result in report["variables"])
    assert report["multiplicity_policy"].startswith("Benjamini-Hochberg")


def test_comparison_requires_explicit_variable_kind() -> None:
    with pytest.raises(ValueError, match="numeric or categorical"):
        compare_participant_cohorts(
            {"a": [{"value": 1}], "b": [{"value": 2}]},
            variables=[{"column": "value", "kind": "auto"}],
        )


def test_constant_groups_are_reported_not_estimable() -> None:
    report = compare_participant_cohorts(
        {"a": [{"value": 1}] * 3, "b": [{"value": 1}] * 3},
        variables=[{"column": "value", "kind": "numeric"}],
    )

    assert report["variables"][0]["status"] == "not_estimable"
    assert report["variables"][0]["primary_test"] is None
