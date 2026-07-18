from __future__ import annotations

from qortex.eda.participants import (
    ParticipantRecord,
    ParticipantsTable,
    summarize_demographics,
)


def test_demographic_summary_separates_dirty_and_invalid_values():
    table = ParticipantsTable(
        columns=["participant_id", "age", "sex"],
        records=[
            ParticipantRecord("sub-01", {"age": "20", "sex": "F"}),
            ParticipantRecord("sub-02", {"age": "24", "sex": "F"}),
            ParticipantRecord("sub-03", {"age": "26", "sex": "M"}),
            ParticipantRecord("sub-04", {"age": "22", "sex": "M,"}),
            ParticipantRecord("sub-05", {"age": "not-recorded", "sex": "M"}),
            ParticipantRecord("sub-06", {"age": "n/a", "sex": "n/a"}),
        ],
        sidecar={"sex": {"Levels": {"F": "female", "M": "male"}}},
    )

    summary = summarize_demographics(table)

    assert summary["categorical"]["valid_counts"] == {"F": 2, "M": 2}
    assert summary["categorical"]["invalid_values"] == {"M,": [4]}
    assert summary["categorical"]["n_missing"] == 1
    assert summary["numeric"]["n_invalid"] == 1
    assert summary["numeric"]["n_missing"] == 1
    assert summary["overall"] == {
        "n": 4, "median": 23.0, "q1": 21.5, "q3": 24.5, "min": 20.0, "max": 26.0,
    }
    groups = {row["group"]: row for row in summary["groups"]}
    assert groups["F"]["values"] == [20.0, 24.0]
    assert groups["M"]["values"] == [26.0]
    assert groups["Invalid"]["values"] == [22.0]


def test_demographic_summary_requires_named_columns():
    table = ParticipantsTable(columns=["participant_id"], records=[])

    try:
        summarize_demographics(table)
    except ValueError as exc:
        assert "sex" in str(exc)
    else:
        raise AssertionError("missing group column was accepted")
