"""Diff normalized validation reports."""

from __future__ import annotations

from qortex.core.entities import ValidationDiff, ValidationIssue, ValidationReport


def diff_validation_reports(
    before: ValidationReport,
    after: ValidationReport,
) -> ValidationDiff:
    """Return added, resolved, and persistent validation issues."""
    before_keys = {_issue_key(issue): issue for issue in before.issues}
    after_keys = {_issue_key(issue): issue for issue in after.issues}
    before_set = set(before_keys)
    after_set = set(after_keys)
    return ValidationDiff(
        before_path=before.dataset_path,
        after_path=after.dataset_path,
        added=[after_keys[key] for key in sorted(after_set - before_set)],
        resolved=[before_keys[key] for key in sorted(before_set - after_set)],
        persisted=[after_keys[key] for key in sorted(before_set & after_set)],
    )


def _issue_key(issue: ValidationIssue) -> tuple[str, str, str, str]:
    return (
        issue.severity,
        issue.code,
        issue.path or "",
        issue.message,
    )
