"""Official BIDS Validator integration.

The validator is an external CLI/JS tool. Qortex keeps it optional and wraps
its JSON output in stable typed models so downstream code does not depend on
validator-specific JSON details.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from qortex.core.entities import ValidationIssue, ValidationReport
from qortex.core.exceptions import ValidationError
from qortex.validation.cache import ValidationCache


class BIDSValidatorRunner:
    """Run the official BIDS Validator and normalize the output."""

    def __init__(
        self,
        executable: str = "bids-validator",
        timeout_s: float = 600.0,
    ) -> None:
        self.executable = executable
        self.timeout_s = timeout_s

    def validate(
        self,
        dataset_path: str | Path,
        *,
        config_path: str | Path | None = None,
        output_json: str | Path | None = None,
        ignore_warnings: bool = False,
        ignore_nifti_headers: bool = False,
        timeout_s: float | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> ValidationReport:
        """Validate a local BIDS dataset using machine-readable JSON output."""
        root = Path(dataset_path).expanduser().resolve()
        if not root.exists():
            raise ValidationError(f"BIDS dataset path does not exist: {root}")
        if not root.is_dir():
            raise ValidationError(f"BIDS dataset path is not a directory: {root}")
        if shutil.which(self.executable) is None:
            raise ValidationError(
                f"Cannot find '{self.executable}'. Install the official BIDS "
                "Validator CLI and make sure it is on PATH."
            )
        cache = ValidationCache()
        cache_key = cache.key(
            root,
            executable=self.executable,
            config_path=config_path,
            ignore_warnings=ignore_warnings,
            ignore_nifti_headers=ignore_nifti_headers,
        )
        if use_cache and not refresh_cache:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="qortex-bids-validator-") as tmp:
            json_path = Path(output_json) if output_json else Path(tmp) / "validation.json"
            cmd = [
                self.executable,
                str(root),
                "--json",
            ]
            if config_path is not None:
                cmd.extend(["--config", str(Path(config_path).expanduser())])
            if ignore_warnings:
                cmd.append("--ignoreWarnings")
            if ignore_nifti_headers:
                cmd.append("--ignoreNiftiHeaders")

            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s or self.timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                raise ValidationError(
                    f"BIDS validation timed out after {timeout_s or self.timeout_s:.1f}s"
                ) from exc

            raw = _read_validator_json(json_path, proc.stdout)
            if output_json is not None and not json_path.exists():
                json_path.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
            elapsed = time.monotonic() - t0
            issues = _normalize_issues(raw)
            valid = _infer_valid(raw, proc.returncode, issues)
            version = _extract_version(raw, proc.stderr, proc.stdout)
            if version is None:
                version = _probe_validator_version(self.executable, timeout_s=timeout_s or self.timeout_s)
            report = ValidationReport(
                dataset_path=str(root),
                valid=valid,
                issues=issues,
                command=cmd,
                return_code=proc.returncode,
                validator_version=version,
                elapsed=elapsed,
                stdout=proc.stdout,
                stderr=proc.stderr,
                raw=raw,
            )
            if use_cache:
                cache.put(cache_key, report)
            return report


def validate_bids(
    dataset_path: str | Path,
    *,
    executable: str = "bids-validator",
    config_path: str | Path | None = None,
    output_json: str | Path | None = None,
    ignore_warnings: bool = False,
    ignore_nifti_headers: bool = False,
    timeout_s: float = 600.0,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> ValidationReport:
    """Convenience wrapper around :class:`BIDSValidatorRunner`."""
    return BIDSValidatorRunner(executable=executable, timeout_s=timeout_s).validate(
        dataset_path,
        config_path=config_path,
        output_json=output_json,
        ignore_warnings=ignore_warnings,
        ignore_nifti_headers=ignore_nifti_headers,
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )


def _read_validator_json(json_path: Path, stdout: str) -> dict[str, Any]:
    payload = ""
    if json_path.exists():
        payload = json_path.read_text(encoding="utf-8")
    elif stdout.strip().startswith("{"):
        payload = stdout
    if not payload.strip():
        raise ValidationError(
            "BIDS Validator did not produce JSON output. Check the validator "
            "version and CLI flags."
        )
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Could not parse BIDS Validator JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValidationError("BIDS Validator JSON root was not an object.")
    return parsed


def _normalize_issues(raw: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issue_root = raw.get("issues", raw)

    buckets = {
        "error": _as_list(issue_root.get("errors")) + _as_list(raw.get("errors")),
        "warning": _as_list(issue_root.get("warnings")) + _as_list(raw.get("warnings")),
        "ignored": _as_list(issue_root.get("ignored")) + _as_list(raw.get("ignored")),
    }
    if not any(buckets.values()) and isinstance(issue_root, list):
        buckets["error"] = issue_root

    seen: set[tuple[str, str, str | None, str]] = set()
    for severity, entries in buckets.items():
        for entry in entries:
            for normalized_entry in _expand_issue_entry(entry):
                issue = _normalize_issue(severity, normalized_entry)
                key = (issue.severity, issue.code, issue.path, issue.message)
                if key in seen:
                    continue
                seen.add(key)
                issues.append(issue)
    return issues


def _expand_issue_entry(entry: Any) -> list[Any]:
    if not isinstance(entry, dict):
        return [entry]
    files = entry.get("files")
    if not isinstance(files, list) or not files:
        return [entry]
    expanded: list[dict[str, Any]] = []
    group = {k: v for k, v in entry.items() if k != "files"}
    for file_issue in files:
        if isinstance(file_issue, dict):
            merged = {**group, **file_issue}
            merged.setdefault("group", group)
            expanded.append(merged)
        else:
            expanded.append({**group, "file": file_issue})
    return expanded


def _normalize_issue(severity: str, entry: Any) -> ValidationIssue:
    if not isinstance(entry, dict):
        return ValidationIssue(
            severity=severity, code="UNKNOWN", message=str(entry), raw={}
        )
    location = entry.get("location") or entry.get("file") or entry.get("path")
    path, line, column = _parse_location(location)
    code = str(
        entry.get("code")
        or entry.get("key")
        or entry.get("reason")
        or entry.get("rule")
        or "UNKNOWN"
    )
    message = str(
        entry.get("message")
        or entry.get("reason")
        or entry.get("description")
        or entry.get("evidence")
        or code
    )
    evidence = entry.get("evidence")
    return ValidationIssue(
        severity=severity,
        code=code,
        message=message,
        path=path,
        line=line,
        column=column,
        evidence=str(evidence) if evidence is not None else None,
        raw=entry,
    )


def _parse_location(location: Any) -> tuple[str | None, int | None, int | None]:
    if isinstance(location, str):
        return location.lstrip("/") or None, None, None
    if isinstance(location, dict):
        path = (
            location.get("path")
            or location.get("file")
            or location.get("filename")
            or location.get("location")
        )
        line = _as_int(location.get("line"))
        column = _as_int(location.get("column"))
        return str(path).lstrip("/") if path else None, line, column
    return None, None, None


def _infer_valid(
    raw: dict[str, Any],
    return_code: int,
    issues: list[ValidationIssue],
) -> bool:
    if isinstance(raw.get("valid"), bool):
        return bool(raw["valid"])
    if isinstance(raw.get("isValid"), bool):
        return bool(raw["isValid"])
    if any(issue.severity == "error" for issue in issues):
        return False
    return return_code == 0


def _extract_version(raw: dict[str, Any], stderr: str, stdout: str) -> str | None:
    for key in ("validatorVersion", "version", "bidsValidatorVersion"):
        value = raw.get(key)
        if value:
            return str(value)
    for text in (stderr, stdout):
        for line in text.splitlines():
            if "bids-validator" in line.lower() and any(ch.isdigit() for ch in line):
                return line.strip()
    return None


def _probe_validator_version(executable: str, *, timeout_s: float) -> str | None:
    try:
        proc = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=min(timeout_s, 10.0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or proc.stderr).strip()
    return text or None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
