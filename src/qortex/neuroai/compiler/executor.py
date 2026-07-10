"""Execution-plan verifier for the Qortex NeuroAI compiler.

The compiler is an offline planner: it never downloads weights or runs
inference. This module is the safety/integrity pre-flight that must pass
before any real execution of a saved plan -- it re-checks that the plan
file wasn't tampered with, that the source hasn't drifted since compile,
and that the license/remote-code gates still pass. It does NOT run model
inference; there are no verified checkpoints in the zoo yet.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.compiler.serialization import sha256_json
from qortex.neuroai.contracts import BaseModel, Field
from qortex.neuroai.models.license import check_license_gate
from qortex.neuroai.models.security import check_remote_code_gate
from qortex.neuroai.models.zoo import registry as zoo_registry

# Keys included in the compiler's plan_hash payload (see result.py build()).
# created_at is deliberately excluded (wall-clock, not part of plan identity),
# and plan_hash itself is excluded because it hashes the payload that produces it.
_PLAN_HASH_FIELDS = (
    "request",
    "source_profile",
    "evidence_graph",
    "acquisition_plan",
    "candidates",
    "runnable",
    "selected_model",
)


class ExecutionCheck(BaseModel):
    name: str
    status: Literal["pass", "fail", "skipped"]
    detail: str


class ExecutionVerification(BaseModel):
    plan_path: str
    plan_hash_recorded: str
    plan_hash_recomputed: str
    plan_hash_matches: bool
    checks: list[ExecutionCheck] = Field(default_factory=list)
    verified: bool
    selected_model: str | None = None


def _load_plan(plan_path: str | Path) -> dict:
    path = Path(plan_path)
    if not path.exists():
        raise ModelAdapterError(f"Execution plan not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelAdapterError(f"Execution plan is not valid JSON: {path} ({exc})") from exc


def _recompute_plan_hash(plan: dict) -> str:
    payload = {key: plan.get(key) for key in _PLAN_HASH_FIELDS}
    return sha256_json(payload)


def _check_source_integrity(source_profile: dict) -> ExecutionCheck:
    source = source_profile.get("source")
    recorded_sha256 = source_profile.get("sha256")
    if not source or recorded_sha256 is None:
        return ExecutionCheck(
            name="source_integrity",
            status="skipped",
            detail="Source is remote or unhashed (sha256 not recorded at compile time).",
        )
    source_path = Path(source)
    if not source_path.exists():
        return ExecutionCheck(
            name="source_integrity",
            status="skipped",
            detail=f"Source path no longer exists: {source_path}",
        )
    actual_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if actual_sha256 == recorded_sha256:
        return ExecutionCheck(name="source_integrity", status="pass", detail="Source sha256 unchanged since compile.")
    return ExecutionCheck(
        name="source_integrity",
        status="fail",
        detail=f"Source sha256 changed since compile: recorded={recorded_sha256} actual={actual_sha256}.",
    )


def _check_gates(candidate_id: str, request: dict) -> list[ExecutionCheck]:
    entry = zoo_registry.lookup(candidate_id)
    if entry is None:
        detail = f"Zoo entry {candidate_id!r} not found in registry; cannot re-verify gates."
        return [
            ExecutionCheck(name="license_gate", status="skipped", detail=detail),
            ExecutionCheck(name="remote_code_gate", status="skipped", detail=detail),
        ]
    checks = []
    try:
        check_license_gate(
            entry, accept_unknown_license_risk=bool(request.get("accept_unknown_license_risk", False))
        )
        checks.append(ExecutionCheck(name="license_gate", status="pass", detail="License gate re-verified."))
    except ModelAdapterError as exc:
        checks.append(ExecutionCheck(name="license_gate", status="fail", detail=str(exc)))
    try:
        check_remote_code_gate(entry, allow_remote_code=bool(request.get("allow_remote_code", False)))
        checks.append(ExecutionCheck(name="remote_code_gate", status="pass", detail="Remote-code gate re-verified."))
    except ModelAdapterError as exc:
        checks.append(ExecutionCheck(name="remote_code_gate", status="fail", detail=str(exc)))
    return checks


def verify_execution_plan(plan_path: str | Path) -> ExecutionVerification:
    """Re-verify a saved execution-plan.json before it is trusted for execution."""

    plan = _load_plan(plan_path)

    recorded_hash = plan.get("plan_hash", "")
    recomputed_hash = _recompute_plan_hash(plan)
    plan_hash_matches = recorded_hash == recomputed_hash

    checks: list[ExecutionCheck] = []
    checks.append(_check_source_integrity(plan.get("source_profile") or {}))

    selected_model = plan.get("selected_model")
    candidates = plan.get("candidates") or []
    candidate_id = selected_model
    if candidate_id is None:
        runnable = [c for c in candidates if c.get("runnable")]
        candidate_id = runnable[0]["id"] if runnable else None

    if candidate_id is None:
        checks.append(
            ExecutionCheck(
                name="license_gate", status="skipped", detail="No runnable/selected candidate to gate-check."
            )
        )
        checks.append(
            ExecutionCheck(
                name="remote_code_gate", status="skipped", detail="No runnable/selected candidate to gate-check."
            )
        )
    else:
        checks.extend(_check_gates(candidate_id, plan.get("request") or {}))

    verified = plan_hash_matches and all(check.status != "fail" for check in checks)

    return ExecutionVerification(
        plan_path=str(plan_path),
        plan_hash_recorded=recorded_hash,
        plan_hash_recomputed=recomputed_hash,
        plan_hash_matches=plan_hash_matches,
        checks=checks,
        verified=verified,
        selected_model=selected_model,
    )


__all__ = ["ExecutionCheck", "ExecutionVerification", "verify_execution_plan"]
