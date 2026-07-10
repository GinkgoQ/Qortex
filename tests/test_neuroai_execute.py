from __future__ import annotations

import json

from typer.testing import CliRunner

from qortex.cli.app import app
from qortex.neuroai.compiler import CompilationRequest, compile_neuroai, verify_execution_plan

runner = CliRunner()


def _compile_and_save(tmp_path, source_bytes: bytes = b"source"):
    source = tmp_path / "image.nii.gz"
    source.write_bytes(source_bytes)
    request = CompilationRequest(
        source=str(source),
        task="segmentation",
        accept_unknown_license_risk=True,
    )
    result = compile_neuroai(request)
    output = tmp_path / "execution-plan.json"
    result.save(output)
    return source, output, result


def test_verify_execution_plan_passes_for_untampered_plan(tmp_path):
    _source, output, result = _compile_and_save(tmp_path)

    verification = verify_execution_plan(output)

    assert verification.plan_hash_matches is True
    checks_by_name = {c.name: c for c in verification.checks}
    assert checks_by_name["source_integrity"].status == "pass"
    assert verification.selected_model == result.selected_model
    # verified should reflect whatever the gate outcomes are (no forced True)
    assert verification.verified == (
        verification.plan_hash_matches and all(c.status != "fail" for c in verification.checks)
    )


def test_verify_execution_plan_detects_tampering(tmp_path):
    _source, output, _result = _compile_and_save(tmp_path)

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["candidates"][0]["provider"] = "tampered-provider"
    output.write_text(json.dumps(payload), encoding="utf-8")

    verification = verify_execution_plan(output)

    assert verification.plan_hash_matches is False
    assert verification.verified is False


def test_verify_execution_plan_detects_source_drift(tmp_path):
    source, output, _result = _compile_and_save(tmp_path)

    source.write_bytes(b"different-bytes-than-compile-time")

    verification = verify_execution_plan(output)

    checks_by_name = {c.name: c for c in verification.checks}
    assert checks_by_name["source_integrity"].status == "fail"
    assert verification.verified is False


def test_execute_cli_exits_zero_on_good_plan_and_nonzero_on_tampered_plan(tmp_path):
    _source, output, _result = _compile_and_save(tmp_path)

    good = runner.invoke(app, ["execute", str(output)])
    assert good.exit_code == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["candidates"][0]["provider"] = "tampered-provider"
    output.write_text(json.dumps(payload), encoding="utf-8")

    tampered = runner.invoke(app, ["execute", str(output)])
    assert tampered.exit_code != 0
