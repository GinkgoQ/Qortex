# Model Zoo Phase 6: Security, License, and Artifacts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a license gate and a remote-code/executable-allowlist gate
that operate on `ZooEntry`'s already-existing `LicenseInfo`/`SecurityPolicy`
fields (built in Phase 1, unused by any real code path until now), wire
both into the two commands that actually construct/run a model
(`prompt-predict`, `run-external-segmentation`), add a lightweight
file-level provenance ledger for external engine runs, and a synthetic-data
notice helper for generative model entries. This is the final phase of the
Model Zoo expansion.

**Architecture:** `license.py` and `security.py` are pure functions over
`ZooEntry` — no new schema fields, since `LicenseInfo`/`SecurityPolicy`
already carry everything needed (Phase 1). They raise the existing
`ModelAdapterError` (reused throughout every phase so far) when a gate
blocks, and require an explicit CLI opt-in flag to override — never a
silent default. Both CLI commands that already look up a zoo entry and
construct an adapter (`prompt-predict` from Phase 5,
`run-external-segmentation` which predates the zoo) get the gates inserted
at the one point that already exists: after lookup, before
load/construction/execution. The provenance ledger extends
`run_external_segmentation`'s existing metadata-writing step (it already
writes one JSON file per run) rather than inventing a new mechanism.

**Tech Stack:** Python 3.10+ stdlib (`hashlib`, `json`), Pydantic (optional,
`_PYDANTIC` fallback), pytest, Typer.

## Post-review hardening addendum

The MONAI comparison review found additional P0 correctness gaps beyond the
original Phase 6 checklist. The implemented corrections are:

- `src/qortex/neuroai/models/zoo/status.py` normalizes `qortex_status` so
  unresolved promptable checkpoints do not read as executable support.
- `monai.vista3d`, `foundation.medsam`, and `foundation.sam_med3d` are now
  `checkpoint_unresolved` until real checkpoint loading, prompt transforms,
  and end-to-end fixtures exist.
- `src/qortex/neuroai/outputs/dicom_seg_out.py` and
  `src/qortex/neuroai/outputs/dicom_sr_out.py` fail closed: no `.npy` or JSON
  fallback is written under DICOM output types, geometry mismatch raises, and
  written DICOM files are reopened for modality validation.
- `src/qortex/neuroai/models/monai.py` rejects unsafe ZIP members, malformed
  bundle JSON, state-dict mismatches, and sliding-window inference failures
  instead of hiding them behind broad fallback behavior.

Remaining P0 work is not closed by this addendum: full MONAI bundle workflow
execution, source-space geometry inversion, real promptable checkpoint
fixtures, TotalSegmentator task discovery, and full medical evaluation/training
systems require separate implementation phases.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  — this phase implements §16.1 (remote-code gate), §16.2 (executable
  allowlist / command provenance — largely already satisfied by
  `external.py`'s existing subprocess boundary; this phase closes the one
  remaining gap, tying the *declared* `SecurityPolicy.executable_names`
  to the *actually resolved* executable path), §16.3 (license gate), a
  bounded reading of §18 (artifact requirements — see the explicit
  non-goals below), §12.5's generative-model notice convention, and the
  Phase 6 line of §20.
- **No new schema fields.** `LicenseInfo` and `SecurityPolicy` (both built
  in Phase 1) already carry every field these gates need
  (`LicenseInfo.evidence_status`, `.requires_registration`,
  `.commercial_use`; `SecurityPolicy.trust_remote_code_required`,
  `.allow_remote_code`, `.executable_names`). This phase only adds
  functions that *read* them and enforce a decision — it does not touch
  `zoo/schema.py`.
- **No silent defaults.** Every gate defaults to the strictest behavior
  (block on unknown license, block on required-but-not-allowed remote
  code) and requires an explicit, named CLI flag to override — matching
  spec §16.1's "Any entry requiring remote code must be blocked unless the
  user explicitly enables it" and §16.3's "`--accept-unknown-license-risk`"
  flag name.
- **Bounded artifact scope.** Spec §18 lists 11+ provenance files a "full"
  production run would write (`compatibility_report.json`,
  `preprocess_plan.json`, `geometry_ledger.json`, etc.) — most of those
  require pipeline machinery (`CompatibilityEngine`, `PreprocessPlanner`)
  that already exists elsewhere in Qortex but is out of scope for a model
  *zoo* phase to wire end-to-end. This phase implements exactly:
  `model_zoo_entry.json` (entry provenance) and a lightweight
  `geometry_ledger.json` limited to file-level facts (existence, size,
  sha256) — NOT NIfTI header parsing (shape/affine/voxel spacing), since
  that would require adding a new dependency (`nibabel`) not currently used
  anywhere in this codebase. This is a documented, deliberate scope
  boundary, the same pattern as Phase 3's HF-pretrained-checkpoint
  deferral — not a silently dropped requirement.
- Do not modify `_base.py`, `_contracts.py`, `contracts.py`, `spec.py`,
  `zoo/schema.py`, or any adapter file's core logic — only the two CLI
  command functions in `app.py` and `external.py`'s existing metadata-
  writing step gain new gate calls.
- No network calls in this phase's tests.
- Follow existing pytest style: flat `tests/test_neuroai_*.py` files.
- Update `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  §0 checklist (Phase 6 section) in the same commit as the code that
  completes each item. This is the LAST phase — when Task 5 finishes, the
  entire §0 checklist should read as either checked or explicitly
  deferred-with-reason; no blank unchecked item without a note.

---

### Task 1: License gate

**Files:**
- Create: `src/qortex/neuroai/models/license.py`
- Test: `tests/test_neuroai_license_gate.py`

**Interfaces:**
- Consumes: `ZooEntry`, `LicenseInfo` from `zoo/schema.py` (existing);
  `EvidenceStatus` from `qortex.neuroai.contracts` (existing);
  `ModelAdapterError` from `qortex.core.exceptions` (existing, reused
  throughout every prior phase).
- Produces (used by Task 3):
  - `class LicenseStatus(str, Enum)` — `safe_for_open_use`,
    `research_only`, `non_commercial_only`, `registration_required`,
    `unknown`, `blocked` (spec §16.3's exact state list).
  - `evaluate_license(license_info: LicenseInfo) -> LicenseStatus` — pure
    mapping function. Rules, in priority order: `commercial_use is False`
    → `non_commercial_only`; `requires_registration is True` →
    `registration_required`; `evidence_status == EvidenceStatus.unknown` →
    `unknown`; `evidence_status == EvidenceStatus.blocked` → `blocked`;
    otherwise (evidence confirmed/inferred, no restrictions declared) →
    `safe_for_open_use`. `research_only` is reserved for a future explicit
    field Qortex does not yet capture — never inferred, since inferring it
    would be exactly the kind of guessed classification the whole zoo
    architecture forbids; note this in the docstring rather than silently
    omitting the state.
  - `check_license_gate(entry: ZooEntry, *, accept_unknown_license_risk:
    bool = False) -> None` — raises `ModelAdapterError` when
    `evaluate_license(entry.license)` is `unknown` and
    `accept_unknown_license_risk` is `False`, or when it is `blocked`
    unconditionally (no override exists for `blocked` — that state means
    Qortex has confirmed evidence the license forbids use, not merely that
    it hasn't checked). Returns `None` (no exception) for every other
    status.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_license_gate.py
from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.license import LicenseStatus, check_license_gate, evaluate_license
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, ZooEntry, ZooEntryType


def _entry(license_info: LicenseInfo) -> ZooEntry:
    return ZooEntry(
        id="test.model",
        display_name="Test Model",
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/model",
        modality=["eeg"],
        task=["classification"],
        license=license_info,
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_unknown_evidence_status_maps_to_unknown():
    assert evaluate_license(LicenseInfo(evidence_status=EvidenceStatus.unknown)) == LicenseStatus.unknown


def test_blocked_evidence_status_maps_to_blocked():
    assert evaluate_license(LicenseInfo(evidence_status=EvidenceStatus.blocked)) == LicenseStatus.blocked


def test_confirmed_with_no_restrictions_maps_to_safe_for_open_use():
    license_info = LicenseInfo(evidence_status=EvidenceStatus.confirmed, name="MIT")
    assert evaluate_license(license_info) == LicenseStatus.safe_for_open_use


def test_commercial_use_false_maps_to_non_commercial_only():
    license_info = LicenseInfo(evidence_status=EvidenceStatus.confirmed, commercial_use=False)
    assert evaluate_license(license_info) == LicenseStatus.non_commercial_only


def test_requires_registration_maps_to_registration_required():
    license_info = LicenseInfo(evidence_status=EvidenceStatus.confirmed, requires_registration=True)
    assert evaluate_license(license_info) == LicenseStatus.registration_required


def test_gate_blocks_unknown_license_by_default():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.unknown))

    with pytest.raises(ModelAdapterError, match="license"):
        check_license_gate(entry)


def test_gate_allows_unknown_license_with_explicit_opt_in():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.unknown))

    check_license_gate(entry, accept_unknown_license_risk=True)  # must not raise


def test_gate_blocks_blocked_license_even_with_opt_in():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.blocked))

    with pytest.raises(ModelAdapterError):
        check_license_gate(entry, accept_unknown_license_risk=True)


def test_gate_allows_safe_license_with_no_flags():
    entry = _entry(LicenseInfo(evidence_status=EvidenceStatus.confirmed, name="MIT"))

    check_license_gate(entry)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_license_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `license.py`**

```python
# src/qortex/neuroai/models/license.py
"""License gate for zoo entries.

Reads the LicenseInfo already present on every ZooEntry (built in Phase 1)
and enforces a decision -- no new schema fields, no silent defaults. Per
docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md section
16.3: unknown license blocks by default, requiring an explicit
--accept-unknown-license-risk opt-in; a confirmed "blocked" license has no
override, since that state means Qortex has confirmed evidence the
license forbids use, not merely that it hasn't checked yet.
"""

from __future__ import annotations

from enum import Enum

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.schema import LicenseInfo, ZooEntry


class LicenseStatus(str, Enum):
    safe_for_open_use = "safe_for_open_use"
    research_only = "research_only"
    non_commercial_only = "non_commercial_only"
    registration_required = "registration_required"
    unknown = "unknown"
    blocked = "blocked"


def evaluate_license(license_info: LicenseInfo) -> LicenseStatus:
    # research_only is intentionally never inferred here -- Qortex does not
    # yet capture a confirmed "research use only" field distinct from
    # commercial_use/requires_registration, and guessing it would be exactly
    # the kind of fabricated classification the zoo forbids.
    if license_info.commercial_use is False:
        return LicenseStatus.non_commercial_only
    if license_info.requires_registration:
        return LicenseStatus.registration_required
    if license_info.evidence_status == EvidenceStatus.blocked:
        return LicenseStatus.blocked
    if license_info.evidence_status == EvidenceStatus.unknown:
        return LicenseStatus.unknown
    return LicenseStatus.safe_for_open_use


def check_license_gate(entry: ZooEntry, *, accept_unknown_license_risk: bool = False) -> None:
    status = evaluate_license(entry.license)
    if status == LicenseStatus.blocked:
        raise ModelAdapterError(
            f"{entry.id}'s license is confirmed blocked for this use -- cannot proceed."
        )
    if status == LicenseStatus.unknown and not accept_unknown_license_risk:
        raise ModelAdapterError(
            f"{entry.id}'s license has not been verified (evidence_status=unknown). "
            "Pass --accept-unknown-license-risk to proceed at your own risk."
        )


__all__ = ["LicenseStatus", "evaluate_license", "check_license_gate"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_license_gate.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/license.py tests/test_neuroai_license_gate.py
git commit -m "feat(neuroai): add license gate reading existing ZooEntry.license"
```

---

### Task 2: Remote-code and executable-allowlist gates

**Files:**
- Create: `src/qortex/neuroai/models/security.py`
- Test: `tests/test_neuroai_security_gate.py`

**Interfaces:**
- Consumes: `ZooEntry`, `SecurityPolicy` from `zoo/schema.py` (existing);
  `ModelAdapterError` (existing).
- Produces (used by Tasks 3-4):
  - `check_remote_code_gate(entry: ZooEntry, *, allow_remote_code: bool =
    False) -> None` — raises `ModelAdapterError` when
    `entry.security.trust_remote_code_required` is `True` and neither
    `allow_remote_code` (the caller's explicit opt-in) nor
    `entry.security.allow_remote_code` (the entry's own declared default,
    which stays `False` unless a curator explicitly set it) is `True`.
  - `check_executable_allowlist(entry: ZooEntry, resolved_executable_path:
    str) -> None` — raises `ModelAdapterError` when
    `entry.security.executable_names` is non-empty and the basename of
    `resolved_executable_path` is not in that list. When
    `executable_names` is empty (not declared), this is a no-op — an
    entry with no declared allowlist has nothing to check against, and
    Phase 4's external-engine entries all populate this field, so this
    only ever no-ops for entries that predate that convention.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_security_gate.py
from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.security import check_executable_allowlist, check_remote_code_gate
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, SecurityPolicy, ZooEntry, ZooEntryType


def _entry(security: SecurityPolicy) -> ZooEntry:
    return ZooEntry(
        id="test.model",
        display_name="Test Model",
        entry_type=ZooEntryType.model,
        provider="plugin",
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/model",
        modality=["eeg"],
        task=["classification"],
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        security=security,
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_remote_code_gate_blocks_by_default_when_required():
    entry = _entry(SecurityPolicy(trust_remote_code_required=True))

    with pytest.raises(ModelAdapterError, match="remote code"):
        check_remote_code_gate(entry)


def test_remote_code_gate_allows_with_caller_opt_in():
    entry = _entry(SecurityPolicy(trust_remote_code_required=True))

    check_remote_code_gate(entry, allow_remote_code=True)  # must not raise


def test_remote_code_gate_allows_when_entry_declares_it_allowed():
    entry = _entry(SecurityPolicy(trust_remote_code_required=True, allow_remote_code=True))

    check_remote_code_gate(entry)  # must not raise


def test_remote_code_gate_noop_when_not_required():
    entry = _entry(SecurityPolicy())

    check_remote_code_gate(entry)  # must not raise


def test_executable_allowlist_blocks_mismatched_path():
    entry = _entry(SecurityPolicy(executable_names=["TotalSegmentator"]))

    with pytest.raises(ModelAdapterError, match="executable"):
        check_executable_allowlist(entry, "/usr/local/bin/some_other_tool")


def test_executable_allowlist_allows_matching_basename():
    entry = _entry(SecurityPolicy(executable_names=["TotalSegmentator"]))

    check_executable_allowlist(entry, "/usr/local/bin/TotalSegmentator")  # must not raise


def test_executable_allowlist_noop_when_not_declared():
    entry = _entry(SecurityPolicy())

    check_executable_allowlist(entry, "/usr/local/bin/anything")  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_security_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `security.py`**

```python
# src/qortex/neuroai/models/security.py
"""Remote-code and executable-allowlist gates for zoo entries.

Reads the SecurityPolicy already present on every ZooEntry (built in
Phase 1) and enforces a decision -- no new schema fields, no silent
defaults. Per docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 16.1-16.2.
"""

from __future__ import annotations

import os

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models.zoo.schema import ZooEntry


def check_remote_code_gate(entry: ZooEntry, *, allow_remote_code: bool = False) -> None:
    if not entry.security.trust_remote_code_required:
        return
    if allow_remote_code or entry.security.allow_remote_code:
        return
    raise ModelAdapterError(
        f"{entry.id} requires remote Python code execution. "
        "Use --allow-remote-code only in a trusted environment."
    )


def check_executable_allowlist(entry: ZooEntry, resolved_executable_path: str) -> None:
    allowed = entry.security.executable_names
    if not allowed:
        return
    basename = os.path.basename(resolved_executable_path)
    if basename not in allowed:
        raise ModelAdapterError(
            f"{entry.id}'s security policy allows executables {allowed!r}, "
            f"but resolved executable is {basename!r}."
        )


__all__ = ["check_remote_code_gate", "check_executable_allowlist"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_security_gate.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/security.py tests/test_neuroai_security_gate.py
git commit -m "feat(neuroai): add remote-code and executable-allowlist gates"
```

---

### Task 3: Wire license and remote-code gates into `prompt-predict`

**Files:**
- Modify: `src/qortex/cli/app.py` (`neuroai_prompt_predict`, added in
  Phase 5 — add `--accept-unknown-license-risk` and `--allow-remote-code`
  options, call both gates after the zoo lookup and before
  `make_model_adapter`)
- Test: `tests/test_neuroai_prompt_predict_cli.py` (existing, from Phase 5
  — append new test cases)

**Interfaces:**
- Consumes: `check_license_gate` (Task 1), `check_remote_code_gate`
  (Task 2).
- Produces: `qortex neuroai prompt-predict ... [--accept-unknown-license-risk]
  [--allow-remote-code]`.

- [ ] **Step 1: Write the failing test**

Read `tests/test_neuroai_prompt_predict_cli.py` first (4 existing tests
from Phase 5). Append:

```python
def test_prompt_predict_blocks_unknown_license_by_default():
    # monai.vista3d has evidence_status=unknown on its LicenseInfo (Phase 1/2
    # seed data never confirmed a specific license).
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.nii.gz", "--model", "monai.vista3d", "--point", "1,2,3"],
    )
    assert result.exit_code != 0
    assert "license" in result.output.lower()


def test_prompt_predict_allows_with_license_risk_accepted():
    result = runner.invoke(
        app,
        [
            "neuroai", "prompt-predict", "input.nii.gz",
            "--model", "monai.vista3d", "--point", "1,2,3",
            "--accept-unknown-license-risk",
        ],
    )
    assert result.exit_code == 0, result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_prompt_predict_cli.py -v -k license`
Expected: FAIL — `test_prompt_predict_blocks_unknown_license_by_default`
fails because no license gate exists yet (command currently succeeds
unconditionally for a valid promptable model + valid prompt).

- [ ] **Step 3: Wire the gates into the command**

Read the current `neuroai_prompt_predict` function in
`src/qortex/cli/app.py` first (added in Phase 5 — has `--model`, `--point`,
`--point-label`, `--box`, `--text` options and looks up the zoo entry,
checks `entry_type`, builds a `Prompt`, constructs the adapter, validates
against `interaction_contract()`). Add two new options to the signature:

```python
    accept_unknown_license_risk: bool = typer.Option(False, "--accept-unknown-license-risk", help="Proceed even if the model's license has not been verified"),
    allow_remote_code: bool = typer.Option(False, "--allow-remote-code", help="Allow remote code execution if the model requires it"),
```

Immediately after the existing `entry_type` check (which already raises
`typer.Exit(1)` if the entry isn't promptable) and before constructing the
adapter via `make_model_adapter`, add:

```python
    from qortex.neuroai.models.license import check_license_gate
    from qortex.neuroai.models.security import check_remote_code_gate
    from qortex.core.exceptions import ModelAdapterError

    try:
        check_license_gate(entry, accept_unknown_license_risk=accept_unknown_license_risk)
        check_remote_code_gate(entry, allow_remote_code=allow_remote_code)
    except ModelAdapterError as exc:
        typer.echo(f"[ERROR] {exc}", err=True)
        raise typer.Exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_prompt_predict_cli.py -v`
Expected: PASS (6 tests: 4 existing + 2 new)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt*.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py tests/test_neuroai_license_gate.py tests/test_neuroai_security_gate.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/qortex/cli/app.py tests/test_neuroai_prompt_predict_cli.py
git commit -m "feat(cli): wire license and remote-code gates into prompt-predict"
```

---

### Task 4: Executable allowlist + zoo provenance in external segmentation runs

**Files:**
- Modify: `src/qortex/neuroai/external.py` (`run_external_segmentation`)
- Test: `tests/test_neuroai_external.py` (existing — append new test
  cases)

**Interfaces:**
- Consumes: `check_executable_allowlist` (Task 2); `qortex.neuroai.models.zoo`
  (existing — needs `lookup`); `hashlib`.
- Produces: `run_external_segmentation` now, when a matching zoo entry
  exists for the request's engine (`external.<engine>`, matching the
  naming convention every Phase 4 entry already uses), additionally writes
  a `model_zoo_entry.json` file alongside the existing metadata file, and
  enforces `check_executable_allowlist` against the resolved command's
  executable path before running the subprocess.

- [ ] **Step 1: Write the failing test**

Read `tests/test_neuroai_external.py` first (has `_write_executable`
helper and tests through Phase 4). Append:

```python
def test_run_totalsegmentator_writes_zoo_provenance_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401  (triggers zoo registration)

    _write_executable(
        tmp_path / "TotalSegmentator",
        '#!/usr/bin/env bash\nprintf "mask" > "${@: -1}"\n',
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "image.nii.gz"
    image.write_text("image", encoding="utf-8")
    output = tmp_path / "mask.nii.gz"

    result = run_external_segmentation(
        ExternalSegmentationRequest(engine="totalsegmentator", image_path=image, output_path=output, task="total")
    )

    # Deterministic: metadata_path.name + ".model_zoo_entry.json", same
    # parent directory as the existing metadata file.
    provenance_path = result.metadata_path.with_name(result.metadata_path.name + ".model_zoo_entry.json")
    assert provenance_path.exists()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert payload["zoo_entry_id"] == "external.totalsegmentator"
    assert payload["provider"] == "external_cli"


def test_run_external_segmentation_enforces_executable_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401

    # Symlink a wrong-named fake executable under the name TotalSegmentator
    # expects to resolve, but monkeypatch shutil.which to return a path whose
    # basename does NOT match the zoo entry's declared executable_names, to
    # exercise the allowlist check deterministically without relying on PATH
    # resolution quirks.
    fake = tmp_path / "not_totalsegmentator"
    _write_executable(fake, "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setattr("shutil.which", lambda name: str(fake) if name == "TotalSegmentator" else None)
    image = tmp_path / "image.nii.gz"
    image.write_text("image", encoding="utf-8")

    with pytest.raises(ExternalSegmentationError, match="executable"):
        run_external_segmentation(
            ExternalSegmentationRequest(engine="totalsegmentator", image_path=image, output_path=tmp_path / "mask.nii.gz", task="total")
        )
```

You will need `import json` if not already present in the test file (check
the top first).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_external.py -v -k "zoo_provenance or allowlist"`
Expected: FAIL — no `model_zoo_entry.json` written yet, and the allowlist
mismatch does not raise yet (both are no-ops today)

- [ ] **Step 3: Implement in `external.py`**

Read the current file first (extended twice already, in Phase 4). In
`run_external_segmentation`, after `command = _build_external_command(...)`
and before the `subprocess.run(...)` call, add the allowlist check —
resolving the actual executable is already done inside each
`_build_*_command` via `_require_executable`, so the check happens against
`command[0]` (the first element of the built argv is always the resolved
executable path, per every existing builder):

```python
    zoo_entry_id = f"external.{request.engine}"
    from qortex.neuroai.models.zoo.registry import lookup as _zoo_lookup
    from qortex.neuroai.models.security import check_executable_allowlist

    zoo_entry = _zoo_lookup(zoo_entry_id)
    if zoo_entry is not None:
        try:
            check_executable_allowlist(zoo_entry, command[0])
        except Exception as exc:  # ModelAdapterError from security.py
            raise ExternalSegmentationError(str(exc)) from exc
```

Place this block immediately after the existing `command = _build_external_command(request, image_path, output_path)` line.

After the existing `result.metadata_path.write_text(...)` line, add zoo
provenance writing (only when a zoo entry was found):

```python
    if zoo_entry is not None:
        # Simple, deterministic append -- always a sibling of metadata_path
        # regardless of whether it's "mask.nii.gz.qortex.json" (file case)
        # or "qortex_external_segmentation.json" (directory case), since
        # with_name() replaces the filename in the same parent directory
        # either way.
        provenance_path = result.metadata_path.with_name(
            result.metadata_path.name + ".model_zoo_entry.json"
        )
        provenance_path.write_text(
            json.dumps(
                {
                    "zoo_entry_id": zoo_entry.id,
                    "provider": zoo_entry.provider,
                    "source_url": zoo_entry.source_url,
                    "license_evidence_status": zoo_entry.license.evidence_status.value,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
```

This produces e.g. `mask.nii.gz.qortex.json.model_zoo_entry.json` next to
`mask.nii.gz.qortex.json` — a longer name than ideal, but simple,
collision-free, and correct for both the file and directory forms of
`metadata_path` without any conditional string surgery.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_external.py -v`
Expected: PASS (all existing + 2 new)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt*.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py tests/test_neuroai_license_gate.py tests/test_neuroai_security_gate.py -v`
Expected: all PASS

- [ ] **Step 6: Update the spec's progress checklist**

Check off "Executable allowlist" and "Model zoo artifact integration"
under Phase 6 in §0. For "Geometry ledger requirement," check it off with
a note: "Implemented as file-level provenance (existence, size, sha256) in
`model_zoo_entry.json`; NIfTI header-level geometry (shape/affine/voxel
spacing) deferred — would require adding `nibabel`, not currently a
dependency anywhere in this codebase."

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/external.py tests/test_neuroai_external.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): enforce executable allowlist and write zoo provenance for external runs"
```

---

### Task 5: Synthetic-data notice and final checklist closeout

**Files:**
- Modify: `src/qortex/neuroai/models/zoo/monai_generative.py` (add a
  `synthetic_data_notice()` helper function — read the file first, it
  already has 7 entries from Phase 2)
- Modify: `src/qortex/cli/app.py` (`neuroai_zoo_show` — append the notice
  when the entry is `entry_type=generative_model`)
- Test: `tests/test_neuroai_zoo_monai_generative.py` (existing — append),
  `tests/test_neuroai_zoo_cli.py` (existing — append)

**Interfaces:**
- Consumes: `ZooEntry`, `ZooEntryType` (existing).
- Produces: `synthetic_data_notice(entry: ZooEntry) -> dict[str, object]`
  — returns `{"clinical_use": "prohibited", "research_use": "allowed",
  "watermark_synthetic": True, "require_generation_metadata": True}` for
  any `entry_type=generative_model` entry, matching spec §12.5's YAML
  convention exactly; raises `ValueError` if called on a non-generative
  entry (this function should never be called for anything else, and a
  loud failure is better than a silently-wrong notice on a diagnostic
  model).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_neuroai_zoo_monai_generative.py
def test_synthetic_data_notice_matches_spec_convention():
    from qortex.neuroai.models.zoo.monai_generative import synthetic_data_notice
    from qortex.neuroai.models.zoo.registry import lookup

    entry = lookup("monai.mednist_gan")
    notice = synthetic_data_notice(entry)

    assert notice == {
        "clinical_use": "prohibited",
        "research_use": "allowed",
        "watermark_synthetic": True,
        "require_generation_metadata": True,
    }


def test_synthetic_data_notice_rejects_non_generative_entry():
    from qortex.neuroai.models.zoo.monai_generative import synthetic_data_notice
    from qortex.neuroai.models.zoo.registry import lookup

    entry = lookup("monai.brats_mri_segmentation")

    with pytest.raises(ValueError):
        synthetic_data_notice(entry)
```

```python
# append to tests/test_neuroai_zoo_cli.py
def test_zoo_show_includes_synthetic_data_notice_for_generative_entry():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "monai.mednist_gan"])
    assert result.exit_code == 0
    assert "clinical_use" in result.output
    assert "prohibited" in result.output


def test_zoo_show_omits_synthetic_data_notice_for_non_generative_entry():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "braindecode.EEGNet"])
    assert result.exit_code == 0
    assert "clinical_use" not in result.output
```

You will need `import pytest` in `test_neuroai_zoo_monai_generative.py` if
not already present — check the top of the file first.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_monai_generative.py tests/test_neuroai_zoo_cli.py -v -k "synthetic or notice"`
Expected: FAIL — `ImportError: cannot import name 'synthetic_data_notice'`

- [ ] **Step 3: Implement `synthetic_data_notice()`**

Add to `src/qortex/neuroai/models/zoo/monai_generative.py` (after
`register_all()`):

```python
def synthetic_data_notice(entry: ZooEntry) -> dict[str, object]:
    """Structured clinical-use notice for a generative model entry, per
    design spec section 12.5. Raises if called on a non-generative entry
    -- this notice must never be attached to a diagnostic model's output.
    """
    if entry.entry_type != ZooEntryType.generative_model:
        raise ValueError(
            f"synthetic_data_notice() called on non-generative entry {entry.id!r} "
            f"(entry_type={entry.entry_type.value})"
        )
    return {
        "clinical_use": "prohibited",
        "research_use": "allowed",
        "watermark_synthetic": True,
        "require_generation_metadata": True,
    }
```

Add `ZooEntry` to this file's existing schema import if not already
present (it currently imports `ExecutionMode`, `LicenseInfo`, `ZooEntry`,
`ZooEntryType` — confirm before editing). Add `"synthetic_data_notice"` to
`__all__`.

- [ ] **Step 4: Wire into `neuroai_zoo_show`**

Read the current `neuroai_zoo_show` function in `src/qortex/cli/app.py`
first (added in Phase 1 — prints id, display_name, entry_type, provider,
execution_mode, source_url, modality, task, evidence_status, license,
qortex_status). After the existing print statements, add:

```python
    if entry.entry_type.value == "generative_model":
        from qortex.neuroai.models.zoo.monai_generative import synthetic_data_notice
        notice = synthetic_data_notice(entry)
        typer.echo(f"synthetic_data_notice: {notice}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_monai_generative.py tests/test_neuroai_zoo_cli.py -v`
Expected: PASS (all)

Then the complete model zoo test suite:
Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt*.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py tests/test_neuroai_license_gate.py tests/test_neuroai_security_gate.py -v`
Expected: all PASS

- [ ] **Step 6: Final spec checklist closeout**

Check off "Synthetic-data notice for generative models" under Phase 6 in
§0 — this is the last unchecked box in the entire document. Read through
all six phases' checklists one more time and confirm every item is either
`[x]` or has an explicit deferral note (there should be exactly three
deferrals in the whole document: HF pretrained registry support in Phase
3, TotalSegmentator task discovery in Phase 4, and NIfTI-header geometry
in Phase 6 — each with a one-line reason already written by prior tasks).
Add a closing line at the end of §0: "**Model Zoo expansion (Phases
1-6): complete.**"

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/models/zoo/monai_generative.py src/qortex/cli/app.py tests/test_neuroai_zoo_monai_generative.py tests/test_neuroai_zoo_cli.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): add synthetic-data notice for generative entries

Completes Phase 6 (Security, license, and artifacts) -- the final phase
of the model zoo expansion."
```

---

## Phase 6 exit criteria

- [ ] `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt*.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py tests/test_neuroai_license_gate.py tests/test_neuroai_security_gate.py -v` — all green.
- [ ] `qortex neuroai prompt-predict input.nii.gz --model monai.vista3d --point 1,2,3` fails with a license-gate error by default, succeeds with `--accept-unknown-license-risk`.
- [ ] `qortex neuroai zoo show monai.mednist_gan` includes the synthetic-data notice; `qortex neuroai zoo show braindecode.EEGNet` does not.
- [ ] `qortex neuroai run-external-segmentation totalsegmentator ...` writes a `model_zoo_entry.json` alongside its existing metadata file.
- [ ] Spec §0 checklist: every phase fully checked or explicitly deferred with a one-line reason; closing "complete" line present.
- [ ] No changes to `_base.py`, `_contracts.py`, `contracts.py`, `spec.py`,
      `zoo/schema.py`, or any adapter's core inference logic.

This is the final phase. Once merged, the Model Zoo expansion (design spec
`2026-07-09-model-zoo-expansion-design.md`) is complete end to end:
registry hardening, MONAI integration, Braindecode expansion, external CLI
engines, promptable segmentation, and security/license/artifact gates.
