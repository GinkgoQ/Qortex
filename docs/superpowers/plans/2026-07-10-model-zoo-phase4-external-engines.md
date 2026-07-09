# Model Zoo Phase 4: External CLI Engines — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `neuroai/external.py`'s existing subprocess-boundary pattern
(already proven for TotalSegmentator and nnU-Net) with 5 more neuroimaging
CLI engines — SynthSeg, SynthStrip, HD-BET, FastSurfer, TractSeg — and
register each as a `ZooEntry(entry_type=external_engine)`, reusing the exact
`ExternalEngineContract` pattern Phase 1 already established for
`external.totalsegmentator`.

**Architecture:** `external.py` gains 5 new `_build_<engine>_command`
functions following the identical shape of the existing
`_build_totalsegmentator_command`/`_build_nnunet_command`, added to the
`ExternalSegmentationEngine` Literal, `_validate_external_request`,
`_build_external_command`'s dispatch, and
`available_external_segmentation_engines()`. FastSurfer's CLI shape differs
from the others (subject-id + subject-directory layout, not a flat
input→output file mapping) — handled the same way nnU-Net's extra required
fields already are: an optional `subject_id` field on
`ExternalSegmentationRequest`, validated as required only for that one
engine. No new module is needed for the registry side — `zoo/schema.py`'s
`ExternalEngineContract` and the `entry_type=external_engine` pattern
already exist and are already proven by Phase 1's
`external.totalsegmentator` seed entry; this phase just adds 5 more
entries the same way.

**Tech Stack:** Python 3.10+ stdlib `subprocess`/`shutil`, pytest (fake
executable shell scripts on `PATH` via `tmp_path`/`monkeypatch`, exactly
matching the existing `tests/test_neuroai_external.py` pattern — no real
binaries required).

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  — this phase implements §11.4, §12.2 (the 5 new engines; `nnU-Net` and
  `TotalSegmentator` are already done), and the Phase 4 line of §20.
- **Build to documented CLI contracts, no binaries required** (per the
  user's own scoping decision during brainstorming). Command builders are
  unit-tested with fake executable shell scripts on `PATH`, exactly like
  the existing `tests/test_neuroai_external.py::test_run_totalsegmentator_external_boundary`.
  Real-binary smoke testing is out of scope for this phase.
- **Never fabricate a CLI flag you are not confident is real.** Where an
  engine's CLI has an option this plan is not fully certain about (e.g.
  exact GPU-selection syntax), the command builder only sets the flags
  explicitly listed in this plan's task code, and forwards anything else
  through the existing `extra_args` passthrough — never invents a flag
  name to "complete" the picture.
- Do not modify `_base.py`, `_contracts.py`, `_registry.py`,
  `contracts.py`, `spec.py`, or any adapter file. `external.py` is the one
  existing file this phase modifies (additively — new engines only, no
  behavior change to the existing `totalsegmentator`/`nnunet` paths).
- No network calls, no binary downloads, anywhere in this phase's code or
  tests.
- Follow existing patterns exactly: `tests/test_neuroai_external.py`'s
  fake-executable-on-PATH style for Task 1; `zoo/seed_examples.py`'s
  `external.totalsegmentator` entry (already registered in Phase 1) as the
  template for Task 2's 5 new entries.
- Update `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  §0 checklist (Phase 4 section + "Registry entries implemented so far") in
  the same commit as the code that completes each item.

---

### Task 1: Extend `external.py` with 5 new CLI engines

**Files:**
- Modify: `src/qortex/neuroai/external.py`
- Test: `tests/test_neuroai_external.py` (existing file from before Phase 1
  — read it first; it already has `_write_executable` helper and tests for
  `totalsegmentator`/`nnunet`. Append new tests using the same helper.)

**Interfaces:**
- Consumes: nothing new — this task only extends existing types in
  `external.py` (`ExternalSegmentationRequest`, `ExternalSegmentationEngine`,
  `_build_external_command`, `_require_executable`, `_clean_extra_args`).
- Produces (used by Task 2 and Task 3):
  - `ExternalSegmentationEngine` Literal extended to `["totalsegmentator",
    "nnunet", "synthseg", "synthstrip", "hdbet", "fastsurfer", "tractseg"]`.
  - `ExternalSegmentationRequest` gains one new optional field:
    `subject_id: str | None = None` (required only for `engine="fastsurfer"`).
  - `_build_synthseg_command`, `_build_synthstrip_command`,
    `_build_hdbet_command`, `_build_fastsurfer_command`,
    `_build_tractseg_command` — same signature shape as the existing two.
  - `available_external_segmentation_engines()` reports 5 more keys:
    `"synthseg"`, `"synthstrip"`, `"hdbet"`, `"fastsurfer"`, `"tractseg"`.

- [ ] **Step 1: Write the failing tests**

Read `tests/test_neuroai_external.py` first to confirm the exact
`_write_executable` helper signature and import style, then append:

```python
def test_external_segmentation_reports_all_7_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", str(tmp_path))

    engines = available_external_segmentation_engines()

    assert set(engines.keys()) == {
        "totalsegmentator", "nnunet", "synthseg", "synthstrip",
        "hdbet", "fastsurfer", "tractseg",
    }
    assert all(v is False for v in engines.values())


def test_build_synthseg_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "mri_synthseg", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="synthseg",
            image_path=image,
            output_path=tmp_path / "seg.nii.gz",
            device="cpu",
        )
    )

    assert command[0].endswith("mri_synthseg")
    assert "--i" in command and str(image) in command
    assert "--o" in command
    assert "--cpu" in command


def test_build_synthstrip_command_gpu_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "mri_synthstrip", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="synthstrip",
            image_path=image,
            output_path=tmp_path / "brain.nii.gz",
            device="cuda",
        )
    )

    assert command[0].endswith("mri_synthstrip")
    assert "-i" in command and str(image) in command
    assert "-o" in command
    assert "--gpu" in command


def test_build_hdbet_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "hd-bet", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="hdbet",
            image_path=image,
            output_path=tmp_path / "brain.nii.gz",
            device="cpu",
        )
    )

    assert command[0].endswith("hd-bet")
    assert "-i" in command and str(image) in command
    assert "-o" in command
    assert "-device" in command and "cpu" in command


def test_build_fastsurfer_command_requires_subject_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "run_fastsurfer.sh", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    with pytest.raises(ExternalSegmentationError, match="subject_id"):
        build_external_segmentation_command(
            ExternalSegmentationRequest(
                engine="fastsurfer",
                image_path=image,
                output_path=tmp_path / "subjects",
            )
        )


def test_build_fastsurfer_command_with_subject_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "run_fastsurfer.sh", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="fastsurfer",
            image_path=image,
            output_path=tmp_path / "subjects",
            subject_id="sub-01",
        )
    )

    assert command[0].endswith("run_fastsurfer.sh")
    assert "--t1" in command and str(image) in command
    assert "--sid" in command and "sub-01" in command
    assert "--sd" in command


def test_build_tractseg_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_executable(tmp_path / "TractSeg", "#!/usr/bin/env bash\nexit 0\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "dwi.nii.gz"
    image.write_text("image", encoding="utf-8")

    command = build_external_segmentation_command(
        ExternalSegmentationRequest(
            engine="tractseg",
            image_path=image,
            output_path=tmp_path / "bundles",
        )
    )

    assert command[0].endswith("TractSeg")
    assert "-i" in command and str(image) in command
    assert "-o" in command


def test_unsupported_engine_still_raises():
    with pytest.raises(ExternalSegmentationError):
        build_external_segmentation_command(
            ExternalSegmentationRequest(
                engine="not_a_real_engine",  # type: ignore[arg-type]
                image_path="x.nii.gz",
                output_path="y.nii.gz",
            )
        )
```

You will need `import os` and `ExternalSegmentationError` imported in the
test file if not already present — check the top of the file first (the
existing tests already import `os`, `Path`, `pytest`, and several names
from `qortex.neuroai` per the file's existing content).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_neuroai_external.py -v -k "synthseg or synthstrip or hdbet or fastsurfer or tractseg or reports_all_7"`
Expected: FAIL — `ExternalSegmentationError: Unsupported external segmentation engine` for each new engine, and the 7-engine count test fails (only 2 keys currently).

- [ ] **Step 3: Implement the extensions in `external.py`**

Read the current file first (276 lines) to confirm line numbers before
editing — line numbers below are current-as-of-Phase-3 but may drift if a
concurrent phase touched this file.

Change the Literal (currently line 23):

```python
ExternalSegmentationEngine = Literal[
    "totalsegmentator", "nnunet", "synthseg", "synthstrip", "hdbet",
    "fastsurfer", "tractseg",
]
```

Add one field to `ExternalSegmentationRequest` (currently lines 26-43),
immediately after the existing `folds` field:

```python
    subject_id: str | None = None  # required only for engine="fastsurfer"
```

Extend `available_external_segmentation_engines()` (currently lines
154-160):

```python
def available_external_segmentation_engines() -> dict[str, bool]:
    """Report which supported external segmentation CLIs are on PATH."""

    return {
        "totalsegmentator": shutil.which("TotalSegmentator") is not None,
        "nnunet": shutil.which("nnUNetv2_predict") is not None,
        "synthseg": shutil.which("mri_synthseg") is not None,
        "synthstrip": shutil.which("mri_synthstrip") is not None,
        "hdbet": shutil.which("hd-bet") is not None,
        "fastsurfer": shutil.which("run_fastsurfer.sh") is not None,
        "tractseg": shutil.which("TractSeg") is not None,
    }
```

Extend `_validate_external_request` (currently lines 163-183) — change the
engine-membership check and add a fastsurfer-specific required-field check
alongside the existing nnunet one:

```python
def _validate_external_request(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
    *,
    check_image_exists: bool = True,
) -> None:
    _SUPPORTED_ENGINES = (
        "totalsegmentator", "nnunet", "synthseg", "synthstrip", "hdbet",
        "fastsurfer", "tractseg",
    )
    if request.engine not in _SUPPORTED_ENGINES:
        raise ExternalSegmentationError(f"Unsupported external segmentation engine: {request.engine!r}")
    if check_image_exists and not image_path.exists():
        raise ExternalSegmentationError(f"Input image does not exist: {image_path}")
    if request.engine == "nnunet":
        missing = []
        if request.dataset_id is None:
            missing.append("dataset_id")
        if request.configuration is None:
            missing.append("configuration")
        if missing:
            raise ExternalSegmentationError(
                f"nnU-Net request is missing required fields: {', '.join(missing)}"
            )
    if request.engine == "fastsurfer" and request.subject_id is None:
        raise ExternalSegmentationError(
            "FastSurfer request is missing required field: subject_id"
        )
```

Extend `_build_external_command`'s dispatch (currently lines 186-193):

```python
def _build_external_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    builders = {
        "totalsegmentator": _build_totalsegmentator_command,
        "nnunet": _build_nnunet_command,
        "synthseg": _build_synthseg_command,
        "synthstrip": _build_synthstrip_command,
        "hdbet": _build_hdbet_command,
        "fastsurfer": _build_fastsurfer_command,
        "tractseg": _build_tractseg_command,
    }
    return builders[request.engine](request, image_path, output_path)
```

Add the 5 new command builders after the existing `_build_nnunet_command`
(currently ends at line 237), before `_require_executable`:

```python
def _build_synthseg_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("mri_synthseg")
    command = [executable, "--i", str(image_path), "--o", str(output_path)]
    if request.device == "cpu":
        command.append("--cpu")
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_synthstrip_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("mri_synthstrip")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    if request.device and request.device != "cpu":
        command.append("--gpu")
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_hdbet_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("hd-bet")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    if request.device:
        command.extend(["-device", request.device])
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_fastsurfer_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    # FastSurfer's CLI shape differs from the others: it writes into a
    # subjects-directory layout keyed by subject_id, not a single output
    # file/dir. subject_id is validated as required in
    # _validate_external_request before this builder ever runs.
    executable = _require_executable("run_fastsurfer.sh")
    command = [
        executable,
        "--t1", str(image_path),
        "--sid", str(request.subject_id),
        "--sd", str(output_path),
    ]
    if request.device:
        command.extend(["--device", request.device])
    command.extend(_clean_extra_args(request.extra_args))
    return command


def _build_tractseg_command(
    request: ExternalSegmentationRequest,
    image_path: Path,
    output_path: Path,
) -> list[str]:
    executable = _require_executable("TractSeg")
    command = [executable, "-i", str(image_path), "-o", str(output_path)]
    command.extend(_clean_extra_args(request.extra_args))
    return command
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_external.py -v`
Expected: PASS (all existing tests + 8 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/external.py tests/test_neuroai_external.py
git commit -m "feat(neuroai): add SynthSeg/SynthStrip/HD-BET/FastSurfer/TractSeg external engines"
```

---

### Task 2: Register the 5 new external engines as zoo entries

**Files:**
- Create: `src/qortex/neuroai/models/zoo/external_engines.py`
- Modify: `src/qortex/neuroai/models/zoo/__init__.py` (add
  `external_engines.register_all()` alongside existing calls)
- Modify: `tests/conftest.py` (add the same call to the shared autouse
  re-seed fixture — read it first, it already calls 4 domain modules)
- Test: `tests/test_neuroai_zoo_external_engines.py`

**Interfaces:**
- Consumes: `ZooEntry`, `ZooEntryType`, `ExecutionMode`, `LicenseInfo`,
  `ExternalEngineContract` from `zoo/schema.py` (all already exist since
  Phase 1); `register` from `zoo/registry.py`.
- Produces: 5 new `entry_type=external_engine` entries, following the
  exact shape of Phase 1's `external.totalsegmentator` seed entry (in
  `zoo/seed_examples.py` — read it first as the template).

Register these 5, each `provider="external_cli"`,
`execution_mode=ExecutionMode.external_cli`, `entry_type=ZooEntryType.external_engine`,
`license=LicenseInfo(evidence_status=EvidenceStatus.unknown,
notes=["requires manual check"])`, `evidence_status=EvidenceStatus.confirmed`,
`priority="P0"`, `qortex_status="runnable_if_executable_available"`:

| id | executable | modality | supported_tasks | source_url |
|---|---|---|---|---|
| `external.synthseg` | `mri_synthseg` | mri | `["whole_brain_segmentation"]` | `https://github.com/BBillot/SynthSeg` |
| `external.synthstrip` | `mri_synthstrip` | mri | `["skull_stripping"]` | `https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/` |
| `external.hdbet` | `hd-bet` | mri | `["skull_stripping"]` | `https://github.com/MIC-DKFZ/HD-BET` |
| `external.fastsurfer` | `run_fastsurfer.sh` | mri | `["whole_brain_segmentation", "cortical_parcellation"]` | `https://github.com/Deep-MI/FastSurfer` |
| `external.tractseg` | `TractSeg` | dwi | `["white_matter_tract_segmentation"]` | `https://github.com/MIC-DKFZ/TractSeg` |

Each `ExternalEngineContract`: `engine=<id without "external." prefix>`,
`executable=<executable from table>`, `input_file_types=["nifti"]`,
`output_file_types=["nifti"]` (`fastsurfer` additionally
`["nifti", "directory"]` since it writes a subjects-directory tree, not a
single file), `supported_modalities=[<modality from table>]`,
`supported_tasks=<from table>`, `command_builder="_build_<engine>_command"`
(matching Task 1's actual function names), `output_manifest_supported=False`
(none of these 5 emit a machine-readable capability/manifest command the
way TotalSegmentator's `totalseg_info --json` does), `evidence_status=EvidenceStatus.confirmed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_external_engines.py
from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries, lookup
from qortex.neuroai.models.zoo.schema import ZooEntryType
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "external.synthseg",
    "external.synthstrip",
    "external.hdbet",
    "external.fastsurfer",
    "external.tractseg",
}


def test_all_5_external_engine_entries_registered():
    registered_ids = {e.id for e in list_entries(entry_type=ZooEntryType.external_engine)}
    # external.totalsegmentator (Phase 1 seed) + these 5 = 6 external engines
    assert _EXPECTED_IDS.issubset(registered_ids)
    assert len(registered_ids) == 6


def test_external_engine_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []


def test_no_external_engine_entry_has_a_tensor_input_contract():
    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        assert entry.input_contract is None
        assert entry.external_engine_contract is not None


def test_fastsurfer_declares_directory_output():
    entry = lookup("external.fastsurfer")
    assert "directory" in entry.external_engine_contract.output_file_types


def test_command_builder_names_match_external_py_function_names():
    from qortex.neuroai import external as external_module

    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        builder_name = entry.external_engine_contract.command_builder
        assert hasattr(external_module, builder_name), (
            f"{entry_id}'s command_builder={builder_name!r} does not exist in external.py"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_external_engines.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `zoo/external_engines.py`**

```python
# src/qortex/neuroai/models/zoo/external_engines.py
"""External CLI neuroimaging engines (design spec section 12.2), following
the exact ExternalEngineContract pattern Phase 1 established for
external.totalsegmentator (zoo/seed_examples.py). These are file-based CLI
tools, not in-process tensor models -- entries never carry an
input_contract, only an external_engine_contract, per spec section 8.2.
"""

from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    LicenseInfo,
    ZooEntry,
    ZooEntryType,
)


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def _engine_entry(
    engine: str,
    display_name: str,
    executable: str,
    source_url: str,
    modality: str,
    supported_tasks: list[str],
    output_file_types: list[str] | None = None,
) -> ZooEntry:
    return ZooEntry(
        id=f"external.{engine}",
        display_name=display_name,
        entry_type=ZooEntryType.external_engine,
        provider="external_cli",
        execution_mode=ExecutionMode.external_cli,
        source_url=source_url,
        modality=[modality],
        task=supported_tasks,
        external_engine_contract=ExternalEngineContract(
            engine=engine,
            executable=executable,
            input_file_types=["nifti"],
            output_file_types=output_file_types or ["nifti"],
            supported_modalities=[modality],
            supported_tasks=supported_tasks,
            command_builder=f"_build_{engine}_command",
            output_manifest_supported=False,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_if_executable_available",
        priority="P0",
    )


def register_all() -> None:
    register(_engine_entry(
        "synthseg", "SynthSeg", "mri_synthseg",
        "https://github.com/BBillot/SynthSeg",
        "mri", ["whole_brain_segmentation"],
    ))
    register(_engine_entry(
        "synthstrip", "SynthStrip", "mri_synthstrip",
        "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/",
        "mri", ["skull_stripping"],
    ))
    register(_engine_entry(
        "hdbet", "HD-BET", "hd-bet",
        "https://github.com/MIC-DKFZ/HD-BET",
        "mri", ["skull_stripping"],
    ))
    register(_engine_entry(
        "fastsurfer", "FastSurfer", "run_fastsurfer.sh",
        "https://github.com/Deep-MI/FastSurfer",
        "mri", ["whole_brain_segmentation", "cortical_parcellation"],
        output_file_types=["nifti", "directory"],
    ))
    register(_engine_entry(
        "tractseg", "TractSeg", "TractSeg",
        "https://github.com/MIC-DKFZ/TractSeg",
        "dwi", ["white_matter_tract_segmentation"],
    ))


__all__ = ["register_all"]
```

- [ ] **Step 4: Wire into `zoo/__init__.py` and `tests/conftest.py`**

Read both files first (each already has 4 domain modules' `register_all()`
calls from Phases 1-3). Add
`from qortex.neuroai.models.zoo import external_engines as
_external_engines` and `_external_engines.register_all()` to both,
alongside (not replacing) the existing calls.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_external_engines.py -v`
Expected: PASS (5 tests)

Then the full zoo+extractor+external suite:
Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py -v`
Expected: all PASS

- [ ] **Step 6: Update the spec's progress checklist**

Check off all of Phase 4's items in §0 (SynthSeg/SynthStrip/HD-BET/
FastSurfer/TractSeg wrappers, "TotalSegmentator task discovery
integration" — note: task discovery via `totalseg_info --json` is NOT
implemented in this phase, since it needs its own capability-parsing code
beyond a bare command builder; leave that one box unchecked with a note
"deferred — needs a --json output parser, not just a command builder",
"External command provenance" — already satisfied by the existing
`run_external_segmentation`'s metadata-file writing, which applies
automatically to all 7 engines including these 5 new ones, so check this
one off). Append the 5 new entries to "Registry entries implemented so
far".

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/models/zoo/external_engines.py src/qortex/neuroai/models/zoo/__init__.py tests/conftest.py tests/test_neuroai_zoo_external_engines.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): register SynthSeg/SynthStrip/HD-BET/FastSurfer/TractSeg as zoo entries"
```

---

### Task 3: Wire the new engines into the `run-external-segmentation` CLI

**Files:**
- Modify: `src/qortex/cli/app.py` (extend the hardcoded engine allowlist
  and add the `--subject-id` option to `neuroai_run_external_segmentation`)
- Test: `tests/test_neuroai_external.py` (append one CLI-level test)

**Interfaces:**
- Consumes: `ExternalSegmentationRequest.subject_id` (Task 1),
  `available_external_segmentation_engines()` (Task 1).
- Produces: `qortex neuroai run-external-segmentation <engine> ...` accepts
  all 7 engine names and a new `--subject-id` option for FastSurfer.

- [ ] **Step 1: Write the failing test**

Read `tests/test_neuroai_external.py`'s existing
`test_cli_run_external_segmentation` test first (it already exercises the
CLI via `CliRunner`) to match its exact style, then append:

```python
def test_cli_rejects_unknown_engine_name():
    from typer.testing import CliRunner
    from qortex.cli.app import app

    result = CliRunner().invoke(
        app,
        ["neuroai", "run-external-segmentation", "not_a_real_engine", "x.nii.gz", "y.nii.gz"],
    )

    assert result.exit_code != 0


def test_cli_accepts_synthseg_engine_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from typer.testing import CliRunner
    from qortex.cli.app import app

    _write_executable(tmp_path / "mri_synthseg", "#!/usr/bin/env bash\nprintf 'seg' > \"${@: -1}\"\n")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    image = tmp_path / "t1.nii.gz"
    image.write_text("image", encoding="utf-8")
    output = tmp_path / "seg.nii.gz"

    result = CliRunner().invoke(
        app,
        ["neuroai", "run-external-segmentation", "synthseg", str(image), str(output)],
    )

    assert result.exit_code == 0, result.output
```

The `mri_synthseg` fake script above writes to the last argument, which is
`str(output_path)` per the `_build_synthseg_command` argument order
(`[executable, "--i", image, "--o", output]`) — matches the existing
`test_run_totalsegmentator_external_boundary` test's fake-script style of
writing to whichever argv position the real output path lands on.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_external.py -v -k "cli_rejects_unknown or cli_accepts_synthseg"`
Expected: FAIL — `test_cli_accepts_synthseg_engine_name` fails because the
CLI's hardcoded `engine not in {"totalsegmentator", "nnunet"}` check
rejects `"synthseg"`.

- [ ] **Step 3: Update the CLI**

In `src/qortex/cli/app.py`, find `neuroai_run_external_segmentation`
(currently around line 2027). Change the hardcoded set check (currently
line 2051):

```python
    if engine not in {
        "totalsegmentator", "nnunet", "synthseg", "synthstrip",
        "hdbet", "fastsurfer", "tractseg",
    }:
        typer.echo(
            "[ERROR] engine must be one of: totalsegmentator, nnunet, "
            "synthseg, synthstrip, hdbet, fastsurfer, tractseg",
            err=True,
        )
        raise typer.Exit(1)
```

Add a `subject_id` parameter to the function signature, alongside the
existing `model_folder`/`dataset_id` nnU-Net-specific options (around line
2032-2037):

```python
    subject_id: str | None = typer.Option(None, "--subject-id", help="FastSurfer subject id (required for engine=fastsurfer)"),
```

And pass it through in the `ExternalSegmentationRequest(...)` construction
(around line 2057-2071), adding `subject_id=subject_id,` alongside the
existing `folds=tuple(fold or ("all",)),` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_external.py -v`
Expected: PASS (all existing + 2 new)

Then confirm the CLI help text reflects the new engines:

Run: `qortex neuroai run-external-segmentation --help`
Expected: help text shows `--subject-id` in the options list.

- [ ] **Step 5: Run the full suite one final time**

Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/qortex/cli/app.py tests/test_neuroai_external.py
git commit -m "feat(cli): accept all 7 external engines in run-external-segmentation

Completes Phase 4 (External CLI engines) of the model zoo expansion."
```

---

## Phase 4 exit criteria

- [ ] `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py -v` — all green.
- [ ] `qortex neuroai zoo list --entry-type external_engine` shows 6 entries
      (5 new + TotalSegmentator from Phase 1).
- [ ] `qortex neuroai zoo validate` reports 0 errors.
- [ ] `qortex neuroai run-external-segmentation --help` lists all 7 engine
      names and the new `--subject-id` option.
- [ ] Spec §0 checklist: Phase 4 checked except "TotalSegmentator task
      discovery integration," which carries an explicit deferral note.
      "Registry entries implemented so far" lists all 5 new entries.
- [ ] No changes to `_base.py`, `_contracts.py`, `_registry.py`,
      `contracts.py`, `spec.py`, or any adapter file.

Once this phase is merged, write
`docs/superpowers/plans/<date>-model-zoo-phase5-promptable-segmentation.md`
covering `Prompt`, `InteractionContract` wiring into adapters,
`PromptableModelAdapter`, and the VISTA3D/MedSAM/SAM-Med3D adapters from
spec §9.2 and §12.4.
