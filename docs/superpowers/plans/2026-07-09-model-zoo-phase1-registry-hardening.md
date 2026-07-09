# Model Zoo Phase 1: Registry Hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `zoo/` package's contract-validated schema (`ZooEntry` and
its sub-contracts), a registry to hold entries, an offline validator, and a
`qortex neuroai zoo {list,show,validate}` CLI — seeded with the three
concrete example entries from the design spec so every layer has real data
to exercise from day one.

**Architecture:** New `src/qortex/neuroai/models/zoo/` subpackage. `schema.py`
defines the entry model; `registry.py` is a simple in-memory list with
register/list/lookup; `validate.py` runs the offline invariant checks from
the spec; three seed entries prove the whole path end to end. This phase
does **not** touch `_contracts.py`, `_registry.py`, or `suggest-models` — the
bridge to the legacy registry is Phase 2 (MONAI integration), per the design
spec §20.

**Tech Stack:** Python 3.10+, Pydantic (already an optional dependency —
`contracts.py` has a `_PYDANTIC` fallback pattern to match), pytest, Typer
(existing CLI framework in `cli/app.py`).

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  — every field and invariant in this plan traces to that document's §7, §8,
  §13, §14, §19.1.
- Do not duplicate or replace `_base.py`, `_contracts.py`, `_registry.py`,
  `contracts.py`, `spec.py` (spec §5). This phase only adds new files.
- Reuse `qortex.neuroai.contracts.EvidenceStatus` rather than defining a
  parallel enum — extend it with one new member (`contradicted`) rather than
  duplicating the concept. `unsupported` semantics are already covered by
  the existing `blocked` member.
- No network calls, no weight downloads, anywhere in this phase's code or
  tests.
- Follow existing pytest style: flat `tests/test_neuroai_*.py` files,
  `from qortex.neuroai import ...` public imports where a public surface
  exists.
- Update `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  §0 checklist in the same commit as the code that completes each item.

---

### Task 1: Evidence status extension + `zoo/schema.py` contracts

**Files:**
- Modify: `src/qortex/neuroai/contracts.py` (add `contradicted` to
  `EvidenceStatus`, around line 38-43)
- Create: `src/qortex/neuroai/models/zoo/__init__.py` (empty package marker
  for now — populated in Task 4)
- Create: `src/qortex/neuroai/models/zoo/schema.py`
- Test: `tests/test_neuroai_zoo_schema.py`

**Interfaces:**
- Consumes: `qortex.neuroai.contracts.EvidenceStatus`,
  `qortex.neuroai.contracts.InputContract`,
  `qortex.neuroai.contracts.OutputContract`.
- Produces (used by Tasks 2-5):
  - `ExecutionMode(str, Enum)`: `in_process`, `external_cli`, `remote_api`,
    `bundle`, `pipeline_app`
  - `ZooEntryType(str, Enum)`: `model`, `foundation_model`,
    `external_engine`, `generative_model`, `promptable_model`, `template`,
    `watchlist`
  - `PromptType(str, Enum)`: `point`, `box`, `text`, `mask`, `scribble`,
    `class_label`
  - `PromptCoordinateFrame(str, Enum)`: `image_2d`, `voxel_3d`, `world_mm`,
    `normalized`
  - `ProvenancedValue` — fields `value: Any`, `evidence_status:
    EvidenceStatus`, `source_url: str | None = None`, `source_field: str |
    None = None`, `checked_at: str | None = None`, `note: str | None = None`
  - `LicenseInfo` — fields `name: str | None = None`, `url: str | None =
    None`, `commercial_use: bool | None = None`, `redistribution_allowed:
    bool | None = None`, `requires_registration: bool = False`,
    `requires_citation: bool = False`, `evidence_status: EvidenceStatus =
    EvidenceStatus.unknown`, `notes: list[str] = []`
  - `SecurityPolicy` — fields `trust_remote_code_required: bool = False`,
    `allow_remote_code: bool = False`, `requires_sandbox: bool = False`,
    `allowed_imports: list[str] = []`, `blocked_imports: list[str] = []`,
    `executable_names: list[str] = []`, `network_required_at_runtime: bool =
    False`, `network_required_for_download: bool = False`
  - `InteractionContract` — fields `supported_prompt_types:
    list[PromptType]`, `prompt_coordinate_frame: PromptCoordinateFrame |
    None = None`, `max_points: int | None = None`, `max_boxes: int | None =
    None`, `supports_negative_points: bool = False`,
    `supports_multiclass_prompting: bool = False`, `supports_automatic_mode:
    bool = False`, `supports_iterative_refinement: bool = False`,
    `requires_label_set: bool = False`, `evidence_status: EvidenceStatus =
    EvidenceStatus.confirmed`
  - `ExternalEngineContract` — fields `engine: str`, `executable: str`,
    `input_file_types: list[str]`, `output_file_types: list[str]`,
    `supported_modalities: list[str]`, `supported_tasks: list[str]`,
    `command_builder: str`, `list_capabilities_command: list[str] | None =
    None`, `output_manifest_supported: bool = False`,
    `geometry_preservation_known: bool | None = None`, `license_required:
    bool = False`, `docker_supported: bool = False`, `evidence_status:
    EvidenceStatus`
  - `ZooEntry` — fields `id: str`, `display_name: str`, `entry_type:
    ZooEntryType`, `provider: str`, `execution_mode: ExecutionMode`,
    `source_url: str`, `paper_url: str | None = None`, `model_url: str |
    None = None`, `docs_url: str | None = None`, `maintainer: str | None =
    None`, `modality: list[str]`, `task: list[str]`, `input_contract:
    InputContract | None = None`, `output_contract: OutputContract | None =
    None`, `interaction_contract: InteractionContract | None = None`,
    `external_engine_contract: ExternalEngineContract | None = None`,
    `license: LicenseInfo`, `security: SecurityPolicy = SecurityPolicy()`,
    `evidence_status: EvidenceStatus`, `provenance:
    dict[str, ProvenancedValue] = {}`, `qortex_status: str`, `priority:
    str`, `notes: list[str] = []`

  Note: `preprocessing_contract` from the spec's §7 sketch is deferred —
  no task in this phase or Phase 2 populates it yet, and an unused required
  field would force fabricated data. Add it as `preprocessing_contract:
  Any | None = None` so the type exists for Phase 2 to fill in without a
  schema migration, but nothing sets it yet.

- [ ] **Step 1: Write the failing test for the EvidenceStatus extension**

```python
# tests/test_neuroai_zoo_schema.py
from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus


def test_evidence_status_has_contradicted_member():
    assert EvidenceStatus.contradicted == "contradicted"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_schema.py -v`
Expected: FAIL with `AttributeError: contradicted`

- [ ] **Step 3: Add the enum member**

In `src/qortex/neuroai/contracts.py`, change:

```python
class EvidenceStatus(str, Enum):
    confirmed  = "confirmed"
    inferred   = "inferred"
    missing    = "missing"
    unknown    = "unknown"
    blocked    = "blocked"
```

to:

```python
class EvidenceStatus(str, Enum):
    confirmed    = "confirmed"
    inferred     = "inferred"
    missing      = "missing"
    unknown      = "unknown"
    blocked      = "blocked"
    contradicted = "contradicted"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_neuroai_zoo_schema.py -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `ZooEntry` construction**

Append to `tests/test_neuroai_zoo_schema.py`:

```python
from qortex.neuroai.contracts import InputContract, AxisConvention
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    InteractionContract,
    LicenseInfo,
    PromptCoordinateFrame,
    PromptType,
    SecurityPolicy,
    ZooEntry,
    ZooEntryType,
)


def test_zoo_entry_minimal_model_construction():
    entry = ZooEntry(
        id="braindecode.EEGNet",
        display_name="EEGNet",
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://braindecode.org/stable/generated/braindecode.models.EEGNet.html",
        modality=["eeg"],
        task=["classification", "eeg_decoding", "bci"],
        input_contract=InputContract(
            modality="eeg",
            axis_convention=AxisConvention.batch_channels_time,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )
    assert entry.id == "braindecode.EEGNet"
    assert entry.entry_type is ZooEntryType.model
    assert entry.interaction_contract is None


def test_zoo_entry_promptable_construction():
    entry = ZooEntry(
        id="foundation.medsam",
        display_name="MedSAM",
        entry_type=ZooEntryType.promptable_model,
        provider="medsam",
        execution_mode=ExecutionMode.in_process,
        source_url="https://github.com/bowang-lab/MedSAM",
        modality=["ct", "mri"],
        task=["segmentation"],
        interaction_contract=InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            prompt_coordinate_frame=PromptCoordinateFrame.image_2d,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    )
    assert PromptType.text not in entry.interaction_contract.supported_prompt_types


def test_zoo_entry_external_engine_construction():
    entry = ZooEntry(
        id="external.totalsegmentator",
        display_name="TotalSegmentator",
        entry_type=ZooEntryType.external_engine,
        provider="external_cli",
        execution_mode=ExecutionMode.external_cli,
        source_url="https://github.com/wasserth/TotalSegmentator",
        modality=["ct", "mri"],
        task=["anatomical_segmentation"],
        external_engine_contract=ExternalEngineContract(
            engine="totalsegmentator",
            executable="TotalSegmentator",
            input_file_types=["nifti"],
            output_file_types=["nifti", "json"],
            supported_modalities=["ct", "mri"],
            supported_tasks=["total", "total_mr"],
            command_builder="_build_totalsegmentator_command",
            list_capabilities_command=["totalseg_info", "--json"],
            output_manifest_supported=True,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        security=SecurityPolicy(executable_names=["TotalSegmentator"]),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_if_executable_available",
        priority="P0",
    )
    assert entry.external_engine_contract.executable == "TotalSegmentator"
    assert entry.input_contract is None
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_neuroai_zoo_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qortex.neuroai.models.zoo'`

- [ ] **Step 7: Implement `zoo/schema.py`**

```python
# src/qortex/neuroai/models/zoo/schema.py
"""Contract-validated schema for the Qortex NeuroAI model zoo.

Every ``ZooEntry`` separates model identity, execution mode, license,
security risk, and scientific I/O contracts, following
docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md §7-8.

Prompt/interaction support is a separate ``InteractionContract`` on
``ZooEntry`` — never folded into ``InputContract`` — because a prompt is an
interaction constraint, not a biomedical input tensor.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from qortex.neuroai.contracts import (
    BaseModel,
    EvidenceStatus,
    Field,
    InputContract,
    OutputContract,
    _PYDANTIC,
)


class ExecutionMode(str, Enum):
    in_process = "in_process"
    external_cli = "external_cli"
    remote_api = "remote_api"
    bundle = "bundle"
    pipeline_app = "pipeline_app"


class ZooEntryType(str, Enum):
    model = "model"
    foundation_model = "foundation_model"
    external_engine = "external_engine"
    generative_model = "generative_model"
    promptable_model = "promptable_model"
    template = "template"
    watchlist = "watchlist"


class PromptType(str, Enum):
    point = "point"
    box = "box"
    text = "text"
    mask = "mask"
    scribble = "scribble"
    class_label = "class_label"


class PromptCoordinateFrame(str, Enum):
    image_2d = "image_2d"
    voxel_3d = "voxel_3d"
    world_mm = "world_mm"
    normalized = "normalized"


def _model(**kwargs):
    """Build the base class + default __init__/model_dump for the
    non-pydantic fallback, matching contracts.py's existing pattern."""
    return kwargs


class ProvenancedValue(BaseModel if _PYDANTIC else object):
    value: Any = None
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    source_url: str | None = None
    source_field: str | None = None
    checked_at: str | None = None
    note: str | None = None

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.evidence_status = EvidenceStatus.unknown
            for k, v in kwargs.items():
                setattr(self, k, v)


class LicenseInfo(BaseModel if _PYDANTIC else object):
    name: str | None = None
    url: str | None = None
    commercial_use: bool | None = None
    redistribution_allowed: bool | None = None
    requires_registration: bool = False
    requires_citation: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    notes: list[str] = Field(default_factory=list) if _PYDANTIC else []

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.requires_registration = False
            self.requires_citation = False
            self.evidence_status = EvidenceStatus.unknown
            self.notes = []
            for k, v in kwargs.items():
                setattr(self, k, v)


class SecurityPolicy(BaseModel if _PYDANTIC else object):
    trust_remote_code_required: bool = False
    allow_remote_code: bool = False
    requires_sandbox: bool = False
    allowed_imports: list[str] = Field(default_factory=list) if _PYDANTIC else []
    blocked_imports: list[str] = Field(default_factory=list) if _PYDANTIC else []
    executable_names: list[str] = Field(default_factory=list) if _PYDANTIC else []
    network_required_at_runtime: bool = False
    network_required_for_download: bool = False

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.trust_remote_code_required = False
            self.allow_remote_code = False
            self.requires_sandbox = False
            self.allowed_imports = []
            self.blocked_imports = []
            self.executable_names = []
            self.network_required_at_runtime = False
            self.network_required_for_download = False
            for k, v in kwargs.items():
                setattr(self, k, v)


class InteractionContract(BaseModel if _PYDANTIC else object):
    supported_prompt_types: list[PromptType]
    prompt_coordinate_frame: PromptCoordinateFrame | None = None
    max_points: int | None = None
    max_boxes: int | None = None
    supports_negative_points: bool = False
    supports_multiclass_prompting: bool = False
    supports_automatic_mode: bool = False
    supports_iterative_refinement: bool = False
    requires_label_set: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.confirmed

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.supported_prompt_types = []
            self.supports_negative_points = False
            self.supports_multiclass_prompting = False
            self.supports_automatic_mode = False
            self.supports_iterative_refinement = False
            self.requires_label_set = False
            self.evidence_status = EvidenceStatus.confirmed
            for k, v in kwargs.items():
                setattr(self, k, v)


class ExternalEngineContract(BaseModel if _PYDANTIC else object):
    engine: str
    executable: str
    input_file_types: list[str] = Field(default_factory=list) if _PYDANTIC else []
    output_file_types: list[str] = Field(default_factory=list) if _PYDANTIC else []
    supported_modalities: list[str] = Field(default_factory=list) if _PYDANTIC else []
    supported_tasks: list[str] = Field(default_factory=list) if _PYDANTIC else []
    command_builder: str = ""
    list_capabilities_command: list[str] | None = None
    output_manifest_supported: bool = False
    geometry_preservation_known: bool | None = None
    license_required: bool = False
    docker_supported: bool = False
    evidence_status: EvidenceStatus = EvidenceStatus.unknown

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.input_file_types = []
            self.output_file_types = []
            self.supported_modalities = []
            self.supported_tasks = []
            self.command_builder = ""
            self.output_manifest_supported = False
            self.license_required = False
            self.docker_supported = False
            self.evidence_status = EvidenceStatus.unknown
            for k, v in kwargs.items():
                setattr(self, k, v)


class ZooEntry(BaseModel if _PYDANTIC else object):
    id: str
    display_name: str
    entry_type: ZooEntryType
    provider: str
    execution_mode: ExecutionMode

    source_url: str
    paper_url: str | None = None
    model_url: str | None = None
    docs_url: str | None = None
    maintainer: str | None = None

    modality: list[str] = Field(default_factory=list) if _PYDANTIC else []
    task: list[str] = Field(default_factory=list) if _PYDANTIC else []

    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None
    # Populated starting Phase 2 (MONAI bundle extractor) — kept on the
    # schema now so later phases don't need a migration.
    preprocessing_contract: Any | None = None
    interaction_contract: InteractionContract | None = None
    external_engine_contract: ExternalEngineContract | None = None

    license: LicenseInfo
    security: SecurityPolicy = SecurityPolicy() if not _PYDANTIC else Field(default_factory=SecurityPolicy)

    evidence_status: EvidenceStatus
    provenance: dict[str, ProvenancedValue] = Field(default_factory=dict) if _PYDANTIC else {}

    qortex_status: str
    priority: str
    notes: list[str] = Field(default_factory=list) if _PYDANTIC else []

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.modality = []
            self.task = []
            self.security = SecurityPolicy()
            self.provenance = {}
            self.notes = []
            for k, v in kwargs.items():
                setattr(self, k, v)


__all__ = [
    "ExecutionMode",
    "ZooEntryType",
    "PromptType",
    "PromptCoordinateFrame",
    "ProvenancedValue",
    "LicenseInfo",
    "SecurityPolicy",
    "InteractionContract",
    "ExternalEngineContract",
    "ZooEntry",
]
```

Check `contracts.py` actually exports `BaseModel`, `Field`, `_PYDANTIC` as
importable names (they are module-level names, not prefixed private) — if
`Field` fallback (`from dataclasses import field as Field`) doesn't behave
identically to `pydantic.Field` for `default_factory=list`, mirror exactly
how `InputContract` in `contracts.py:273-314` uses
`Field(default_factory=list) if _PYDANTIC else []`, which this file already
copies. Delete the unused `_model()` helper from the draft above before
committing — it was scaffolding, not part of the design (ponytail: no dead
code).

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_schema.py -v`
Expected: PASS (4 tests)

- [ ] **Step 9: Commit**

```bash
git add src/qortex/neuroai/contracts.py src/qortex/neuroai/models/zoo/__init__.py src/qortex/neuroai/models/zoo/schema.py tests/test_neuroai_zoo_schema.py
git commit -m "feat(neuroai): add ZooEntry contract-validated schema

Adds ExecutionMode, ZooEntryType, PromptType, InteractionContract,
ExternalEngineContract, LicenseInfo, SecurityPolicy, and ZooEntry per
the model zoo design spec. Interaction/prompt support lives on ZooEntry,
not InputContract, per spec section 8.1."
```

---

### Task 2: `zoo/registry.py` — register / list / lookup

**Files:**
- Create: `src/qortex/neuroai/models/zoo/registry.py`
- Test: `tests/test_neuroai_zoo_registry.py`

**Interfaces:**
- Consumes: `ZooEntry`, `ZooEntryType` from
  `qortex.neuroai.models.zoo.schema` (Task 1).
- Produces (used by Task 3 validate, Task 4 seeds, Task 5 CLI):
  - `register(entry: ZooEntry) -> None` — appends to the module-level
    registry list; raises `ValueError` if `entry.id` is already registered.
  - `list_entries(*, entry_type: ZooEntryType | str | None = None, provider:
    str | None = None, modality: str | None = None, task: str | None =
    None, priority: str | None = None) -> list[ZooEntry]` — filtered view,
    always returns entries sorted by `id`.
  - `lookup(entry_id: str) -> ZooEntry | None`.
  - `clear_registry() -> None` — test-only helper to reset module state
    between tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_registry.py
from __future__ import annotations

import pytest

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import (
    clear_registry,
    lookup,
    list_entries,
    register,
)
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    LicenseInfo,
    ZooEntry,
    ZooEntryType,
)


def _entry(entry_id: str, entry_type=ZooEntryType.model, provider="braindecode") -> ZooEntry:
    return ZooEntry(
        id=entry_id,
        display_name=entry_id,
        entry_type=entry_type,
        provider=provider,
        execution_mode=ExecutionMode.in_process,
        source_url="https://example.org/model",
        modality=["eeg"],
        task=["classification"],
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_register_and_lookup():
    register(_entry("braindecode.EEGNet"))
    found = lookup("braindecode.EEGNet")
    assert found is not None
    assert found.id == "braindecode.EEGNet"


def test_lookup_missing_returns_none():
    assert lookup("nonexistent.model") is None


def test_register_duplicate_id_raises():
    register(_entry("braindecode.EEGNet"))
    with pytest.raises(ValueError):
        register(_entry("braindecode.EEGNet"))


def test_list_entries_filters_by_provider_and_sorts():
    register(_entry("braindecode.Deep4Net", provider="braindecode"))
    register(_entry("monai.vista3d", provider="monai_bundle"))
    register(_entry("braindecode.EEGNet", provider="braindecode"))

    bd = list_entries(provider="braindecode")
    assert [e.id for e in bd] == ["braindecode.Deep4Net", "braindecode.EEGNet"]


def test_list_entries_filters_by_entry_type():
    register(_entry("external.totalsegmentator", entry_type=ZooEntryType.external_engine, provider="external_cli"))
    register(_entry("braindecode.EEGNet"))

    engines = list_entries(entry_type=ZooEntryType.external_engine)
    assert [e.id for e in engines] == ["external.totalsegmentator"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named
'qortex.neuroai.models.zoo.registry'`

- [ ] **Step 3: Implement `zoo/registry.py`**

```python
# src/qortex/neuroai/models/zoo/registry.py
"""In-memory registry for ZooEntry records.

This is the Phase-1 zoo registry, deliberately separate from the legacy
qortex.neuroai.models._contracts registry (13 curated entries). Phase 2
adds the compatibility bridge that lets suggest-models read from both.
"""

from __future__ import annotations

from qortex.neuroai.models.zoo.schema import ZooEntry, ZooEntryType

_REGISTRY: dict[str, ZooEntry] = {}


def register(entry: ZooEntry) -> None:
    if entry.id in _REGISTRY:
        raise ValueError(f"ZooEntry id already registered: {entry.id!r}")
    _REGISTRY[entry.id] = entry


def lookup(entry_id: str) -> ZooEntry | None:
    return _REGISTRY.get(entry_id)


def list_entries(
    *,
    entry_type: ZooEntryType | str | None = None,
    provider: str | None = None,
    modality: str | None = None,
    task: str | None = None,
    priority: str | None = None,
) -> list[ZooEntry]:
    results = list(_REGISTRY.values())
    if entry_type is not None:
        want = entry_type.value if isinstance(entry_type, ZooEntryType) else str(entry_type)
        results = [e for e in results if e.entry_type.value == want]
    if provider is not None:
        results = [e for e in results if e.provider == provider]
    if modality is not None:
        results = [e for e in results if modality in e.modality]
    if task is not None:
        results = [e for e in results if task in e.task]
    if priority is not None:
        results = [e for e in results if e.priority == priority]
    return sorted(results, key=lambda e: e.id)


def clear_registry() -> None:
    """Test-only: reset registry state between test modules."""
    _REGISTRY.clear()


__all__ = ["register", "lookup", "list_entries", "clear_registry"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/zoo/registry.py tests/test_neuroai_zoo_registry.py
git commit -m "feat(neuroai): add ZooEntry registry with register/list/lookup"
```

---

### Task 3: `zoo/validate.py` — offline invariant checks

**Files:**
- Create: `src/qortex/neuroai/models/zoo/validate.py`
- Test: `tests/test_neuroai_zoo_validate.py`

**Interfaces:**
- Consumes: `list_entries()` from Task 2's `registry.py`; `ZooEntry`,
  `ZooEntryType` from Task 1's `schema.py`;
  `qortex.neuroai.models._registry.make_model_adapter` (existing) for
  provider-dispatch checking.
- Produces (used by Task 5 CLI):
  - `class ValidationIssue` — fields `entry_id: str`, `severity: str` (`"error"`
    or `"warning"`), `message: str`.
  - `validate_registry() -> list[ValidationIssue]` — runs every check below
    across all registered entries; empty list means fully valid.

Checks implemented (spec §19.1, restricted to what Phase 1's schema can
actually validate — promptable/external-specific checks only apply to
entries of that type):

1. Entry IDs unique — guaranteed by `register()` already, but re-checked
   here in case entries were constructed by mutating the dict directly in a
   future refactor (belt-and-suspenders, cheap to check).
2. `source_url` is a well-formed URL (`urllib.parse.urlparse` has a scheme
   and netloc).
3. `paper_url` / `model_url` / `docs_url`, if set, are well-formed URLs.
4. Every entry has `license` set with a non-`None` `evidence_status`.
5. Every entry has `evidence_status` set (not `None`).
6. Every `promptable_model` / `foundation_model` entry has a non-`None`
   `interaction_contract` with at least one `supported_prompt_types` entry.
7. Every `external_engine` entry has a non-`None` `external_engine_contract`.
8. Every entry's `provider` string resolves through
   `make_model_adapter`'s dispatch — except `external_cli` (external
   engines are dispatched through `neuroai/external.py`, not
   `make_model_adapter`, so `ValueError` for that one provider string is
   expected and skipped). A provider that raises `ImportError` (missing
   optional dependency) is fine — only `ValueError` (unknown provider) is a
   violation. `make_model_adapter` needs a `ModelSpec`; to check offline
   without importing the actual optional backend, catch `ImportError` and
   treat as pass.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_validate.py
from __future__ import annotations

import pytest

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.zoo.registry import clear_registry, register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    InteractionContract,
    LicenseInfo,
    PromptType,
    ZooEntry,
    ZooEntryType,
)
from qortex.neuroai.models.zoo.validate import validate_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def _base_kwargs(entry_id: str) -> dict:
    return dict(
        id=entry_id,
        display_name=entry_id,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://braindecode.org/stable/generated/braindecode.models.EEGNet.html",
        modality=["eeg"],
        task=["classification"],
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def test_valid_registry_has_no_issues():
    register(ZooEntry(entry_type=ZooEntryType.model, **_base_kwargs("braindecode.EEGNet")))
    assert validate_registry() == []


def test_malformed_source_url_is_an_error():
    kwargs = _base_kwargs("braindecode.Bad")
    kwargs["source_url"] = "not-a-url"
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))
    issues = validate_registry()
    assert any(i.entry_id == "braindecode.Bad" and i.severity == "error" for i in issues)


def test_promptable_entry_without_interaction_contract_is_an_error():
    register(ZooEntry(entry_type=ZooEntryType.promptable_model, **_base_kwargs("foundation.medsam")))
    issues = validate_registry()
    assert any(
        i.entry_id == "foundation.medsam" and "interaction_contract" in i.message
        for i in issues
    )


def test_promptable_entry_with_interaction_contract_passes():
    kwargs = _base_kwargs("foundation.medsam")
    kwargs["interaction_contract"] = InteractionContract(
        supported_prompt_types=[PromptType.point, PromptType.box]
    )
    register(ZooEntry(entry_type=ZooEntryType.promptable_model, **kwargs))
    assert validate_registry() == []


def test_external_engine_without_contract_is_an_error():
    kwargs = _base_kwargs("external.badengine")
    kwargs["provider"] = "external_cli"
    kwargs["execution_mode"] = ExecutionMode.external_cli
    register(ZooEntry(entry_type=ZooEntryType.external_engine, **kwargs))
    issues = validate_registry()
    assert any(
        i.entry_id == "external.badengine" and "external_engine_contract" in i.message
        for i in issues
    )


def test_unknown_provider_string_is_an_error():
    kwargs = _base_kwargs("bogus.model")
    kwargs["provider"] = "not_a_real_provider"
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))
    issues = validate_registry()
    assert any(i.entry_id == "bogus.model" and "provider" in i.message for i in issues)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_validate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `zoo/validate.py`**

```python
# src/qortex/neuroai/models/zoo/validate.py
"""Offline self-checks for the zoo registry — no network, no weights.

Run via ``qortex neuroai zoo validate`` or directly in CI to catch a
registry entry that fabricates or omits required contract data before it
ships. See docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 19.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from qortex.neuroai.models.zoo.registry import list_entries
from qortex.neuroai.models.zoo.schema import ZooEntry, ZooEntryType

# Providers that are not dispatched through make_model_adapter — they run
# through neuroai/external.py's own command-builder dispatch instead.
_EXTERNAL_ONLY_PROVIDERS = {"external_cli"}


@dataclass
class ValidationIssue:
    entry_id: str
    severity: str  # "error" | "warning"
    message: str


def _is_well_formed_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return bool(parsed.scheme) and bool(parsed.netloc)


def _check_urls(entry: ZooEntry) -> list[ValidationIssue]:
    issues = []
    for field_name in ("source_url", "paper_url", "model_url", "docs_url"):
        value = getattr(entry, field_name, None)
        if value and not _is_well_formed_url(value):
            issues.append(ValidationIssue(
                entry.id, "error", f"{field_name} is not a well-formed URL: {value!r}",
            ))
    return issues


def _check_license_and_evidence(entry: ZooEntry) -> list[ValidationIssue]:
    issues = []
    if entry.license is None:
        issues.append(ValidationIssue(entry.id, "error", "missing license info"))
    if entry.evidence_status is None:
        issues.append(ValidationIssue(entry.id, "error", "missing evidence_status"))
    return issues


def _check_interaction_contract(entry: ZooEntry) -> list[ValidationIssue]:
    if entry.entry_type not in (ZooEntryType.promptable_model, ZooEntryType.foundation_model):
        return []
    if entry.interaction_contract is None:
        return [ValidationIssue(
            entry.id, "error",
            "promptable/foundation entry missing interaction_contract",
        )]
    if not entry.interaction_contract.supported_prompt_types:
        return [ValidationIssue(
            entry.id, "error",
            "interaction_contract.supported_prompt_types is empty",
        )]
    return []


def _check_external_engine_contract(entry: ZooEntry) -> list[ValidationIssue]:
    if entry.entry_type != ZooEntryType.external_engine:
        return []
    if entry.external_engine_contract is None:
        return [ValidationIssue(
            entry.id, "error",
            "external_engine entry missing external_engine_contract",
        )]
    return []


def _check_provider_dispatch(entry: ZooEntry) -> list[ValidationIssue]:
    if entry.provider in _EXTERNAL_ONLY_PROVIDERS:
        return []
    from qortex.neuroai.models._registry import make_model_adapter
    from qortex.neuroai.spec import ModelSpec

    try:
        make_model_adapter(ModelSpec(provider=entry.provider, id=entry.id))
    except ImportError:
        return []  # optional dependency missing — not a registry defect
    except ValueError:
        return [ValidationIssue(
            entry.id, "error", f"provider {entry.provider!r} has no adapter dispatch",
        )]
    except Exception:
        # Constructing the adapter touched something else offline (e.g. a
        # missing local path) — that's a runtime concern, not a registry
        # validity concern.
        return []
    return []


def validate_registry() -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for entry in list_entries():
        issues.extend(_check_urls(entry))
        issues.extend(_check_license_and_evidence(entry))
        issues.extend(_check_interaction_contract(entry))
        issues.extend(_check_external_engine_contract(entry))
        issues.extend(_check_provider_dispatch(entry))
    return issues


__all__ = ["ValidationIssue", "validate_registry"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_validate.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/zoo/validate.py tests/test_neuroai_zoo_validate.py
git commit -m "feat(neuroai): add offline zoo registry validator"
```

---

### Task 4: Seed the three spec example entries

**Files:**
- Create: `src/qortex/neuroai/models/zoo/seed_examples.py`
- Modify: `src/qortex/neuroai/models/zoo/__init__.py` (import
  `seed_examples` so entries register on package import)
- Test: `tests/test_neuroai_zoo_seed_examples.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: importing `qortex.neuroai.models.zoo` registers exactly three
  entries: `monai.brats_mri_segmentation`, `braindecode.EEGNet`,
  `external.totalsegmentator` — transcribed verbatim from spec §13.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_seed_examples.py
from __future__ import annotations

import pytest

from qortex.neuroai.models.zoo.registry import clear_registry, lookup
from qortex.neuroai.models.zoo.validate import validate_registry


@pytest.fixture(autouse=True)
def _reset_registry():
    clear_registry()
    yield
    clear_registry()


def test_importing_zoo_package_registers_seed_examples():
    import importlib
    import qortex.neuroai.models.zoo as zoo_pkg
    importlib.reload(zoo_pkg)

    assert lookup("monai.brats_mri_segmentation") is not None
    assert lookup("braindecode.EEGNet") is not None
    assert lookup("external.totalsegmentator") is not None


def test_seed_examples_pass_offline_validation():
    import importlib
    import qortex.neuroai.models.zoo as zoo_pkg
    importlib.reload(zoo_pkg)

    issues = validate_registry()
    assert issues == [], f"seed examples must be fully valid, got: {issues}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_seed_examples.py -v`
Expected: FAIL — `lookup("monai.brats_mri_segmentation")` is `None`

- [ ] **Step 3: Implement `zoo/seed_examples.py`**

```python
# src/qortex/neuroai/models/zoo/seed_examples.py
"""The three worked examples from the design spec (section 13), registered
verbatim as real ZooEntry instances so every layer of Phase 1 has real data
to validate against. Domain-specific entries (MONAI imaging bundles,
Braindecode expansion, external engines) land in Phases 2-4.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, OutputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    ExternalEngineContract,
    LicenseInfo,
    SecurityPolicy,
    ZooEntry,
    ZooEntryType,
)


def _register_brats_mri_segmentation() -> None:
    register(ZooEntry(
        id="monai.brats_mri_segmentation",
        display_name="BraTS MRI Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url="https://huggingface.co/MONAI/brats_mri_segmentation",
        docs_url="https://project-monai.github.io/model-zoo.html",
        maintainer="Project MONAI",
        modality=["mri"],
        task=["segmentation", "brain_tumor_segmentation"],
        input_contract=InputContract(
            modality="mri",
            axis_convention=AxisConvention.channels_first,
            required_channels=["T1", "T1c", "T2", "FLAIR"],
            n_channels=4,
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="segmentation_mask",
            classes=["tumor_core", "whole_tumor", "enhancing_tumor"],
            produces_probabilities=False,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"]),
        security=SecurityPolicy(
            network_required_for_download=True,
            network_required_at_runtime=False,
        ),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))


def _register_braindecode_eegnet() -> None:
    register(ZooEntry(
        id="braindecode.EEGNet",
        display_name="EEGNet",
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url="https://braindecode.org/stable/generated/braindecode.models.EEGNet.html",
        modality=["eeg"],
        task=["classification", "eeg_decoding", "bci"],
        input_contract=InputContract(
            modality="eeg",
            axis_convention=AxisConvention.batch_channels_time,
            required_metadata=["n_chans", "n_times"],
            evidence_status=EvidenceStatus.inferred,
        ),
        output_contract=OutputContract(
            output_type="class_logits",
            produces_probabilities=False,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    ))


def _register_external_totalsegmentator() -> None:
    register(ZooEntry(
        id="external.totalsegmentator",
        display_name="TotalSegmentator",
        entry_type=ZooEntryType.external_engine,
        provider="external_cli",
        execution_mode=ExecutionMode.external_cli,
        source_url="https://github.com/wasserth/TotalSegmentator",
        modality=["ct", "mri"],
        task=["anatomical_segmentation"],
        external_engine_contract=ExternalEngineContract(
            engine="totalsegmentator",
            executable="TotalSegmentator",
            input_file_types=["nifti"],
            output_file_types=["nifti", "json"],
            supported_modalities=["ct", "mri"],
            supported_tasks=["total", "total_mr"],
            command_builder="_build_totalsegmentator_command",
            list_capabilities_command=["totalseg_info", "--json"],
            output_manifest_supported=True,
            evidence_status=EvidenceStatus.confirmed,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown),
        security=SecurityPolicy(executable_names=["TotalSegmentator"]),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_if_executable_available",
        priority="P0",
    ))


def register_all() -> None:
    _register_brats_mri_segmentation()
    _register_braindecode_eegnet()
    _register_external_totalsegmentator()


__all__ = ["register_all"]
```

Note: `provider="monai"` (not `"monai_bundle"`) for the BraTS entry — this
matches the existing `_registry.py` dispatch table (`"monai"` /
`"monai_bundle"` are both accepted aliases there already, confirmed in
`src/qortex/neuroai/models/_registry.py:38-40`). Use the alias that already
has adapter dispatch so `_check_provider_dispatch` passes without adding a
new provider string in this phase.

- [ ] **Step 4: Wire `register_all()` into package import**

```python
# src/qortex/neuroai/models/zoo/__init__.py
"""Qortex NeuroAI model zoo — contract-validated capability registry.

Importing this package registers all curated ZooEntry instances (seed
examples now; MONAI/Braindecode/vision/external-engine domain modules as
each phase lands — see docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md).
"""

from __future__ import annotations

from qortex.neuroai.models.zoo import seed_examples as _seed_examples

_seed_examples.register_all()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_seed_examples.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full zoo test suite to confirm no regressions**

Run: `python -m pytest tests/test_neuroai_zoo_schema.py tests/test_neuroai_zoo_registry.py tests/test_neuroai_zoo_validate.py tests/test_neuroai_zoo_seed_examples.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Update the spec's progress checklist**

In `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md` §0,
check off:
- `zoo/registry.py` (register/list/lookup)
- `zoo/validate.py` offline self-checks

Under "Registry entries implemented so far", replace `- (none yet)` with:

```markdown
- `monai.brats_mri_segmentation` — provider `monai`, entry_type `model` (Phase 1 seed)
- `braindecode.EEGNet` — provider `braindecode`, entry_type `model` (Phase 1 seed)
- `external.totalsegmentator` — provider `external_cli`, entry_type `external_engine` (Phase 1 seed)
```

- [ ] **Step 8: Commit**

```bash
git add src/qortex/neuroai/models/zoo/seed_examples.py src/qortex/neuroai/models/zoo/__init__.py tests/test_neuroai_zoo_seed_examples.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): seed the zoo registry with the three spec example entries"
```

---

### Task 5: CLI — `qortex neuroai zoo {list,show,validate}`

**Files:**
- Modify: `src/qortex/cli/app.py` (add `zoo_app` Typer sub-app after
  `neuroai_app` definition at line 1478, following the exact
  `app.add_typer(neuroai_app, name="neuroai")` pattern already used there
  and the `app.add_typer(check_app, name="check")` pattern at line 2299)
- Test: `tests/test_neuroai_zoo_cli.py`

**Interfaces:**
- Consumes: `list_entries`, `lookup` from `zoo/registry.py`;
  `validate_registry` from `zoo/validate.py`; importing
  `qortex.neuroai.models.zoo` triggers seed registration (Task 4).
- Produces: three Typer commands reachable as `qortex neuroai zoo list`,
  `qortex neuroai zoo show <id>`, `qortex neuroai zoo validate`. Uses
  `typer.testing.CliRunner` for tests, matching how Typer CLIs are
  conventionally tested (check `tests/` for an existing CLI test file to
  confirm the project's runner pattern before writing this test — if none
  exists, use `from typer.testing import CliRunner` and `from
  qortex.cli.app import app`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_cli.py
from __future__ import annotations

from typer.testing import CliRunner

from qortex.cli.app import app

runner = CliRunner()


def test_zoo_list_shows_seed_entries():
    result = runner.invoke(app, ["neuroai", "zoo", "list"])
    assert result.exit_code == 0
    assert "monai.brats_mri_segmentation" in result.stdout
    assert "braindecode.EEGNet" in result.stdout
    assert "external.totalsegmentator" in result.stdout


def test_zoo_list_filters_by_provider():
    result = runner.invoke(app, ["neuroai", "zoo", "list", "--provider", "braindecode"])
    assert result.exit_code == 0
    assert "braindecode.EEGNet" in result.stdout
    assert "monai.brats_mri_segmentation" not in result.stdout


def test_zoo_show_prints_entry_detail():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "braindecode.EEGNet"])
    assert result.exit_code == 0
    assert "EEGNet" in result.stdout
    assert "braindecode" in result.stdout


def test_zoo_show_unknown_id_exits_nonzero():
    result = runner.invoke(app, ["neuroai", "zoo", "show", "nonexistent.model"])
    assert result.exit_code != 0


def test_zoo_validate_passes_on_seed_registry():
    result = runner.invoke(app, ["neuroai", "zoo", "validate"])
    assert result.exit_code == 0
    assert "0 issue" in result.stdout.lower() or "no issues" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_cli.py -v`
Expected: FAIL — `zoo` is not a registered command under `neuroai`

- [ ] **Step 3: Implement the CLI commands**

In `src/qortex/cli/app.py`, immediately after line 1478
(`app.add_typer(neuroai_app, name="neuroai")`), insert:

```python
zoo_app = typer.Typer(
    name="zoo",
    help="Model zoo — contract-validated capability registry.",
    no_args_is_help=True,
)
neuroai_app.add_typer(zoo_app, name="zoo")


@zoo_app.command("list")
def neuroai_zoo_list(
    provider: str = typer.Option(None, "--provider", help="Filter by provider string"),
    modality: str = typer.Option(None, "--modality", help="Filter by modality"),
    task: str = typer.Option(None, "--task", help="Filter by task"),
    entry_type: str = typer.Option(None, "--entry-type", help="Filter by entry type"),
    priority: str = typer.Option(None, "--priority", help="Filter by priority (P0/P1/P2)"),
) -> None:
    """List model zoo entries, optionally filtered."""
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401  (triggers registration)
    from qortex.neuroai.models.zoo.registry import list_entries

    entries = list_entries(
        provider=provider, modality=modality, task=task,
        entry_type=entry_type, priority=priority,
    )
    if not entries:
        typer.echo("No matching entries.")
        return
    for entry in entries:
        typer.echo(
            f"{entry.id:<40} {entry.provider:<14} {entry.entry_type.value:<18} "
            f"{','.join(entry.modality):<12} {entry.priority}"
        )


@zoo_app.command("show")
def neuroai_zoo_show(entry_id: str = typer.Argument(..., help="Zoo entry id")) -> None:
    """Show full detail for one model zoo entry."""
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401
    from qortex.neuroai.models.zoo.registry import lookup

    entry = lookup(entry_id)
    if entry is None:
        typer.echo(f"Unknown zoo entry: {entry_id!r}", err=True)
        raise typer.Exit(1)

    typer.echo(f"id:              {entry.id}")
    typer.echo(f"display_name:    {entry.display_name}")
    typer.echo(f"entry_type:      {entry.entry_type.value}")
    typer.echo(f"provider:        {entry.provider}")
    typer.echo(f"execution_mode:  {entry.execution_mode.value}")
    typer.echo(f"source_url:      {entry.source_url}")
    typer.echo(f"modality:        {', '.join(entry.modality)}")
    typer.echo(f"task:            {', '.join(entry.task)}")
    typer.echo(f"evidence_status: {entry.evidence_status.value}")
    typer.echo(f"license:         {entry.license.evidence_status.value}")
    typer.echo(f"qortex_status:   {entry.qortex_status}")


@zoo_app.command("validate")
def neuroai_zoo_validate() -> None:
    """Run offline self-checks across the whole zoo registry. No network."""
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401
    from qortex.neuroai.models.zoo.validate import validate_registry

    issues = validate_registry()
    if not issues:
        typer.echo("0 issues — registry is valid.")
        return
    for issue in issues:
        typer.echo(f"[{issue.severity.upper()}] {issue.entry_id}: {issue.message}")
    if any(i.severity == "error" for i in issues):
        raise typer.Exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_cli.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Manual smoke check**

Run: `python -m qortex.cli.app neuroai zoo list` (or `qortex neuroai zoo
list` if the package is installed in editable mode)
Expected: prints the three seed entries in a table.

Run: `qortex neuroai zoo validate`
Expected: `0 issues — registry is valid.`

- [ ] **Step 6: Update the spec's progress checklist**

In `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md` §0,
check off:
- `ZooEntry` schema (`zoo/schema.py`)
- `LicenseInfo`
- `SecurityPolicy`
- `InteractionContract`
- `ExternalEngineContract`
- CLI: `zoo list`
- CLI: `zoo show`
- CLI: `zoo validate`

Every Phase 1 checklist item is now checked.

- [ ] **Step 7: Commit**

```bash
git add src/qortex/cli/app.py tests/test_neuroai_zoo_cli.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(cli): add qortex neuroai zoo list/show/validate commands

Completes Phase 1 (Registry hardening) of the model zoo expansion."
```

---

## Phase 1 exit criteria

- [ ] `python -m pytest tests/test_neuroai_zoo_*.py -v` — all green.
- [ ] `qortex neuroai zoo list` shows 3 entries.
- [ ] `qortex neuroai zoo validate` reports 0 issues.
- [ ] `qortex neuroai zoo show braindecode.EEGNet` prints full detail.
- [ ] Spec §0 checklist fully checked for Phase 1, "Registry entries
      implemented so far" lists the 3 seeds.
- [ ] No changes to `_contracts.py`, `_registry.py`, `suggest-models`, or
      any existing adapter file (Phase 1 is purely additive).

Once this phase is merged, write
`docs/superpowers/plans/<date>-model-zoo-phase2-monai-integration.md`
covering the MONAI bundle extractor and the P0 MONAI imaging entries from
spec §12.1 — do not pre-plan it now; the extractor's exact shape depends on
what Phase 1's `ZooEntry.preprocessing_contract` placeholder needs to become,
which is easier to decide with Phase 1 actually merged and real MONAI bundle
metadata in hand.
