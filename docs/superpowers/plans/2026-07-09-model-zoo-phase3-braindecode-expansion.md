# Model Zoo Phase 3: Braindecode Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline Braindecode model-config extractor, register the 11
remaining P0 EEG entries from spec §12.3 (only `braindecode.EEGNet` exists
so far, seeded in Phase 1), and add an EEG-specific offline validation
check that catches an entry claiming `confirmed` evidence without the
minimum EEG parameters that would actually justify it.

**Architecture:** `extractors/braindecode_model.py` mirrors Phase 2's
`extractors/monai_bundle.py` — a pure function turning an already-loaded HF
`config.json`-shaped dict (`n_chans`, `n_times`, `sfreq`, `n_outputs`,
`id2label`) into contract fields, no network. `zoo/braindecode_eeg.py`
follows the exact `zoo/monai_imaging.py` pattern. All 11 new entries are
registered as architecture-only (no fabricated pretrained-checkpoint IDs —
see Task 2's rationale for why "pretrained" entries are deferred, not
silently dropped).

**Tech Stack:** Python 3.10+, Pydantic (optional, `_PYDANTIC` fallback),
pytest, Typer.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  — this phase implements §11.2 and §12.3, and the Phase 3 line of §20.
- **No guessed contracts.** None of the 11 new models have a channel count,
  sample count, or sampling rate confirmed by the design spec's own text —
  unlike Phase 2's MONAI entries, spec §12.3 gives no numeric facts for any
  of these 11 models, only pretraining-scale facts for LaBraM (2,500 hours,
  ~20 datasets) and REVE (60,000+ hours, 92 datasets, 25,000 subjects).
  Those go in `notes`, never into `n_channels`/`sampling_rate_hz`/etc.
- **No fabricated pretrained checkpoints.** Braindecode's own docs state
  several of these models have HF Hub pretrained weights, but the exact HF
  repo IDs are not confirmable offline in this environment (no network
  access in this phase's implementation or tests). Register all 11 as
  architecture-only entries (`qortex_status="architecture_available"`,
  matching Phase 1's `braindecode.EEGNet` precedent exactly). This is a
  documented, deliberate deferral — see Task 2's file docstring — not a
  silently dropped requirement.
- Do not modify `_base.py`, `_contracts.py`, `_registry.py`, `contracts.py`,
  `spec.py`, `braindecode.py` (the existing adapter), or any zoo file from
  Phases 1-2 except the additive `zoo/__init__.py` and `tests/conftest.py`
  edits Task 2 requires.
- No network calls anywhere in this phase's code or tests.
- Follow existing pytest style: flat `tests/test_neuroai_*.py` files.
- Update `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  §0 checklist (Phase 3 section + "Registry entries implemented so far") in
  the same commit as the code that completes each item.

---

### Task 1: Braindecode model config extractor (offline)

**Files:**
- Create: `src/qortex/neuroai/models/extractors/braindecode_model.py`
- Test: `tests/test_neuroai_extractors_braindecode_model.py`

**Interfaces:**
- Consumes: `qortex.neuroai.contracts.{InputContract, OutputContract,
  AxisConvention, EvidenceStatus}`.
- Produces (used by Task 2, optionally — Task 2's entries do not require
  this extractor since no HF config.json is available offline for any of
  the 11 new models, but the extractor itself must exist, be tested, and be
  ready for the adapter or a future live-fetch tool to call):
  - `@dataclass ExtractedBraindecodeContract` — fields `model_id: str`,
    `input_contract: InputContract | None`, `output_contract: OutputContract
    | None`.
  - `extract_braindecode_contract(model_id: str, config: dict) ->
    ExtractedBraindecodeContract` — pure function, no I/O. Reads
    Braindecode/HF Hub's documented config keys: `n_chans` (or the older
    alias `n_channels`), `n_times` (or `input_window_seconds` combined with
    `sfreq` to derive it), `sfreq`, `n_outputs`, `id2label`. This mirrors
    what `src/qortex/neuroai/models/braindecode.py`'s existing adapter
    already does inline when it reads `config.json` from HF Hub — this
    extractor is a standalone, testable version of that same logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_extractors_braindecode_model.py
from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.extractors.braindecode_model import (
    ExtractedBraindecodeContract,
    extract_braindecode_contract,
)


def test_extract_full_config_populates_contracts():
    config = {
        "n_chans": 22,
        "n_times": 1000,
        "sfreq": 250.0,
        "n_outputs": 4,
        "id2label": {"0": "left_hand", "1": "right_hand", "2": "feet", "3": "tongue"},
    }

    extracted = extract_braindecode_contract("test/model", config)

    assert extracted.model_id == "test/model"
    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels == 22
    assert extracted.input_contract.sampling_rate_hz == 250.0
    assert extracted.input_contract.window_duration_s == 4.0
    assert extracted.input_contract.evidence_status == EvidenceStatus.confirmed
    assert extracted.output_contract is not None
    assert extracted.output_contract.n_classes == 4
    assert extracted.output_contract.classes == ["left_hand", "right_hand", "feet", "tongue"]


def test_extract_accepts_legacy_n_channels_alias():
    config = {"n_channels": 64, "n_times": 500, "sfreq": 125.0}

    extracted = extract_braindecode_contract("legacy/model", config)

    assert extracted.input_contract.n_channels == 64


def test_extract_empty_config_returns_none_contracts():
    extracted = extract_braindecode_contract("bare/model", {})

    assert extracted.input_contract is None
    assert extracted.output_contract is None


def test_extract_partial_config_does_not_guess_missing_fields():
    config = {"n_chans": 22}

    extracted = extract_braindecode_contract("partial/model", config)

    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels == 22
    assert extracted.input_contract.sampling_rate_hz is None
    assert extracted.input_contract.window_duration_s is None
    assert extracted.input_contract.evidence_status == EvidenceStatus.inferred
    assert extracted.output_contract is None


def test_extract_without_id2label_still_populates_n_classes():
    config = {"n_chans": 22, "n_times": 1000, "sfreq": 250.0, "n_outputs": 4}

    extracted = extract_braindecode_contract("no_labels/model", config)

    assert extracted.output_contract.n_classes == 4
    assert extracted.output_contract.classes == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_extractors_braindecode_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `extractors/braindecode_model.py`**

```python
# src/qortex/neuroai/models/extractors/braindecode_model.py
"""Offline Braindecode model config extractor.

Turns an already-loaded HF Hub config.json-shaped dict into Qortex
contract fields. Pure function — no network access, no HF Hub download.
Missing fields are left unknown, never guessed, per
docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md section
11.2.

Mirrors the inline config-reading logic already present in
src/qortex/neuroai/models/braindecode.py's BrainDecodeAdapter — this is a
standalone, independently-testable version of that same extraction.
"""

from __future__ import annotations

from dataclasses import dataclass

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    OutputContract,
)


@dataclass
class ExtractedBraindecodeContract:
    model_id: str
    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None


def _extract_input_contract(config: dict) -> InputContract | None:
    n_channels = config.get("n_chans", config.get("n_channels"))
    n_times = config.get("n_times")
    sfreq = config.get("sfreq")

    if n_channels is None and n_times is None and sfreq is None:
        return None

    window_duration_s = None
    if n_times is not None and sfreq:
        window_duration_s = n_times / sfreq

    confirmed = n_channels is not None and sfreq is not None and n_times is not None
    return InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        n_channels=n_channels,
        sampling_rate_hz=sfreq,
        window_duration_s=window_duration_s,
        evidence_status=EvidenceStatus.confirmed if confirmed else EvidenceStatus.inferred,
    )


def _extract_output_contract(config: dict) -> OutputContract | None:
    n_outputs = config.get("n_outputs")
    if n_outputs is None:
        return None
    id2label = config.get("id2label") or {}
    classes = [id2label[k] for k in sorted(id2label, key=lambda x: int(x))] if id2label else []
    return OutputContract(
        output_type="classification",
        n_classes=n_outputs,
        classes=classes,
        produces_probabilities=False,
    )


def extract_braindecode_contract(model_id: str, config: dict) -> ExtractedBraindecodeContract:
    return ExtractedBraindecodeContract(
        model_id=model_id,
        input_contract=_extract_input_contract(config),
        output_contract=_extract_output_contract(config),
    )


__all__ = ["ExtractedBraindecodeContract", "extract_braindecode_contract"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_extractors_braindecode_model.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/extractors/braindecode_model.py tests/test_neuroai_extractors_braindecode_model.py
git commit -m "feat(neuroai): add offline Braindecode model config extractor"
```

---

### Task 2: Register the 11 remaining P0 Braindecode entries

**Files:**
- Create: `src/qortex/neuroai/models/zoo/braindecode_eeg.py`
- Modify: `src/qortex/neuroai/models/zoo/__init__.py` (add
  `braindecode_eeg.register_all()` alongside existing calls)
- Modify: `tests/conftest.py` (add the same call to the shared autouse
  re-seed fixture — read it first, it already calls 3 other domain
  modules' `register_all()`)
- Test: `tests/test_neuroai_zoo_braindecode_eeg.py`

**Interfaces:**
- Consumes: `ZooEntry`, `ZooEntryType`, `ExecutionMode`, `LicenseInfo` from
  `zoo/schema.py`; `register` from `zoo/registry.py`; `InputContract`,
  `AxisConvention`, `EvidenceStatus` from `qortex.neuroai.contracts`.
- Produces: importing `qortex.neuroai.models.zoo` registers 11 additional
  `braindecode`-provider entries.

Register these 11 (spec §12.3, minus `braindecode.EEGNet` already seeded in
Phase 1): `Deep4Net`, `ShallowFBCSPNet`, `EEGConformer`, `BENDR`, `BIOT`,
`Labram`, `REVE`, `USleep`, `AttnSleep`, `DeepSleepNet`, `SignalJEPA`. All
`provider="braindecode"`, `execution_mode=ExecutionMode.in_process`,
`entry_type=ZooEntryType.model`, `source_url` is each model's Braindecode
API doc page (`https://braindecode.org/stable/generated/braindecode.models.<ClassName>.html`),
`modality=["eeg"]`, `task=["classification", "eeg_decoding"]` except
`USleep`/`AttnSleep`/`DeepSleepNet` get `task=["classification",
"sleep_staging"]` (they are documented sleep-staging architectures, not
BCI/motor-imagery ones — this distinction is itself confirmed by each
model's well-known purpose, not a numeric field, so it's safe to state).
`input_contract=InputContract(modality="eeg",
axis_convention=AxisConvention.batch_channels_time,
evidence_status=EvidenceStatus.unknown)` (no channel/rate/window fields —
none confirmed by spec text), no `output_contract` (classification head
size varies by dataset/task and is not a property of the architecture
itself — leave `None`, consistent with how an untrained architecture has no
fixed output shape). `license=LicenseInfo(evidence_status=EvidenceStatus.unknown,
notes=["requires manual check"])`, `evidence_status=EvidenceStatus.confirmed`
(the entry's own modality/task metadata is confirmed, even though the
tensor contract fields are not), `priority="P0"`,
`qortex_status="architecture_available"`. `LaBraM` and `REVE` additionally
get a `paper_url` and a `notes` entry citing their pretraining scale
(exact figures from spec §12.3): LaBraM → `arXiv:2405.18765`, "~2,500 hours
of EEG from ~20 datasets"; REVE → `arXiv:2510.21585`, "60,000+ hours of EEG
from 92 datasets and 25,000 subjects".

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_braindecode_eeg.py
from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries, lookup
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "braindecode.Deep4Net",
    "braindecode.ShallowFBCSPNet",
    "braindecode.EEGConformer",
    "braindecode.BENDR",
    "braindecode.BIOT",
    "braindecode.Labram",
    "braindecode.REVE",
    "braindecode.USleep",
    "braindecode.AttnSleep",
    "braindecode.DeepSleepNet",
    "braindecode.SignalJEPA",
}


def test_all_11_braindecode_entries_registered():
    registered_ids = {e.id for e in list_entries(provider="braindecode")}
    # braindecode.EEGNet (Phase 1 seed) + these 11 = 12 braindecode entries
    assert _EXPECTED_IDS.issubset(registered_ids)
    assert len(registered_ids) == 12


def test_braindecode_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []


def test_sleep_staging_models_get_sleep_task():
    for model_id in ("braindecode.USleep", "braindecode.AttnSleep", "braindecode.DeepSleepNet"):
        entry = lookup(model_id)
        assert "sleep_staging" in entry.task


def test_bci_models_do_not_get_sleep_task():
    entry = lookup("braindecode.Deep4Net")
    assert "sleep_staging" not in entry.task


def test_no_entry_has_fabricated_channel_count():
    for model_id in _EXPECTED_IDS:
        entry = lookup(model_id)
        assert entry.input_contract.n_channels is None
        assert entry.input_contract.evidence_status.value == "unknown"


def test_labram_and_reve_cite_pretraining_scale_in_notes_not_fields():
    labram = lookup("braindecode.Labram")
    reve = lookup("braindecode.REVE")
    assert any("2,500 hours" in n for n in labram.notes)
    assert any("60,000" in n for n in reve.notes)
    # Pretraining scale is a fact about training data, not the model's
    # tensor contract — must never appear as a fabricated n_channels/etc.
    assert labram.input_contract.n_channels is None
    assert reve.input_contract.n_channels is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_braindecode_eeg.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `zoo/braindecode_eeg.py`**

```python
# src/qortex/neuroai/models/zoo/braindecode_eeg.py
"""P0 Braindecode EEG entries (design spec section 12.3), excluding
braindecode.EEGNet which was seeded in Phase 1 (zoo/seed_examples.py).

All 11 entries here are registered architecture-only
(qortex_status="architecture_available") — Braindecode's own docs state
several of these have HF Hub pretrained checkpoints, but the exact HF repo
IDs are not confirmable offline in this environment. Registering a
"pretrained" variant with a guessed repo id would violate the "no
fabricated contracts" invariant, so pretrained entries are deferred until
a real, confirmed checkpoint id is available — not silently dropped.

No entry here carries a fabricated n_channels/sampling_rate_hz/n_classes:
the design spec's own text gives no numeric facts for any of these 11
models (unlike Phase 2's MONAI entries, where a few had spec-confirmed
counts). Only LaBraM and REVE get a pretraining-scale fact, and it's
recorded in notes, never coerced into a tensor-contract field.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, ZooEntry, ZooEntryType

_DOCS_BASE = "https://braindecode.org/stable/generated/braindecode.models."


def _doc_url(class_name: str) -> str:
    return f"{_DOCS_BASE}{class_name}.html"


def _unknown_eeg_input() -> InputContract:
    return InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        evidence_status=EvidenceStatus.unknown,
    )


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def _bci_entry(class_name: str, display_name: str, extra_notes: list[str] | None = None, paper_url: str | None = None) -> ZooEntry:
    return ZooEntry(
        id=f"braindecode.{class_name}",
        display_name=display_name,
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url=_doc_url(class_name),
        paper_url=paper_url,
        modality=["eeg"],
        task=["classification", "eeg_decoding"],
        input_contract=_unknown_eeg_input(),
        output_contract=None,
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
        notes=extra_notes or [],
    )


def _sleep_entry(class_name: str, display_name: str) -> ZooEntry:
    return ZooEntry(
        id=f"braindecode.{class_name}",
        display_name=display_name,
        entry_type=ZooEntryType.model,
        provider="braindecode",
        execution_mode=ExecutionMode.in_process,
        source_url=_doc_url(class_name),
        modality=["eeg"],
        task=["classification", "sleep_staging"],
        input_contract=_unknown_eeg_input(),
        output_contract=None,
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="architecture_available",
        priority="P0",
    )


def register_all() -> None:
    register(_bci_entry("Deep4Net", "Deep4Net"))
    register(_bci_entry("ShallowFBCSPNet", "ShallowFBCSPNet"))
    register(_bci_entry("EEGConformer", "EEGConformer"))
    register(_bci_entry("BENDR", "BENDR"))
    register(_bci_entry("BIOT", "BIOT"))
    register(_bci_entry(
        "Labram", "LaBraM",
        paper_url="https://arxiv.org/abs/2405.18765",
        extra_notes=[
            "LaBraM (arXiv:2405.18765) reports pretraining on approximately "
            "2,500 hours of EEG from around 20 datasets. Pretraining scale "
            "only -- not a tensor contract fact.",
        ],
    ))
    register(_bci_entry(
        "REVE", "REVE",
        paper_url="https://arxiv.org/abs/2510.21585",
        extra_notes=[
            "REVE (arXiv:2510.21585) reports pretraining on over 60,000 "
            "hours of EEG from 92 datasets and 25,000 subjects. Pretraining "
            "scale only -- not a tensor contract fact.",
        ],
    ))
    register(_sleep_entry("USleep", "USleep"))
    register(_sleep_entry("AttnSleep", "AttnSleep"))
    register(_sleep_entry("DeepSleepNet", "DeepSleepNet"))
    register(_bci_entry("SignalJEPA", "SignalJEPA"))


__all__ = ["register_all"]
```

- [ ] **Step 4: Wire into `zoo/__init__.py` and `tests/conftest.py`**

Read both files first — each already has real content from Phases 1-2 (3
domain modules' `register_all()` calls in each). Add
`from qortex.neuroai.models.zoo import braindecode_eeg as
_braindecode_eeg` and `_braindecode_eeg.register_all()` to both, alongside
(not replacing) the existing calls.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_braindecode_eeg.py -v`
Expected: PASS (6 tests)

Then the full zoo+extractor suite:
Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py -v`
Expected: all PASS

- [ ] **Step 6: Update the spec's progress checklist**

Check off "Expanded Braindecode entries (§12.3 — list grows below)" under
Phase 3 in §0. Append the 11 new entries to "Registry entries implemented
so far".

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/models/zoo/braindecode_eeg.py src/qortex/neuroai/models/zoo/__init__.py tests/conftest.py tests/test_neuroai_zoo_braindecode_eeg.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): register the 11 remaining P0 Braindecode entries"
```

---

### Task 3: EEG contract consistency check in the validator

**Files:**
- Modify: `src/qortex/neuroai/models/zoo/validate.py` (add one new check
  function, wire it into `validate_registry()`)
- Test: `tests/test_neuroai_zoo_validate.py` (add new test cases)

**Interfaces:**
- Consumes: `ZooEntry` from `zoo/schema.py` (already imported in
  `validate.py`).
- Produces: `validate_registry()` now also flags any `eeg`-modality entry
  that claims `entry.evidence_status == EvidenceStatus.confirmed` while its
  `input_contract.evidence_status` is `unknown` AND neither `n_channels`
  nor `sampling_rate_hz` is set — i.e. an entry cannot claim overall
  "confirmed" status while its EEG tensor shape is entirely unconfirmed
  without at least flagging that inconsistency for review. This is
  deliberately a **warning**, not an error (Phase 3's own 11 new entries
  are examples of a legitimate, honest use of this pattern — the entry's
  *metadata* is confirmed even though its *tensor contract* is not — so
  this check surfaces the distinction for a human to review, it does not
  block the registry).

- [ ] **Step 1: Write the failing test**

Read `tests/test_neuroai_zoo_validate.py` first — it already has 6 tests
from Phase 1 using a `_base_kwargs` helper and `ZooEntry`/`ExecutionMode`
imports. Append these new tests, reusing that existing helper:

```python
def test_eeg_entry_with_unknown_shape_and_confirmed_status_gets_a_warning():
    kwargs = _base_kwargs("braindecode.Unconfirmed")
    kwargs["input_contract"] = InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        evidence_status=EvidenceStatus.unknown,
    )
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))

    issues = validate_registry()

    matches = [i for i in issues if i.entry_id == "braindecode.Unconfirmed" and i.severity == "warning"]
    assert len(matches) == 1
    assert "evidence_status=confirmed" in matches[0].message


def test_eeg_entry_with_confirmed_channels_and_rate_gets_no_warning():
    kwargs = _base_kwargs("braindecode.Confirmed")
    kwargs["input_contract"] = InputContract(
        modality="eeg",
        axis_convention=AxisConvention.batch_channels_time,
        n_channels=64,
        sampling_rate_hz=250.0,
        evidence_status=EvidenceStatus.confirmed,
    )
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))

    issues = validate_registry()

    assert [i for i in issues if i.entry_id == "braindecode.Confirmed"] == []


def test_non_eeg_entry_is_not_subject_to_the_eeg_check():
    kwargs = _base_kwargs("monai.SomeImaging")
    kwargs["input_contract"] = InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        evidence_status=EvidenceStatus.unknown,
    )
    register(ZooEntry(entry_type=ZooEntryType.model, **kwargs))

    issues = validate_registry()

    assert [i for i in issues if i.entry_id == "monai.SomeImaging"] == []
```

You will also need to add `InputContract`, `AxisConvention` to this test
file's existing imports from `qortex.neuroai.contracts` if not already
present — check the top of the file first.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_validate.py -v`
Expected: FAIL — the two new "gets a warning" assertions fail (no warning
produced yet); the "no warning" and "not subject to" tests will pass
vacuously since no check exists yet, so specifically confirm the RED state
is `test_eeg_entry_with_unknown_shape_and_confirmed_status_gets_a_warning`
failing on `assert len(matches) == 1` (actual: 0).

- [ ] **Step 3: Implement the check**

Read the current `src/qortex/neuroai/models/zoo/validate.py` first (it has
5 existing `_check_*` functions and `validate_registry()` calling all of
them). Add this new function following the same signature pattern
(`entry: ZooEntry -> list[ValidationIssue]`):

```python
def _check_eeg_contract_consistency(entry: ZooEntry) -> list[ValidationIssue]:
    ic = entry.input_contract
    if ic is None or str(ic.modality).lower() != "eeg":
        return []
    if entry.evidence_status != EvidenceStatus.confirmed:
        return []
    has_confirmed_shape = ic.n_channels is not None or ic.sampling_rate_hz is not None
    if has_confirmed_shape:
        return []
    return [ValidationIssue(
        entry.id, "warning",
        "entry.evidence_status=confirmed but input_contract has no "
        "confirmed n_channels or sampling_rate_hz -- confirm this is "
        "intentional (metadata confirmed, tensor shape architecture-only)",
    )]
```

Add `_check_eeg_contract_consistency(entry)` to the `validate_registry()`
loop's list of `issues.extend(...)` calls, alongside the 5 existing checks.
`EvidenceStatus` is NOT currently imported in this file (verified: only
`dataclasses.dataclass`, `urllib.parse.urlparse`, and the two zoo-local
imports are present) — add
`from qortex.neuroai.contracts import EvidenceStatus` to the file's import
block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_validate.py -v`
Expected: PASS (9 tests: 6 existing + 3 new)

Then run the full zoo suite — Task 2's 11 new Braindecode entries will now
each produce exactly one **warning** (not error) from this new check, since
they intentionally have `evidence_status=confirmed` with unconfirmed shape.
Confirm `zoo validate`'s exit code stays 0 for warnings (only `severity ==
"error"` triggers `typer.Exit(1)`, per the existing Phase 1 CLI
implementation) — this is expected and correct, not a regression:

Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py -v`
Expected: all PASS

Run: `qortex neuroai zoo validate`
Expected: exit code 0, but the output now lists 11 `[WARNING]` lines (one
per Task 2 entry) alongside `0 issues` no longer being literally true — the
CLI's `validate_registry()` caller should still report success since no
`error`-severity issue exists. Manually confirm the CLI's message text
still makes sense with warnings present (it may currently only special-case
the fully-empty case) — if the existing `neuroai_zoo_validate` CLI command
prints `"0 issues — registry is valid."` only when `not issues`, it will
now instead fall through to printing each issue line-by-line including the
11 warnings, followed by no `typer.Exit(1)` since none are errors. This is
correct behavior, not a bug — just confirm it by reading
`src/qortex/cli/app.py`'s `neuroai_zoo_validate` function (added in Phase
1) rather than assuming.

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/zoo/validate.py tests/test_neuroai_zoo_validate.py
git commit -m "feat(neuroai): add EEG contract consistency check to zoo validator

Completes Phase 3 (Braindecode expansion) of the model zoo expansion."
```

- [ ] **Step 6: Update the spec's progress checklist**

Check off "EEG shape/channel/sampling contract validation" under Phase 3.
Note: "Braindecode extractor" and "Architecture vs. pretrained separation"
are also satisfied by Tasks 1-2 (architecture-only registration IS the
separation — pretrained variants are the deferred half) — check those off
too. "HF pretrained registry support" stays unchecked with a one-line note
in §0: "Deferred — requires confirmed HF repo IDs per model, not available
offline; see zoo/braindecode_eeg.py module docstring." This keeps the
checklist honest rather than claiming a deferred item is done.

Commit this doc update together with Step 5's commit (amend the file list,
not a separate commit).

---

## Phase 3 exit criteria

- [ ] `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py -v` — all green.
- [ ] `qortex neuroai zoo list --provider braindecode` shows 12 entries (11 new + EEGNet from Phase 1).
- [ ] `qortex neuroai zoo validate` exits 0 (11 warnings expected and correct, 0 errors).
- [ ] Spec §0 checklist: Phase 3 fully checked except "HF pretrained
      registry support," which carries an explicit deferral note instead of
      a checkmark. "Registry entries implemented so far" lists all 11 new
      entries.
- [ ] No changes to `_base.py`, `_contracts.py`, `_registry.py`,
      `contracts.py`, `spec.py`, `braindecode.py`, or any Phase 1/2 zoo file
      except the additive `__init__.py`/`conftest.py` edits.

Once this phase is merged, write
`docs/superpowers/plans/<date>-model-zoo-phase4-external-engines.md`
covering the SynthSeg/SynthStrip/HD-BET/FastSurfer/TractSeg CLI wrappers
from spec §12.2 and §11.4.
