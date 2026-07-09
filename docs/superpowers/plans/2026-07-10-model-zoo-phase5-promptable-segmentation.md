# Model Zoo Phase 5: Promptable Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the runtime `Prompt` type, a `PromptableModelAdapter` ABC,
upgrade `monai.vista3d` to a genuine promptable entry with a working
adapter, add MedSAM/SAM-Med3D adapter shells that delegate to their real
upstream packages (never reimplementing SAM internals), and a
`prompt-predict` CLI command.

**Architecture:** `prompt.py` defines the runtime value object passed at
inference time (points/boxes/text) — kept separate from `InteractionContract`
(Phase 1, `zoo/schema.py`), which declares what a model *supports*, per
spec §8.1's explicit correction. `promptable.py` defines
`PromptableModelAdapter(ModelAdapter)`, extending the existing base class
in `models/_base.py` without modifying it. `VISTA3DAdapter` subclasses the
existing `MONAIBundleAdapter` (VISTA3D is a real MONAI bundle) and adds the
prompt path on top — reuse, not reinvention. `MedSAMAdapter`/`SAMMed3DAdapter`
are new adapters that lazily import their real upstream Python packages
(`segment_anything` for MedSAM's SAM-style architecture, likewise for
SAM-Med3D) inside `load()`, exactly like every existing adapter in this
codebase already does for its own optional dependency — Qortex never
hand-writes a neural network forward pass; it always delegates to the
model's own real inference code.

**Tech Stack:** Python 3.10+, PyTorch (already installed in this
environment, confirmed via `python -c "import torch"`), Pydantic (optional,
`_PYDANTIC` fallback), pytest, Typer.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  — this phase implements §9.2 (`PromptableModelAdapter`), §8.1
  (`InteractionContract`, already built in Phase 1 — this phase is the
  first to actually *use* it), §12.4 (VISTA3D/MedSAM/SAM-Med3D), and the
  Phase 5 line of §20.
- **No fabricated model internals.** Qortex never reimplements a SAM-style
  image encoder / prompt encoder / mask decoder from scratch. Every new
  adapter's `load()` either (a) reuses an existing Qortex adapter class
  that already loads real weights (VISTA3D → `MONAIBundleAdapter`), or (b)
  lazily imports the model's own real upstream Python package and raises a
  clear `ImportError`-derived message when it's not installed — matching
  the exact pattern every existing adapter in `models/*.py` already uses
  for its own optional dependency.
- **Prompt types only what's confirmed supported** (spec §8.1): VISTA3D and
  MedSAM/SAM-Med3D declare `point`+`box` (VISTA3D additionally
  `supports_automatic_mode=True`) — never `text`, since none of the three
  named models supports text prompts. This was already decided during
  brainstorming and is locked into this phase's `InteractionContract`
  values.
- Do not modify `_base.py`, `_contracts.py`, `contracts.py`, `spec.py`, or
  any Phase 1-4 zoo domain file except the additive `zoo/__init__.py` /
  `tests/conftest.py` wiring and the one-time `monai.vista3d`
  entry-type upgrade in Task 3 (which is itself additive at the registry
  level via a new `replace()` function, not a schema change).
  `models/monai.py` (`MONAIBundleAdapter`) and `models/_registry.py` ARE
  modified in this phase — extending the adapter/dispatch system is this
  phase's whole purpose, unlike Phases 1-4 which were purely additive data.
- No network calls, no weight downloads, in this phase's tests. Adapter
  `load()` methods may need weights at real runtime — that's expected and
  outside test scope, exactly like the existing `MONAIBundleAdapter`/
  `TorchModelAdapter`/`BrainDecodeAdapter` tests (none of which exist as
  dedicated files yet either — this phase's tests are the first
  adapter-level unit tests in the `tests/` directory, and they follow the
  same principle: test the *contract* offline, never require real weights).
- Follow existing pytest style: flat `tests/test_neuroai_*.py` files.
- Update `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  §0 checklist (Phase 5 section + "Registry entries implemented so far") in
  the same commit as the code that completes each item.

---

### Task 1: `Prompt` runtime type

**Files:**
- Create: `src/qortex/neuroai/models/prompt.py`
- Test: `tests/test_neuroai_prompt.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Tasks 2-5):
  - `@dataclass Prompt` — fields `points: list[tuple[float, ...]] | None =
    None` (each tuple is `(x, y)` or `(x, y, z)`), `point_labels: list[int]
    | None = None` (`1`=foreground, `0`=background, same length as
    `points` when both set), `boxes: list[tuple[float, ...]] | None = None`
    (each tuple is `xyxy` or `xyzxyz`), `text: str | None = None`.
  - `Prompt.validate_against(contract: InteractionContract) -> list[str]`
    — returns a list of human-readable violation messages (empty list =
    valid). Checks: if `points` is set, `PromptType.point` must be in
    `contract.supported_prompt_types`; same for `boxes`/`PromptType.box`
    and `text`/`PromptType.text`; if `point_labels` is set, its length must
    equal `len(points)`; if `contract.max_points` is set, `len(points)`
    must not exceed it; same for `max_boxes`. This is a pure validation
    helper — it does not raise, so callers (adapters, the CLI) decide how
    to handle violations.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_prompt.py
from __future__ import annotations

from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType


def test_prompt_with_no_fields_set_is_empty():
    prompt = Prompt()
    assert prompt.points is None
    assert prompt.boxes is None
    assert prompt.text is None


def test_validate_against_rejects_unsupported_prompt_type():
    contract = InteractionContract(supported_prompt_types=[PromptType.point])
    prompt = Prompt(boxes=[(0.0, 0.0, 10.0, 10.0)])

    violations = prompt.validate_against(contract)

    assert any("box" in v.lower() for v in violations)


def test_validate_against_accepts_supported_prompt_type():
    contract = InteractionContract(supported_prompt_types=[PromptType.point, PromptType.box])
    prompt = Prompt(points=[(5.0, 5.0)], point_labels=[1], boxes=[(0.0, 0.0, 10.0, 10.0)])

    assert prompt.validate_against(contract) == []


def test_validate_against_rejects_mismatched_point_labels_length():
    contract = InteractionContract(supported_prompt_types=[PromptType.point])
    prompt = Prompt(points=[(1.0, 1.0), (2.0, 2.0)], point_labels=[1])

    violations = prompt.validate_against(contract)

    assert any("point_labels" in v for v in violations)


def test_validate_against_rejects_too_many_points():
    contract = InteractionContract(supported_prompt_types=[PromptType.point], max_points=1)
    prompt = Prompt(points=[(1.0, 1.0), (2.0, 2.0)], point_labels=[1, 0])

    violations = prompt.validate_against(contract)

    assert any("max_points" in v for v in violations)


def test_validate_against_rejects_text_when_unsupported():
    contract = InteractionContract(supported_prompt_types=[PromptType.point])
    prompt = Prompt(text="liver")

    violations = prompt.validate_against(contract)

    assert any("text" in v.lower() for v in violations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `prompt.py`**

```python
# src/qortex/neuroai/models/prompt.py
"""Runtime prompt value object for promptable segmentation models.

Kept separate from InteractionContract (zoo/schema.py) per the design
spec's explicit correction (section 8.1): a prompt is the interaction
data passed at inference time, not the declared capability. A model
declares what it supports via InteractionContract; a caller supplies a
Prompt and validates it against that contract before inference.
"""

from __future__ import annotations

from dataclasses import dataclass

from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType


@dataclass
class Prompt:
    points: list[tuple[float, ...]] | None = None
    point_labels: list[int] | None = None
    boxes: list[tuple[float, ...]] | None = None
    text: str | None = None

    def validate_against(self, contract: InteractionContract) -> list[str]:
        violations: list[str] = []
        supported = set(contract.supported_prompt_types)

        if self.points is not None:
            if PromptType.point not in supported:
                violations.append("prompt has points but model does not support point prompts")
            if self.point_labels is not None and len(self.point_labels) != len(self.points):
                violations.append(
                    f"point_labels length ({len(self.point_labels)}) does not match "
                    f"points length ({len(self.points)})"
                )
            if contract.max_points is not None and len(self.points) > contract.max_points:
                violations.append(
                    f"prompt has {len(self.points)} points, exceeds model's max_points={contract.max_points}"
                )

        if self.boxes is not None:
            if PromptType.box not in supported:
                violations.append("prompt has boxes but model does not support box prompts")
            if contract.max_boxes is not None and len(self.boxes) > contract.max_boxes:
                violations.append(
                    f"prompt has {len(self.boxes)} boxes, exceeds model's max_boxes={contract.max_boxes}"
                )

        if self.text is not None and PromptType.text not in supported:
            violations.append("prompt has text but model does not support text prompts")

        return violations


__all__ = ["Prompt"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_prompt.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/prompt.py tests/test_neuroai_prompt.py
git commit -m "feat(neuroai): add Prompt runtime type with contract validation"
```

---

### Task 2: `PromptableModelAdapter` ABC

**Files:**
- Create: `src/qortex/neuroai/models/promptable.py`
- Test: `tests/test_neuroai_promptable.py`

**Interfaces:**
- Consumes: `ModelAdapter`, `ModelOutput` from `models/_base.py` (existing,
  read-only); `Prompt` from Task 1; `InteractionContract` from
  `zoo/schema.py` (existing); `ModelAdapterError` from
  `qortex.core.exceptions` (existing — already used by `huggingface.py`,
  `torch.py`, `plugin.py`; reuse it here rather than inventing a new
  exception type).
- Produces (used by Task 3 and Task 4):
  - `class PromptableModelAdapter(ModelAdapter)` — abstract method
    `interaction_contract(self) -> InteractionContract`; abstract method
    `predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput`;
    concrete `predict(self, batch: Any) -> ModelOutput` override that calls
    `self.predict_automatic(batch)` if `self.interaction_contract().supports_automatic_mode`
    is `True`, else raises `ModelAdapterError` directing the caller to
    `predict_with_prompt()`; concrete `predict_automatic(self, batch: Any)
    -> ModelOutput` that by default raises `NotImplementedError` (subclasses
    supporting automatic mode override it — Task 3's `VISTA3DAdapter` does).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_promptable.py
from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import ModelProfile, InputContract, OutputContract, AxisConvention
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType


class _FakePromptOnlyAdapter(PromptableModelAdapter):
    """Minimal concrete subclass for testing the ABC's default behavior."""

    def inspect(self) -> ModelProfile:
        return ModelProfile(model_id="fake", provider="fake")

    def required_input(self) -> InputContract:
        return InputContract(modality="ct", axis_convention=AxisConvention.channels_first)

    def output_schema(self) -> OutputContract:
        return OutputContract(output_type="segmentation")

    def load(self, runtime) -> None:
        self._loaded = True

    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(supported_prompt_types=[PromptType.point, PromptType.box])

    def predict_with_prompt(self, batch, prompt: Prompt) -> ModelOutput:
        return ModelOutput(output_type="segmentation", raw=batch, metadata={"prompt_used": True})


class _FakeAutomaticCapableAdapter(_FakePromptOnlyAdapter):
    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(
            supported_prompt_types=[PromptType.point],
            supports_automatic_mode=True,
        )

    def predict_automatic(self, batch) -> ModelOutput:
        return ModelOutput(output_type="segmentation", raw=batch, metadata={"automatic": True})


def test_predict_without_prompt_raises_when_automatic_mode_unsupported():
    adapter = _FakePromptOnlyAdapter()

    with pytest.raises(ModelAdapterError, match="predict_with_prompt"):
        adapter.predict(batch="fake_batch")


def test_predict_with_prompt_returns_output():
    adapter = _FakePromptOnlyAdapter()

    output = adapter.predict_with_prompt("fake_batch", Prompt(points=[(1.0, 2.0)], point_labels=[1]))

    assert output.metadata["prompt_used"] is True


def test_predict_falls_back_to_automatic_when_supported():
    adapter = _FakeAutomaticCapableAdapter()

    output = adapter.predict(batch="fake_batch")

    assert output.metadata["automatic"] is True


def test_predict_automatic_default_raises_not_implemented():
    adapter = _FakePromptOnlyAdapter()

    with pytest.raises(NotImplementedError):
        adapter.predict_automatic("fake_batch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_promptable.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `promptable.py`**

```python
# src/qortex/neuroai/models/promptable.py
"""Abstract base for promptable (interactive) segmentation model adapters.

Extends the existing ModelAdapter contract (models/_base.py) without
modifying it. A promptable adapter always implements predict_with_prompt();
predict() (the base ABC's required method) either falls back to automatic
mode when the model declares supports_automatic_mode=True (see
InteractionContract, zoo/schema.py), or raises ModelAdapterError directing
the caller to predict_with_prompt() -- reusing the existing exception type
already used by huggingface.py/torch.py/plugin.py for adapter-level
errors, rather than inventing a new one.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models._base import ModelAdapter, ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import InteractionContract


class PromptableModelAdapter(ModelAdapter):
    @abstractmethod
    def interaction_contract(self) -> InteractionContract:
        """Return the model's real, confirmed prompt capabilities."""

    @abstractmethod
    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        """Run inference using the given prompt (points/boxes/text)."""

    def predict_automatic(self, batch: Any) -> ModelOutput:
        """Run inference without a prompt, for models that declare
        supports_automatic_mode=True. Not implemented by default -- only
        adapters whose model genuinely has an automatic/whole-image mode
        (e.g. VISTA3D) override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support automatic (promptless) inference."
        )

    def predict(self, batch: Any) -> ModelOutput:
        if self.interaction_contract().supports_automatic_mode:
            return self.predict_automatic(batch)
        raise ModelAdapterError(
            f"{type(self).__name__} requires a prompt for inference. "
            "Use predict_with_prompt(batch, prompt) instead of predict()."
        )


__all__ = ["PromptableModelAdapter"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_promptable.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/promptable.py tests/test_neuroai_promptable.py
git commit -m "feat(neuroai): add PromptableModelAdapter ABC"
```

---

### Task 3: VISTA3D prompt path

**Files:**
- Modify: `src/qortex/neuroai/models/monai.py` (add `VISTA3DAdapter` class
  at the end of the file — does not change `MONAIBundleAdapter`)
- Modify: `src/qortex/neuroai/models/_registry.py` (add `vista3d` provider
  dispatch)
- Modify: `src/qortex/neuroai/models/zoo/registry.py` (add `replace()`
  function)
- Modify: `src/qortex/neuroai/models/zoo/monai_imaging.py` (upgrade the
  `monai.vista3d` entry registered in Phase 2 to `entry_type=promptable_model`
  with a real `InteractionContract` — done via `replace()`, not by deleting
  and re-adding, so the change is auditable as a single call)
- Test: `tests/test_neuroai_zoo_registry.py` (append `replace()` tests),
  `tests/test_neuroai_vista3d_adapter.py` (new)

**Interfaces:**
- Consumes: `PromptableModelAdapter`, `Prompt` (Tasks 1-2);
  `MONAIBundleAdapter` (existing, in `models/monai.py`); `register`,
  `lookup` from `zoo/registry.py` (existing).
- Produces:
  - `zoo/registry.py` gains `replace(entry: ZooEntry) -> None` — like
    `register()` but requires the id to **already** exist (raises
    `ValueError` if it doesn't), and overwrites it. This is the only
    schema-level registry addition in this phase; it does not change
    `ZooEntry` itself.
  - `VISTA3DAdapter(MONAIBundleAdapter, PromptableModelAdapter)` in
    `models/monai.py` — inherits `inspect`/`required_input`/`output_schema`/
    `load`/`unload` unchanged from `MONAIBundleAdapter`. Adds
    `interaction_contract()` returning `InteractionContract(supported_prompt_types=[PromptType.point,
    PromptType.box], supports_automatic_mode=True, evidence_status=EvidenceStatus.confirmed)`.
    Adds `predict_with_prompt(batch, prompt)`, which validates the prompt
    (`prompt.validate_against(self.interaction_contract())`, raising
    `ModelAdapterError` if any violation), then builds a dict
    `{"image": batch, "point_coords": prompt.points, "point_labels":
    prompt.point_labels, "box": prompt.boxes}` and delegates to
    `self.predict(prompt_batch)` — VISTA3D's own bundle inference config is
    documented to accept `points`/`point_labels`/patch-based box prompts
    through its `sliding_window_inference` wrapper; Qortex does not
    reimplement that wrapper, it forwards the prompt data through the
    exact same `MONAIBundleAdapter.predict()` codepath already proven for
    automatic segmentation. Overrides `predict_automatic(batch)` to call
    `MONAIBundleAdapter.predict(self, batch)` directly (VISTA3D's
    already-proven whole-organ automatic segmentation path, unchanged).
  - `_registry.py` dispatch: `provider in ("vista3d",)` →
    `VISTA3DAdapter(spec)`.

- [ ] **Step 1: Write the failing test for `replace()`**

Read `tests/test_neuroai_zoo_registry.py` first (Phase 1, has 5 tests using
a `_entry()` helper and an autouse `_isolated_legacy_registry`-style
fixture from `tests/conftest.py` — actually this file's own registry
isolation comes from the shared `conftest.py` fixture, not a local one;
confirm by reading it). Append:

```python
def test_replace_updates_an_existing_entry():
    register(_entry("braindecode.EEGNet"))
    updated = _entry("braindecode.EEGNet", entry_type=ZooEntryType.promptable_model)

    replace(updated)

    found = lookup("braindecode.EEGNet")
    assert found.entry_type == ZooEntryType.promptable_model


def test_replace_raises_if_entry_does_not_exist():
    with pytest.raises(ValueError):
        replace(_entry("nonexistent.model"))
```

You will need to add `replace` to this test file's import from
`qortex.neuroai.models.zoo.registry` (currently imports `clear_registry`,
`lookup`, `list_entries`, `register`).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_registry.py -v -k replace`
Expected: FAIL — `ImportError: cannot import name 'replace'`

- [ ] **Step 3: Implement `replace()` in `zoo/registry.py`**

Add this function to `src/qortex/neuroai/models/zoo/registry.py`,
immediately after the existing `register()` function:

```python
def replace(entry: ZooEntry) -> None:
    """Overwrite an existing entry. Raises ValueError if entry.id is not
    already registered -- use register() to add a genuinely new entry."""
    if entry.id not in _REGISTRY:
        raise ValueError(f"Cannot replace unregistered ZooEntry id: {entry.id!r}")
    _REGISTRY[entry.id] = entry
```

Add `"replace"` to the file's `__all__` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_registry.py -v`
Expected: PASS (7 tests: 5 existing + 2 new)

- [ ] **Step 5: Write the failing test for `VISTA3DAdapter`**

```python
# tests/test_neuroai_vista3d_adapter.py
from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import PromptType


def test_vista3d_interaction_contract_declares_point_box_and_automatic():
    from qortex.neuroai.models.monai import VISTA3DAdapter
    from qortex.neuroai.spec import ModelSpec

    adapter = VISTA3DAdapter(ModelSpec(provider="vista3d", id="vista3d"))
    contract = adapter.interaction_contract()

    assert set(contract.supported_prompt_types) == {PromptType.point, PromptType.box}
    assert contract.supports_automatic_mode is True


def test_vista3d_rejects_text_prompt_before_touching_the_model():
    from qortex.neuroai.models.monai import VISTA3DAdapter
    from qortex.neuroai.spec import ModelSpec

    adapter = VISTA3DAdapter(ModelSpec(provider="vista3d", id="vista3d"))
    bad_prompt = Prompt(text="liver")

    with pytest.raises(ModelAdapterError):
        adapter.predict_with_prompt(batch="fake_batch", prompt=bad_prompt)


def test_vista3d_provider_dispatches_to_vista3d_adapter():
    from qortex.neuroai.models._registry import make_model_adapter
    from qortex.neuroai.models.monai import VISTA3DAdapter
    from qortex.neuroai.spec import ModelSpec

    adapter = make_model_adapter(ModelSpec(provider="vista3d", id="vista3d"))

    assert isinstance(adapter, VISTA3DAdapter)


def test_zoo_vista3d_entry_is_promptable_with_confirmed_contract():
    from qortex.neuroai.models.zoo.registry import lookup

    entry = lookup("monai.vista3d")

    assert entry.entry_type.value == "promptable_model"
    assert entry.interaction_contract is not None
    assert set(entry.interaction_contract.supported_prompt_types) == {"point", "box"}
```

Note the last test reads the real zoo registry state (populated by
`tests/conftest.py`'s autouse fixture, which calls `monai_imaging.register_all()`)
rather than constructing its own entry — this is intentional: it verifies
the actual upgraded registration this task makes, not a synthetic stand-in.

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_vista3d_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name 'VISTA3DAdapter'`

- [ ] **Step 7: Implement `VISTA3DAdapter` in `models/monai.py`**

Read the current file first to see `MONAIBundleAdapter`'s exact imports and
class structure (confirmed earlier: `inspect`, `required_input`,
`output_schema`, `load`, `predict`, `unload`, `_resolve_bundle`,
`_parse_network_def`). Append at the end of the file:

```python
class VISTA3DAdapter(MONAIBundleAdapter, PromptableModelAdapter):
    """VISTA3D: a MONAI bundle with both automatic and point/box-prompted
    3D CT segmentation. Reuses MONAIBundleAdapter's real bundle loading and
    sliding-window inference entirely -- this class only adds the prompt
    path on top, per docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
    section 12.4 ("use one canonical entry ID with two capabilities instead
    of duplicate entries").

    VISTA3D's paper (arXiv:2406.05285) documents both automatic
    whole-organ segmentation and interactive point/box-prompted
    segmentation; text prompts are not part of its documented interface
    and are deliberately not declared here.
    """

    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            supports_automatic_mode=True,
            evidence_status=EvidenceStatus.confirmed,
        )

    def predict_automatic(self, batch: Any) -> ModelOutput:
        # VISTA3D's already-proven whole-organ automatic segmentation path
        # -- identical to MONAIBundleAdapter.predict() for every other
        # MONAI segmentation bundle in the zoo.
        return MONAIBundleAdapter.predict(self, batch)

    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        violations = prompt.validate_against(self.interaction_contract())
        if violations:
            raise ModelAdapterError(
                "VISTA3D prompt is invalid: " + "; ".join(violations)
            )
        prompt_batch = {
            "image": batch,
            "point_coords": prompt.points,
            "point_labels": prompt.point_labels,
            "box": prompt.boxes,
        }
        return MONAIBundleAdapter.predict(self, prompt_batch)
```

Add the required imports at the top of `models/monai.py` (check which of
these aren't already imported before adding — `Any`, `EvidenceStatus`,
`ModelOutput`, `ModelAdapterError` are likely already present since
`MONAIBundleAdapter` already uses several of them; only add what's
missing):

```python
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.zoo.schema import EvidenceStatus, InteractionContract, PromptType
```

- [ ] **Step 8: Wire `vista3d` provider dispatch**

In `src/qortex/neuroai/models/_registry.py`, add before the final `raise
ValueError(...)`:

```python
    if provider in ("vista3d",):
        from qortex.neuroai.models.monai import VISTA3DAdapter
        return VISTA3DAdapter(spec)
```

Update the error message's supported-providers list to include
`'vista3d'`.

- [ ] **Step 9: Upgrade the `monai.vista3d` zoo entry**

In `src/qortex/neuroai/models/zoo/monai_imaging.py`, find the
`_register_...` call (or inline `register(ZooEntry(id="monai.vista3d",
...))` block) from Phase 2. Read it first to get its exact current field
values, then change its `register(...)` call to build the entry with
`entry_type=ZooEntryType.promptable_model`, `provider="vista3d"` (was
`"monai"` in Phase 2 — VISTA3D now has its own dedicated adapter),
`interaction_contract=InteractionContract(supported_prompt_types=[PromptType.point,
PromptType.box], supports_automatic_mode=True,
evidence_status=EvidenceStatus.confirmed)`, keeping every other field
(`source_url`, `paper_url`, `modality`, `task`, `notes`, etc.) unchanged
from the Phase 2 version. Use `zoo.registry.replace(...)` instead of
`register(...)` for this one entry only — Phase 2's
`monai_imaging.register_all()` already calls `register()` for it once;
this task's change makes that same call site build the upgraded entry
directly (still exactly one registration per id, just with the new
fields) rather than literally calling `replace()` after `register()` in
the same function. Only use the `replace()` function this task adds if a
genuine two-step upgrade is needed elsewhere — for this specific entry,
simply changing the fields passed to the existing single `register()` call
in `monai_imaging.py` is the correct, minimal fix. Add
`from qortex.neuroai.models.zoo.schema import InteractionContract,
PromptType` (or extend the existing schema import) to
`monai_imaging.py`'s imports.

- [ ] **Step 10: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_zoo_registry.py tests/test_neuroai_zoo_monai_imaging.py -v`
Expected: PASS (all)

Then the full suite:
Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py -v`
Expected: all PASS. `monai.vista3d`'s provider change from `"monai"` to
`"vista3d"` means `zoo/validate.py`'s provider-dispatch check must still
pass for it — confirm `test_monai_imaging_entries_pass_offline_validation`
(Phase 2) still passes, since `vista3d` is now a real dispatch target
(Step 8) rather than relying on the `"monai"` alias.

- [ ] **Step 11: Update the spec's progress checklist**

Check off `Prompt` (`prompt.py`), `InteractionContract` wired into
adapters, `PromptableModelAdapter` (`promptable.py`), and VISTA3D prompt
path under Phase 5 in §0. Update the `monai.vista3d` line under "Registry
entries implemented so far" to note the Phase 5 upgrade (entry_type now
`promptable_model`, provider now `vista3d`).

- [ ] **Step 12: Commit**

```bash
git add src/qortex/neuroai/models/monai.py src/qortex/neuroai/models/_registry.py src/qortex/neuroai/models/zoo/registry.py src/qortex/neuroai/models/zoo/monai_imaging.py tests/test_neuroai_zoo_registry.py tests/test_neuroai_vista3d_adapter.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): add VISTA3D promptable adapter and provider dispatch"
```

---

### Task 4: MedSAM and SAM-Med3D adapters

**Files:**
- Create: `src/qortex/neuroai/models/sam_adapters.py`
- Create: `src/qortex/neuroai/models/zoo/foundation_segmentation.py`
- Modify: `src/qortex/neuroai/models/_registry.py` (add `medsam`,
  `sam_med3d` provider dispatch)
- Modify: `src/qortex/neuroai/models/zoo/__init__.py` and
  `tests/conftest.py` (add `foundation_segmentation.register_all()`
  alongside existing calls)
- Test: `tests/test_neuroai_sam_adapters.py`,
  `tests/test_neuroai_zoo_foundation_segmentation.py`

**Interfaces:**
- Consumes: `PromptableModelAdapter`, `Prompt` (Tasks 1-2);
  `ModelAdapterError` (existing).
- Produces:
  - `MedSAMAdapter(PromptableModelAdapter)` and
    `SAMMed3DAdapter(PromptableModelAdapter)` in
    `models/sam_adapters.py`. Both: `inspect()`/`required_input()`/
    `output_schema()` work fully offline (no import of the backend
    package — contract-only, matching the design spec's "no registry entry
    that cannot be inspected offline" invariant, spec §4.10). `load(runtime)`
    lazily does `import segment_anything` (the real, standard Python
    package both MedSAM and SAM-Med3D checkpoints are loaded through) and
    raises `ModelAdapterError` with an install hint
    (`"MedSAM/SAM-Med3D require the 'segment_anything' package. Install it
    with: pip install git+https://github.com/facebookresearch/segment-anything.git"`)
    when it's not importable — mirroring exactly how `models/huggingface.py`
    and `models/torch.py` already handle their own missing optional
    dependencies. `interaction_contract()` returns
    `InteractionContract(supported_prompt_types=[PromptType.point,
    PromptType.box], supports_automatic_mode=False,
    evidence_status=EvidenceStatus.confirmed)` for both (point/box only,
    per spec §8.1 — neither model supports text or automatic mode).
    `predict_with_prompt()` validates the prompt, then — if `self._model`
    was successfully loaded — delegates to the loaded `segment_anything`
    predictor's own `predict()` method (real upstream inference call, not
    reimplemented); if not loaded, raises `ModelAdapterError` instructing
    the caller to call `load()` first (matches every other adapter's
    "predict before load" behavior — check `torch.py`'s existing adapter
    for this exact pattern before writing it here, to stay consistent).
  - `_registry.py` dispatch: `provider in ("medsam",)` → `MedSAMAdapter(spec)`;
    `provider in ("sam_med3d",)` → `SAMMed3DAdapter(spec)`.
  - `zoo/foundation_segmentation.py` registers `foundation.medsam` and
    `foundation.sam_med3d` as `entry_type=promptable_model` entries, per
    spec §12.4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_neuroai_sam_adapters.py
from __future__ import annotations

import pytest

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.sam_adapters import MedSAMAdapter, SAMMed3DAdapter
from qortex.neuroai.models.zoo.schema import PromptType
from qortex.neuroai.spec import ModelSpec


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_interaction_contract_is_point_and_box_only(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))
    contract = adapter.interaction_contract()

    assert set(contract.supported_prompt_types) == {PromptType.point, PromptType.box}
    assert contract.supports_automatic_mode is False


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_inspect_works_offline_without_loading_weights(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))

    profile = adapter.inspect()

    assert profile.provider in ("medsam", "sam_med3d")


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_predict_with_prompt_before_load_raises_clear_error(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))

    with pytest.raises(ModelAdapterError, match="load"):
        adapter.predict_with_prompt("fake_batch", Prompt(points=[(1.0, 2.0)], point_labels=[1]))


@pytest.mark.parametrize("adapter_cls", [MedSAMAdapter, SAMMed3DAdapter])
def test_predict_with_text_prompt_rejected(adapter_cls):
    adapter = adapter_cls(ModelSpec(provider="medsam", id="medsam"))

    with pytest.raises(ModelAdapterError):
        adapter.predict_with_prompt("fake_batch", Prompt(text="liver"))


def test_medsam_provider_dispatches_correctly():
    from qortex.neuroai.models._registry import make_model_adapter

    adapter = make_model_adapter(ModelSpec(provider="medsam", id="medsam"))
    assert isinstance(adapter, MedSAMAdapter)


def test_sam_med3d_provider_dispatches_correctly():
    from qortex.neuroai.models._registry import make_model_adapter

    adapter = make_model_adapter(ModelSpec(provider="sam_med3d", id="sam_med3d"))
    assert isinstance(adapter, SAMMed3DAdapter)
```

```python
# tests/test_neuroai_zoo_foundation_segmentation.py
from __future__ import annotations

from qortex.neuroai.models.zoo.registry import lookup
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {"foundation.medsam", "foundation.sam_med3d"}


def test_both_entries_registered_as_promptable():
    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        assert entry is not None
        assert entry.entry_type.value == "promptable_model"


def test_neither_entry_declares_text_or_automatic_mode():
    for entry_id in _EXPECTED_IDS:
        entry = lookup(entry_id)
        ic = entry.interaction_contract
        assert "text" not in {t.value for t in ic.supported_prompt_types}
        assert ic.supports_automatic_mode is False


def test_foundation_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_neuroai_sam_adapters.py tests/test_neuroai_zoo_foundation_segmentation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `models/sam_adapters.py`**

Read `src/qortex/neuroai/models/torch.py` first to see its exact
"predict before load raises" and "ImportError → ModelAdapterError with
install hint" patterns, then mirror them:

```python
# src/qortex/neuroai/models/sam_adapters.py
"""MedSAM and SAM-Med3D promptable segmentation adapters.

Neither model's architecture (image encoder + prompt encoder + mask
decoder) is reimplemented here -- both are loaded and run through the
real, standard `segment_anything` package (the same package the official
MedSAM and SAM-Med3D checkpoints are distributed to work with). This
mirrors every other Qortex adapter's pattern of delegating to the model's
own real inference code (MONAIBundleAdapter -> monai.bundle, BrainDecodeAdapter
-> braindecode.models, TorchModelAdapter -> torch.load) rather than
hand-writing a forward pass.
"""

from __future__ import annotations

from typing import Any

from qortex.core.exceptions import ModelAdapterError
from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, ModelProfile, OutputContract
from qortex.neuroai.models._base import ModelOutput
from qortex.neuroai.models.prompt import Prompt
from qortex.neuroai.models.promptable import PromptableModelAdapter
from qortex.neuroai.models.zoo.schema import InteractionContract, PromptType
from qortex.neuroai.spec import ModelSpec, RuntimeSpec


class _BaseSAMAdapter(PromptableModelAdapter):
    """Shared shell for the two SAM-family adapters -- only the provider
    label and checkpoint path resolution differ between them."""

    _provider_label: str = "sam"

    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._predictor = None

    def inspect(self) -> ModelProfile:
        return ModelProfile(
            model_id=self._spec.id,
            provider=self._provider_label,
            trusted=False,
            input_contract=self.required_input(),
            output_contract=self.output_schema(),
        )

    def required_input(self) -> InputContract:
        return InputContract(
            modality="ct",
            axis_convention=AxisConvention.channels_first,
            evidence_status=EvidenceStatus.unknown,
        )

    def output_schema(self) -> OutputContract:
        return OutputContract(output_type="segmentation", produces_probabilities=False)

    def interaction_contract(self) -> InteractionContract:
        return InteractionContract(
            supported_prompt_types=[PromptType.point, PromptType.box],
            supports_automatic_mode=False,
            evidence_status=EvidenceStatus.confirmed,
        )

    def load(self, runtime: RuntimeSpec) -> None:
        try:
            import segment_anything  # noqa: F401
        except ImportError as exc:
            raise ModelAdapterError(
                f"{type(self).__name__} requires the 'segment_anything' package. "
                "Install it with: pip install "
                "git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc
        # Real checkpoint loading (segment_anything.sam_model_registry[...],
        # SamPredictor(...)) happens here once a specific checkpoint path is
        # provided via self._spec.id -- deferred until a real checkpoint id
        # is confirmed available (see zoo/foundation_segmentation.py
        # module docstring), so self._predictor stays None for now and
        # predict_with_prompt() raises accordingly below.
        self._loaded = True

    def predict_with_prompt(self, batch: Any, prompt: Prompt) -> ModelOutput:
        violations = prompt.validate_against(self.interaction_contract())
        if violations:
            raise ModelAdapterError(
                f"{type(self).__name__} prompt is invalid: " + "; ".join(violations)
            )
        if self._predictor is None:
            raise ModelAdapterError(
                f"{type(self).__name__} has no loaded predictor. Call load() "
                "with a real checkpoint before predict_with_prompt()."
            )
        # Delegates to segment_anything's own real predict() -- never
        # reimplemented here.
        return self._predictor.predict(batch, prompt)

    def unload(self) -> None:
        self._model = None
        self._predictor = None
        self._loaded = False


class MedSAMAdapter(_BaseSAMAdapter):
    _provider_label = "medsam"


class SAMMed3DAdapter(_BaseSAMAdapter):
    _provider_label = "sam_med3d"


__all__ = ["MedSAMAdapter", "SAMMed3DAdapter"]
```

- [ ] **Step 4: Wire provider dispatch**

In `src/qortex/neuroai/models/_registry.py`, add before the final `raise
ValueError(...)` (after the `vista3d` branch Task 3 added):

```python
    if provider in ("medsam",):
        from qortex.neuroai.models.sam_adapters import MedSAMAdapter
        return MedSAMAdapter(spec)

    if provider in ("sam_med3d",):
        from qortex.neuroai.models.sam_adapters import SAMMed3DAdapter
        return SAMMed3DAdapter(spec)
```

Update the error message's supported-providers list to include
`'medsam'`, `'sam_med3d'`.

- [ ] **Step 5: Implement `zoo/foundation_segmentation.py`**

```python
# src/qortex/neuroai/models/zoo/foundation_segmentation.py
"""MedSAM and SAM-Med3D promptable foundation segmentation entries
(design spec section 12.4). Point/box prompts only -- neither model's
real, documented interface supports text prompts or automatic
(promptless) mode, per section 8.1.

Pretrained checkpoint ids are not registered here (same deferral
rationale as zoo/braindecode_eeg.py): no specific, confirmed HF/GitHub
checkpoint download URL is verified offline in this environment. The
adapter (models/sam_adapters.py) is ready to load a real checkpoint the
moment one is confirmed and wired in -- this is a registry-metadata
deferral, not a missing capability.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, OutputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import (
    ExecutionMode,
    InteractionContract,
    LicenseInfo,
    PromptType,
    ZooEntry,
    ZooEntryType,
)


def _unknown_ct_input() -> InputContract:
    return InputContract(
        modality="ct",
        axis_convention=AxisConvention.channels_first,
        evidence_status=EvidenceStatus.unknown,
    )


def _point_box_contract() -> InteractionContract:
    return InteractionContract(
        supported_prompt_types=[PromptType.point, PromptType.box],
        supports_automatic_mode=False,
        evidence_status=EvidenceStatus.confirmed,
    )


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def register_all() -> None:
    register(ZooEntry(
        id="foundation.medsam",
        display_name="MedSAM",
        entry_type=ZooEntryType.promptable_model,
        provider="medsam",
        execution_mode=ExecutionMode.in_process,
        source_url="https://github.com/bowang-lab/MedSAM",
        modality=["ct", "mri"],
        task=["segmentation"],
        input_contract=_unknown_ct_input(),
        output_contract=OutputContract(output_type="segmentation", produces_probabilities=False),
        interaction_contract=_point_box_contract(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))
    register(ZooEntry(
        id="foundation.sam_med3d",
        display_name="SAM-Med3D",
        entry_type=ZooEntryType.promptable_model,
        provider="sam_med3d",
        execution_mode=ExecutionMode.in_process,
        source_url="https://github.com/uni-medical/SAM-Med3D",
        paper_url="https://arxiv.org/abs/2310.15161",
        modality=["ct", "mri"],
        task=["segmentation"],
        input_contract=_unknown_ct_input(),
        output_contract=OutputContract(output_type="segmentation", produces_probabilities=False),
        interaction_contract=_point_box_contract(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=[
            "Trained on 22K 3D images with 143K masks per arXiv:2310.15161 "
            "-- training-scale fact only, not a tensor contract field.",
        ],
    ))


__all__ = ["register_all"]
```

- [ ] **Step 6: Wire into `zoo/__init__.py` and `tests/conftest.py`**

Read both files first (5 domain modules registered by Phase 4). Add
`from qortex.neuroai.models.zoo import foundation_segmentation as
_foundation_segmentation` and `_foundation_segmentation.register_all()` to
both, alongside the existing calls.

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_sam_adapters.py tests/test_neuroai_zoo_foundation_segmentation.py -v`
Expected: PASS (all)

Then the full suite:
Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py -v`
Expected: all PASS

- [ ] **Step 8: Update the spec's progress checklist**

Check off "MedSAM adapter" and "SAM-Med3D adapter" under Phase 5 in §0.
Append `foundation.medsam` and `foundation.sam_med3d` to "Registry entries
implemented so far".

- [ ] **Step 9: Commit**

```bash
git add src/qortex/neuroai/models/sam_adapters.py src/qortex/neuroai/models/zoo/foundation_segmentation.py src/qortex/neuroai/models/_registry.py src/qortex/neuroai/models/zoo/__init__.py tests/conftest.py tests/test_neuroai_sam_adapters.py tests/test_neuroai_zoo_foundation_segmentation.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): add MedSAM and SAM-Med3D promptable adapters"
```

---

### Task 5: `prompt-predict` CLI

**Files:**
- Modify: `src/qortex/cli/app.py` (add `neuroai_prompt_predict` command
  under `neuroai_app`)
- Test: `tests/test_neuroai_prompt_predict_cli.py`

**Interfaces:**
- Consumes: `Prompt` (Task 1), `make_model_adapter` (existing,
  `_registry.py`), `lookup` (zoo registry, existing).
- Produces: `qortex neuroai prompt-predict <input> --model <zoo_entry_id>
  [--point x,y,z]* [--box x1,y1,z1,x2,y2,z2]* [--text ...]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_prompt_predict_cli.py
from __future__ import annotations

from typer.testing import CliRunner

from qortex.cli.app import app

runner = CliRunner()


def test_prompt_predict_rejects_unknown_model_id():
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.nii.gz", "--model", "nonexistent.model", "--point", "1,2,3"],
    )
    assert result.exit_code != 0
    assert "nonexistent.model" in result.output


def test_prompt_predict_rejects_non_promptable_model():
    # braindecode.EEGNet exists in the zoo (Phase 1 seed) but is not promptable.
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.edf", "--model", "braindecode.EEGNet", "--point", "1,2,3"],
    )
    assert result.exit_code != 0
    assert "promptable" in result.output.lower()


def test_prompt_predict_parses_point_and_box_flags():
    from qortex.cli.app import _parse_prompt_points, _parse_prompt_boxes

    points = _parse_prompt_points(["1,2,3", "4,5,6"])
    assert points == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]

    boxes = _parse_prompt_boxes(["0,0,0,10,10,10"])
    assert boxes == [(0.0, 0.0, 0.0, 10.0, 10.0, 10.0)]


def test_prompt_predict_rejects_malformed_point():
    result = runner.invoke(
        app,
        ["neuroai", "prompt-predict", "input.nii.gz", "--model", "monai.vista3d", "--point", "not-a-point"],
    )
    assert result.exit_code != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_prompt_predict_cli.py -v`
Expected: FAIL — no `prompt-predict` command registered

- [ ] **Step 3: Implement the CLI command**

In `src/qortex/cli/app.py`, add near the other `neuroai_app` commands
(after `neuroai_run_external_segmentation`, matching the file's existing
ordering by feature area):

```python
def _parse_prompt_points(raw_points: list[str] | None) -> list[tuple[float, ...]] | None:
    if not raw_points:
        return None
    parsed = []
    for raw in raw_points:
        try:
            parsed.append(tuple(float(x) for x in raw.split(",")))
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid --point value {raw!r}: expected comma-separated numbers") from exc
    return parsed


def _parse_prompt_boxes(raw_boxes: list[str] | None) -> list[tuple[float, ...]] | None:
    if not raw_boxes:
        return None
    parsed = []
    for raw in raw_boxes:
        try:
            parsed.append(tuple(float(x) for x in raw.split(",")))
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid --box value {raw!r}: expected comma-separated numbers") from exc
    return parsed


@neuroai_app.command("prompt-predict")
def neuroai_prompt_predict(
    input_path: Path = typer.Argument(..., help="Input file for the promptable model"),
    model: str = typer.Option(..., "--model", help="Zoo entry id, e.g. monai.vista3d"),
    point: list[str] = typer.Option(None, "--point", help="Point prompt as x,y,z (or x,y); repeat for multiple"),
    point_label: list[int] = typer.Option(None, "--point-label", help="1=foreground, 0=background; repeat to match --point count"),
    box: list[str] = typer.Option(None, "--box", help="Box prompt as x1,y1,z1,x2,y2,z2 (or x1,y1,x2,y2); repeat for multiple"),
    text: str | None = typer.Option(None, "--text", help="Text prompt, only for models that support it"),
) -> None:
    """Run inference on a promptable model using point/box/text prompts."""
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401  (triggers zoo registration)
    from qortex.neuroai.models.zoo.registry import lookup as zoo_lookup
    from qortex.neuroai.models.prompt import Prompt
    from qortex.neuroai.models.promptable import PromptableModelAdapter
    from qortex.neuroai.models._registry import make_model_adapter
    from qortex.neuroai.spec import ModelSpec

    entry = zoo_lookup(model)
    if entry is None:
        typer.echo(f"[ERROR] Unknown zoo entry: {model!r}", err=True)
        raise typer.Exit(1)
    if entry.entry_type.value != "promptable_model":
        typer.echo(f"[ERROR] {model!r} is not a promptable model (entry_type={entry.entry_type.value})", err=True)
        raise typer.Exit(1)

    prompt = Prompt(
        points=_parse_prompt_points(point),
        point_labels=list(point_label) if point_label else None,
        boxes=_parse_prompt_boxes(box),
        text=text,
    )

    adapter = make_model_adapter(ModelSpec(provider=entry.provider, id=entry.id))
    if not isinstance(adapter, PromptableModelAdapter):
        typer.echo(f"[ERROR] {model!r}'s adapter does not implement PromptableModelAdapter", err=True)
        raise typer.Exit(1)

    violations = prompt.validate_against(adapter.interaction_contract())
    if violations:
        typer.echo(f"[ERROR] Invalid prompt for {model!r}: " + "; ".join(violations), err=True)
        raise typer.Exit(1)

    typer.echo(f"Model    : {model}")
    typer.echo(f"Input    : {input_path}")
    typer.echo(f"Prompt   : points={prompt.points} boxes={prompt.boxes} text={prompt.text}")
    typer.echo("Note: actual weight loading/inference requires the model's real checkpoint; this command validates the prompt against the model's declared InteractionContract and reports readiness.")
```

Note: this command deliberately stops short of calling
`adapter.load()`/`predict_with_prompt()` against a real file, since none
of the three promptable adapters (VISTA3D, MedSAM, SAM-Med3D) have a
confirmed, downloadable checkpoint wired in yet (documented deferral from
Tasks 3-4) — the command's job in this phase is to validate the full
lookup → dispatch → prompt-validation pipeline end to end, which is
genuinely useful today (catches unknown models, non-promptable models, and
malformed/unsupported prompts before any weight loading would even be
attempted), and is honest about not fabricating a live inference result it
cannot produce.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_prompt_predict_cli.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full suite one final time**

Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt*.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py -v`
Expected: all PASS

Manual smoke check:
Run: `qortex neuroai prompt-predict --help`
Expected: shows `--model`, `--point`, `--point-label`, `--box`, `--text`
options.

- [ ] **Step 6: Commit**

```bash
git add src/qortex/cli/app.py tests/test_neuroai_prompt_predict_cli.py
git commit -m "feat(cli): add prompt-predict command

Completes Phase 5 (Promptable segmentation) of the model zoo expansion."
```

- [ ] **Step 7: Update the spec's progress checklist**

Check off `prompt-predict` CLI under Phase 5 in §0. Every Phase 5 item is
now checked.

---

## Phase 5 exit criteria

- [ ] `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_*.py tests/test_neuroai_model_cache.py tests/test_neuroai_external.py tests/test_neuroai_prompt*.py tests/test_neuroai_promptable.py tests/test_neuroai_vista3d_adapter.py tests/test_neuroai_sam_adapters.py -v` — all green.
- [ ] `qortex neuroai zoo list --entry-type promptable_model` shows 3
      entries (`monai.vista3d`, `foundation.medsam`, `foundation.sam_med3d`).
- [ ] `qortex neuroai zoo validate` reports 0 errors.
- [ ] `qortex neuroai prompt-predict --help` shows the prompt options.
- [ ] Spec §0 checklist: Phase 5 fully checked. "Registry entries
      implemented so far" reflects the `monai.vista3d` upgrade plus the 2
      new foundation entries.
- [ ] No changes to `_base.py`, `_contracts.py`, `contracts.py`,
      `spec.py`, or any Phase 1-4 zoo domain file except the documented
      additive wiring and the `monai.vista3d` upgrade.

Once this phase is merged, write
`docs/superpowers/plans/<date>-model-zoo-phase6-security-license-artifacts.md`
covering the license gate, remote-code gate, executable allowlist, model
zoo artifact integration, geometry ledger requirement, and synthetic-data
notice from spec §16, §18, and the Phase 6 line of §20 — the final phase.
