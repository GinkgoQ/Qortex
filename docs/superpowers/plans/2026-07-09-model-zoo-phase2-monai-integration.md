# Model Zoo Phase 2: MONAI Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the P0 MONAI imaging bundles and generative bundles as
`ZooEntry` records, add an offline MONAI bundle metadata extractor, add a
model weight cache/provenance layer, and bridge the zoo registry into the
existing `suggest-models` CLI so the expanded catalog is actually reachable
from it.

**Architecture:** `extractors/monai_bundle.py` is a pure function that turns
an already-loaded MONAI bundle `metadata.json` dict into contract fields —
no network, no bundle download, fixture-tested against MONAI's documented
`network_data_format` schema shape. `zoo/monai_imaging.py` and
`zoo/monai_generative.py` are domain files, following the exact pattern
`zoo/seed_examples.py` established in Phase 1. `cache.py` is a provenance
manifest layered on top of each backend's own download cache, not a
downloader. `zoo/bridge.py` converts fully-contracted `ZooEntry` records
into the legacy `_contracts.ModelContractEntry` shape so `suggest-models`
picks them up, without modifying `_contracts.py` itself.

**Tech Stack:** Python 3.10+, Pydantic (optional, `_PYDANTIC` fallback
pattern), pytest, Typer.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  — this phase implements §11.1, §12.1, §12.5, §15, and the "compatibility
  bridge into `suggest-models`" line of §20 Phase 2.
- **No guessed contracts (spec §4.1).** For every new registry entry, only
  set a quantitative field (`n_channels`, `n_classes`, `spatial_shape`,
  etc.) when the design spec's own rationale text states it, or when an
  existing curated entry in `_contracts.py` already confirms it for the
  same real-world model (reuse, not fabrication — see Task 2's
  `wholeBody_ct_segmentation` note). Otherwise leave the field `None` and
  set `evidence_status=EvidenceStatus.unknown`.
- Do not duplicate the `monai.brats_mri_segmentation` id — it was already
  registered in Phase 1 (`zoo/seed_examples.py`). This phase registers the
  other 13 P0 MONAI imaging bundles.
- Do not modify `_base.py`, `_contracts.py`, `_registry.py`, `contracts.py`,
  `spec.py`, or any existing adapter file — Task 5's bridge only *calls*
  `_contracts.register()`/`_contracts.lookup()`, never edits that module.
- No network calls, no weight/bundle downloads, anywhere in this phase's
  code or tests.
- Follow existing pytest style: flat `tests/test_neuroai_zoo_*.py` files.
- Update `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md`
  §0 checklist (Phase 2 section + "Registry entries implemented so far") in
  the same commit as the code that completes each item.
- The zoo package already exists at `src/qortex/neuroai/models/zoo/`
  (`schema.py`, `registry.py`, `validate.py`, `seed_examples.py`,
  `__init__.py`) — this phase only adds new files inside it plus two new
  top-level modules (`extractors/`, `cache.py`).

---

### Task 1: MONAI bundle metadata extractor (offline)

**Files:**
- Create: `src/qortex/neuroai/models/extractors/__init__.py` (empty package
  marker)
- Create: `src/qortex/neuroai/models/extractors/monai_bundle.py`
- Test: `tests/test_neuroai_extractors_monai_bundle.py`

**Interfaces:**
- Consumes: `qortex.neuroai.contracts.{InputContract, OutputContract,
  AxisConvention, EvidenceStatus}`.
- Produces (used by Task 2, optionally):
  - `@dataclass ExtractedMONAIContract` — fields `model_id: str`,
    `input_contract: InputContract | None`, `output_contract: OutputContract
    | None`, `unresolved_transforms: list[str]`.
  - `extract_monai_contract(model_id: str, metadata: dict, inference: dict |
    None = None) -> ExtractedMONAIContract` — pure function, no I/O. Reads
    MONAI's documented bundle metadata schema
    (`metadata["network_data_format"]["inputs"]`/`["outputs"]`, each a dict
    keyed by tensor name with `"type"`, `"format"`, `"num_channels"`,
    `"spatial_shape"`, `"dtype"` sub-keys — this is MONAI's public, stable
    bundle metadata convention). Missing keys are treated as unknown, never
    guessed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_extractors_monai_bundle.py
from __future__ import annotations

from qortex.neuroai.contracts import EvidenceStatus
from qortex.neuroai.models.extractors.monai_bundle import (
    ExtractedMONAIContract,
    extract_monai_contract,
)


def test_extract_full_metadata_populates_contracts():
    metadata = {
        "network_data_format": {
            "inputs": {
                "image": {
                    "type": "image",
                    "format": "magnitude",
                    "num_channels": 4,
                    "spatial_shape": [240, 240, 155],
                    "dtype": "float32",
                }
            },
            "outputs": {
                "pred": {
                    "type": "image",
                    "format": "segmentation",
                    "num_channels": 4,
                    "dtype": "float32",
                }
            },
        }
    }

    extracted = extract_monai_contract("test.bundle", metadata)

    assert extracted.model_id == "test.bundle"
    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels == 4
    assert extracted.input_contract.spatial_shape == (240, 240, 155)
    assert extracted.input_contract.evidence_status == EvidenceStatus.confirmed
    assert extracted.output_contract is not None
    assert extracted.output_contract.n_classes == 4
    assert extracted.unresolved_transforms == []


def test_extract_missing_network_data_format_returns_unknown_contracts():
    extracted = extract_monai_contract("bare.bundle", {})

    assert extracted.input_contract is None
    assert extracted.output_contract is None


def test_extract_partial_metadata_does_not_guess_missing_fields():
    metadata = {
        "network_data_format": {
            "inputs": {"image": {"type": "image", "format": "magnitude"}},
            "outputs": {},
        }
    }

    extracted = extract_monai_contract("partial.bundle", metadata)

    assert extracted.input_contract is not None
    assert extracted.input_contract.n_channels is None
    assert extracted.input_contract.spatial_shape is None
    assert extracted.input_contract.evidence_status == EvidenceStatus.inferred
    assert extracted.output_contract is None


def test_extract_flags_unresolved_custom_transforms():
    metadata = {
        "network_data_format": {
            "inputs": {"image": {"type": "image", "format": "magnitude", "num_channels": 1}},
            "outputs": {},
        }
    }
    inference = {
        "preprocessing": [
            {"_target_": "LoadImaged"},
            {"_target_": "my_custom_module.WeirdTransform"},
        ]
    }

    extracted = extract_monai_contract("custom.bundle", metadata, inference)

    assert "my_custom_module.WeirdTransform" in extracted.unresolved_transforms
    assert "LoadImaged" not in extracted.unresolved_transforms
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_extractors_monai_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `extractors/monai_bundle.py`**

```python
# src/qortex/neuroai/models/extractors/monai_bundle.py
"""Offline MONAI bundle metadata extractor.

Turns an already-loaded MONAI bundle metadata.json (and optionally
inference.json) dict into Qortex contract fields. Pure function — no
network access, no bundle download. Missing fields are left unknown, never
guessed, per docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 11.1.

MONAI bundles conventionally describe I/O under
metadata["network_data_format"]["inputs"/"outputs"], each keyed by tensor
name with "type"/"format"/"num_channels"/"spatial_shape"/"dtype" — this is
MONAI's own public, stable bundle metadata convention, not something Qortex
invents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qortex.neuroai.contracts import (
    AxisConvention,
    EvidenceStatus,
    InputContract,
    OutputContract,
)

# MONAI transform class names that Qortex already knows how to translate
# into its own preprocessing plan. Anything outside this set in a bundle's
# inference.json preprocessing chain is a custom callable Qortex cannot
# safely auto-apply.
_KNOWN_MONAI_TRANSFORMS = {
    "LoadImaged", "EnsureChannelFirstd", "Orientationd", "Spacingd",
    "ScaleIntensityRanged", "NormalizeIntensityd", "CropForegroundd",
    "Resized", "ToTensord", "EnsureTyped",
}


@dataclass
class ExtractedMONAIContract:
    model_id: str
    input_contract: InputContract | None = None
    output_contract: OutputContract | None = None
    unresolved_transforms: list[str] = field(default_factory=list)


def _extract_input_contract(inputs: dict) -> InputContract | None:
    if not inputs:
        return None
    # Bundles may declare multiple named inputs; Qortex's InputContract
    # models a single primary tensor, so take the first declared input.
    _, spec = next(iter(inputs.items()))
    n_channels = spec.get("num_channels")
    spatial_shape = spec.get("spatial_shape")
    dtype = spec.get("dtype", "float32")

    confirmed = n_channels is not None and spatial_shape is not None
    return InputContract(
        modality="mri",  # MONAI bundle inputs are volumetric medical images;
                          # the specific modality (mri/ct) is not encoded in
                          # network_data_format and must come from the zoo
                          # entry's own modality field, not this extractor.
        axis_convention=AxisConvention.channels_first,
        n_channels=n_channels,
        spatial_shape=tuple(spatial_shape) if spatial_shape else None,
        dtype=dtype,
        evidence_status=EvidenceStatus.confirmed if confirmed else EvidenceStatus.inferred,
    )


def _extract_output_contract(outputs: dict) -> OutputContract | None:
    if not outputs:
        return None
    _, spec = next(iter(outputs.items()))
    n_channels = spec.get("num_channels")
    if n_channels is None:
        return None
    return OutputContract(
        output_type="segmentation",
        n_classes=n_channels,
        produces_probabilities=False,
    )


def _find_unresolved_transforms(inference: dict | None) -> list[str]:
    if not inference:
        return []
    unresolved = []
    for step in inference.get("preprocessing", []):
        target = step.get("_target_", "")
        class_name = target.rsplit(".", 1)[-1]
        if class_name not in _KNOWN_MONAI_TRANSFORMS:
            unresolved.append(target)
    return unresolved


def extract_monai_contract(
    model_id: str,
    metadata: dict,
    inference: dict | None = None,
) -> ExtractedMONAIContract:
    ndf = metadata.get("network_data_format", {})
    return ExtractedMONAIContract(
        model_id=model_id,
        input_contract=_extract_input_contract(ndf.get("inputs", {})),
        output_contract=_extract_output_contract(ndf.get("outputs", {})),
        unresolved_transforms=_find_unresolved_transforms(inference),
    )


__all__ = ["ExtractedMONAIContract", "extract_monai_contract"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_extractors_monai_bundle.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/extractors/__init__.py src/qortex/neuroai/models/extractors/monai_bundle.py tests/test_neuroai_extractors_monai_bundle.py
git commit -m "feat(neuroai): add offline MONAI bundle metadata extractor"
```

---

### Task 2: Register the 13 remaining P0 MONAI imaging bundles

**Files:**
- Create: `src/qortex/neuroai/models/zoo/monai_imaging.py`
- Modify: `src/qortex/neuroai/models/zoo/__init__.py` (import and call
  `monai_imaging.register_all()` alongside the existing
  `seed_examples.register_all()` call)
- Test: `tests/test_neuroai_zoo_monai_imaging.py`

**Interfaces:**
- Consumes: `ZooEntry`, `ZooEntryType`, `ExecutionMode`, `LicenseInfo`,
  `InteractionContract`, `PromptType` from `zoo/schema.py`; `register` from
  `zoo/registry.py`; `InputContract`, `OutputContract`, `AxisConvention`,
  `EvidenceStatus` from `qortex.neuroai.contracts`.
- Produces: importing `qortex.neuroai.models.zoo` registers 13 additional
  entries (listed below), on top of Phase 1's 3.

Register these 13 entries (spec §12.1, minus `brats_mri_segmentation`
already seeded in Phase 1). For each, only quantitative fields explicitly
confirmed by spec text or an existing `_contracts.py` entry are set — every
other field is `None` with `evidence_status=EvidenceStatus.unknown`, per
this plan's Global Constraints:

| id | modality | task | quantitative facts confirmed by spec/existing registry |
|---|---|---|---|
| `monai.wholeBrainSeg_Large_UNEST_segmentation` | mri | segmentation | "133 structures" (spec §12.1) — put in `notes`, not `n_classes` (ambiguous whether background is included) |
| `monai.vista3d` | ct | segmentation | none confirmed as exact counts; §8.1 cites "127 automatic classes" from the VISTA3D paper — cite in `notes` with the arXiv link as `paper_url`, not `n_classes` |
| `monai.swin_unetr_btcv_segmentation` | ct | segmentation | none |
| `monai.wholeBody_ct_segmentation` | ct | segmentation | **reuse**, not fabrication: the existing legacy entry `wholeBody_ct_segmentation` in `src/qortex/neuroai/models/_contracts.py` (already in the codebase, confirmed) has `n_channels=1`, `intensity_range=(-1024.0, 3071.0)`, `n_classes=105` — copy those exact values here since they describe the same real model |
| `monai.spleen_ct_segmentation` | ct | segmentation | none |
| `monai.multi_organ_segmentation` | ct | segmentation | none |
| `monai.pancreas_ct_dints_segmentation` | ct | segmentation | none |
| `monai.prostate_mri_anatomy` | mri | segmentation | none |
| `monai.renalStructures_CECT_segmentation` | ct | segmentation | none |
| `monai.renalStructures_UNEST_segmentation` | ct | segmentation | none |
| `monai.ventricular_short_axis_3label` | mri | segmentation | "3label" in the model name confirms 3 output classes — `n_classes=3` |
| `monai.valve_landmarks` | mri | landmark_detection | none |
| `monai.retinalOCT_RPD_segmentation` | oct | segmentation | none |

All entries: `provider="monai"`, `execution_mode=ExecutionMode.bundle`,
`maintainer="Project MONAI"`, `docs_url="https://project-monai.github.io/model-zoo.html"`,
`source_url` is each bundle's Hugging Face page
(`https://huggingface.co/MONAI/<bundle_name>`), `license=LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])`
(matching Phase 1's BraTS entry — MONAI Model Zoo bundle licenses are
per-bundle and Qortex has not verified each one), `evidence_status` on the
entry itself is `EvidenceStatus.confirmed` when the entry's modality/task
are stated by the spec (all of them are), `priority="P0"`,
`qortex_status="runnable_after_contract_validation"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_monai_imaging.py
from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries, lookup
from qortex.neuroai.models.zoo.schema import ZooEntryType
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "monai.wholeBrainSeg_Large_UNEST_segmentation",
    "monai.vista3d",
    "monai.swin_unetr_btcv_segmentation",
    "monai.wholeBody_ct_segmentation",
    "monai.spleen_ct_segmentation",
    "monai.multi_organ_segmentation",
    "monai.pancreas_ct_dints_segmentation",
    "monai.prostate_mri_anatomy",
    "monai.renalStructures_CECT_segmentation",
    "monai.renalStructures_UNEST_segmentation",
    "monai.ventricular_short_axis_3label",
    "monai.valve_landmarks",
    "monai.retinalOCT_RPD_segmentation",
}


def test_all_13_monai_imaging_entries_registered():
    registered_ids = {e.id for e in list_entries(provider="monai")}
    # brats_mri_segmentation (Phase 1 seed) + these 13 = 14 monai-provider entries
    assert _EXPECTED_IDS.issubset(registered_ids)
    assert len(registered_ids) == 14


def test_monai_imaging_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []


def test_wholebody_ct_reuses_confirmed_legacy_contract():
    entry = lookup("monai.wholeBody_ct_segmentation")
    assert entry is not None
    assert entry.input_contract.n_channels == 1
    assert entry.input_contract.intensity_range == (-1024.0, 3071.0)
    assert entry.output_contract.n_classes == 105


def test_ventricular_short_axis_has_confirmed_3_classes():
    entry = lookup("monai.ventricular_short_axis_3label")
    assert entry.output_contract.n_classes == 3


def test_entries_without_confirmed_shape_leave_fields_unknown():
    entry = lookup("monai.swin_unetr_btcv_segmentation")
    assert entry.input_contract.n_channels is None
    assert entry.input_contract.evidence_status.value == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_monai_imaging.py -v`
Expected: FAIL — `ModuleNotFoundError` / 0 of 14 entries registered

- [ ] **Step 3: Implement `zoo/monai_imaging.py`**

```python
# src/qortex/neuroai/models/zoo/monai_imaging.py
"""P0 MONAI imaging bundle entries (design spec section 12.1), excluding
brats_mri_segmentation which was seeded in Phase 1
(zoo/seed_examples.py). Every quantitative field is only set when
confirmed by the design spec's own text or reused from an existing
confirmed entry in qortex.neuroai.models._contracts — everything else is
left unknown rather than guessed.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, OutputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, ZooEntry, ZooEntryType

_MAINTAINER = "Project MONAI"
_CATALOG_URL = "https://project-monai.github.io/model-zoo.html"


def _unknown_input(modality: str) -> InputContract:
    return InputContract(
        modality=modality,
        axis_convention=AxisConvention.channels_first,
        evidence_status=EvidenceStatus.unknown,
    )


def _unknown_output(output_type: str = "segmentation") -> OutputContract:
    return OutputContract(output_type=output_type, produces_probabilities=False)


def _unlicensed() -> LicenseInfo:
    return LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"])


def _hub_url(bundle_name: str) -> str:
    return f"https://huggingface.co/MONAI/{bundle_name}"


def register_all() -> None:
    register(ZooEntry(
        id="monai.wholeBrainSeg_Large_UNEST_segmentation",
        display_name="Whole Brain Segmentation (Large UNEST)",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("wholeBrainSeg_Large_UNEST_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["segmentation", "whole_brain_segmentation"],
        input_contract=_unknown_input("mri"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["T1w whole-brain structural segmentation with 133 structures per design spec section 12.1; exact class-index count not confirmed offline."],
    ))

    register(ZooEntry(
        id="monai.vista3d",
        display_name="VISTA3D",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("vista3d"),
        paper_url="https://arxiv.org/abs/2406.05285",
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation", "foundation_segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=[
            "Foundation-style 3D CT segmentation and annotation.",
            "VISTA3D paper (arXiv:2406.05285) reports 127 automatic classes; "
            "not encoded as n_classes here since exact figure is unconfirmed offline.",
            "Registered here as entry_type=model (segmentation only). Phase 5 "
            "(promptable segmentation) upgrades this entry to entry_type="
            "promptable_model with a populated InteractionContract once the "
            "VISTA3D prompt adapter lands — see plan Task list Phase 5.",
        ],
    ))

    register(ZooEntry(
        id="monai.swin_unetr_btcv_segmentation",
        display_name="Swin UNETR BTCV Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("swin_unetr_btcv_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["Transformer CT segmentation baseline per design spec section 12.1."],
    ))

    # Reuses confirmed contract data from the existing legacy
    # qortex.neuroai.models._contracts entry "wholeBody_ct_segmentation"
    # (n_channels, intensity_range, n_classes) describing the same real model.
    register(ZooEntry(
        id="monai.wholeBody_ct_segmentation",
        display_name="Whole Body CT Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("wholeBody_ct_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation", "whole_body_segmentation"],
        input_contract=InputContract(
            modality="ct",
            axis_convention=AxisConvention.channels_first,
            n_channels=1,
            intensity_range=(-1024.0, 3071.0),
            dtype="float32",
            evidence_status=EvidenceStatus.confirmed,
        ),
        output_contract=OutputContract(
            output_type="segmentation",
            n_classes=105,
            produces_probabilities=False,
        ),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["Contract reused from the existing curated qortex.neuroai.models._contracts entry for the same model."],
    ))

    register(ZooEntry(
        id="monai.spleen_ct_segmentation",
        display_name="Spleen CT Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("spleen_ct_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
        notes=["Small MONAI bundle useful for tests/demos per design spec section 12.1."],
    ))

    register(ZooEntry(
        id="monai.multi_organ_segmentation",
        display_name="Multi-Organ Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("multi_organ_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.pancreas_ct_dints_segmentation",
        display_name="Pancreas CT DiNTS Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("pancreas_ct_dints_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.prostate_mri_anatomy",
        display_name="Prostate MRI Anatomy",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("prostate_mri_anatomy"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["segmentation"],
        input_contract=_unknown_input("mri"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.renalStructures_CECT_segmentation",
        display_name="Renal Structures CECT Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("renalStructures_CECT_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.renalStructures_UNEST_segmentation",
        display_name="Renal Structures UNEST Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("renalStructures_UNEST_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["ct"],
        task=["segmentation"],
        input_contract=_unknown_input("ct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    # "3label" in the bundle name confirms exactly 3 output classes.
    register(ZooEntry(
        id="monai.ventricular_short_axis_3label",
        display_name="Ventricular Short Axis (3-label)",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("ventricular_short_axis_3label"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["segmentation", "cardiac_segmentation"],
        input_contract=_unknown_input("mri"),
        output_contract=OutputContract(
            output_type="segmentation",
            n_classes=3,
            produces_probabilities=False,
        ),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.valve_landmarks",
        display_name="Valve Landmarks",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("valve_landmarks"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["mri"],
        task=["landmark_detection"],
        input_contract=_unknown_input("mri"),
        output_contract=OutputContract(output_type="landmark_detection", produces_probabilities=False),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))

    register(ZooEntry(
        id="monai.retinalOCT_RPD_segmentation",
        display_name="Retinal OCT RPD Segmentation",
        entry_type=ZooEntryType.model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url("retinalOCT_RPD_segmentation"),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=["oct"],
        task=["segmentation"],
        input_contract=_unknown_input("oct"),
        output_contract=_unknown_output(),
        license=_unlicensed(),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P0",
    ))


__all__ = ["register_all"]
```

- [ ] **Step 4: Wire into `zoo/__init__.py`**

Read the current file first — it already has a `_seed_examples.register_all()`
call from Phase 1. Add the new import and call alongside it, do not remove
the existing one:

```python
from qortex.neuroai.models.zoo import monai_imaging as _monai_imaging
```

and after the existing `_seed_examples.register_all()` line:

```python
_monai_imaging.register_all()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_monai_imaging.py -v`
Expected: PASS (5 tests)

Then run the full zoo suite to confirm no regressions:
Run: `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_monai_bundle.py -v`
Expected: all PASS

- [ ] **Step 6: Update the spec's progress checklist**

In `docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md` §0,
under Phase 2, check off "P0 MONAI imaging entries (see §12.1 — list grows
below as each lands)". Append the 13 new entries to "Registry entries
implemented so far", one line each (id — provider `monai`, entry_type
`model`, Phase 2).

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/models/zoo/monai_imaging.py src/qortex/neuroai/models/zoo/__init__.py tests/test_neuroai_zoo_monai_imaging.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): register the 13 remaining P0 MONAI imaging bundles"
```

---

### Task 3: Register the 7 P0 MONAI generative bundles

**Files:**
- Create: `src/qortex/neuroai/models/zoo/monai_generative.py`
- Modify: `src/qortex/neuroai/models/zoo/__init__.py` (add
  `monai_generative.register_all()` alongside the existing calls)
- Test: `tests/test_neuroai_zoo_monai_generative.py`

**Interfaces:**
- Consumes: same as Task 2.
- Produces: 7 additional entries with `output_contract.output_type ==
  "image_generation"` and `produces_probabilities=False`, per spec §12.5's
  convention (generative models must never be tagged as
  segmentation/classification).

Entries (spec §12.5), all `provider="monai"`,
`execution_mode=ExecutionMode.bundle`, `maintainer="Project MONAI"`,
`entry_type=ZooEntryType.generative_model`, `task=["image_generation",
"synthesis"]`, `license=LicenseInfo(evidence_status=EvidenceStatus.unknown,
notes=["requires manual check"])`, `priority="P1"` (spec labels this whole
section P1), `qortex_status="runnable_after_contract_validation"`, and a
`notes` entry stating `"clinical_use=prohibited, research_use=allowed"` per
spec §12.5 (full `artifact_policy` structured fields are Phase 6 scope —
see design spec §18, not modeled on `ZooEntry` yet):

| id | modality |
|---|---|
| `monai.brain_image_synthesis_latent_diffusion_model` | mri |
| `monai.brats_mri_generative_diffusion` | mri |
| `monai.brats_mri_axial_slices_generative_diffusion` | mri |
| `monai.maisi_ct_generative` | ct |
| `monai.cxr_image_synthesis_latent_diffusion_model` | xray |
| `monai.mednist_ddpm` | mixed |
| `monai.mednist_gan` | mixed |

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_monai_generative.py
from __future__ import annotations

from qortex.neuroai.models.zoo.registry import list_entries
from qortex.neuroai.models.zoo.schema import ZooEntryType
from qortex.neuroai.models.zoo.validate import validate_registry

_EXPECTED_IDS = {
    "monai.brain_image_synthesis_latent_diffusion_model",
    "monai.brats_mri_generative_diffusion",
    "monai.brats_mri_axial_slices_generative_diffusion",
    "monai.maisi_ct_generative",
    "monai.cxr_image_synthesis_latent_diffusion_model",
    "monai.mednist_ddpm",
    "monai.mednist_gan",
}


def test_all_7_generative_entries_registered():
    entries = list_entries(entry_type=ZooEntryType.generative_model)
    assert {e.id for e in entries} == _EXPECTED_IDS


def test_generative_entries_are_never_tagged_as_segmentation_or_classification():
    entries = list_entries(entry_type=ZooEntryType.generative_model)
    for entry in entries:
        assert entry.output_contract.output_type == "image_generation"
        assert entry.output_contract.produces_probabilities is False


def test_generative_entries_pass_offline_validation():
    issues = validate_registry()
    relevant = [i for i in issues if i.entry_id in _EXPECTED_IDS]
    assert relevant == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_monai_generative.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `zoo/monai_generative.py`**

```python
# src/qortex/neuroai/models/zoo/monai_generative.py
"""P0 MONAI generative bundle entries (design spec section 12.5).

Generative models are never tagged as segmentation/classification —
output_type is always "image_generation" and produces_probabilities is
always False, per the spec's explicit invariant that a generative model
must not be mistaken for a diagnostic one.
"""

from __future__ import annotations

from qortex.neuroai.contracts import AxisConvention, EvidenceStatus, InputContract, OutputContract
from qortex.neuroai.models.zoo.registry import register
from qortex.neuroai.models.zoo.schema import ExecutionMode, LicenseInfo, ZooEntry, ZooEntryType

_MAINTAINER = "Project MONAI"
_CATALOG_URL = "https://project-monai.github.io/model-zoo.html"
_CLINICAL_USE_NOTE = "clinical_use=prohibited, research_use=allowed (design spec section 12.5)."


def _hub_url(bundle_name: str) -> str:
    return f"https://huggingface.co/MONAI/{bundle_name}"


def _generative_entry(bundle_name: str, display_name: str, modality: str, extra_notes: list[str] | None = None) -> ZooEntry:
    return ZooEntry(
        id=f"monai.{bundle_name}",
        display_name=display_name,
        entry_type=ZooEntryType.generative_model,
        provider="monai",
        execution_mode=ExecutionMode.bundle,
        source_url=_hub_url(bundle_name),
        docs_url=_CATALOG_URL,
        maintainer=_MAINTAINER,
        modality=[modality],
        task=["image_generation", "synthesis"],
        input_contract=InputContract(
            modality=modality,
            axis_convention=AxisConvention.channels_first,
            evidence_status=EvidenceStatus.unknown,
        ),
        output_contract=OutputContract(
            output_type="image_generation",
            produces_probabilities=False,
        ),
        license=LicenseInfo(evidence_status=EvidenceStatus.unknown, notes=["requires manual check"]),
        evidence_status=EvidenceStatus.confirmed,
        qortex_status="runnable_after_contract_validation",
        priority="P1",
        notes=[_CLINICAL_USE_NOTE] + (extra_notes or []),
    )


def register_all() -> None:
    register(_generative_entry(
        "brain_image_synthesis_latent_diffusion_model",
        "Brain Image Synthesis (Latent Diffusion)",
        "mri",
    ))
    register(_generative_entry(
        "brats_mri_generative_diffusion",
        "BraTS MRI Generative Diffusion",
        "mri",
    ))
    register(_generative_entry(
        "brats_mri_axial_slices_generative_diffusion",
        "BraTS MRI Axial Slices Generative Diffusion",
        "mri",
    ))
    register(_generative_entry(
        "maisi_ct_generative",
        "MAISI CT Generative",
        "ct",
        extra_notes=[
            "MAISI: diffusion-based synthetic 3D CT with anatomical control, "
            "up to 512x512x768 voxels conditioned on organ segmentations "
            "(design spec section 12.5).",
        ],
    ))
    register(_generative_entry(
        "cxr_image_synthesis_latent_diffusion_model",
        "Chest X-Ray Image Synthesis (Latent Diffusion)",
        "xray",
    ))
    register(_generative_entry(
        "mednist_ddpm",
        "MedNIST DDPM",
        "mixed",
    ))
    register(_generative_entry(
        "mednist_gan",
        "MedNIST GAN",
        "mixed",
    ))


__all__ = ["register_all"]
```

- [ ] **Step 4: Wire into `zoo/__init__.py`**

Add `from qortex.neuroai.models.zoo import monai_generative as
_monai_generative` and `_monai_generative.register_all()` alongside the
existing calls, same pattern as Task 2 Step 4.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_monai_generative.py -v`
Expected: PASS (3 tests)

Then the full zoo suite: `python -m pytest tests/test_neuroai_zoo_*.py -v`
Expected: all PASS

- [ ] **Step 6: Update the spec's progress checklist**

Check off "MONAI generative entries (§12.5)" under Phase 2 in §0. Append
the 7 new entries to "Registry entries implemented so far".

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/models/zoo/monai_generative.py src/qortex/neuroai/models/zoo/__init__.py tests/test_neuroai_zoo_monai_generative.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): register the 7 P0 MONAI generative bundles"
```

---

### Task 4: Model cache / provenance layer

**Files:**
- Create: `src/qortex/neuroai/models/cache.py`
- Test: `tests/test_neuroai_model_cache.py`

**Interfaces:**
- Consumes: nothing from the zoo package — this is a standalone provenance
  layer any adapter can call into later.
- Produces (used by Task 5's CLI wiring, and future phases):
  - `@dataclass CacheEntry` — fields `model_id: str`, `provider: str`,
    `local_path: str`, `size_bytes: int`, `sha256: str | None`,
    `downloaded_at: str`, `source_url: str | None`.
  - `class ModelCache` — `__init__(self, cache_dir: Path | str | None =
    None)` (default: `Path(os.environ.get("QORTEX_CACHE_DIR",
    Path.home() / ".qortex" / "model_cache"))`), `is_cached(self, model_id:
    str) -> bool`, `record(self, entry: CacheEntry) -> None` (writes/updates
    `manifest.json`), `lookup(self, model_id: str) -> CacheEntry | None`,
    `list_cached(self) -> list[CacheEntry]`, `disk_usage(self) -> int`
    (sum of `size_bytes` across all entries), `remove(self, model_id: str)
    -> None` (removes from manifest only — does not delete the backend's
    own cached files, since Qortex does not own that storage).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_model_cache.py
from __future__ import annotations

from pathlib import Path

import pytest

from qortex.neuroai.models.cache import CacheEntry, ModelCache


def _entry(model_id: str = "monai.brats_mri_segmentation") -> CacheEntry:
    return CacheEntry(
        model_id=model_id,
        provider="monai",
        local_path="/tmp/fake/bundle",
        size_bytes=1024,
        sha256="deadbeef",
        downloaded_at="2026-07-09T00:00:00Z",
        source_url="https://huggingface.co/MONAI/brats_mri_segmentation",
    )


def test_record_and_is_cached(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    assert cache.is_cached("monai.brats_mri_segmentation") is False

    cache.record(_entry())

    assert cache.is_cached("monai.brats_mri_segmentation") is True


def test_lookup_returns_recorded_entry(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry())

    found = cache.lookup("monai.brats_mri_segmentation")

    assert found is not None
    assert found.sha256 == "deadbeef"
    assert found.size_bytes == 1024


def test_lookup_missing_returns_none(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    assert cache.lookup("nonexistent") is None


def test_manifest_persists_across_instances(tmp_path: Path):
    ModelCache(cache_dir=tmp_path).record(_entry())

    reopened = ModelCache(cache_dir=tmp_path)

    assert reopened.is_cached("monai.brats_mri_segmentation") is True
    assert (tmp_path / "manifest.json").exists()


def test_list_cached_and_disk_usage(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry("monai.brats_mri_segmentation"))
    cache.record(_entry("braindecode.EEGNet"))

    listed = cache.list_cached()

    assert {e.model_id for e in listed} == {"monai.brats_mri_segmentation", "braindecode.EEGNet"}
    assert cache.disk_usage() == 2048


def test_record_overwrites_existing_entry_for_same_id(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry())
    updated = _entry()
    updated.size_bytes = 9999
    cache.record(updated)

    assert cache.disk_usage() == 9999
    assert len(cache.list_cached()) == 1


def test_remove_drops_entry_from_manifest(tmp_path: Path):
    cache = ModelCache(cache_dir=tmp_path)
    cache.record(_entry())

    cache.remove("monai.brats_mri_segmentation")

    assert cache.is_cached("monai.brats_mri_segmentation") is False


def test_default_cache_dir_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QORTEX_CACHE_DIR", str(tmp_path))
    cache = ModelCache()

    cache.record(_entry())

    assert (tmp_path / "manifest.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_model_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `cache.py`**

```python
# src/qortex/neuroai/models/cache.py
"""Model weight cache / provenance layer.

This is a manifest on top of each backend's own download cache (HF hub
cache, MONAI bundle directory, torch hub cache) — NOT a downloader. Qortex
records what it knows was downloaded and its checksum; it never fetches
weights itself. See docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
section 15.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CacheEntry:
    model_id: str
    provider: str
    local_path: str
    size_bytes: int
    sha256: str | None
    downloaded_at: str
    source_url: str | None = None


def _default_cache_dir() -> Path:
    override = os.environ.get("QORTEX_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".qortex" / "model_cache"


class ModelCache:
    """JSON-manifest-backed provenance record of downloaded model weights."""

    SCHEMA_VERSION = "1.0"

    def __init__(self, cache_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
        self.manifest_path = self.cache_dir / "manifest.json"

    def _load(self) -> dict[str, dict]:
        if not self.manifest_path.exists():
            return {}
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {e["model_id"]: e for e in data.get("entries", [])}

    def _save(self, entries: dict[str, dict]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": self.SCHEMA_VERSION, "entries": list(entries.values())}
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_cached(self, model_id: str) -> bool:
        return model_id in self._load()

    def lookup(self, model_id: str) -> CacheEntry | None:
        raw = self._load().get(model_id)
        return CacheEntry(**raw) if raw else None

    def record(self, entry: CacheEntry) -> None:
        entries = self._load()
        entries[entry.model_id] = asdict(entry)
        self._save(entries)

    def remove(self, model_id: str) -> None:
        entries = self._load()
        entries.pop(model_id, None)
        self._save(entries)

    def list_cached(self) -> list[CacheEntry]:
        return [CacheEntry(**raw) for raw in self._load().values()]

    def disk_usage(self) -> int:
        return sum(e.size_bytes for e in self.list_cached())


__all__ = ["CacheEntry", "ModelCache"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_model_cache.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/qortex/neuroai/models/cache.py tests/test_neuroai_model_cache.py
git commit -m "feat(neuroai): add ModelCache provenance manifest layer"
```

---

### Task 5: Bridge the zoo registry into `suggest-models`

**Files:**
- Create: `src/qortex/neuroai/models/zoo/bridge.py`
- Modify: `src/qortex/cli/app.py` (call the bridge once at the top of
  `neuroai_suggest_models`, before it builds its candidate list — see exact
  location below)
- Test: `tests/test_neuroai_zoo_bridge.py`

**Interfaces:**
- Consumes: `list_entries` from `zoo/registry.py`; `ModelContractEntry`,
  `register`, `lookup` from `qortex.neuroai.models._contracts` (existing,
  read-only usage — never edit that file).
- Produces: `sync_into_legacy_registry() -> int` — converts every `ZooEntry`
  that has both `input_contract` and `output_contract` populated into a
  `ModelContractEntry` and registers it into the legacy `_contracts`
  registry, **idempotently** (skips ids already present there, so calling
  it more than once — e.g. once per CLI invocation — never raises
  `ValueError` on duplicate registration). Returns the count of entries
  newly synced. Entries without both contracts (e.g.
  `external.totalsegmentator`, which has no tensor `input_contract`) are
  skipped — `suggest-models` only makes sense for in-process tensor models.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuroai_zoo_bridge.py
from __future__ import annotations

from qortex.neuroai.models._contracts import lookup as legacy_lookup
from qortex.neuroai.models.zoo.bridge import sync_into_legacy_registry


def test_sync_registers_contracted_zoo_entries_into_legacy_registry():
    synced_count = sync_into_legacy_registry()

    assert synced_count > 0
    # braindecode.EEGNet has both input_contract and output_contract (Phase 1 seed)
    assert legacy_lookup("braindecode.EEGNet") is not None


def test_sync_skips_entries_without_both_contracts():
    sync_into_legacy_registry()

    # external.totalsegmentator has no input_contract (external CLI engine)
    assert legacy_lookup("external.totalsegmentator") is None


def test_sync_is_idempotent():
    first = sync_into_legacy_registry()
    second = sync_into_legacy_registry()

    assert second == 0
    assert first > 0


def test_synced_entry_preserves_original_contracts():
    sync_into_legacy_registry()

    entry = legacy_lookup("braindecode.EEGNet")

    assert entry.input_contract.modality == "eeg"
    assert entry.provider == "braindecode"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_neuroai_zoo_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `zoo/bridge.py`**

```python
# src/qortex/neuroai/models/zoo/bridge.py
"""Bridge from the zoo registry into the legacy curated contract registry.

qortex.neuroai.models._contracts.ModelContractEntry predates the zoo
registry and is what suggest-models actually reads. This module is a
one-way, additive sync: it never modifies _contracts.py, only calls its
public register()/lookup() functions, so suggest-models sees zoo entries
without either registry needing to know about the other's internals.

Entries without both an input_contract and output_contract (external
engines, generative models with no classification/segmentation output) are
skipped — ModelContractEntry requires both, and suggest-models's ranking
logic assumes output_contract.output_type is always present.
"""

from __future__ import annotations

from qortex.neuroai.models import _contracts
from qortex.neuroai.models.zoo.registry import list_entries


def sync_into_legacy_registry() -> int:
    """Register every fully-contracted ZooEntry into the legacy registry.

    Idempotent: entries already present in the legacy registry (by id) are
    skipped, so this is safe to call on every suggest-models invocation.

    Returns
    -------
    int
        Number of entries newly registered by this call.
    """
    synced = 0
    for entry in list_entries():
        if entry.input_contract is None or entry.output_contract is None:
            continue
        if _contracts.lookup(entry.id) is not None:
            continue
        _contracts.register(_contracts.ModelContractEntry(
            model_id=entry.id,
            provider=entry.provider,
            input_contract=entry.input_contract,
            output_contract=entry.output_contract,
            estimated_memory_mb=None,
            notes=entry.display_name,
        ))
        synced += 1
    return synced


__all__ = ["sync_into_legacy_registry"]
```

- [ ] **Step 4: Wire into `suggest-models`**

In `src/qortex/cli/app.py`, find `def neuroai_suggest_models(` (currently
around line 2108). Immediately after its docstring and the initial `try:
from qortex.neuroai.sources._registry import make_source_adapter ...`
import block (i.e. right before the `# 1. Probe source` comment, currently
around line 2135), insert:

```python
    # Pull in the model zoo's fully-contracted entries so suggest-models
    # ranks them alongside the original curated registry.
    from qortex.neuroai.models import zoo as _zoo  # noqa: F401  (triggers zoo registration)
    from qortex.neuroai.models.zoo.bridge import sync_into_legacy_registry
    sync_into_legacy_registry()

```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_neuroai_zoo_bridge.py -v`
Expected: PASS (4 tests)

Then confirm `suggest-models` actually surfaces a zoo entry:

Run: `python -c "
from typer.testing import CliRunner
from qortex.cli.app import app
result = CliRunner().invoke(app, ['neuroai', 'suggest-models', 'nonexistent.edf', '--task', 'classification', '--modality', 'eeg'])
print(result.output)
"`

This will fail to probe `nonexistent.edf` (expected — no test fixture file
exists), but the traceback/output should show the command reaching the
model-search stage, not failing at import. If a real `.edf` fixture already
exists in the test suite, prefer using that path instead so the full
ranked-list output can be inspected — check `tests/test_neuroai_pipeline.py`
or similar for an existing fixture path convention.

- [ ] **Step 6: Update the spec's progress checklist**

Check off "Compatibility bridge into `suggest-models`" under Phase 2 in §0.
Every Phase 2 checklist item is now checked.

- [ ] **Step 7: Commit**

```bash
git add src/qortex/neuroai/models/zoo/bridge.py src/qortex/cli/app.py tests/test_neuroai_zoo_bridge.py docs/superpowers/specs/2026-07-09-model-zoo-expansion-design.md
git commit -m "feat(neuroai): bridge zoo registry into suggest-models

Completes Phase 2 (MONAI integration) of the model zoo expansion."
```

---

## Phase 2 exit criteria

- [ ] `python -m pytest tests/test_neuroai_zoo_*.py tests/test_neuroai_extractors_monai_bundle.py tests/test_neuroai_model_cache.py -v` — all green.
- [ ] `qortex neuroai zoo list --provider monai` shows 14 entries (13 new + BraTS from Phase 1).
- [ ] `qortex neuroai zoo list --entry-type generative_model` shows 7 entries.
- [ ] `qortex neuroai zoo validate` reports 0 issues across the full registry.
- [ ] `suggest-models` picks up zoo entries via the bridge without any
      `_contracts.py` edits.
- [ ] Spec §0 checklist fully checked for Phase 2, "Registry entries
      implemented so far" lists all 24 entries (3 Phase 1 + 13 + 7 + 1
      already-counted... i.e. 3 + 13 + 7 = 23 total).
- [ ] No changes to `_base.py`, `_contracts.py`, `_registry.py`,
      `contracts.py`, `spec.py`, or any existing adapter file.

Once this phase is merged, write
`docs/superpowers/plans/<date>-model-zoo-phase3-braindecode-expansion.md`
covering the Braindecode extractor and the P0 EEG entries from spec §12.3.
