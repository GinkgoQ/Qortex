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


class ProvenancedValue(BaseModel if _PYDANTIC else object):
    value: Any = None
    evidence_status: EvidenceStatus = EvidenceStatus.unknown
    source_url: str | None = None
    source_field: str | None = None
    checked_at: str | None = None
    note: str | None = None

    if not _PYDANTIC:
        def __init__(self, **kwargs):
            self.value = None
            self.evidence_status = EvidenceStatus.unknown
            self.source_url = None
            self.source_field = None
            self.checked_at = None
            self.note = None
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
            self.name = None
            self.url = None
            self.commercial_use = None
            self.redistribution_allowed = None
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
            self.prompt_coordinate_frame = None
            self.max_points = None
            self.max_boxes = None
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
            self.list_capabilities_command = None
            self.output_manifest_supported = False
            self.geometry_preservation_known = None
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
