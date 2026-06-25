from qortex.manifest.bids import (
    BIDS_DATATYPES,
    BIDS_ENTITY_KEYS,
    extract_datatype,
    infer_modality,
    parse_entities,
    parse_filename,
    sidecar_group_key,
)
from qortex.manifest.diff import ManifestDiff, diff_manifests
from qortex.manifest.graph import ManifestGraph
from qortex.manifest.sidecar import find_events_files, group_sidecars

__all__ = [
    "ManifestBuilder",
    "load_manifest",
    "save_manifest",
    "ManifestDiff",
    "diff_manifests",
    "ManifestGraph",
    "group_sidecars",
    "find_events_files",
    "parse_filename",
    "parse_entities",
    "extract_datatype",
    "infer_modality",
    "sidecar_group_key",
    "BIDS_DATATYPES",
    "BIDS_ENTITY_KEYS",
]


def __getattr__(name: str):
    if name in {"ManifestBuilder", "load_manifest", "save_manifest"}:
        from qortex.manifest.builder import ManifestBuilder, load_manifest, save_manifest

        return {
            "ManifestBuilder": ManifestBuilder,
            "load_manifest": load_manifest,
            "save_manifest": save_manifest,
        }[name]
    raise AttributeError(name)
