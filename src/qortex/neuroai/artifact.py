"""ArtifactWriter — full artifact directory writer for Qortex NeuroAI pipeline runs.

Implements AGENT.md §15 artifact directory layout.  Every completed pipeline run
should call ``ArtifactWriter.write()`` to produce a self-contained, auditable
artifact directory.

Directory layout::

    <artifact_dir>/
        artifact_manifest.json   — index of all files with SHA-256 + size
        artifact_contract.json   — formal output contract
        provenance.json          — full lineage record
        warnings.json            — all warnings and unknowns
        pipeline.yaml            — original pipeline spec (YAML)
        compatibility_report.json
        preprocess_plan.json
        runtime_report.json
        latency_report.json
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _get_qortex_version() -> str:
    try:
        from qortex._version import __version__
        return __version__
    except Exception:
        return "unknown"


def _to_serialisable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serialisable types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serialisable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _to_serialisable(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _to_serialisable(obj.__dict__)
    if hasattr(obj, "value"):          # Enum
        return obj.value
    return str(obj)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactWriter:
    """Writes a complete Qortex artifact directory for a pipeline run.

    The artifact directory contains:

    * ``artifact_manifest.json``  — index of all artifact files
    * ``artifact_contract.json``  — formal output contract
    * ``provenance.json``         — full lineage record
    * ``warnings.json``           — all warnings and unknowns
    * ``pipeline.yaml``           — original pipeline spec
    * ``compatibility_report.json``
    * ``preprocess_plan.json``
    * ``runtime_report.json``
    * ``latency_report.json``

    Parameters
    ----------
    output_dir:
        Directory to write the artifact into (created if absent).
    pipeline_ref:
        Optional short reference string (e.g. first 12 chars of spec hash)
        to embed in provenance for traceability.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        pipeline_ref: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.pipeline_ref = pipeline_ref

    # ── Public API ────────────────────────────────────────────────────────────

    def write(
        self,
        *,
        spec: Any,
        compat_report: Any,
        preprocess_plan: Any,
        run_report: Any,
        source_profile: Any,
        model_profile: Any,
    ) -> Path:
        """Write complete artifact directory.

        Parameters
        ----------
        spec:
            ``PipelineSpec`` instance.
        compat_report:
            ``CompatibilityReport`` instance.
        preprocess_plan:
            ``PreprocessPlan`` instance.
        run_report:
            ``PipelineRunReport`` instance.
        source_profile:
            ``SourceProfile`` instance.
        model_profile:
            ``ModelProfile`` instance.

        Returns
        -------
        Path
            Path to ``artifact_manifest.json``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        log.info("ArtifactWriter: writing artifact directory to %s", self.output_dir)

        self._write_provenance(spec, source_profile, model_profile, run_report)
        self._write_compatibility_report(compat_report)
        self._write_preprocess_plan(preprocess_plan)
        self._write_runtime_report(run_report)
        self._write_latency_report(run_report)
        self._write_warnings(compat_report, preprocess_plan, run_report)
        self._write_pipeline_yaml(spec)
        self._write_artifact_contract(run_report)
        manifest_path = self._write_manifest()
        log.info("ArtifactWriter: done — manifest at %s", manifest_path)
        return manifest_path

    # ── Private writers ───────────────────────────────────────────────────────

    def _write_provenance(
        self,
        spec: Any,
        source_profile: Any,
        model_profile: Any,
        run_report: Any,
    ) -> None:
        """Write provenance.json with full lineage."""
        runtime = getattr(spec, "runtime", None)
        data: dict[str, Any] = {
            "qortex_version": _get_qortex_version(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_spec_hash": spec.content_hash() if hasattr(spec, "content_hash") else None,
            "pipeline_name": getattr(spec, "name", None),
            "source": {
                "source_id": getattr(source_profile, "source_id", None),
                "modality": _to_serialisable(getattr(source_profile, "modality", None)),
                "n_channels": getattr(source_profile, "n_channels", None),
                "sampling_rate_hz": getattr(source_profile, "sampling_rate_hz", None),
                "path": getattr(source_profile, "path", None),
            },
            "model": {
                "model_id": getattr(model_profile, "model_id", None),
                "provider": getattr(model_profile, "provider", None),
                "task": getattr(model_profile, "task", None),
                "revision": getattr(model_profile, "revision", None),
                "model_hash": getattr(model_profile, "model_hash", None),
            },
            "runtime": {
                "device": getattr(runtime, "device", None) if runtime else None,
                "fp16": getattr(runtime, "fp16", None) if runtime else None,
                "latency_budget_ms": getattr(runtime, "latency_budget_ms", None) if runtime else None,
            },
            "outputs_written": getattr(run_report, "n_outputs_written", 0),
            "windows_processed": getattr(run_report, "n_windows_processed", 0),
            "errors": getattr(run_report, "errors", []),
            "pipeline_ref": self.pipeline_ref,
        }
        self._write_json("provenance.json", data)

    def _write_compatibility_report(self, report: Any) -> None:
        """Write compatibility_report.json."""
        data = _to_serialisable(report)
        self._write_json("compatibility_report.json", data)

    def _write_preprocess_plan(self, plan: Any) -> None:
        """Write preprocess_plan.json."""
        data = _to_serialisable(plan)
        self._write_json("preprocess_plan.json", data)

    def _write_runtime_report(self, run_report: Any) -> None:
        """Write runtime_report.json — high-level run summary."""
        data: dict[str, Any] = {
            "success": getattr(run_report, "success", None),
            "n_outputs_written": getattr(run_report, "n_outputs_written", 0),
            "n_windows_processed": getattr(run_report, "n_windows_processed", 0),
            "outputs": _to_serialisable(getattr(run_report, "outputs", [])),
            "errors": getattr(run_report, "errors", []),
            "warnings": _to_serialisable(getattr(run_report, "warnings", [])),
        }
        self._write_json("runtime_report.json", data)

    def _write_latency_report(self, run_report: Any) -> None:
        """Write latency_report.json."""
        latency = getattr(run_report, "latency_report", None)
        if latency is not None:
            data = _to_serialisable(latency)
        else:
            data = {"status": "UNKNOWN", "note": "No latency report available"}
        self._write_json("latency_report.json", data)

    def _write_warnings(
        self,
        compat_report: Any,
        preprocess_plan: Any,
        run_report: Any,
    ) -> None:
        """Write warnings.json — consolidated list of all warnings and unknowns."""
        warnings: list[dict] = []

        for source_name, obj in (
            ("compatibility_report", compat_report),
            ("preprocess_plan", preprocess_plan),
            ("run_report", run_report),
        ):
            for w in getattr(obj, "warnings", []) or []:
                item = _to_serialisable(w)
                if isinstance(item, dict):
                    item.setdefault("source", source_name)
                else:
                    item = {"message": str(item), "source": source_name}
                warnings.append(item)

        unknowns: list[dict] = []
        for source_name, obj in (
            ("compatibility_report", compat_report),
            ("preprocess_plan", preprocess_plan),
        ):
            for u in getattr(obj, "unknowns", []) or []:
                unknowns.append({"unknown": str(u), "source": source_name})

        blockers: list[dict] = []
        for b in getattr(compat_report, "blockers", []) or []:
            item = _to_serialisable(b)
            if not isinstance(item, dict):
                item = {"message": str(item)}
            blockers.append(item)

        data: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "n_warnings": len(warnings),
            "n_unknowns": len(unknowns),
            "n_blockers": len(blockers),
            "warnings": warnings,
            "unknowns": unknowns,
            "blockers": blockers,
        }
        self._write_json("warnings.json", data)

    def _write_pipeline_yaml(self, spec: Any) -> None:
        """Write pipeline.yaml — the original pipeline spec."""
        yaml_path = self.output_dir / "pipeline.yaml"
        try:
            if hasattr(spec, "to_yaml"):
                yaml_path.write_text(spec.to_yaml(), encoding="utf-8")
                return
        except Exception as exc:
            log.debug("spec.to_yaml() failed: %s", exc)

        try:
            import yaml
            if hasattr(spec, "to_dict"):
                d = spec.to_dict()
            elif hasattr(spec, "__dict__"):
                d = _to_serialisable(spec.__dict__)
            else:
                d = {}
            yaml_path.write_text(yaml.dump(d, default_flow_style=False), encoding="utf-8")
        except ImportError:
            # Fall back to JSON-in-yaml
            d = _to_serialisable(spec.__dict__) if hasattr(spec, "__dict__") else {}
            yaml_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        except Exception as exc:
            log.warning("Could not write pipeline.yaml: %s", exc)
            yaml_path.write_text(f"# Error serialising spec: {exc}\n", encoding="utf-8")

    def _write_artifact_contract(self, run_report: Any) -> None:
        """Write artifact_contract.json — formal output contract."""
        contract = getattr(run_report, "artifact_contract", None)
        if contract is not None:
            data = _to_serialisable(contract)
        else:
            data = {
                "qortex_version": _get_qortex_version(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "success": getattr(run_report, "success", None),
                "n_outputs_written": getattr(run_report, "n_outputs_written", 0),
                "note": "No formal ArtifactContract attached to run report",
            }
        self._write_json("artifact_contract.json", data)

    def _write_manifest(self) -> Path:
        """Write artifact_manifest.json listing all files with SHA-256 and size.

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        files: dict[str, dict] = {}
        created_at = datetime.now(timezone.utc).isoformat()

        for p in sorted(self.output_dir.rglob("*")):
            if p.is_file() and p.name != "artifact_manifest.json":
                try:
                    sha = _sha256_file(p)
                    size = p.stat().st_size
                    rel = p.relative_to(self.output_dir).as_posix()
                    files[rel] = {
                        "sha256": sha,
                        "size_bytes": size,
                        "created_at": created_at,
                    }
                except Exception as exc:
                    rel = p.relative_to(self.output_dir).as_posix()
                    log.warning("Could not hash file %s: %s", rel, exc)
                    files[rel] = {"sha256": None, "size_bytes": None, "created_at": created_at}

        manifest_path = self.output_dir / "artifact_manifest.json"
        manifest_data: dict[str, Any] = {
            "qortex_version": _get_qortex_version(),
            "created_at": created_at,
            "pipeline_ref": self.pipeline_ref,
            "n_files": len(files),
            "files": files,
        }
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_path

    def _write_json(self, filename: str, data: Any) -> None:
        path = self.output_dir / filename
        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            log.debug("Wrote %s (%d bytes)", filename, path.stat().st_size)
        except Exception as exc:
            log.warning("Could not write %s: %s", filename, exc)
