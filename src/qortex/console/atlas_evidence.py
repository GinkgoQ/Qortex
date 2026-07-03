"""Evidence normalization bridge for Qortex Atlas.

Qortex's own codebase has at least three independent "how sure are we"
concepts:

  * ``qortex.checks.EvidenceState`` — a 7-state enum (confirmed / inferred /
    claimed / missing / contradicted / unknown / blocked) used by the
    ``qortex check ...`` subsystem.
  * ``qortex.neuroai.contracts.EvidenceStatus`` — a similar but distinct enum
    used only inside the NeuroAI source/model compatibility engine.
  * ``CanTrainReport.label_status`` — a 3-state ``Literal["confirmed",
    "candidate", "missing"]`` used only on that one report.

None of these span the whole "can I use this dataset" decision the Atlas UI
needs to render on one Evidence tab. This module is the single place that
reads real ``ReadinessReport``, ``CanTrainReport``, and (optionally)
``LabelLandscape``/``SignalBudget`` objects and produces one consistent
4-state claim list — confirmed / inferred / unknown / blocked — the same
vocabulary the Atlas frontend already renders as icon+text badges.

Every claim traces back to a real Qortex finding or computed field; nothing
here is invented. ``cost_hint`` is a plain-language estimate of what it would
take to resolve an unknown (derived from which Qortex call would supply it),
matching the "cheapest next check" pattern from ``qortex doctor``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EvidenceGroup = Literal["confirmed", "inferred", "unknown", "blocked"]


@dataclass
class Claim:
    group: EvidenceGroup
    text: str
    source: str
    cost_hint: str | None = None


@dataclass
class EvidenceBundle:
    claims: list[Claim] = field(default_factory=list)

    def add(self, group: EvidenceGroup, text: str, source: str, cost_hint: str | None = None) -> None:
        self.claims.append(Claim(group, text, source, cost_hint))

    def counts(self) -> dict[str, int]:
        out = {"confirmed": 0, "inferred": 0, "unknown": 0, "blocked": 0}
        for c in self.claims:
            out[c.group] += 1
        return out

    def as_dict(self) -> dict[str, Any]:
        groups: dict[str, list[dict[str, Any]]] = {"confirmed": [], "inferred": [], "unknown": [], "blocked": []}
        for c in self.claims:
            groups[c.group].append({"text": c.text, "source": c.source, "cost": c.cost_hint})
        return {"groups": groups, "counts": self.counts()}


def _pct(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def build_evidence(
    *,
    dataset_id: str,
    manifest_summary: dict[str, Any],
    readiness: Any | None = None,
    can_train: Any | None = None,
    label_landscape: Any | None = None,
    ingestion_level: str = "manifest",
) -> EvidenceBundle:
    """Compose the Atlas evidence bundle from real Qortex report objects.

    Parameters accept the raw Qortex objects (``ReadinessReport``,
    ``CanTrainReport``, ``LabelLandscape``) — not pre-serialized dicts —
    since we need their typed fields (``.findings``, ``.label_status``, ...).
    """
    ev = EvidenceBundle()

    # ── Catalog-level facts: always confirmed, they came straight off the
    # OpenNeuro snapshot/manifest, no inference involved. ──────────────────
    # ManifestSummary carries a `subjects` list, not a precomputed count.
    n_subjects = len(manifest_summary.get("subjects") or [])
    ev.add("confirmed", f"{n_subjects} subjects present in snapshot manifest.", "OpenNeuro snapshot manifest")
    if manifest_summary.get("modalities"):
        ev.add("confirmed", f"Modalities present: {', '.join(manifest_summary['modalities'])}.", "manifest file-type scan")
    if manifest_summary.get("license"):
        ev.add("confirmed", f"License: {manifest_summary['license']}.", "dataset_description.json")

    # ── Readiness findings → the bulk of the evidence table. ───────────────
    if readiness is not None:
        n_rec = readiness.n_recordings
        if n_rec:
            ev_pct = _pct(readiness.n_event_complete, n_rec)
            if ev_pct >= 0.999:
                ev.add("confirmed", f"All {n_rec} recordings have complete events.", "computed from manifest + events.tsv presence")
            elif ev_pct > 0:
                ev.add("inferred", f"{readiness.n_event_complete}/{n_rec} recordings have events ({ev_pct:.0%}).", "computed from manifest", cost_hint="Fetch remaining events.tsv to confirm the rest")
            else:
                ev.add("unknown", "No recordings have confirmed events yet.", "computed from manifest", cost_hint="Run label-check plan to fetch events.tsv")

            label_pct = _pct(readiness.n_label_ready, n_rec)
            if label_pct >= 0.999:
                ev.add("confirmed", f"All {n_rec} recordings are label-ready.", "compute_readiness()")
            elif label_pct > 0:
                ev.add("inferred", f"{readiness.n_label_ready}/{n_rec} recordings are label-ready ({label_pct:.0%}).", "compute_readiness()", cost_hint="Inspect remaining event label columns")

            load_pct = _pct(readiness.n_loadable, n_rec)
            if load_pct < 0.999 and n_rec:
                ev.add("unknown" if load_pct == 0 else "inferred",
                       f"{readiness.n_loadable}/{n_rec} recordings confirmed loadable (have a download URL).",
                       "compute_readiness()")

        # Real datasets can carry one finding per recording (a 19-subject MEG
        # dataset repeats "channels.tsv missing" ~480 times) — group by code
        # so the evidence table stays one row per distinct issue, with the
        # true instance count and one representative path, rather than
        # flooding the UI with hundreds of near-duplicate rows.
        by_code: dict[str, list] = {}
        for finding in readiness.findings:
            by_code.setdefault(finding.code, []).append(finding)
        for code, group_findings in by_code.items():
            first = group_findings[0]
            n = len(group_findings)
            group: EvidenceGroup = "blocked" if first.severity == "error" else ("unknown" if first.severity == "warning" else "confirmed")
            text = first.message if n == 1 else f"{first.message} ({n} recordings affected)"
            example = first.path or (group_findings[0].recording_id if hasattr(group_findings[0], "recording_id") else None)
            if example and n == 1:
                text = f"{text} ({example})"
            elif example:
                text = f"{text} — e.g. {example}"
            ev.add(group, text, f"qortex check — {code}", first.recommendation)

    # ── CanTrainReport.label_status: the library's own 3-state confidence
    # flag, folded into the same 4-state vocabulary. ───────────────────────
    if can_train is not None:
        status_map: dict[str, EvidenceGroup] = {"confirmed": "confirmed", "candidate": "inferred", "missing": "blocked"}
        group = status_map.get(getattr(can_train, "label_status", "missing"), "unknown")
        ev.add(group, f"Training target status: {can_train.label_status}.", "decision.can_train()")
        for risk in getattr(can_train, "leakage_risks", []) or []:
            ev.add("unknown", f"Potential leakage risk: {risk}", "decision.can_train() leakage scan", "Run leakage-check after conversion")

    # ── LabelLandscape: real, remotely-computed class-balance evidence
    # (zero bytes downloaded — CDN events.tsv fetch only). ─────────────────
    if label_landscape is not None:
        n_fetched = label_landscape.n_files_fetched
        n_total = label_landscape.n_events_files
        if n_total:
            if label_landscape.n_files_failed:
                ev.add("inferred", f"Label landscape computed from {n_fetched}/{n_total} events files ({label_landscape.n_files_failed} fetch failures).", "label_landscape() (remote CDN scan)")
            else:
                ev.add("confirmed", f"Label landscape computed from all {n_total} events files.", "label_landscape() (remote CDN scan)")
            ev.add("confirmed", f"Class imbalance ratio: {label_landscape.imbalance_ratio:.2f}x (label column '{label_landscape.label_column}').", "label_landscape()")
            ev.add("confirmed" if label_landscape.cross_subject_consistency >= 0.99 else "inferred",
                   f"Cross-subject label consistency: {label_landscape.cross_subject_consistency:.0%}.", "label_landscape()")
        for rec in label_landscape.recommendations:
            ev.add("unknown", rec, "label_landscape() recommendation")
    elif ingestion_level in ("manifest", "catalog"):
        ev.add("unknown", "Class balance not yet computed — requires a remote events scan.", "not yet fetched", cost_hint="Run label_landscape() (no download required)")

    return ev


def mlreadiness_dims(profile_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn ``DatasetProfile.ml_readiness`` weighted components into the
    0-100-normalized per-dimension rows the Atlas readiness chart renders.
    Weights come straight from ``qortex.inspect.dataset.MLReadinessScore``.
    """
    ml = profile_dict.get("ml_readiness") or {}
    weights = {"events": 30, "subjects": 20, "license": 15, "modality": 15, "structure": 10, "companion": 10}
    labels = {
        "events": "Event completeness", "subjects": "Subject count", "license": "License openness",
        "modality": "Modality richness", "structure": "BIDS structure", "companion": "Companion completeness",
    }
    rows = []
    for key, max_pts in weights.items():
        raw = ml.get(f"{key}_score", 0) or 0
        rows.append({
            "key": key, "label": labels[key],
            "value": round((raw / max_pts) * 100) if max_pts else 0,
            "status": "confirmed",
            "note": f"{raw:g}/{max_pts} pts (weighted component of MLReadinessScore)",
        })
    return rows
