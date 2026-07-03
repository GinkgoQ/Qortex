"""Query compiler — turns free text (+ structured kwargs) into a typed
``QueryPlan`` that separates HARD constraints (must pass; drive the structured
retriever's SQL) from SOFT signals (contribute to ranking only).

The flat substring scorer this replaces conflated the two: a query like "at
least 40 subjects" degraded into two meaningless text tokens ("40" and
"subjects") because there was no notion of a quantitative constraint at all.

The grammar is deterministic (regex + unit parsing), with zero LLM dependency
in the hot path — this keeps ``compile_query`` sub-millisecond and fully
reproducible/testable, which matters because it is the one stage every other
retriever's precision depends on. An LLM slot-filler is a valid future
addition for genuinely unstructured free text, but only as a *proposer*
layered on top of this deterministic core (see qortex-atlas-search-engine.md
§3c) — never as a replacement for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from qortex.search.ontology import Ontology

_NUM = r"(\d+(?:\.\d+)?)"

# (regex, target field, comparator). Order matters: more specific phrasings
# are tried before generic ones so "at least" isn't partially eaten by a
# looser pattern first.
_GRAMMAR: list[tuple[re.Pattern, str, str]] = [
    (re.compile(rf"\bat least {_NUM}\s*subjects?\b"), "min_subjects", "ge"),
    (re.compile(rf"\bmin(?:imum)?\.?\s*{_NUM}\s*subjects?\b"), "min_subjects", "ge"),
    (re.compile(rf"\b(?:>=|≥)\s*{_NUM}\s*subjects?\b"), "min_subjects", "ge"),
    (re.compile(rf"\b{_NUM}\+\s*subjects?\b"), "min_subjects", "ge"),
    (re.compile(rf"\bunder {_NUM}\s*gb\b"), "max_size_gb", "le"),
    (re.compile(rf"\bless than {_NUM}\s*gb\b"), "max_size_gb", "le"),
    (re.compile(rf"\bat most {_NUM}\s*gb\b"), "max_size_gb", "le"),
    (re.compile(rf"\b(?:<=|≤)\s*{_NUM}\s*gb\b"), "max_size_gb", "le"),
    (re.compile(rf"\b{_NUM}\s*gb\s*(?:max|or less)\b"), "max_size_gb", "le"),
    (re.compile(rf"\bat least {_NUM}\s*classes?\b"), "min_n_classes", "ge"),
    (re.compile(rf"\b{_NUM}\+\s*classes?\b"), "min_n_classes", "ge"),
]

_OPEN_LICENSE_RE = re.compile(r"\bopen[- ]?licen[cs]e[d]?\b|\bcc0\b|\bpublic domain\b")
_HAS_EVENTS_RE = re.compile(r"\bwith events\b|\bevent[- ]related\b|\bhas events\b")


@dataclass(frozen=True)
class Constraint:
    field: str
    op: str  # "eq" | "ge" | "le" | "in"
    value: Any
    hard: bool = True  # False = soft-only even though grammar-extracted (e.g. has_events)


@dataclass
class QueryPlan:
    raw_query: str
    hard: dict[str, Constraint] = field(default_factory=dict)
    soft_terms: list[str] = field(default_factory=list)
    lexical_terms: list[str] = field(default_factory=list)
    semantic_text: str = ""
    modality_tokens: set[str] = field(default_factory=set)
    provenance: dict[str, str] = field(default_factory=dict)  # field -> grammar|ontology|passthrough

    def describe(self) -> str:
        parts = [f"{c.field}{'≥' if c.op == 'ge' else '≤' if c.op == 'le' else '='}{c.value}" for c in self.hard.values()]
        if self.soft_terms:
            parts.append(f"soft={self.soft_terms}")
        return "QueryPlan(" + ", ".join(parts) + ")"


def compile_query(
    text: str | None,
    *,
    ontology: Ontology | None = None,
    modality: str | None = None,
    min_subjects: int | None = None,
    max_size_gb: float | None = None,
    license_open: bool | None = None,
    has_events: bool | None = None,
) -> QueryPlan:
    ontology = ontology or Ontology.default()
    raw = text or ""
    plan = QueryPlan(raw_query=raw)
    remainder = raw.lower()

    # 1. explicit structured kwargs always win, marked "passthrough"
    if modality:
        expanded = ontology.canonical_modalities(modality)
        plan.hard["modality"] = Constraint("modality", "in", expanded or {modality.lower()})
        plan.modality_tokens |= expanded
        plan.provenance["modality"] = "passthrough"
    if min_subjects is not None:
        plan.hard["min_subjects"] = Constraint("min_subjects", "ge", min_subjects)
        plan.provenance["min_subjects"] = "passthrough"
    if max_size_gb is not None:
        plan.hard["max_size_gb"] = Constraint("max_size_gb", "le", max_size_gb)
        plan.provenance["max_size_gb"] = "passthrough"
    if license_open:
        plan.hard["license_open"] = Constraint("license_open", "eq", True)
        plan.provenance["license_open"] = "passthrough"
    if has_events is not None:
        plan.hard["has_events"] = Constraint("has_events", "eq", has_events, hard=False)
        plan.provenance["has_events"] = "passthrough"

    # 2. deterministic grammar over free text (only fields not already set)
    for pattern, target_field, op in _GRAMMAR:
        if target_field in plan.hard:
            continue
        m = pattern.search(remainder)
        if not m:
            continue
        value: float | int = float(m.group(1))
        if target_field in {"min_subjects", "min_n_classes"}:
            value = int(value)
        plan.hard[target_field] = Constraint(target_field, op, value)
        plan.provenance[target_field] = "grammar"
        remainder = remainder[: m.start()] + " " + remainder[m.end():]

    if "license_open" not in plan.hard and _OPEN_LICENSE_RE.search(remainder):
        plan.hard["license_open"] = Constraint("license_open", "eq", True)
        plan.provenance["license_open"] = "grammar"
        remainder = _OPEN_LICENSE_RE.sub(" ", remainder)

    if "has_events" not in plan.hard and _HAS_EVENTS_RE.search(remainder):
        plan.hard["has_events"] = Constraint("has_events", "eq", True, hard=False)
        plan.provenance["has_events"] = "grammar"
        remainder = _HAS_EVENTS_RE.sub(" ", remainder)

    # 3. modality detection from remaining free text via ontology (canonical + fuzzy)
    if "modality" not in plan.hard:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]*", remainder)
        found: set[str] = set()
        matched_spans: list[str] = []
        for tok in tokens:
            hit = ontology.canonical_modalities(tok)
            if hit:
                found |= hit
                matched_spans.append(tok)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            hit = ontology.canonical_modalities(bigram)
            if hit:
                found |= hit
                matched_spans.append(bigram)
        if found:
            plan.hard["modality"] = Constraint("modality", "in", found)
            plan.modality_tokens |= found
            plan.provenance["modality"] = "ontology"
            for span in matched_spans:
                remainder = remainder.replace(span, " ")

    # 4. everything left over becomes soft lexical/semantic signal, synonym-expanded
    leftover_tokens = [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_+.-]*", remainder) if len(t) > 1]
    plan.soft_terms = leftover_tokens
    plan.semantic_text = " ".join(leftover_tokens) if leftover_tokens else raw

    expanded_terms: set[str] = set(leftover_tokens)
    for tok in leftover_tokens:
        expanded_terms |= ontology.task_synonyms(tok)
    # also expand contiguous bigrams (task labels are often multi-word)
    for i in range(len(leftover_tokens) - 1):
        bigram = f"{leftover_tokens[i]}-{leftover_tokens[i + 1]}"
        expanded_terms |= ontology.task_synonyms(bigram)
    plan.lexical_terms = sorted(expanded_terms) if expanded_terms else leftover_tokens

    return plan
