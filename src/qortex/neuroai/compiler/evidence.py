"""Evidence graph primitives for NeuroAI compilation."""

from __future__ import annotations

from typing import Any, Literal

from qortex.neuroai.contracts import BaseModel, EvidenceStatus, Field


class EvidenceNode(BaseModel):
    id: str
    kind: str
    status: EvidenceStatus
    source: str | None = None
    value: Any = None
    note: str | None = None


class EvidenceEdge(BaseModel):
    source: str
    target: str
    relation: Literal["supports", "blocks", "requires", "derives"] = "supports"


class EvidenceGraph(BaseModel):
    nodes: list[EvidenceNode] = Field(default_factory=list)
    edges: list[EvidenceEdge] = Field(default_factory=list)

    def add_node(
        self,
        *,
        node_id: str,
        kind: str,
        status: EvidenceStatus,
        source: str | None = None,
        value: Any = None,
        note: str | None = None,
    ) -> None:
        if any(node.id == node_id for node in self.nodes):
            return
        self.nodes.append(EvidenceNode(
            id=node_id,
            kind=kind,
            status=status,
            source=source,
            value=value,
            note=note,
        ))

    def add_edge(self, source: str, target: str, relation: str = "supports") -> None:
        edge = EvidenceEdge(source=source, target=target, relation=relation)
        if edge not in self.edges:
            self.edges.append(edge)


__all__ = ["EvidenceEdge", "EvidenceGraph", "EvidenceNode"]
