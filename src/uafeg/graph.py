"""A deterministic claim-centred star graph, without learned reasoning or gates.

Graph metadata holds IDs for traceability. Labels, condition metadata and paths
are never copied into nodes. Each present modality produces one evidence node.
"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.schemas.packet import EvidencePacket
from src.schemas.evidence import NarrativeEvidence, VisualEvidence, ImageQuality, ForensicResult

from uafeg.relations import narrative_visual_relation

_PAYLOAD_TYPES = {"narrative": NarrativeEvidence, "visual": VisualEvidence,
                  "image_quality": ImageQuality, "forensic": ForensicResult}


class Node(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str = Field(min_length=1)
    kind: Literal["claim", "narrative", "visual", "image_quality", "forensic"]
    evidence: NarrativeEvidence | VisualEvidence | ImageQuality | ForensicResult | None = None

    @model_validator(mode="after")
    def valid_payload(self):
        if self.kind == "claim":
            if self.evidence is not None:
                raise ValueError("Claim node must not contain evidence or ground truth")
        elif not isinstance(self.evidence, _PAYLOAD_TYPES[self.kind]):
            raise ValueError("Evidence type does not match node kind")
        if self.kind == "forensic" and not self.evidence.available:
            raise ValueError("Unavailable forensic output is not an evidence node")
        return self


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: Literal["evidence_for", "support", "contradiction", "overlap", "unknown"] = "evidence_for"


class EvidenceGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str = Field(min_length=1)
    source_image_id: str | None = None
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @model_validator(mode="after")
    def valid_graph(self):
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate node IDs")
        claims = [node for node in self.nodes if node.kind == "claim"]
        if len(claims) != 1:
            raise ValueError("Exactly one claim node is required")
        kinds = [node.kind for node in self.nodes]
        if len(kinds) != len(set(kinds)):
            raise ValueError("MVP permits one node per modality")
        claim_id = claims[0].id
        id_set = set(ids)

        links = [(edge.source, edge.target) for edge in self.edges]
        if len(links) != len(set(links)):
            raise ValueError("Duplicate edges")

        # Star edges (claim-ward) must exist for every evidence node, with
        # relation=evidence_for, exactly as before this was a plain star graph.
        expected_star = {(node.id, claim_id) for node in self.nodes if node.kind != "claim"}
        star_links = {link for link in links if link[1] == claim_id}
        if star_links != expected_star:
            raise ValueError("Each evidence must point to the claim; dangling/self/missing edges forbidden")

        for edge in self.edges:
            if edge.source not in id_set or edge.target not in id_set:
                raise ValueError("Edge endpoints must reference existing nodes")
            if edge.source == edge.target:
                raise ValueError("Self-loop edges are forbidden")
            if edge.target == claim_id and edge.relation != "evidence_for":
                raise ValueError("Edges into the claim node must use relation=evidence_for")
            if edge.target != claim_id and edge.relation == "evidence_for":
                raise ValueError("relation=evidence_for is reserved for edges into the claim node")
        return self


def build_graph(packet: EvidencePacket) -> EvidenceGraph:
    if not isinstance(packet, EvidencePacket):
        raise TypeError("Expected canonical EvidencePacket from baseline repository")
    # Revalidate even instances created through model_copy/model_construct.
    packet = EvidencePacket.model_validate(packet.model_dump())
    nodes = [Node(id="claim", kind="claim")]
    edges = []
    for kind in _PAYLOAD_TYPES:
        evidence = getattr(packet, kind)
        if evidence is None or (kind == "forensic" and not evidence.available):
            continue
        nodes.append(Node(id=kind, kind=kind, evidence=evidence.model_copy(deep=True)))
        edges.append(Edge(source=kind, target="claim"))

    node_kinds = {node.kind for node in nodes}
    if "narrative" in node_kinds and "visual" in node_kinds:
        relation = narrative_visual_relation(packet.narrative, packet.visual)
        edges.append(Edge(source="narrative", target="visual", relation=relation))

    return EvidenceGraph(claim_id=packet.claim.claim_id,
                         source_image_id=packet.claim.source_image_id,
                         nodes=tuple(nodes), edges=tuple(edges))
