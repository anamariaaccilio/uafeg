"""P1: EvidenceGraph -> relational GATv2 -> pooling -> clasificador de 3
clases. SIN reliability gate (ver uafeg.models.p2_uafeg para la version con
compuerta, que reutiliza GraphClassifier como base).

Mismo encoder de evidencia (uafeg.models.encoders.EvidenceEncoderBank),
mismo espacio de labels y mismo protocolo de entrenamiento
(uafeg.models.training) que B6 (uafeg.models.b6_deepsets): la comparacion
B6 vs P1 debe aislar el aporte EXCLUSIVO de la estructura de grafo, no el
de un encoder o protocolo distinto.

Implementacion manual de GATv2 (Brody et al. 2021), sin torch_geometric: los
grafos por claim son pequenos (<=5 nodos: claim + hasta 4 modalidades), por
lo que una capa de atencion densa es suficiente y evita el riesgo de
instalar extensiones compiladas (torch-scatter/torch-sparse) que deben
coincidir exactamente con la version de torch/CPU/CUDA instalada, para un
beneficio marginal a esta escala de grafo.
"""
from __future__ import annotations

import torch
from torch import nn

from src.evaluation.metrics import CLASS_ORDER
from uafeg.graph import build_graph
from uafeg.models.encoders import DEFAULT_HIDDEN_DIM, DEFAULT_SHARED_DIM, EvidenceEncoderBank
from uafeg.models.features import packet_to_modality_features

NUM_CLASSES = len(CLASS_ORDER)
RELATION_TYPES = ("evidence_for", "support", "contradiction", "overlap", "unknown")
RELATION_INDEX = {relation: i for i, relation in enumerate(RELATION_TYPES)}


class GATv2Layer(nn.Module):
    """Una capa de atencion GATv2 relacional, calculada densamente (sin
    torch_geometric) sobre grafos pequenos. El tipo de relacion de cada
    arista sesga el score de atencion (via `W_rel`), no el contenido del
    mensaje: la relacion decide CUANTO pesar a un vecino, no que valor
    propagar."""

    def __init__(self, dim: int, heads: int = 2, relation_dim: int = 8):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim debe ser divisible por heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.relation_embedding = nn.Embedding(len(RELATION_TYPES), relation_dim)
        self.W_src = nn.Linear(dim, dim, bias=False)
        self.W_dst = nn.Linear(dim, dim, bias=False)
        self.W_rel = nn.Linear(relation_dim, dim, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, node_features: torch.Tensor, edges: list[tuple[int, int, int]]) -> torch.Tensor:
        """node_features: (N, dim). edges: lista de (src_idx, dst_idx,
        relation_idx), mensajes src->dst. Devuelve (N, dim): cada nodo tiene
        SIEMPRE un self-loop implicito ademas de sus aristas entrantes reales
        (atencion estandar GAT con self-loop), para que un nodo sin vecinos
        nunca pierda su propia senal."""
        n = node_features.shape[0]
        out = node_features.clone()
        projected_src = self.W_src(node_features)
        projected_dst = self.W_dst(node_features)
        zero_rel_bias = self.W_rel(torch.zeros(1, self.relation_embedding.embedding_dim)).squeeze(0)

        by_dst: dict[int, list[tuple[int, torch.Tensor]]] = {i: [(i, zero_rel_bias)] for i in range(n)}
        for src, dst, relation in edges:
            rel_bias = self.W_rel(self.relation_embedding(torch.tensor([relation]))).squeeze(0)
            by_dst[dst].append((src, rel_bias))

        for dst, candidates in by_dst.items():
            src_idx = torch.tensor([c[0] for c in candidates], dtype=torch.long)
            rel_bias = torch.stack([c[1] for c in candidates], dim=0)

            src_proj = projected_src[src_idx] + rel_bias
            dst_proj = projected_dst[dst].unsqueeze(0).expand_as(src_proj)

            combined = self.leaky_relu(src_proj + dst_proj).view(-1, self.heads, self.head_dim)
            scores = torch.einsum("khd,hd->kh", combined, self.attn)
            alpha = torch.softmax(scores, dim=0)

            values = projected_src[src_idx].view(-1, self.heads, self.head_dim)
            aggregated = (alpha.unsqueeze(-1) * values).sum(dim=0).reshape(-1)
            out[dst] = aggregated
        return out


class GraphClassifier(nn.Module):
    """P1 (sin compuerta). `message_scale(packet, graph, index_of)` devuelve
    1.0 para cada nodo en esta clase base; P2 la sobreescribe para aplicar
    g_j = clip(c*q*r*(1-u), 0, 1) a los mensajes antes de la atencion."""

    def __init__(
        self,
        shared_dim: int = DEFAULT_SHARED_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        heads: int = 2,
        include_cross_relations: bool = True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.include_cross_relations = include_cross_relations
        """Ablacion 'no_explicit_relations' (ver experiments/ablations.py):
        cuando es False, se descartan las aristas que no sean evidence_for
        (es decir, las relaciones narrativa-visual observables), dejando
        solo la estrella claim-centrica."""
        self.encoder_bank = EvidenceEncoderBank(shared_dim=shared_dim, hidden_dim=hidden_dim)
        self.claim_embedding = nn.Parameter(torch.zeros(shared_dim))
        self.gat_layer = GATv2Layer(shared_dim, heads=heads)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def message_scale(self, packet, graph, index_of: dict[str, int]) -> torch.Tensor:
        """Factor multiplicativo por nodo aplicado a los mensajes salientes
        antes de la atencion GATv2. P1: siempre 1.0 (sin compuerta)."""
        return torch.ones(len(index_of))

    def _node_features_and_edges(self, packet):
        graph = build_graph(packet)
        if len(graph.nodes) == 1:
            raise ValueError("Al menos una modalidad de evidencia debe estar presente")

        raw_features = packet_to_modality_features(packet)
        tensors = {kind: torch.tensor(vec, dtype=torch.float32) for kind, vec in raw_features.items()}
        encoded = self.encoder_bank(tensors)

        node_order = [node.id for node in graph.nodes]
        index_of = {node_id: i for i, node_id in enumerate(node_order)}
        node_features = torch.stack(
            [self.claim_embedding if node_id == "claim" else encoded[node_id] for node_id in node_order],
            dim=0,
        )

        scale = self.message_scale(packet, graph, index_of)
        scaled_features = node_features * scale.unsqueeze(-1)

        edges = [
            (index_of[edge.source], index_of[edge.target], RELATION_INDEX[edge.relation])
            for edge in graph.edges
            if self.include_cross_relations or edge.relation == "evidence_for"
        ]
        return node_features, scaled_features, edges

    def forward(self, packet) -> torch.Tensor:
        _, scaled_features, edges = self._node_features_and_edges(packet)
        # scaled_features ya incorpora message_scale (P1: 1.0 siempre; P2:
        # g_j) como factor sobre el mensaje que cada nodo propaga, incluido
        # su propio self-loop (ver GATv2Layer.forward).
        updated = self.gat_layer(scaled_features, edges)
        pooled = updated.mean(dim=0)
        return self.classifier(pooled)

    def forward_batch(self, packets: list) -> torch.Tensor:
        return torch.stack([self.forward(packet) for packet in packets], dim=0)
