"""P2: UAFEG completo. Reutiliza P1 (uafeg.models.p1_graph.GraphClassifier)
como arquitectura base — mismo encoder, misma atencion GATv2, mismo pooling
y clasificador. La UNICA diferencia es el reliability gate g_j
(uafeg.models.gate.node_gate), aplicado al mensaje de cada nodo de
evidencia ANTES de la atencion/agregacion (ver
GraphClassifier.message_scale). gate=1 para todos los nodos reproduce
EXACTAMENTE a P1 con los mismos pesos: la comparacion P1 vs P2 aisla el
aporte exclusivo de la compuerta de confiabilidad.
"""
from __future__ import annotations

import torch

from uafeg.models.gate import node_gate
from uafeg.models.p1_graph import GraphClassifier


class ReliabilityGatedGraphClassifier(GraphClassifier):
    """P2 = P1 + reliability gate. No reimplementa encoder/atencion/pooling:
    solo sobreescribe `message_scale`.

    `disable_gate_terms`: SOLO para la ablacion 'no_uncertainty' (ver
    experiments/ablations.py) — fuerza a su valor neutro uno o mas de
    {coverage, quality, reliability, uncertainty} en todos los nodos. Vacio
    por defecto (P2 completo)."""

    def __init__(self, *args, disable_gate_terms=frozenset(), **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_gate_terms = frozenset(disable_gate_terms)  # acepta list (p.ej. desde YAML) o frozenset

    def message_scale(self, packet, graph, index_of: dict[str, int]) -> torch.Tensor:
        scale = torch.ones(len(index_of))
        for node_id, idx in index_of.items():
            if node_id == "claim":
                continue  # el nodo claim no es evidencia; sin gate (factor 1.0)
            scale[idx] = node_gate(packet, node_id, disable=self.disable_gate_terms)
        return scale
