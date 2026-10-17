"""Evidence encoders compartidos por B6, P1 y P2.

Misma entrada (EvidencePacket -> features por modalidad, ver
uafeg.models.features), mismos encoders por modalidad, mismo espacio de
labels (SUPPORTED/SUSPICIOUS/INSUFFICIENT). Solo cambia la arquitectura
AGUAS ABAJO del encoder: pooling permutation-invariante (B6), relational
GATv2 sin compuerta (P1), o relational GATv2 con reliability gate (P2). Esto
es intencional: B6 vs P1 vs P2 deben medir el aporte del grafo y de la
compuerta, no el de un encoder distinto.
"""
from __future__ import annotations

import torch
from torch import nn

from uafeg.models.features import MODALITY_DIMS

DEFAULT_SHARED_DIM = 32
DEFAULT_HIDDEN_DIM = 64


class ModalityEncoder(nn.Module):
    """MLP(raw_dim -> hidden -> shared_dim) para UNA modalidad."""

    def __init__(self, input_dim: int, hidden_dim: int = DEFAULT_HIDDEN_DIM, shared_dim: int = DEFAULT_SHARED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EvidenceEncoderBank(nn.Module):
    """Un ModalityEncoder independiente por modalidad, todos proyectando al
    mismo shared_dim, para que las representaciones sean comparables/
    agregables sin importar que modalidades esten presentes."""

    def __init__(self, shared_dim: int = DEFAULT_SHARED_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.shared_dim = shared_dim
        self.encoders = nn.ModuleDict(
            {kind: ModalityEncoder(dim, hidden_dim, shared_dim) for kind, dim in MODALITY_DIMS.items()}
        )

    def forward(self, modality_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """modality_features: {kind: tensor shape (dim,)}. Devuelve
        {kind: tensor shape (shared_dim,)}. Solo procesa las modalidades
        presentes en el dict de entrada (no rellena las ausentes)."""
        return {kind: self.encoders[kind](tensor) for kind, tensor in modality_features.items()}
