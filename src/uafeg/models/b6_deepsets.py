"""B6: DeepSets baseline.

evidence encoders (compartidos, ver uafeg.models.encoders)
  -> pooling permutation-invariante (suma) sobre las modalidades presentes
  -> clasificador de 3 clases (SUPPORTED, SUSPICIOUS, INSUFFICIENT)

Mismos EvidencePackets, mismo encoder y mismo espacio de labels que P1/P2
(ver uafeg/configs/b6_deepsets.yaml); la comparacion B6 vs P1 aisla el
aporte del grafo porque ambos comparten este mismo frente de entrada.
"""
from __future__ import annotations

import torch
from torch import nn

from src.evaluation.metrics import CLASS_ORDER
from uafeg.models.encoders import DEFAULT_HIDDEN_DIM, DEFAULT_SHARED_DIM, EvidenceEncoderBank
from uafeg.models.features import packet_to_modality_features

NUM_CLASSES = len(CLASS_ORDER)


class DeepSetsClassifier(nn.Module):
    def __init__(
        self,
        shared_dim: int = DEFAULT_SHARED_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.encoder_bank = EvidenceEncoderBank(shared_dim=shared_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward_from_features(self, modality_features: dict[str, torch.Tensor]) -> torch.Tensor:
        """modality_features: {kind: tensor shape (shared_dim,) tras encoder,
        o aqui se pasa PRE-encoder: {kind: tensor shape (raw_dim,)}}.
        Devuelve logits shape (num_classes,)."""
        if not modality_features:
            raise ValueError("Al menos una modalidad de evidencia debe estar presente")
        encoded = self.encoder_bank(modality_features)
        pooled = torch.stack(list(encoded.values()), dim=0).sum(dim=0)  # permutation-invariant
        return self.classifier(pooled)

    def forward(self, packet) -> torch.Tensor:
        """Conveniencia: acepta un EvidencePacket directamente (una sola
        claim) y devuelve logits shape (num_classes,)."""
        raw_features = packet_to_modality_features(packet)
        tensors = {kind: torch.tensor(vec, dtype=torch.float32) for kind, vec in raw_features.items()}
        return self.forward_from_features(tensors)

    def forward_batch(self, packets: list) -> torch.Tensor:
        """Forward por lotes SIN compartir computo entre muestras (cada
        EvidencePacket puede tener un conjunto distinto de modalidades
        presentes, por lo que no hay un tensor denso comun para batchear).
        Devuelve logits shape (batch, num_classes)."""
        return torch.stack([self.forward(packet) for packet in packets], dim=0)
