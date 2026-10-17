"""Conversion de EvidencePacket a vectores de features por modalidad.

Usado por B6 (DeepSets), P1 y P2 para compartir EXACTAMENTE el mismo
encoder de entrada, tal como exige la comparacion B6 vs P1 vs P2 (solo debe
cambiar la arquitectura aguas abajo del encoder, no la entrada).

None (modalidad ausente) significa que el elemento no entra al conjunto de
evidencia; nunca se rellena con un vector de ceros para simular presencia
(null != 0, ver EvidencePacket). Ground truth (`expected_label`,
`condition`, etc.) nunca se lee aqui: estas funciones solo ven el objeto de
evidencia de una modalidad, no el ClaimRecord completo.
"""
from __future__ import annotations

import numpy as np

from src.schemas.evidence import CANONICAL_DAMAGE_CLASSES
from src.schemas.packet import EvidencePacket

NARRATIVE_DIM = len(CANONICAL_DAMAGE_CLASSES)
VISUAL_DIM = len(CANONICAL_DAMAGE_CLASSES)
IMAGE_QUALITY_DIM = 4  # aspect_ratio, log1p(file_size_bytes), laplacian_variance, is_corrupted
FORENSIC_DIM = 2  # score, reliability

MODALITY_DIMS: dict[str, int] = {
    "narrative": NARRATIVE_DIM,
    "visual": VISUAL_DIM,
    "image_quality": IMAGE_QUALITY_DIM,
    "forensic": FORENSIC_DIM,
}


def narrative_features(evidence) -> np.ndarray:
    """Vector de score NLI normalizado por clase canonica, orden fijo."""
    return np.array(
        [evidence.normalized_scores.get(c, 0.0) for c in CANONICAL_DAMAGE_CLASSES],
        dtype=np.float32,
    )


def visual_features(evidence) -> np.ndarray:
    """Confianza maxima detectada por clase canonica (0.0 si no detectada)."""
    best = {c: 0.0 for c in CANONICAL_DAMAGE_CLASSES}
    for detection in evidence.detections:
        confidence = detection.confidence if detection.confidence is not None else 1.0
        if confidence > best[detection.damage_class]:
            best[detection.damage_class] = confidence
    return np.array([best[c] for c in CANONICAL_DAMAGE_CLASSES], dtype=np.float32)


def image_quality_features(evidence) -> np.ndarray:
    laplacian = evidence.laplacian_variance if evidence.laplacian_variance is not None else 0.0
    return np.array(
        [
            evidence.aspect_ratio,
            float(np.log1p(evidence.file_size_bytes)),
            float(laplacian),
            float(evidence.is_corrupted),
        ],
        dtype=np.float32,
    )


def forensic_features(evidence) -> np.ndarray:
    score = evidence.score if evidence.score is not None else 0.0
    reliability = evidence.reliability if evidence.reliability is not None else 0.0
    return np.array([score, reliability], dtype=np.float32)


_EXTRACTORS = {
    "narrative": narrative_features,
    "visual": visual_features,
    "image_quality": image_quality_features,
    "forensic": forensic_features,
}


def packet_to_modality_features(packet: EvidencePacket) -> dict[str, np.ndarray]:
    """{modalidad: vector} SOLO para modalidades presentes (no None; para
    forensic solo si available=True). No produce entradas para modalidades
    ausentes."""
    features: dict[str, np.ndarray] = {}
    for kind, extractor in _EXTRACTORS.items():
        evidence = getattr(packet, kind)
        if evidence is None:
            continue
        if kind == "forensic" and not evidence.available:
            continue
        features[kind] = extractor(evidence)
    return features
