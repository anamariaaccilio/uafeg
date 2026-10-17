"""Reliability gate: g_j = clip(c_j * q_j * r_j * (1 - u_j), 0, 1) por nodo
de evidencia j, donde:

  c_j = coverage      (que tanto de la modalidad fue observado)
  q_j = source_quality (calidad de la fuente/imagen subyacente)
  r_j = reliability   (confiabilidad propia reportada por el especialista)
  u_j = uncertainty   (incertidumbre de la prediccion del especialista, en [0,1])

Todos los factores se derivan SOLO de las salidas observables del propio
EvidencePacket (narrative/visual/forensic/image_quality) — nunca de
expected_label, condition u otro metadato de ground truth. Donde el
contrato actual (EvidencePacket) no expone un factor para una modalidad, se
usa el valor neutro (1.0 para c/q/r, 0.0 para u) para no penalizar ni
beneficiar artificialmente el gate; es una limitacion documentada del
contrato (ver PROJECT_COMPLETION_GAP.md), no evidencia fabricada.
"""
from __future__ import annotations

import math

from src.schemas.evidence import ForensicResult, ImageQuality, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket

GateFactors = tuple[float, float, float, float]  # (coverage, quality, reliability, uncertainty)

LAPLACIAN_QUALITY_SCALE = 100.0
"""Normalizacion simple y explicita de laplacian_variance a [0,1] (no
calibrada estadisticamente); documentado para que una futura calibracion
pueda reemplazarla sin ambiguedad."""


def _narrative_factors(evidence: NarrativeEvidence) -> GateFactors:
    scores = list(evidence.normalized_scores.values())
    coverage = 1.0 if scores else 0.0
    quality = 1.0
    reliability = 1.0
    total = sum(scores)
    if scores and total > 0:
        probs = [s / total for s in scores if s > 0]
        max_entropy = math.log(len(scores)) if len(scores) > 1 else 1.0
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        uncertainty = 1.0  # sin senal util -> maxima incertidumbre, nunca se asume certeza
    return coverage, quality, reliability, uncertainty


def _visual_factors(evidence: VisualEvidence, image_quality: ImageQuality | None) -> GateFactors:
    coverage = 1.0 if evidence.detections else 0.0
    if image_quality is not None and image_quality.is_corrupted:
        quality = 0.0
    elif image_quality is not None:
        laplacian = image_quality.laplacian_variance or 0.0
        quality = max(0.0, min(1.0, laplacian / LAPLACIAN_QUALITY_SCALE))
    else:
        quality = 1.0
    reliability = 1.0
    confidences = [d.confidence for d in evidence.detections if d.confidence is not None]
    uncertainty = 1.0 - (max(confidences) if confidences else 0.0)
    return coverage, quality, reliability, uncertainty


def _forensic_factors(evidence: ForensicResult) -> GateFactors:
    coverage = 1.0  # el nodo forense solo existe en el grafo si available=True
    quality = 1.0
    reliability = evidence.reliability if evidence.reliability is not None else 1.0
    uncertainty = 1.0 - (evidence.score if evidence.score is not None else 0.5)
    return coverage, quality, reliability, uncertainty


def _image_quality_factors(evidence: ImageQuality) -> GateFactors:
    coverage = 1.0
    quality = 0.0 if evidence.is_corrupted else 1.0
    reliability = 1.0
    uncertainty = 0.0
    return coverage, quality, reliability, uncertainty


_FACTOR_FUNCS = {
    "narrative": lambda packet: _narrative_factors(packet.narrative),
    "visual": lambda packet: _visual_factors(packet.visual, packet.image_quality),
    "forensic": lambda packet: _forensic_factors(packet.forensic),
    "image_quality": lambda packet: _image_quality_factors(packet.image_quality),
}


def reliability_gate(coverage: float, quality: float, reliability: float, uncertainty: float) -> float:
    """g_j = clip(c_j * q_j * r_j * (1 - u_j), 0, 1)."""
    raw = coverage * quality * reliability * (1.0 - uncertainty)
    return max(0.0, min(1.0, raw))


GATE_TERMS = ("coverage", "quality", "reliability", "uncertainty")
_NEUTRAL_VALUE = {"coverage": 1.0, "quality": 1.0, "reliability": 1.0, "uncertainty": 0.0}


def node_gate(packet: EvidencePacket, kind: str, disable: frozenset[str] = frozenset()) -> float:
    """g_j para el nodo de evidencia `kind` de este packet. El nodo claim
    no es evidencia y no tiene gate (ver p2_uafeg.ReliabilityGatedGraphClassifier,
    que siempre usa factor 1.0 para 'claim').

    `disable`: nombres en GATE_TERMS a forzar a su valor neutro (usado por
    la ablacion 'no_uncertainty', ver experiments/ablations.py). No es parte
    del diseno principal de P2; es exclusivamente para el experimento de
    ablacion explicito."""
    if kind not in _FACTOR_FUNCS:
        raise ValueError(f"Modalidad desconocida para reliability gate: {kind}")
    unknown = disable - set(GATE_TERMS)
    if unknown:
        raise ValueError(f"Terminos de gate desconocidos: {unknown}")
    coverage, quality, reliability, uncertainty = _FACTOR_FUNCS[kind](packet)
    values = {"coverage": coverage, "quality": quality, "reliability": reliability, "uncertainty": uncertainty}
    for term in disable:
        values[term] = _NEUTRAL_VALUE[term]
    return reliability_gate(values["coverage"], values["quality"], values["reliability"], values["uncertainty"])
