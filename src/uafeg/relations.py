"""Relaciones entre nodos de evidencia derivadas SOLO de senales observables
(NLI, deteccion visual). Nunca desde expected_label ni otras etiquetas de
ground truth: el relation builder no debe poder "ver" el label para decidir
una relacion.

Hoy solo existe una senal cruzada computable con el contrato actual:
narrativa (clases de danio afirmadas via NLI, src/narrative/nli_extractor.py)
vs visual (clases de danio efectivamente detectadas). Una relacion tipo
"overlap" (IoU/Jaccard sobre regiones) requeriria localizacion espacial del
resultado forense, que `ForensicResult` no expone todavia (ver
PROJECT_COMPLETION_GAP.md); no se fabrica esa relacion sin esa senal.
"""
from __future__ import annotations

from typing import Literal

from src.schemas.evidence import NarrativeEvidence, VisualEvidence

NARRATIVE_CLAIM_THRESHOLD = 0.5
"""Un damage_class se considera 'afirmado' por la narrativa si su
normalized_score (probabilidad de entailment NLI) alcanza este umbral."""

CrossEvidenceRelation = Literal["support", "contradiction", "unknown"]


def narrative_claimed_classes(narrative: NarrativeEvidence, threshold: float = NARRATIVE_CLAIM_THRESHOLD) -> set[str]:
    """Clases de danio que la narrativa afirma con score NLI >= threshold."""
    return {cls for cls, score in narrative.normalized_scores.items() if score >= threshold}


def visual_detected_classes(visual: VisualEvidence) -> set[str]:
    """Clases de danio efectivamente detectadas por el especialista visual."""
    return {detection.damage_class for detection in visual.detections}


def narrative_visual_relation(
    narrative: NarrativeEvidence,
    visual: VisualEvidence,
    threshold: float = NARRATIVE_CLAIM_THRESHOLD,
) -> CrossEvidenceRelation:
    """Compara clases afirmadas por narrativa vs detectadas por el
    especialista visual. Nunca usa expected_label.

    - support: al menos una clase afirmada coincide con una detectada.
    - contradiction: la narrativa afirma clases especificas y el especialista
      visual detecto danio de OTRAS clases (sin coincidencia alguna).
    - unknown: la narrativa no afirma ninguna clase con suficiente confianza,
      o el especialista visual no detecto nada (ausencia de deteccion no es
      evidencia concluyente de ausencia de danio).
    """
    claimed = narrative_claimed_classes(narrative, threshold)
    detected = visual_detected_classes(visual)

    if not claimed:
        return "unknown"
    if claimed & detected:
        return "support"
    if detected:
        return "contradiction"
    return "unknown"
