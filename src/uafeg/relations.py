"""Relaciones entre nodos de evidencia derivadas SOLO de senales observables
(NLI, deteccion visual). Nunca desde expected_label ni otras etiquetas de
ground truth: el relation builder no debe poder "ver" el label para decidir
una relacion.

Hoy solo existe una senal cruzada computable con el contrato actual:
narrativa (clases de danio afirmadas via NLI, src/narrative/nli_extractor.py)
vs visual (clases de danio efectivamente detectadas). Una relacion tipo
"overlap" (IoU/Jaccard sobre regiones) requeriria localizacion espacial del
resultado forense, que `ForensicResult` no expone todavia (ver
docs/TRAINING.md); no se fabrica esa relacion sin esa senal.

Ademas del tipo categorico de relacion, cada arista transporta el vector de
atributos continuos de la tesis (eq:aristas):

    a_ij = [p_support, p_contradiction, p_unknown, sim_sem, overlap_spatial]

acompanado de una MASCARA de observabilidad del mismo largo. Un atributo que
el contrato actual no permite computar vale 0.0 con mascara 0.0: es un valor
ausente declarado, nunca un valor inventado. Hoy `overlap_spatial` esta
siempre enmascarado porque ninguna de las dos puntas de la relacion
narrativa-visual expone una region comparable (la narrativa no localiza y
`ForensicResult` no publica localizacion).
"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


EDGE_ATTRIBUTE_NAMES = ("p_support", "p_contradiction", "p_unknown", "sim_sem", "overlap_spatial")
EDGE_ATTRIBUTE_DIM = len(EDGE_ATTRIBUTE_NAMES)


class EdgeAttributes(BaseModel):
    """a_ij de la tesis (eq:aristas) mas su mascara de observabilidad.

    `mask[k] == 0.0` significa que el atributo k NO es computable con el
    contrato actual y que su valor es 0.0 por convencion, no por medicion.
    Un consumidor nunca debe interpretar un 0.0 enmascarado como "se midio
    cero"."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    values: tuple[float, float, float, float, float] = (0.0,) * EDGE_ATTRIBUTE_DIM
    mask: tuple[float, float, float, float, float] = (0.0,) * EDGE_ATTRIBUTE_DIM

    @classmethod
    def unobserved(cls) -> "EdgeAttributes":
        """Todos los atributos ausentes. Es lo que corresponde a las aristas
        `evidence_for` hacia el nodo claim: no son una relacion entre dos
        evidencias, de modo que p_support/p_contradiction/p_unknown y las
        similitudes no estan definidas para ellas."""
        return cls()

    def without_attributes(self) -> "EdgeAttributes":
        """A4 de la tesis: conserva el TIPO de arista pero pierde los
        puntajes continuos. Devuelve el vector completamente enmascarado."""
        return EdgeAttributes.unobserved()

    def as_vector(self) -> tuple[float, ...]:
        """Concatenacion (valores, mascara), largo 2*EDGE_ATTRIBUTE_DIM."""
        return tuple(self.values) + tuple(self.mask)


def _class_confidence(visual: VisualEvidence) -> tuple[dict[str, float], bool]:
    """Mejor confianza por clase detectada. El segundo elemento indica si
    TODAS las detecciones traian confidence; si alguna venia en None, el
    especialista no reporto esa senal y los atributos que dependen de ella
    quedan enmascarados en vez de asumir un valor."""
    best: dict[str, float] = {}
    complete = True
    for detection in visual.detections:
        if detection.confidence is None:
            complete = False
            continue
        best[detection.damage_class] = max(best.get(detection.damage_class, 0.0), detection.confidence)
    return best, complete


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    dot = sum(left.get(k, 0.0) * right.get(k, 0.0) for k in keys)
    norm_left = math.sqrt(sum(v * v for v in left.values()))
    norm_right = math.sqrt(sum(v * v for v in right.values()))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def narrative_visual_attributes(
    narrative: NarrativeEvidence,
    visual: VisualEvidence,
    threshold: float = NARRATIVE_CLAIM_THRESHOLD,
) -> EdgeAttributes:
    """a_ij observable para la arista narrativa-visual. Nunca lee
    expected_label.

    Reparto de masa, consistente con `narrative_visual_relation`:

      - p_support:       masa NLI de las clases afirmadas Y detectadas.
      - p_contradiction: masa de las clases detectadas que la narrativa NO
        afirma (danio presente que el relato no menciona).
      - p_unknown:       masa NLI de las clases afirmadas y NO detectadas.
        La ausencia de deteccion no es evidencia concluyente de ausencia de
        danio, de modo que esa masa va a unknown, no a contradiction.

    Si la narrativa no afirma ninguna clase y no hay detecciones utilizables,
    p_unknown = 1.0: es una conclusion del propio reparto, no un relleno.

    sim_sem: coseno entre el vector de scores NLI por clase y el vector de
    confianzas visuales por clase. Requiere confianzas; si el especialista no
    las reporto, queda enmascarado.

    overlap_spatial: siempre enmascarado hoy. La narrativa no localiza, de
    modo que no existe una region comparable entre las dos puntas.
    """
    claimed = narrative_claimed_classes(narrative, threshold)
    detected_conf, confidence_complete = _class_confidence(visual)
    detected = visual_detected_classes(visual)

    support_mass = sum(narrative.normalized_scores.get(c, 0.0) for c in claimed & detected)
    unknown_mass = sum(narrative.normalized_scores.get(c, 0.0) for c in claimed - detected)
    contradiction_mass = sum(detected_conf.get(c, 0.0) for c in detected - claimed)

    total = support_mass + unknown_mass + contradiction_mass
    if total > 0.0:
        probabilities = (support_mass / total, contradiction_mass / total, unknown_mass / total)
    else:
        # Ni clases afirmadas ni detecciones con confianza: todo el peso va a
        # unknown, que es exactamente lo que reporta narrative_visual_relation.
        probabilities = (0.0, 0.0, 1.0)

    # Las tres probabilidades son computables siempre que exista alguna de las
    # dos senales; si la confianza visual falta, el reparto sigue siendo valido
    # (contradiction solo pierde la masa no reportada), pero sim_sem no lo es.
    sim_observed = bool(claimed) and bool(detected_conf) and confidence_complete
    sim_sem = _cosine(narrative.normalized_scores, detected_conf) if sim_observed else 0.0

    return EdgeAttributes(
        values=(probabilities[0], probabilities[1], probabilities[2], sim_sem, 0.0),
        mask=(1.0, 1.0, 1.0, 1.0 if sim_observed else 0.0, 0.0),
    )


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
