"""Protocolo oracle. Corregido en la pasada de consistencia de 2026-10-17
contra el .tex real de la tesis (capitulo4.tex, "Protocolo oraculo frente a
extremo a extremo"): "Evaluo cuatro configuraciones que combinan percepcion
oraculo o predicha para las afirmaciones del relato y para las
observaciones visuales y forenses" — es decir, visual+forense se togglean
JUNTOS como una sola dimension (no narrativa x visual x forense = 8
combinaciones; son narrativa x [visual+forense] = 4). La version anterior
de este modulo trataba oracle_visual de forma independiente y nunca tocaba
forensic; se corrige aqui porque la tesis es inequivoca en este punto.

Toda evidencia oracle deriva DIRECTAMENTE de campos de procedencia/
anotacion de ClaimRecord (`annotated_damage_classes`, `transformation`),
NUNCA de `expected_label` (la etiqueta de fraude que el modelo debe
predecir).

Las 4 combinaciones: (predicho, predicho) | (oracle narrativa, predicho
visual+forense) | (predicho narrativa, oracle visual+forense) | (oracle,
oracle). La evaluacion PRINCIPAL del proyecto sigue siendo end-to-end
predicho; oracle_* solo mide cuanto limita la imperfeccion de los
especialistas el techo de desempeno del sistema (diagnostico, no resultado
principal).
"""
from __future__ import annotations

from src.schemas.evidence import CANONICAL_DAMAGE_CLASSES, DamageAnnotation, ForensicResult, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket


def oracle_narrative_evidence(packet: EvidencePacket) -> NarrativeEvidence:
    """NarrativeEvidence 'perfecta': score 1.0 para cada clase anotada, 0.0
    para el resto. Deriva SOLO de claim.annotated_damage_classes."""
    annotated = set(packet.claim.annotated_damage_classes)
    scores = {c: (1.0 if c in annotated else 0.0) for c in CANONICAL_DAMAGE_CLASSES}
    return NarrativeEvidence(
        raw_scores=scores, normalized_scores=scores, model_name="oracle-annotated-damage", device="cpu"
    )


def oracle_visual_evidence(packet: EvidencePacket) -> VisualEvidence:
    """VisualEvidence 'perfecta': una deteccion de confianza 1.0 por cada
    clase anotada. Deriva SOLO de claim.annotated_damage_classes."""
    detections = [DamageAnnotation(damage_class=c, confidence=1.0) for c in packet.claim.annotated_damage_classes]
    width = height = 0
    if packet.visual is not None:
        width, height = packet.visual.image_width, packet.visual.image_height
    return VisualEvidence(
        detections=detections, model_name="oracle-annotated-damage", image_width=width, image_height=height
    )


def oracle_forensic_evidence(packet: EvidencePacket) -> ForensicResult:
    """ForensicResult 'perfecto': manipulado (score=1.0) si claim.transformation
    esta presente (la imagen fue sinteticamente manipulada por el generador
    del benchmark), no manipulado (score=0.0) en otro caso. Deriva SOLO de
    claim.transformation (procedencia de generacion), nunca de expected_label."""
    manipulated = packet.claim.transformation is not None
    return ForensicResult(
        available=True,
        method="oracle-transformation-ground-truth",
        score=1.0 if manipulated else 0.0,
        reliability=1.0,
    )


def build_oracle_packet(
    packet: EvidencePacket, oracle_narrative: bool = False, oracle_visual_forensic: bool = False
) -> EvidencePacket:
    """Construye una de las 4 combinaciones del protocolo oracle.
    oracle_visual_forensic togglea visual Y forense JUNTOS (ver docstring
    del modulo — asi lo especifica la tesis). No modifica image_quality ni
    ningun campo de ClaimRecord (expected_label incluido)."""
    updates = {}
    if oracle_narrative:
        updates["narrative"] = oracle_narrative_evidence(packet)
    if oracle_visual_forensic:
        updates["visual"] = oracle_visual_evidence(packet)
        updates["forensic"] = oracle_forensic_evidence(packet)
    return packet.model_copy(update=updates) if updates else packet
