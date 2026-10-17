"""Protocolo oracle: las 4 combinaciones (predicho/oracle narrativa) x
(predicho/oracle visual+forense, togglados JUNTOS per el .tex de la tesis:
"percepcion oraculo o predicha para las afirmaciones del relato y para las
observaciones visuales y forenses"), derivadas SOLO de
annotated_damage_classes/transformation, nunca de expected_label."""
import inspect

from src.schemas.claim import ClaimRecord, ClaimTransformation
from src.schemas.evidence import DamageAnnotation, DecisionLabel, ForensicResult, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket
from uafeg.models.oracle import (
    build_oracle_packet,
    oracle_forensic_evidence,
    oracle_narrative_evidence,
    oracle_visual_evidence,
)


def _predicted_packet(manipulated: bool = False):
    transformation = (
        ClaimTransformation(operation="copy_move", parameters={}, seed=42) if manipulated else None
    )
    return EvidencePacket(
        claim=ClaimRecord(claim_id="fixture", expected_label=DecisionLabel.SUSPICIOUS,
                          annotated_damage_classes=["dent", "scratch"], transformation=transformation),
        narrative=NarrativeEvidence(raw_scores={"dent": 0.3}, normalized_scores={"dent": 0.3},
                                    model_name="predicted-fixture", device="cpu"),
        visual=VisualEvidence(detections=[DamageAnnotation(damage_class="crack", confidence=0.4)],
                              model_name="predicted-fixture", image_width=32, image_height=32),
        forensic=ForensicResult(available=True, method="predicted-fixture", score=0.1, reliability=0.5),
    )


def test_oracle_narrative_matches_annotated_damage_classes():
    packet = _predicted_packet()
    oracle = oracle_narrative_evidence(packet)
    assert oracle.normalized_scores["dent"] == 1.0
    assert oracle.normalized_scores["scratch"] == 1.0
    assert oracle.normalized_scores["crack"] == 0.0


def test_oracle_visual_matches_annotated_damage_classes():
    packet = _predicted_packet()
    oracle = oracle_visual_evidence(packet)
    detected = {d.damage_class for d in oracle.detections}
    assert detected == {"dent", "scratch"}
    assert all(d.confidence == 1.0 for d in oracle.detections)


def test_oracle_forensic_reflects_transformation_ground_truth():
    manipulated = oracle_forensic_evidence(_predicted_packet(manipulated=True))
    unmanipulated = oracle_forensic_evidence(_predicted_packet(manipulated=False))
    assert manipulated.score == 1.0
    assert unmanipulated.score == 0.0
    assert manipulated.reliability == 1.0 and unmanipulated.reliability == 1.0


def test_build_oracle_packet_four_combinations():
    packet = _predicted_packet(manipulated=True)

    predicted_predicted = build_oracle_packet(packet, oracle_narrative=False, oracle_visual_forensic=False)
    assert predicted_predicted.narrative == packet.narrative
    assert predicted_predicted.visual == packet.visual
    assert predicted_predicted.forensic == packet.forensic

    oracle_predicted = build_oracle_packet(packet, oracle_narrative=True, oracle_visual_forensic=False)
    assert oracle_predicted.narrative != packet.narrative
    assert oracle_predicted.visual == packet.visual
    assert oracle_predicted.forensic == packet.forensic

    predicted_oracle = build_oracle_packet(packet, oracle_narrative=False, oracle_visual_forensic=True)
    assert predicted_oracle.narrative == packet.narrative
    assert predicted_oracle.visual != packet.visual
    assert predicted_oracle.forensic != packet.forensic  # visual+forense togglan juntos

    oracle_oracle = build_oracle_packet(packet, oracle_narrative=True, oracle_visual_forensic=True)
    assert oracle_oracle.narrative != packet.narrative
    assert oracle_oracle.visual != packet.visual
    assert oracle_oracle.forensic != packet.forensic


def test_build_oracle_packet_never_touches_expected_label():
    packet = _predicted_packet(manipulated=True)
    oracle_oracle = build_oracle_packet(packet, oracle_narrative=True, oracle_visual_forensic=True)
    assert oracle_oracle.claim.expected_label == packet.claim.expected_label


def test_oracle_functions_never_read_expected_label():
    for fn in (oracle_narrative_evidence, oracle_visual_evidence, oracle_forensic_evidence, build_oracle_packet):
        source = inspect.getsource(fn)
        assert ".expected_label" not in source
