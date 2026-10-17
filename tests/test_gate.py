"""Tests del reliability gate g_j = clip(c*q*r*(1-u), 0, 1). Todas las
propiedades exigidas: rango [0,1], cada factor bajo reduce el gate, y el
gate nunca lee expected_label/ground truth."""
import inspect

import pytest

from src.schemas.claim import ClaimRecord
from src.schemas.evidence import DamageAnnotation, ForensicResult, ImageQuality, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket
from uafeg.models import gate as gate_module
from uafeg.models.gate import node_gate, reliability_gate


@pytest.mark.parametrize(
    "coverage,quality,reliability,uncertainty",
    [(1.0, 1.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.5, 0.5, 0.5, 0.5), (1.0, 0.0, 1.0, 0.0), (0.3, 0.9, 0.2, 0.8)],
)
def test_gate_always_in_unit_range(coverage, quality, reliability, uncertainty):
    g = reliability_gate(coverage, quality, reliability, uncertainty)
    assert 0.0 <= g <= 1.0


def test_high_uncertainty_lowers_gate():
    low_u = reliability_gate(1.0, 1.0, 1.0, uncertainty=0.1)
    high_u = reliability_gate(1.0, 1.0, 1.0, uncertainty=0.9)
    assert high_u < low_u


def test_low_quality_lowers_gate():
    high_q = reliability_gate(1.0, quality=0.9, reliability=1.0, uncertainty=0.0)
    low_q = reliability_gate(1.0, quality=0.1, reliability=1.0, uncertainty=0.0)
    assert low_q < high_q


def test_low_coverage_lowers_gate():
    high_c = reliability_gate(coverage=0.9, quality=1.0, reliability=1.0, uncertainty=0.0)
    low_c = reliability_gate(coverage=0.1, quality=1.0, reliability=1.0, uncertainty=0.0)
    assert low_c < high_c


def test_low_reliability_lowers_gate():
    high_r = reliability_gate(1.0, 1.0, reliability=0.9, uncertainty=0.0)
    low_r = reliability_gate(1.0, 1.0, reliability=0.1, uncertainty=0.0)
    assert low_r < high_r


def test_zero_coverage_forces_zero_gate_regardless_of_other_factors():
    g = reliability_gate(coverage=0.0, quality=1.0, reliability=1.0, uncertainty=0.0)
    assert g == 0.0


def test_full_uncertainty_forces_zero_gate():
    g = reliability_gate(coverage=1.0, quality=1.0, reliability=1.0, uncertainty=1.0)
    assert g == 0.0


def test_node_gate_narrative_high_entropy_lower_than_confident():
    confident = NarrativeEvidence(raw_scores={"dent": 0.95}, normalized_scores={"dent": 0.95},
                                  model_name="fixture", device="cpu")
    ambiguous = NarrativeEvidence(
        raw_scores={"dent": 0.5, "scratch": 0.5}, normalized_scores={"dent": 0.5, "scratch": 0.5},
        model_name="fixture", device="cpu",
    )
    packet_confident = EvidencePacket(claim=ClaimRecord(claim_id="a"), narrative=confident)
    packet_ambiguous = EvidencePacket(claim=ClaimRecord(claim_id="b"), narrative=ambiguous)
    assert node_gate(packet_ambiguous, "narrative") < node_gate(packet_confident, "narrative")


def test_node_gate_visual_zero_when_no_detections():
    empty_visual = VisualEvidence(detections=[], model_name="fixture", image_width=32, image_height=32)
    packet = EvidencePacket(claim=ClaimRecord(claim_id="c"), visual=empty_visual)
    assert node_gate(packet, "visual") == 0.0


def test_node_gate_visual_zero_when_image_corrupted():
    visual = VisualEvidence(detections=[DamageAnnotation(damage_class="dent", confidence=0.9)],
                            model_name="fixture", image_width=32, image_height=32)
    corrupted_quality = ImageQuality(width=1, height=1, aspect_ratio=1.0, file_size_bytes=10,
                                     file_format="PNG", is_corrupted=True)
    packet = EvidencePacket(claim=ClaimRecord(claim_id="d"), visual=visual, image_quality=corrupted_quality)
    assert node_gate(packet, "visual") == 0.0


def test_node_gate_forensic_uses_reliability_and_score():
    high_reliability = ForensicResult(available=True, score=0.9, reliability=0.9)
    low_reliability = ForensicResult(available=True, score=0.9, reliability=0.1)
    packet_high = EvidencePacket(claim=ClaimRecord(claim_id="e"), forensic=high_reliability)
    packet_low = EvidencePacket(claim=ClaimRecord(claim_id="f"), forensic=low_reliability)
    assert node_gate(packet_low, "forensic") < node_gate(packet_high, "forensic")


def test_gate_never_reads_expected_label_or_ground_truth():
    for fn in (gate_module.node_gate, gate_module.reliability_gate, gate_module._narrative_factors,
               gate_module._visual_factors, gate_module._forensic_factors, gate_module._image_quality_factors):
        source = inspect.getsource(fn)
        assert "expected_label" not in source
        assert ".claim" not in source
        assert "ground_truth" not in source.lower()


def test_unknown_modality_raises():
    packet = EvidencePacket(claim=ClaimRecord(claim_id="g"))
    with pytest.raises(ValueError):
        node_gate(packet, "not_a_modality")
