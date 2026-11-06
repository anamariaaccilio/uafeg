"""Atributos continuos de arista a_ij = [p_support, p_contradiction,
p_unknown, sim_sem, overlap_spatial] (eq:aristas de la tesis) y su mascara
de observabilidad.

Estos tests comprueban el CONTRATO de los atributos (reparto de masa,
enmascarado de lo no computable, ausencia de fuga de etiquetas), no una
metrica cientifica: el benchmark real sigue bloqueado (ver docs/TRAINING.md).
"""
import pytest

from src.schemas.evidence import DamageAnnotation, DecisionLabel, NarrativeEvidence, VisualEvidence
from uafeg.graph import build_graph
from uafeg.relations import (
    EDGE_ATTRIBUTE_DIM,
    EDGE_ATTRIBUTE_NAMES,
    EdgeAttributes,
    narrative_visual_attributes,
)

from tests.test_graph import packet as make_packet  # fixture canonica ya usada por el resto


def narrative(scores):
    return NarrativeEvidence(
        raw_scores=dict(scores),
        normalized_scores=dict(scores),
        model_name="test-nli",
        device="cpu",
    )


def visual(detections):
    return VisualEvidence(
        detections=[DamageAnnotation(damage_class=c, confidence=conf) for c, conf in detections],
        model_name="test-yolo",
        image_width=640,
        image_height=480,
    )


def test_attribute_vector_has_thesis_length_and_order():
    assert EDGE_ATTRIBUTE_NAMES == ("p_support", "p_contradiction", "p_unknown", "sim_sem", "overlap_spatial")
    assert EDGE_ATTRIBUTE_DIM == 5
    assert len(EdgeAttributes.unobserved().as_vector()) == 2 * EDGE_ATTRIBUTE_DIM


def test_probabilities_sum_to_one_and_support_dominates_on_agreement():
    attributes = narrative_visual_attributes(narrative({"dent": 0.9}), visual([("dent", 0.8)]))
    p_support, p_contradiction, p_unknown = attributes.values[:3]
    assert pytest.approx(p_support + p_contradiction + p_unknown, abs=1e-6) == 1.0
    assert p_support > p_contradiction and p_support > p_unknown


def test_detected_class_not_claimed_counts_as_contradiction():
    attributes = narrative_visual_attributes(narrative({"dent": 0.9}), visual([("scratch", 0.9)]))
    p_support, p_contradiction, p_unknown = attributes.values[:3]
    assert p_contradiction > 0.0
    assert p_support == 0.0
    # La clase afirmada y no detectada aporta a unknown, nunca a contradiction:
    # la ausencia de deteccion no es evidencia de ausencia de danio.
    assert p_unknown > 0.0


def test_claimed_but_undetected_mass_goes_to_unknown_not_contradiction():
    attributes = narrative_visual_attributes(narrative({"dent": 0.9}), visual([]))
    p_support, p_contradiction, p_unknown = attributes.values[:3]
    assert (p_support, p_contradiction) == (0.0, 0.0)
    assert p_unknown == pytest.approx(1.0)


def test_no_signal_at_all_is_fully_unknown():
    attributes = narrative_visual_attributes(narrative({"dent": 0.1}), visual([]))
    assert attributes.values[2] == pytest.approx(1.0)


def test_missing_confidence_masks_sim_sem_instead_of_inventing_it():
    attributes = narrative_visual_attributes(narrative({"dent": 0.9}), visual([("dent", None)]))
    sim_index = EDGE_ATTRIBUTE_NAMES.index("sim_sem")
    assert attributes.mask[sim_index] == 0.0
    assert attributes.values[sim_index] == 0.0


def test_overlap_spatial_is_always_masked_today():
    """Ninguna de las dos puntas expone una region comparable: la narrativa
    no localiza y ForensicResult no publica localizacion."""
    attributes = narrative_visual_attributes(narrative({"dent": 0.9}), visual([("dent", 0.8)]))
    overlap_index = EDGE_ATTRIBUTE_NAMES.index("overlap_spatial")
    assert attributes.mask[overlap_index] == 0.0
    assert attributes.values[overlap_index] == 0.0


def test_evidence_for_edges_carry_a_fully_masked_vector():
    graph = build_graph(make_packet())
    star_edges = [edge for edge in graph.edges if edge.relation == "evidence_for"]
    assert star_edges
    for edge in star_edges:
        assert edge.attributes.mask == (0.0,) * EDGE_ATTRIBUTE_DIM


def test_cross_edge_in_built_graph_carries_observed_attributes():
    graph = build_graph(make_packet())
    cross = [edge for edge in graph.edges if edge.relation != "evidence_for"]
    assert len(cross) == 1
    assert cross[0].attributes.mask[:3] == (1.0, 1.0, 1.0)


def test_without_attributes_keeps_relation_type_and_drops_scores():
    """A4 de la tesis opera sobre el vector, no sobre el tipo de arista."""
    attributes = narrative_visual_attributes(narrative({"dent": 0.9}), visual([("dent", 0.8)]))
    stripped = attributes.without_attributes()
    assert stripped.values == (0.0,) * EDGE_ATTRIBUTE_DIM
    assert stripped.mask == (0.0,) * EDGE_ATTRIBUTE_DIM


def test_attributes_never_depend_on_expected_label():
    packet = make_packet()
    baseline = build_graph(packet)
    flipped = packet.model_copy(deep=True)
    flipped.claim.expected_label = DecisionLabel.SUPPORTED
    assert [edge.attributes for edge in build_graph(flipped).edges] == [edge.attributes for edge in baseline.edges]
