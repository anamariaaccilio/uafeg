"""Ablaciones: no_uncertainty, no_forensic, no_explicit_relations deben
cambiar el comportamiento del modelo/datos de forma verificable."""
import torch

from src.schemas.claim import ClaimRecord
from src.schemas.evidence import DamageAnnotation, DecisionLabel, ForensicResult, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket
from uafeg.models.gate import node_gate
from uafeg.models.p1_graph import GraphClassifier
from uafeg.models.p2_uafeg import ReliabilityGatedGraphClassifier

from experiments.ablations import ABLATIONS, apply_ablation, strip_forensic


def _packet_with_forensic():
    return EvidencePacket(
        claim=ClaimRecord(claim_id="fixture", expected_label=DecisionLabel.SUSPICIOUS),
        narrative=NarrativeEvidence(raw_scores={"dent": 0.9}, normalized_scores={"dent": 0.9},
                                    model_name="fixture", device="cpu"),
        visual=VisualEvidence(detections=[DamageAnnotation(damage_class="dent", confidence=0.7)],
                              model_name="fixture", image_width=32, image_height=32),
        forensic=ForensicResult(available=True, method="fixture", score=0.6, reliability=0.8),
    )


def test_no_uncertainty_ablation_matches_manual_disable():
    packet = _packet_with_forensic()
    ambiguous_narrative = NarrativeEvidence(
        raw_scores={"dent": 0.5, "scratch": 0.5}, normalized_scores={"dent": 0.5, "scratch": 0.5},
        model_name="fixture", device="cpu",
    )
    ambiguous_packet = packet.model_copy(update={"narrative": ambiguous_narrative})

    with_uncertainty = node_gate(ambiguous_packet, "narrative")
    without_uncertainty = node_gate(ambiguous_packet, "narrative", disable=frozenset({"uncertainty"}))
    assert without_uncertainty > with_uncertainty  # al quitar (1-u) con u>0, el gate solo puede subir o quedar igual


def test_no_forensic_ablation_strips_forensic_field():
    packets = [_packet_with_forensic()]
    stripped = strip_forensic(packets)
    assert packets[0].forensic is not None
    assert stripped[0].forensic is None


def test_no_explicit_relations_ablation_removes_cross_edge():
    packet = EvidencePacket(
        claim=ClaimRecord(claim_id="fixture"),
        narrative=NarrativeEvidence(raw_scores={"dent": 0.9}, normalized_scores={"dent": 0.9},
                                    model_name="fixture", device="cpu"),
        visual=VisualEvidence(detections=[DamageAnnotation(damage_class="dent", confidence=0.7)],
                              model_name="fixture", image_width=32, image_height=32),
    )
    with_relations = GraphClassifier(include_cross_relations=True)
    without_relations = GraphClassifier(include_cross_relations=False)

    _, _, edges_with = with_relations._node_features_and_edges(packet)
    _, _, edges_without = without_relations._node_features_and_edges(packet)

    assert len(edges_with) == 3  # narrative->claim, visual->claim, narrative->visual
    assert len(edges_without) == 2  # solo evidence_for


def test_apply_ablation_no_uncertainty_sets_model_kwarg():
    model_kwargs, packets = apply_ablation("no_uncertainty", {}, [_packet_with_forensic()])
    assert model_kwargs["disable_gate_terms"] == frozenset({"uncertainty"})
    model = ReliabilityGatedGraphClassifier(**model_kwargs)
    assert model.disable_gate_terms == frozenset({"uncertainty"})


def test_apply_ablation_no_forensic_strips_data_not_model():
    model_kwargs, packets = apply_ablation("no_forensic", {}, [_packet_with_forensic()])
    assert model_kwargs == {}
    assert packets[0].forensic is None


def test_unknown_ablation_raises():
    import pytest

    with pytest.raises(ValueError):
        apply_ablation("not_a_real_ablation", {}, [])


def test_thesis_ablations_a1_a2_a3_are_registered_with_correct_thesis_id():
    assert ABLATIONS["no_gate"]["thesis_id"] == "A1"
    assert ABLATIONS["no_forensic"]["thesis_id"] == "A2"
    assert ABLATIONS["graph_to_deepsets"]["thesis_id"] == "A3"
    assert ABLATIONS["graph_to_deepsets"]["equivalent_system"] == "B6"


def test_thesis_ablations_a4_a5_are_explicitly_not_implemented():
    import pytest

    assert ABLATIONS["no_edge_attributes"]["thesis_id"] == "A4"
    assert ABLATIONS["no_benign_no_ood_training"]["thesis_id"] == "A5"
    with pytest.raises(NotImplementedError):
        apply_ablation("no_edge_attributes", {}, [])
    with pytest.raises(NotImplementedError):
        apply_ablation("no_benign_no_ood_training", {}, [])


def test_extra_non_thesis_ablations_are_explicitly_labeled():
    assert ABLATIONS["no_uncertainty"]["thesis_id"] is None
    assert ABLATIONS["no_explicit_relations"]["thesis_id"] is None


def test_all_seven_ablation_entries_are_registered():
    assert set(ABLATIONS) == {
        "no_gate", "no_forensic", "graph_to_deepsets",
        "no_edge_attributes", "no_benign_no_ood_training",
        "no_uncertainty", "no_explicit_relations",
    }
