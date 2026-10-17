"""Smoke tests de P1 (EvidenceGraph -> GATv2 relacional, sin reliability
gate) sobre fixtures sinteticas pequenas. Validan que el codigo funciona
(forward/train/checkpoint), no una metrica cientifica real."""
import numpy as np
import pytest
import torch

from src.schemas.claim import ClaimRecord
from src.schemas.evidence import DamageAnnotation, DecisionLabel, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket
from src.evaluation.metrics import CLASS_ORDER
from uafeg.models.p1_graph import GraphClassifier, NUM_CLASSES
from uafeg.models.training import predict_proba, train


def _packet(claim_id: str, label: DecisionLabel, with_visual=True, narrative_class="dent", visual_class="dent"):
    narrative = NarrativeEvidence(
        raw_scores={narrative_class: 0.9}, normalized_scores={narrative_class: 0.9},
        model_name="fixture", device="cpu",
    )
    visual = None
    if with_visual:
        visual = VisualEvidence(
            detections=[DamageAnnotation(damage_class=visual_class, confidence=0.7)],
            model_name="fixture", image_width=32, image_height=32,
        )
    return EvidencePacket(claim=ClaimRecord(claim_id=claim_id, expected_label=label), narrative=narrative, visual=visual)


def _toy_dataset(n=12, seed=0):
    rng = np.random.default_rng(seed)
    labels = [CLASS_ORDER[i] for i in rng.integers(0, NUM_CLASSES, size=n)]
    packets = [_packet(f"claim-{i}", DecisionLabel(label), with_visual=(i % 2 == 0)) for i, label in enumerate(labels)]
    return packets, labels


def test_forward_single_packet_returns_logits():
    model = GraphClassifier()
    packet = _packet("fixture", DecisionLabel.SUSPICIOUS)
    logits = model.forward(packet)
    assert logits.shape == (NUM_CLASSES,)


def test_forward_rejects_packet_with_no_evidence():
    model = GraphClassifier()
    empty_packet = EvidencePacket(claim=ClaimRecord(claim_id="empty"))
    with pytest.raises(ValueError):
        model.forward(empty_packet)


def test_forward_batch_handles_variable_graphs():
    model = GraphClassifier()
    packets, _ = _toy_dataset(n=6)
    logits = model.forward_batch(packets)
    assert logits.shape == (6, NUM_CLASSES)


def test_contradiction_and_support_graphs_produce_different_logits():
    model = GraphClassifier()
    support_packet = _packet("support", DecisionLabel.SUPPORTED, narrative_class="dent", visual_class="dent")
    contradiction_packet = _packet("contradiction", DecisionLabel.SUPPORTED, narrative_class="dent", visual_class="scratch")
    support_logits = model.forward(support_packet)
    contradiction_logits = model.forward(contradiction_packet)
    assert not torch.allclose(support_logits, contradiction_logits)


def test_training_reduces_loss_on_toy_dataset():
    packets, labels = _toy_dataset(n=16, seed=1)
    model = GraphClassifier()
    history = train(model, packets, labels, epochs=30, lr=0.05, seed=42)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_predict_proba_sums_to_one():
    packets, _ = _toy_dataset(n=5, seed=2)
    model = GraphClassifier()
    probabilities = predict_proba(model, packets)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_graph_relations_never_read_expected_label():
    import inspect

    from uafeg.models.p1_graph import GraphClassifier as GC

    source = inspect.getsource(GC._node_features_and_edges)
    assert "expected_label" not in source
