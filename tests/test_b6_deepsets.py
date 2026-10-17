"""Smoke tests de B6 (DeepSets) sobre fixtures sinteticas pequenas. Estos
tests validan que el codigo FUNCIONA (forward/train/checkpoint/inferencia),
NO que el modelo tenga una metrica cientifica real: el benchmark real sigue
bloqueado (ver PROJECT_COMPLETION_GAP.md). Cualquier numero aqui es
SMOKE_RESULT, nunca REAL_RESULT."""
import numpy as np
import pytest
import torch

from src.schemas.claim import ClaimRecord
from src.schemas.evidence import (
    DamageAnnotation,
    DecisionLabel,
    ForensicResult,
    ImageQuality,
    NarrativeEvidence,
    VisualEvidence,
)
from src.schemas.packet import EvidencePacket
from src.evaluation.metrics import CLASS_ORDER
from uafeg.models.b6_deepsets import DeepSetsClassifier, NUM_CLASSES
from uafeg.models.features import packet_to_modality_features
from uafeg.models.training import (
    labels_to_indices,
    load_checkpoint,
    predict_logits,
    predict_proba,
    save_checkpoint,
    train,
)


def _packet(claim_id: str, label: DecisionLabel, with_visual: bool = True, with_forensic: bool = False) -> EvidencePacket:
    narrative = NarrativeEvidence(
        raw_scores={"dent": 0.8}, normalized_scores={"dent": 0.8}, model_name="fixture", device="cpu"
    )
    visual = None
    if with_visual:
        visual = VisualEvidence(
            detections=[DamageAnnotation(damage_class="dent", confidence=0.7)],
            model_name="fixture", image_width=32, image_height=32,
        )
    forensic = None
    if with_forensic:
        forensic = ForensicResult(available=True, method="fixture", score=0.4, reliability=0.6)
    return EvidencePacket(
        claim=ClaimRecord(claim_id=claim_id, expected_label=label),
        narrative=narrative,
        visual=visual,
        forensic=forensic,
    )


def _toy_dataset(n=12, seed=0):
    rng = np.random.default_rng(seed)
    labels = [CLASS_ORDER[i] for i in rng.integers(0, NUM_CLASSES, size=n)]
    packets = [_packet(f"claim-{i}", DecisionLabel(label), with_visual=(i % 2 == 0)) for i, label in enumerate(labels)]
    return packets, labels


def test_packet_to_modality_features_excludes_absent_modalities():
    packet = _packet("only-narrative", DecisionLabel.SUPPORTED, with_visual=False)
    features = packet_to_modality_features(packet)
    assert set(features.keys()) == {"narrative"}


def test_forward_single_packet_returns_logits_for_present_modalities_only():
    model = DeepSetsClassifier()
    packet = _packet("fixture", DecisionLabel.SUSPICIOUS, with_visual=True, with_forensic=True)
    logits = model.forward(packet)
    assert logits.shape == (NUM_CLASSES,)


def test_forward_rejects_packet_with_no_evidence_at_all():
    model = DeepSetsClassifier()
    empty_packet = EvidencePacket(claim=ClaimRecord(claim_id="empty"))
    with pytest.raises(ValueError):
        model.forward(empty_packet)


def test_forward_batch_handles_variable_modality_sets():
    model = DeepSetsClassifier()
    packets, _ = _toy_dataset(n=6)
    logits = model.forward_batch(packets)
    assert logits.shape == (6, NUM_CLASSES)


def test_training_reduces_loss_on_toy_dataset():
    packets, labels = _toy_dataset(n=20, seed=1)
    model = DeepSetsClassifier()
    history = train(model, packets, labels, epochs=30, lr=0.05, seed=42)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_training_reports_validation_without_updating_on_it():
    train_packets, train_labels = _toy_dataset(n=10, seed=2)
    val_packets, val_labels = _toy_dataset(n=5, seed=3)
    model = DeepSetsClassifier()
    history = train(model, train_packets, train_labels, val_packets, val_labels, epochs=5, lr=0.05)
    assert len(history["val_loss"]) == 5
    assert all(0.0 <= acc <= 1.0 for acc in history["val_accuracy"])


def test_predict_proba_sums_to_one():
    packets, _ = _toy_dataset(n=4, seed=4)
    model = DeepSetsClassifier()
    probabilities = predict_proba(model, packets)
    assert probabilities.shape == (4, NUM_CLASSES)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_predict_logits_match_manual_forward():
    packets, _ = _toy_dataset(n=3, seed=5)
    model = DeepSetsClassifier()
    logits = predict_logits(model, packets)
    manual = model.forward_batch(packets).detach().numpy()
    assert np.allclose(logits, manual)


def test_checkpoint_roundtrip_preserves_predictions(tmp_path):
    packets, labels = _toy_dataset(n=8, seed=6)
    model = DeepSetsClassifier()
    train(model, packets, labels, epochs=5, lr=0.05)
    before = predict_proba(model, packets)

    checkpoint_path = tmp_path / "b6.pt"
    metadata = {"system": "B6", "seed": 42, "run_id": "smoke-test"}
    save_checkpoint(model, checkpoint_path, metadata)

    restored = DeepSetsClassifier()
    loaded_metadata = load_checkpoint(restored, checkpoint_path)
    after = predict_proba(restored, packets)

    assert loaded_metadata == metadata
    assert np.allclose(before, after)


def test_labels_to_indices_rejects_unknown_label():
    with pytest.raises(ValueError):
        labels_to_indices(["NOT_A_LABEL"])


def test_model_never_receives_expected_label_as_a_feature():
    packet = _packet("fixture", DecisionLabel.SUSPICIOUS)
    features = packet_to_modality_features(packet)
    for vector in features.values():
        assert vector.dtype == np.float32
    # packet_to_modality_features nunca debe leer packet.claim (ahi vive
    # expected_label); solo lee los campos de evidencia por modalidad.
    import inspect

    from uafeg.models.features import packet_to_modality_features as fn

    source = inspect.getsource(fn)
    assert "expected_label" not in source
    assert ".claim" not in source
