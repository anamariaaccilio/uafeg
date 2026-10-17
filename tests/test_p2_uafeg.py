"""P2 = P1 + reliability gate. Tests exigidos: gate=1 reproduce P1 exacto,
gradientes fluyen, y no hay leakage de target."""
import inspect

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
from uafeg.models import p2_uafeg as p2_module
from uafeg.models.p1_graph import GraphClassifier
from uafeg.models.p2_uafeg import ReliabilityGatedGraphClassifier


def _packet(claim_id: str, label: DecisionLabel, all_modalities: bool = False) -> EvidencePacket:
    narrative = NarrativeEvidence(raw_scores={"dent": 0.9}, normalized_scores={"dent": 0.9},
                                  model_name="fixture", device="cpu")
    visual = VisualEvidence(detections=[DamageAnnotation(damage_class="dent", confidence=0.7)],
                            model_name="fixture", image_width=32, image_height=32)
    forensic = None
    image_quality = None
    if all_modalities:
        forensic = ForensicResult(available=True, method="fixture", score=0.3, reliability=0.6)
        image_quality = ImageQuality(width=100, height=100, aspect_ratio=1.0, file_size_bytes=1000,
                                     file_format="PNG", laplacian_variance=50.0, is_corrupted=False)
    return EvidencePacket(
        claim=ClaimRecord(claim_id=claim_id, expected_label=label),
        narrative=narrative, visual=visual, forensic=forensic, image_quality=image_quality,
    )


def test_gate_equal_one_reproduces_p1_exactly(monkeypatch):
    monkeypatch.setattr(p2_module, "node_gate", lambda packet, kind, disable=frozenset(): 1.0)
    packet = _packet("fixture", DecisionLabel.SUSPICIOUS)

    p1 = GraphClassifier()
    p2 = ReliabilityGatedGraphClassifier()
    p2.load_state_dict(p1.state_dict())  # mismos pesos exactos

    p1_logits = p1.forward(packet)
    p2_logits = p2.forward(packet)
    assert torch.allclose(p1_logits, p2_logits, atol=1e-6)


def test_gate_below_one_changes_output_relative_to_p1(monkeypatch):
    monkeypatch.setattr(p2_module, "node_gate", lambda packet, kind, disable=frozenset(): 0.1)
    packet = _packet("fixture", DecisionLabel.SUSPICIOUS)

    p1 = GraphClassifier()
    p2 = ReliabilityGatedGraphClassifier()
    p2.load_state_dict(p1.state_dict())

    p1_logits = p1.forward(packet)
    p2_logits = p2.forward(packet)
    assert not torch.allclose(p1_logits, p2_logits, atol=1e-6)


def test_gradients_flow_through_gated_pipeline():
    # Packet con las 4 modalidades presentes para que TODOS los encoders
    # participen en este forward (un encoder de una modalidad ausente no
    # recibe gradiente ese paso; eso es esperado, no un bug).
    model = ReliabilityGatedGraphClassifier()
    packet = _packet("fixture", DecisionLabel.SUSPICIOUS, all_modalities=True)
    logits = model.forward(packet)
    loss = logits.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_p2_never_reads_expected_label():
    source = inspect.getsource(p2_module)
    assert "expected_label" not in source
