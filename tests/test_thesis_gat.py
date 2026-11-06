"""Invariantes de la capa GATv2 de la tesis (eq:mensaje, eq:update) y de la
comparacion controlada B6 / P1 / P2.

Estos tests verifican que el CODIGO respeta el diseno y que las tres
arquitecturas solo difieren en lo que la tesis quiere aislar. No miden
desempeno: el benchmark real sigue bloqueado (ver docs/TRAINING.md) y
cualquier cifra producida aqui es SMOKE_RESULT.
"""
import inspect

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
from uafeg.models.b6_deepsets import DeepSetsClassifier
from uafeg.models.p1_graph import (
    EDGE_FEATURE_DIM,
    GAT_VARIANTS,
    THESIS_RELATION_INDEX,
    GraphClassifier,
    ThesisGATv2Layer,
)
from uafeg.models.p2_uafeg import ReliabilityGatedGraphClassifier

DIM = 8


def packet(all_modalities: bool = True) -> EvidencePacket:
    return EvidencePacket(
        claim=ClaimRecord(claim_id="fixture", source_image_id="src", expected_label=DecisionLabel.SUSPICIOUS),
        narrative=NarrativeEvidence(raw_scores={"dent": 0.9}, normalized_scores={"dent": 0.9},
                                    model_name="fixture", device="cpu"),
        visual=VisualEvidence(detections=[DamageAnnotation(damage_class="dent", confidence=0.7)],
                              model_name="fixture", image_width=32, image_height=32),
        image_quality=ImageQuality(width=32, height=32, aspect_ratio=1.0, file_size_bytes=100,
                                   file_format="JPEG", laplacian_variance=80.0) if all_modalities else None,
        forensic=ForensicResult(available=True, method="fixture", score=0.6,
                                reliability=0.8) if all_modalities else None,
    )


def toy_graph():
    """3 nodos, una arista con atributos observados y otra enmascarada."""
    features = torch.randn(3, DIM)
    edges = [
        (1, 0, THESIS_RELATION_INDEX["evidence_for"], (0.0,) * EDGE_FEATURE_DIM),
        (1, 2, THESIS_RELATION_INDEX["support"], (0.6, 0.1, 0.3, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0)),
    ]
    return features, edges


def test_layer_preserves_shape():
    features, edges = toy_graph()
    out = ThesisGATv2Layer(DIM, heads=2)(features, edges)
    assert out.shape == features.shape


def test_layer_gradients_are_finite_and_reach_used_relation_weights():
    torch.manual_seed(0)
    layer = ThesisGATv2Layer(DIM, heads=2)
    features, edges = toy_graph()
    layer(features, edges).sum().backward()

    grads = [p.grad for p in layer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    for _, _, relation, _ in edges:
        assert layer.relation_value[relation].weight.grad is not None


def test_layer_rejects_dim_not_divisible_by_heads():
    with pytest.raises(ValueError):
        ThesisGATv2Layer(7, heads=2)


def test_edge_attributes_change_the_output():
    """Si a_ij no alterase el resultado, la atencion no lo estaria usando."""
    torch.manual_seed(0)
    layer = ThesisGATv2Layer(DIM, heads=2)
    features, edges = toy_graph()
    masked = [(s, d, r, (0.0,) * EDGE_FEATURE_DIM) for s, d, r, _ in edges]
    assert not torch.allclose(layer(features, edges), layer(features, masked))


def test_relation_type_changes_the_message():
    """W_r por relacion: cambiar el tipo cambia lo que se propaga."""
    torch.manual_seed(0)
    layer = ThesisGATv2Layer(DIM, heads=2)
    features, edges = toy_graph()
    retyped = [(s, d, THESIS_RELATION_INDEX["contradiction"], a) for s, d, _, a in edges]
    assert not torch.allclose(layer(features, edges), layer(features, retyped))


def test_residual_keeps_isolated_node_signal():
    """Un nodo sin vecinos conserva su propia senal via el residual."""
    torch.manual_seed(0)
    layer = ThesisGATv2Layer(DIM, heads=1)
    features = torch.randn(2, DIM)
    out = layer(features, [])
    assert out.shape == features.shape
    assert torch.isfinite(out).all()


def test_gate_equal_one_reproduces_p1_exactly_under_thesis_variant(monkeypatch):
    """Invariante central: P1 y P2 solo difieren en g."""
    import uafeg.models.p2_uafeg as p2_module

    monkeypatch.setattr(p2_module, "node_gate", lambda *args, **kwargs: 1.0)
    torch.manual_seed(7)
    p1 = GraphClassifier(gat_variant="thesis")
    torch.manual_seed(7)
    p2 = ReliabilityGatedGraphClassifier(gat_variant="thesis")

    sample = packet()
    assert torch.allclose(p1.forward(sample), p2.forward(sample), atol=1e-6)


def test_gate_below_one_changes_output_under_thesis_variant(monkeypatch):
    import uafeg.models.p2_uafeg as p2_module

    monkeypatch.setattr(p2_module, "node_gate", lambda *args, **kwargs: 0.2)
    torch.manual_seed(7)
    p1 = GraphClassifier(gat_variant="thesis")
    torch.manual_seed(7)
    p2 = ReliabilityGatedGraphClassifier(gat_variant="thesis")

    sample = packet()
    assert not torch.allclose(p1.forward(sample), p2.forward(sample), atol=1e-6)


def test_gate_scales_the_message_not_the_residual():
    """eq:update aplica el residual sobre h_i sin compuerta: con g=0 el nodo
    conserva su propia representacion en vez de anularse."""
    torch.manual_seed(0)
    layer = ThesisGATv2Layer(DIM, heads=2)
    features, edges = toy_graph()
    zero_gate = layer(features, edges, message_scale=torch.zeros(features.shape[0]))
    assert torch.isfinite(zero_gate).all()
    assert not torch.allclose(zero_gate, torch.zeros_like(zero_gate))


def test_thesis_is_the_default_variant_and_legacy_remains_available():
    assert GAT_VARIANTS == ("thesis", "legacy")
    assert GraphClassifier().gat_variant == "thesis"
    assert ReliabilityGatedGraphClassifier().gat_variant == "thesis"
    assert isinstance(GraphClassifier(gat_variant="legacy").gat_layer, torch.nn.Module)
    with pytest.raises(ValueError):
        GraphClassifier(gat_variant="not_a_variant")


def test_variants_produce_different_outputs():
    sample = packet()
    torch.manual_seed(3)
    thesis = GraphClassifier(gat_variant="thesis")
    torch.manual_seed(3)
    legacy = GraphClassifier(gat_variant="legacy")
    assert not torch.allclose(thesis.forward(sample), legacy.forward(sample), atol=1e-6)


def test_b6_p1_and_p2_all_pool_by_mean():
    """B6 vs P1 debe aislar el grafo, no un pooling distinto: si uno sumara,
    su magnitud escalaria con la cantidad de modalidades presentes."""
    assert ".mean(dim=0)" in inspect.getsource(DeepSetsClassifier.forward_from_features)
    assert ".mean(dim=0)" in inspect.getsource(GraphClassifier.forward)
    # P2 no reimplementa forward: hereda el pooling de P1.
    assert "forward" not in ReliabilityGatedGraphClassifier.__dict__


def test_b6_p1_p2_share_encoder_and_classifier_shapes():
    b6, p1, p2 = DeepSetsClassifier(), GraphClassifier(), ReliabilityGatedGraphClassifier()
    sample = packet()
    shapes = {tuple(model.forward(sample).shape) for model in (b6, p1, p2)}
    assert shapes == {(3,)}  # SUPPORTED / SUSPICIOUS / INSUFFICIENT
    encoder_shapes = [
        {name: tuple(param.shape) for name, param in model.encoder_bank.state_dict().items()}
        for model in (b6, p1, p2)
    ]
    assert encoder_shapes[0] == encoder_shapes[1] == encoder_shapes[2]


def test_thesis_layer_never_reads_expected_label():
    source = inspect.getsource(ThesisGATv2Layer)
    assert "expected_label" not in source
    assert "condition" not in source
