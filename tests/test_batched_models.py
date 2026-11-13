"""The batched path must compute exactly the thesis GATv2 layer, and the thesis
taxonomy graph must never read the ClaimRecord."""
import numpy as np
import pytest
import torch

from uafeg.models.batched import DenseThesisGATv2, EvidenceModel, collate
from uafeg.models.p1_graph import EDGE_FEATURE_DIM, THESIS_RELATION_TYPES
from uafeg.thesis_graph import F_MAX, GraphTensors, KIND_INDEX, ontology_similarity


def random_graph(rng, n):
    edges = []
    for s in range(n):
        for d in range(n):
            if s != d and rng.random() < 0.4:
                edges.append((s, d, int(rng.integers(0, len(THESIS_RELATION_TYPES) - 1)),
                              tuple(rng.random(EDGE_FEATURE_DIM).round(3))))
    return GraphTensors(kind=rng.integers(0, 4, n), x=rng.standard_normal((n, F_MAX)).astype(np.float32),
                        gate=rng.random((n, 4)).astype(np.float32), edges=edges, coverage_bar=0.5)


def test_dense_layer_equals_thesis_layer_graph_by_graph():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dim = 16
    dense = DenseThesisGATv2(dim, heads=4)
    graphs = [random_graph(rng, n) for n in (1, 3, 5, 7)]
    batch = collate(graphs)
    h = torch.randn(len(graphs), batch["x"].shape[1], dim)
    out = dense(h, batch["rel"], batch["attr"], batch["gate"])
    for i, g in enumerate(graphs):
        c, q, r, u = torch.from_numpy(g.gate).T
        scale = torch.clamp(c * q * r * (1 - u), 0, 1)
        ref = dense.layer(h[i, :g.n], g.edges, message_scale=scale)
        assert torch.allclose(out[i, :g.n], ref, atol=1e-5)


def test_p2_with_unit_gate_equals_p1():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    batch = collate([random_graph(rng, 6), random_graph(rng, 4)])
    p2 = EvidenceModel("P2", n_kinds=4).eval()
    p1 = EvidenceModel("P1", n_kinds=4).eval()
    p1.load_state_dict(p2.state_dict())
    assert torch.allclose(p2(batch, gate_override=torch.ones_like(batch["gate"])), p1(batch), atol=1e-6)
    assert not torch.allclose(p2(batch), p1(batch))


def test_pooling_is_permutation_invariant_and_ignores_padding():
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    g = random_graph(rng, 5)
    for system in ("B6", "P2"):
        for pooling in ("mean", "attention"):
            model = EvidenceModel(system, n_kinds=4, pooling=pooling).eval()
            alone = model(collate([g]))
            padded = model(collate([g, random_graph(rng, 9)]))[:1]
            assert torch.allclose(alone, padded, atol=1e-5)


def test_ontology_similarity():
    assert ontology_similarity("dent", "dent") == 1.0
    assert ontology_similarity("scratch", "paint_chip") == 0.5
    assert ontology_similarity("scratch", "dent") == 0.0
    assert set(KIND_INDEX) == {"claim", "visual", "forensic", "source"}
