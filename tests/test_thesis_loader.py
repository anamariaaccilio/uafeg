"""Anti-leakage controls of the real loader and of the thesis-taxonomy graph."""
import json

import numpy as np
import pytest

from experiments import thesis_data
from experiments.thesis_data import FORBIDDEN_FEATURE_KEYS, flat_features, training_view
from src.schemas.claim import ClaimRecord
from src.schemas.evidence import (DamageAnnotation, DecisionLabel, ImageQuality, NarrativeAssertion,
                                  NarrativeEvidence, VisualEvidence, CANONICAL_DAMAGE_CLASSES)
from src.schemas.packet import EvidencePacket
from uafeg.models.batched import collate
from uafeg.thesis_graph import EvidenceView, build_thesis_graph, evidence_view
from tests.test_graph import packet as base_packet


def thesis_packet():
    p = base_packet()
    nli = {k: ((0.9, 0.05, 0.05) if k == "dent" else (0.05, 0.9, 0.05)) for k in CANONICAL_DAMAGE_CLASSES}
    narrative = NarrativeEvidence(raw_scores=p.narrative.raw_scores, normalized_scores=p.narrative.normalized_scores,
                                  model_name="fixture", device="cpu", assertions=[NarrativeAssertion(
                                      text="x", damage_type="dent", polarity="affirmed", p=(0.9, 0.05, 0.05),
                                      u=0.3, coverage=0.25, reliability=0.8, class_nli=nli)])
    visual = VisualEvidence(model_name="fixture", image_width=32, image_height=32, detections=[
        DamageAnnotation(damage_class="dent", confidence=0.9, bbox_norm=(0.1, 0.1, 0.5, 0.5), area_fraction=0.16,
                         uncertainty=0.47, reliability=0.7, region_quality=0.6, coverage=1.0,
                         region_quality_factors={"resolution": 1.0, "sharpness": 0.6, "exposure": 1.0}),
        DamageAnnotation(damage_class="scratch", confidence=0.4, bbox_norm=(0.2, 0.2, 0.6, 0.6), area_fraction=0.16,
                         uncertainty=0.97, reliability=0.5, region_quality=0.5, coverage=1.0)])
    iq = ImageQuality(width=32, height=32, aspect_ratio=1.0, file_size_bytes=100, file_format="PNG",
                      laplacian_variance=50.0, quality_q=0.5,
                      quality_factors={"resolution": 0.5, "sharpness": 1.0, "exposure": 1.0},
                      source_s=None, source_s_status="NOT_OBSERVABLE_NO_EXIF")
    return p.model_copy(update={"narrative": narrative, "visual": visual, "image_quality": iq})


def test_graph_never_reads_the_claim_record():
    p = thesis_packet()
    g1 = build_thesis_graph(evidence_view(p))
    tampered = p.model_copy(deep=True)
    tampered.claim.expected_label = DecisionLabel.SUSPICIOUS
    tampered.claim.claim_id = "other"
    tampered.claim.source_image_id = "hitl:other"
    tampered.claim.image_path = "elsewhere.png"
    g2 = build_thesis_graph(evidence_view(tampered))
    assert np.array_equal(g1.x, g2.x) and np.array_equal(g1.gate, g2.gate) and g1.edges == g2.edges
    with pytest.raises(TypeError):
        build_thesis_graph(p)  # a full packet (with its ClaimRecord) is refused


def test_taxonomy_and_gate_follow_the_thesis():
    g = build_thesis_graph(evidence_view(thesis_packet()))
    from uafeg.thesis_graph import NODE_KINDS
    kinds = [NODE_KINDS[k] for k in g.kind]
    assert kinds == ["claim", "source", "visual", "visual"]
    c, q, r, u = g.gate[0]
    assert (c, r) == (0.25, pytest.approx(0.8)) and u == pytest.approx(0.3)
    assert g.gate[2].tolist() == pytest.approx([1.0, 0.6, 0.7, 0.47])
    relations = {(s, d) for s, d, _, _ in g.edges}
    assert (0, 2) in relations and (2, 0) in relations and (1, 2) in relations
    support = [a for s, d, rel, a in g.edges if (s, d) == (0, 2)][0]
    assert support[:3] == (0.9, 0.05, 0.05) and support[3] == 1.0  # entail, contradiction, neutral, sim_sem


def test_tensor_batch_contains_no_label_or_identifier_fields():
    batch = collate([build_thesis_graph(evidence_view(thesis_packet()))])
    assert not set(batch) & FORBIDDEN_FEATURE_KEYS
    assert set(flat_features(build_thesis_graph(evidence_view(thesis_packet())))) == {"text", "visual", "forensic"}


def test_loader_verifies_lock_and_packet_checksums(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(thesis_data, "verify_benchmark", lambda lock: calls.append(lock) or {"benchmark_version": "x", "git_commit": "y"})
    (tmp_path / "train.jsonl").write_text("{}\n")
    (tmp_path / "packets_manifest.json").write_text(json.dumps({"files": {"train.jsonl": {"sha256": "0" * 64}}}))
    monkeypatch.setattr(thesis_data, "packets_dir", lambda: tmp_path)
    with pytest.raises(ValueError, match="EvidencePacket file changed"):
        thesis_data.verify_inputs()
    assert calls, "the lock must be verified before reading any packet"


def test_one_source_one_split_and_a5_filter(monkeypatch):
    g = build_thesis_graph(evidence_view(thesis_packet()))
    mk = lambda name, src, conds: thesis_data.Split(name, [g] * len(conds), np.array([0] * len(conds)),
                                                    [{"source_image_id": src, "condition": c} for c in conds])
    fake = {"train": mk("train", "a", ["C0", "C8"]), "validation": mk("validation", "b", ["C0"]),
            "test": mk("test", "a", ["C0"])}
    monkeypatch.setattr(thesis_data, "load_split", lambda s, **k: fake[s])
    with pytest.raises(ValueError, match="leakage"):
        thesis_data.load_all()
    assert len(training_view(fake["train"], ("C8",)).labels) == 1
