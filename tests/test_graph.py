import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from src.schemas.claim import Claim, ClaimRecord
from src.schemas.packet import EvidencePacket
from src.schemas.evidence import (
    DecisionLabel, NarrativeEvidence, VisualEvidence, DamageAnnotation, ForensicResult,
)
from uafeg import build_graph
from uafeg.graph import EvidenceGraph, Node, Edge
from uafeg.benchmark import verify_benchmark

LOCK = Path(__file__).resolve().parents[1] / "benchmark.lock.json"


def packet():
    return EvidencePacket(
        claim=ClaimRecord(claim_id="fixture-claim", source_image_id="fixture-source",
                          expected_label=DecisionLabel.SUSPICIOUS,
                          metadata={"condition": "fixture-only"}),
        narrative=NarrativeEvidence(raw_scores={"dent": .8}, normalized_scores={"dent": 1.},
                                    model_name="synthetic-fixture", device="cpu"),
        visual=VisualEvidence(detections=[DamageAnnotation(damage_class="dent", confidence=.7)],
                              model_name="synthetic-fixture", image_width=32, image_height=32),
    )


def test_benchmark_contract_compatibility():
    assert ClaimRecord is Claim
    lock = verify_benchmark(LOCK)
    assert lock["schema_version"] == packet().schema_version


def test_lock_rejects_wrong_commit_and_hash(tmp_path):
    for key in ["git_commit", "contract_files_sha256"]:
        lock = json.loads(LOCK.read_text())
        if key == "git_commit":
            lock[key] = "0" * 40
        else:
            path = next(iter(lock[key]))
            lock[key][path] = "0" * 64
        path = tmp_path / "lock.json"
        path.write_text(json.dumps(lock))
        with pytest.raises(ValueError):
            verify_benchmark(path)


def test_packet_compatibility_and_graph_construction():
    original = packet()
    loaded = EvidencePacket.model_validate_json(original.model_dump_json())
    graph = build_graph(loaded)
    assert graph.claim_id == original.claim.claim_id
    assert graph.source_image_id == original.claim.source_image_id
    assert [n.kind for n in graph.nodes] == ["claim", "narrative", "visual"]
    assert [(e.source, e.target) for e in graph.edges] == [("narrative", "claim"), ("visual", "claim")]
    assert EvidenceGraph.model_validate_json(graph.model_dump_json()) == graph
    assert build_graph(loaded) == graph


def test_graph_never_copies_ground_truth_or_condition_metadata():
    original = packet()
    graph = build_graph(original)
    original.claim.expected_label = DecisionLabel.SUPPORTED
    original.claim.metadata = {"condition": "changed"}
    assert build_graph(original) == graph
    assert "expected_label" not in graph.model_dump_json()
    assert "fixture-only" not in graph.model_dump_json()


def test_missing_evidence_and_unavailable_forensic():
    graph = build_graph(EvidencePacket(claim=ClaimRecord(claim_id="missing"),
                                       forensic=ForensicResult(available=False)))
    assert len(graph.nodes) == 1
    assert graph.edges == ()


def test_available_empty_visual_is_not_missing():
    original = packet()
    original.visual.detections = []
    assert "visual" in [n.kind for n in build_graph(original).nodes]


def test_available_forensic_is_retained():
    original = packet()
    original.forensic = ForensicResult(available=True, method="fixture", score=.3)
    assert "forensic" in [n.kind for n in build_graph(original).nodes]


@pytest.mark.parametrize("change", ["duplicate_node", "no_claim", "dangling_edge", "self_edge", "missing_edge", "duplicate_edge"])
def test_graph_rejects_invalid_nodes_and_edges(change):
    data = build_graph(packet()).model_dump()
    if change == "duplicate_node": data["nodes"] += (data["nodes"][0],)
    if change == "no_claim": data["nodes"] = data["nodes"][1:]
    if change == "dangling_edge": data["edges"][0]["target"] = "absent"
    if change == "self_edge": data["edges"][0]["target"] = "narrative"
    if change == "missing_edge": data["edges"] = ()
    if change == "duplicate_edge": data["edges"] += (data["edges"][0],)
    with pytest.raises(ValidationError):
        EvidenceGraph.model_validate(data)


def test_node_rejects_mismatched_evidence():
    with pytest.raises(ValidationError):
        Node(id="visual", kind="visual", evidence=packet().narrative)
    with pytest.raises(ValidationError):
        Node(id="bad", kind="unknown")
    with pytest.raises(ValidationError):
        Edge(source="visual", target="claim", relation="invented")


def test_builder_requires_canonical_packet():
    with pytest.raises(TypeError):
        build_graph({"claim": {"claim_id": "x"}})
