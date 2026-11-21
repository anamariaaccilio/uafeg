"""Los tres controles imprescindibles del cierre (sobre los datos reales de v3)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.thesis_data import LOCK, packets_dir
from src.schemas.packet import EvidencePacket
from uafeg.aggregator import claim_features
from uafeg.benchmark import verify_benchmark
from uafeg.thesis_graph import build_thesis_graph, evidence_view

pytestmark = pytest.mark.skipif(not (packets_dir() / "validation.jsonl").exists(), reason="real packets not built")


def _sources(split):
    return {json.loads(l)["claim"]["source_image_id"] for l in (packets_dir() / f"{split}.jsonl").read_text().splitlines()}


def test_no_split_leakage():
    s = {k: _sources(k) for k in ("train", "validation", "test")}
    assert not (s["train"] & s["validation"]) and not (s["train"] & s["test"]) and not (s["validation"] & s["test"])


def test_no_label_in_features():
    lines = (packets_dir() / "validation.jsonl").read_text().splitlines()[:50]
    for line in lines:
        p = EvidencePacket.model_validate_json(line)
        other = "SUPPORTED" if p.claim.expected_label.value != "SUPPORTED" else "SUSPICIOUS"
        claim2 = p.claim.model_copy(update={"expected_label": other, "condition": "C0", "claim_id": "x",
                                            "source_image_id": "y"})
        p2 = p.model_copy(update={"claim": claim2})
        for gate in (True, False):
            a = claim_features(build_thesis_graph(evidence_view(p)), gate=gate)
            b = claim_features(build_thesis_graph(evidence_view(p2)), gate=gate)
            assert np.array_equal(a, b)


def test_lock_valid():
    lock = verify_benchmark(LOCK)
    assert lock["benchmark_version"]
