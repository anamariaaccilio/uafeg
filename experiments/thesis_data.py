"""Cargador real del benchmark v3 con controles anti-fuga.

Antes de leer un solo paquete:
  1. verifica el lock (hashes del contrato y artefactos de v3; git_commit informativo);
  2. verifica los checksums de los EvidencePackets reales (packets_manifest.json);
  3. comprueba que ningún source_image_id aparezca en dos splits.
Los tensores se construyen SOLO desde ``evidence_view`` (sin ClaimRecord). La etiqueta,
la condición, el claim_id y el source_image_id viajan en ``meta``, separados, y se usan
únicamente como objetivo, para filtrar el train de A5 y para agrupar el bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation.metrics import CLASS_ORDER
from src.schemas.packet import EvidencePacket
from uafeg.benchmark import canonical_repository, verify_benchmark
from uafeg.thesis_graph import (GraphTensors, KIND_INDEX, build_legacy_graph, build_thesis_graph,
                                evidence_view)

REPO_B = Path(__file__).resolve().parents[1]
LOCK = REPO_B / "benchmark.lock.json"
ART = REPO_B / "artifacts/thesis"
SPLITS = ("train", "validation", "test")
LABEL_INDEX = {c: i for i, c in enumerate(CLASS_ORDER)}
FORBIDDEN_FEATURE_KEYS = {"label", "expected_label", "condition", "claim_id", "source_image_id",
                          "image_path", "annotation_reference", "split"}


def packets_dir() -> Path:
    return canonical_repository() / "artifacts/packets/v3"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs() -> dict:
    lock = verify_benchmark(LOCK)
    manifest = json.loads((packets_dir() / "packets_manifest.json").read_text())
    for name, info in manifest["files"].items():
        if sha256(packets_dir() / name) != info["sha256"]:
            raise ValueError(f"EvidencePacket file changed since it was built: {name}")
    return {"lock_benchmark_version": lock["benchmark_version"], "lock_git_commit_informative": lock["git_commit"],
            "packets_manifest_sha256": sha256(packets_dir() / "packets_manifest.json"),
            "packets_sha256": {k: v["sha256"] for k, v in manifest["files"].items()}}


@dataclass
class Split:
    name: str
    graphs: list[GraphTensors]
    labels: np.ndarray                  # -1 for C10 (OOD, not a class)
    meta: list[dict] = field(default_factory=list)

    def subset(self, keep: np.ndarray) -> "Split":
        idx = np.flatnonzero(keep)
        return Split(self.name, [self.graphs[i] for i in idx], self.labels[idx], [self.meta[i] for i in idx])


def _read_packets(split: str) -> list[EvidencePacket]:
    return [EvidencePacket.model_validate_json(line)
            for line in (packets_dir() / f"{split}.jsonl").read_text().splitlines()]


def load_split(split: str, taxonomy: str = "thesis", oracle: tuple[bool, bool] = (False, False),
               use_forensic: bool = True) -> Split:
    if taxonomy not in ("thesis", "legacy"):
        raise ValueError(taxonomy)
    inputs = verify_inputs()
    key = f"{split}_{taxonomy}_o{int(oracle[0])}{int(oracle[1])}_f{int(use_forensic)}_{inputs['packets_sha256'][split + '.jsonl'][:12]}"
    cache = ART / "cache" / f"{key}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    ambiguous = set(json.loads((packets_dir() / "ambiguous_claim_ids.json").read_text()))
    graphs, labels, meta = [], [], []
    for packet in _read_packets(split):
        if oracle != (False, False):
            from uafeg.thesis_oracle import build_thesis_oracle_packet
            packet = build_thesis_oracle_packet(packet, *oracle)
        if taxonomy == "thesis":
            g = build_thesis_graph(evidence_view(packet, use_forensic=use_forensic))
        else:
            g = build_legacy_graph(packet, use_forensic=use_forensic)
        c = packet.claim
        graphs.append(g)
        labels.append(LABEL_INDEX.get(c.expected_label.value if c.expected_label else None, -1))
        meta.append({"claim_id": c.claim_id, "source_image_id": c.source_image_id, "condition": c.condition,
                     "label": c.expected_label.value, "ambiguous": c.claim_id in ambiguous})
    out = Split(split, graphs, np.array(labels), meta)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def load_all(**kwargs) -> dict[str, Split]:
    data = {s: load_split(s, **kwargs) for s in SPLITS}
    sources = {s: {m["source_image_id"] for m in d.meta} for s, d in data.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and sources[a] & sources[b]:
                raise ValueError(f"source_image_id leakage between {a} and {b}")
    for s in ("train", "validation"):
        if any(m["condition"] == "C10" for m in data[s].meta):
            raise ValueError("C10 must be test-only")
    return data


def training_view(split: Split, exclude_conditions: tuple[str, ...] = ()) -> Split:
    """Rows usable as supervised targets (no C10). A5 removes the benign controls (C8)
    from TRAINING; C10 never is in train, so for C10 A5 changes nothing."""
    keep = (split.labels >= 0) & np.array([m["condition"] not in exclude_conditions for m in split.meta])
    return split.subset(keep)


# --------------------------------------------------------------------- flat features
def flat_features(g: GraphTensors) -> dict[str, np.ndarray]:
    """Per-modality fixed vectors for the non-graph baselines, built from the SAME nodes."""
    x, kind = g.x, g.kind
    def rows(k):
        return x[kind == KIND_INDEX[k]]
    c, v, f, s = rows("claim"), rows("visual"), rows("forensic"), rows("source")
    text = np.concatenate([[float(len(c) > 0), len(c) / 4.0], c[:, :17].mean(0) if len(c) else np.zeros(17)])
    if len(v):
        best = np.zeros(8)
        for r in v:
            k = int(np.argmax(r[:8]))
            best[k] = max(best[k], r[8])
        vis = np.concatenate([[1.0, len(v) / 10.0], best, [v[:, 9].mean()], v[:, 14:17].mean(0)])
    else:
        vis = np.concatenate([[0.0, 0.0], np.zeros(8), [0.0], np.zeros(3)])
    src = np.concatenate([[float(len(s) > 0)], s[0, :6] if len(s) else np.zeros(6)])
    fr = np.concatenate([[float(len(f) > 0)], f[0, :4] if len(f) else np.zeros(4)])
    return {"text": text.astype(np.float32), "visual": np.concatenate([vis, src]).astype(np.float32),
            "forensic": fr.astype(np.float32)}


def flat_matrix(split: Split, parts: tuple[str, ...]) -> np.ndarray:
    return np.stack([np.concatenate([flat_features(g)[p] for p in parts]) for g in split.graphs])
