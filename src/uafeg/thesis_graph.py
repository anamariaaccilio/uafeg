"""Grafo de evidencia con la taxonomía de nodos de la tesis (eq:nodos), aprobada
por la asesora (2026-11-13):

    V = V_claim ∪ V_visual ∪ V_forensic ∪ V_source

- ``claim``: un nodo por AFIRMACIÓN del relato (no un hub por reclamo).
- ``visual``: un nodo por observación de daño localizada (detección YOLO-seg).
- ``forensic``: un nodo por detección de integridad (TruFor), si existe.
- ``source``: un nodo con la calidad de la fuente (q) y su origen (s).

Aristas (cada una con tipo de relación y a_ij de eq:aristas + máscara):

- claim_i <-> visual_k: NLI de la afirmación frente a la hipótesis de la clase
  observada: p_support = entail, p_contradiction = contradiction, p_unknown =
  neutral. sim_sem = compatibilidad en la ontología de daños entre el tipo
  afirmado y el observado. Tipo = argmax(support, contradiction, unknown).
- visual_k <-> visual_l y visual_k <-> forensic: ``overlap`` con
  overlap_spatial = índice de Jaccard entre sus máscaras (ℓ), si > 0.
- source -> visual_k, source -> forensic, source -> claim_i: ``evidence_for``
  (la calidad de la fuente informa a la observación), sin atributos.

Todo se construye SOLO con la evidencia del paquete (narrative, visual,
image_quality, forensic). ``packet.claim`` (etiqueta, condición, ids, rutas)
no se lee nunca: ``evidence_view`` lo descarta antes de construir nada.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass

import numpy as np

from src.schemas.evidence import CANONICAL_DAMAGE_CLASSES
from src.schemas.packet import EvidencePacket
from uafeg.relations import EDGE_ATTRIBUTE_DIM

NODE_KINDS = ("claim", "visual", "forensic", "source")
KIND_INDEX = {k: i for i, k in enumerate(NODE_KINDS)}
KIND_DIMS = {"claim": 17, "visual": 18, "forensic": 4, "source": 6}
F_MAX = max(KIND_DIMS.values())
MAX_ASSERTIONS = 4
MAX_VISUAL = 10
CLASS_INDEX = {c: i for i, c in enumerate(CANONICAL_DAMAGE_CLASSES)}

# Ontología de daños: familias de categorías compatibles (sim_sem = 0.5 dentro de
# la misma familia, 1.0 misma clase, 0.0 en otro caso).
DAMAGE_FAMILIES = {
    "surface": {"scratch", "paint_chip", "flaking"},
    "structural": {"dent", "crack", "broken_part", "missing_part"},
    "chemical": {"corrosion"},
}


def ontology_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return 0.5 if any(a in f and b in f for f in DAMAGE_FAMILIES.values()) else 0.0


def _onehot(cls: str) -> list[float]:
    v = [0.0] * len(CANONICAL_DAMAGE_CLASSES)
    v[CLASS_INDEX[cls]] = 1.0
    return v


def decode_mask(encoded: str | None, shape=(48, 64)) -> np.ndarray | None:
    if not encoded:
        return None
    bits = np.unpackbits(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8))
    return bits[: shape[0] * shape[1]].reshape(shape).astype(bool)


def jaccard(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass(frozen=True)
class EvidenceView:
    """Lo único que ve el constructor: la evidencia, sin el ClaimRecord."""
    narrative: object | None
    visual: object | None
    image_quality: object | None
    forensic: object | None


def evidence_view(packet: EvidencePacket, use_forensic: bool = True) -> EvidenceView:
    forensic = packet.forensic if (use_forensic and packet.forensic is not None and packet.forensic.available) else None
    return EvidenceView(packet.narrative, packet.visual, packet.image_quality, forensic)


@dataclass
class GraphTensors:
    kind: np.ndarray        # (N,) int
    x: np.ndarray           # (N, F_MAX) float32, zero-padded per kind
    gate: np.ndarray        # (N, 4) c, q, r, u
    edges: list             # (src, dst, relation, attr_vector(10))
    coverage_bar: float     # c̄(x): cobertura agregada del expediente

    @property
    def n(self) -> int:
        return len(self.kind)


def _pad(v: list[float]) -> list[float]:
    return list(v) + [0.0] * (F_MAX - len(v))


def build_thesis_graph(view: EvidenceView, max_visual: int = MAX_VISUAL) -> GraphTensors:
    if not isinstance(view, EvidenceView):
        raise TypeError("build_thesis_graph only accepts an EvidenceView (no ClaimRecord)")
    kinds, xs, gates, edges = [], [], [], []
    masks: list[np.ndarray | None] = []

    def add(kind, x, gate, mask=None):
        assert len(x) == KIND_DIMS[kind], (kind, len(x))
        kinds.append(KIND_INDEX[kind])
        xs.append(_pad(x))
        gates.append([float(min(max(g, 0.0), 1.0)) for g in gate])
        masks.append(mask)
        return len(kinds) - 1

    def attr(values, mask):
        return tuple(values) + tuple(mask)

    claims = []
    assertions = list((view.narrative.assertions or []) if view.narrative is not None else [])[:MAX_ASSERTIONS]
    for a in assertions:
        polarity = [float(a.polarity == p) for p in ("affirmed", "denied", "undetermined")]
        filled = [float(a.part is not None), float(a.severity is not None), float(a.circumstance is not None)]
        r = a.reliability if a.reliability is not None else 1.0
        claims.append((add("claim", _onehot(a.damage_type) + list(a.p) + polarity + filled,
                           (a.coverage, 1.0, r, a.u)), a))

    if view.narrative is not None and not claims:
        # Relato presente sin ninguna afirmación extraíble: se representa como tal (tipo
        # vacío, p neutral, u = 1, c = 0, polaridad indeterminada), sin inventar contenido.
        add("claim", [0.0] * 8 + [0.0, 1.0, 0.0] + [0.0, 0.0, 1.0] + [0.0, 0.0, 0.0], (0.0, 1.0, 1.0, 1.0))

    source_idx = None
    q_source = 1.0
    if view.image_quality is not None:
        iq = view.image_quality
        f = iq.quality_factors or {}
        q_source = iq.quality_q if iq.quality_q is not None else 1.0
        s_obs = iq.source_s is not None
        source_idx = add("source", [f.get("resolution", 0.0), f.get("sharpness", 0.0), f.get("exposure", 0.0),
                                    q_source, float(iq.source_s or 0.0), float(s_obs)],
                         (0.0 if iq.is_corrupted else 1.0, q_source, 1.0, 0.0))

    visuals = []
    if view.visual is not None:
        dets = sorted(view.visual.detections, key=lambda d: -(d.confidence or 0.0))[:max_visual]
        for d in dets:
            rqf = d.region_quality_factors
            box = list(d.bbox_norm) if d.bbox_norm is not None else [0.0] * 4
            x = (_onehot(d.damage_class) + [d.confidence or 0.0, d.area_fraction or 0.0] + box
                 + ([rqf["resolution"], rqf["sharpness"], rqf["exposure"], 1.0] if rqf else [0.0, 0.0, 0.0, 0.0]))
            u = d.uncertainty if d.uncertainty is not None else 1.0 - (d.confidence or 0.0)
            q = d.region_quality if d.region_quality is not None else q_source
            visuals.append((add("visual", x, (d.coverage if d.coverage is not None else 1.0, q,
                                              d.reliability if d.reliability is not None else 1.0, u),
                                decode_mask(d.mask_64x48)), d))

    forensic_idx = None
    if view.forensic is not None:
        fr = view.forensic
        forensic_idx = add("forensic", [fr.score or 0.0, fr.uncertainty or 0.0, fr.localization_area or 0.0,
                                        fr.reliability or 0.0],
                           (1.0, q_source, fr.reliability if fr.reliability is not None else 1.0,
                            fr.uncertainty if fr.uncertainty is not None else 0.5),
                           decode_mask(fr.mask_64x48))

    from uafeg.models.p1_graph import THESIS_RELATION_INDEX as R
    unobserved = attr((0.0,) * EDGE_ATTRIBUTE_DIM, (0.0,) * EDGE_ATTRIBUTE_DIM)
    for ci, a in claims:
        for vi, d in visuals:
            e, n, c = a.class_nli.get(d.damage_class, (0.0, 1.0, 0.0))
            relation = "support" if e >= max(n, c) else "contradiction" if c >= max(e, n) else "unknown"
            a_ij = attr((e, c, n, ontology_similarity(a.damage_type, d.damage_class), 0.0), (1, 1, 1, 1, 0))
            edges.append((ci, vi, R[relation], a_ij))
            edges.append((vi, ci, R[relation], a_ij))
    spatial = [(vi, masks[vi]) for vi, _ in visuals] + ([(forensic_idx, masks[forensic_idx])] if forensic_idx is not None else [])
    for i in range(len(spatial)):
        for j in range(i + 1, len(spatial)):
            ov = jaccard(spatial[i][1], spatial[j][1])
            if ov:
                a_ij = attr((0.0, 0.0, 0.0, 0.0, ov), (0, 0, 0, 0, 1))
                edges.append((spatial[i][0], spatial[j][0], R["overlap"], a_ij))
                edges.append((spatial[j][0], spatial[i][0], R["overlap"], a_ij))
    if source_idx is not None:
        for target in [vi for vi, _ in visuals] + [ci for ci, _ in claims] + ([forensic_idx] if forensic_idx is not None else []):
            edges.append((source_idx, target, R["evidence_for"], unobserved))

    if not kinds:
        raise ValueError("Al menos una evidencia debe estar presente")
    c_narr = max([a.coverage for _, a in claims], default=0.0)
    c_img = 0.0 if source_idx is None else gates[source_idx][0]
    return GraphTensors(kind=np.array(kinds, dtype=np.int64), x=np.array(xs, dtype=np.float32),
                        gate=np.array(gates, dtype=np.float32), edges=edges,
                        coverage_bar=float((c_narr + c_img) / 2.0))


# ------------------------------------------------------------------ legacy taxonomy
LEGACY_KINDS = ("claim_hub", "narrative", "visual", "image_quality", "forensic")


def build_legacy_graph(packet: EvidencePacket, use_forensic: bool = True) -> GraphTensors:
    """Taxonomía previa (hub claim + un nodo por modalidad), solo para comparar.
    Reutiliza ``uafeg.graph.build_graph``, ``packet_to_modality_features`` y
    ``node_gate`` tal cual; el único cambio es que el tamaño de archivo (huella de
    generación en v3) se sustituye por q en las features de image_quality."""
    from uafeg.graph import build_graph
    from uafeg.models.features import packet_to_modality_features
    from uafeg.models.gate import node_gate
    from uafeg.models.p1_graph import THESIS_RELATION_INDEX as R
    if not use_forensic and packet.forensic is not None:
        packet = packet.model_copy(update={"forensic": None})
    graph = build_graph(packet)
    feats = packet_to_modality_features(packet)
    if "image_quality" in feats and packet.image_quality is not None:
        feats["image_quality"][1] = packet.image_quality.quality_q or 0.0  # never file size
    order = [n.id for n in graph.nodes]
    index = {k: i for i, k in enumerate(order)}
    kinds, xs, gates = [], [], []
    for node_id in order:
        kinds.append(LEGACY_KINDS.index("claim_hub" if node_id == "claim" else node_id))
        vec = [] if node_id == "claim" else list(feats[node_id])
        xs.append(_pad(vec))
        g = 1.0 if node_id == "claim" else node_gate(packet, node_id)
        gates.append([g, 1.0, 1.0, 0.0])  # gate folded in c so that c*q*r*(1-u) == node_gate
    edges = [(index[e.source], index[e.target], R[e.relation], e.attributes.as_vector()) for e in graph.edges]
    present = sum(k in feats for k in ("narrative", "visual")) / 2.0
    return GraphTensors(kind=np.array(kinds), x=np.array(xs, dtype=np.float32),
                        gate=np.array(gates, dtype=np.float32), edges=edges, coverage_bar=present)
