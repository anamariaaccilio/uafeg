"""Agregador de evidencia interpretable SIN GATv2 (diseño final de la tesis, 2026-11-20).

Sobre el grafo de evidencia (``uafeg.thesis_graph.build_thesis_graph``) no hay
propagación de mensajes aprendida. Para cada afirmación i (nodo ``claim``) y sus
vecinos visuales j:

    S_apoyo(i)  = Σ_j g_j · p_support(i, j)
    S_contra(i) = Σ_j g_j · p_contradiction(i, j)
    S_desc(i)   = Σ_j g_j · p_unknown(i, j)
    S_sim(i)    = Σ_j g_j · sim_sem(i, j)
    F(i)        = max_j g_f · p_edit · IoU(región visual j, región forense)

con g_j = clip(c_j q_j r_j (1 - u_j), 0, 1) (compuerta de confiabilidad). A nivel
de reclamo se agregan máximo, media y conteo sobre las afirmaciones, más la
cobertura c̄, la calidad media q̄ y el puntaje forense global g_f · p_edit. Ese
vector entra a un clasificador liviano (regresión logística multinomial o
gradient boosting, elegido en validación).

Variantes:
  gate=False          G0 / A1: el mismo agregador con g = 1 en todos los nodos.
  edge_attributes=False  A4: p_support/p_contradiction/p_unknown se sustituyen por el
                         indicador del TIPO de arista (argmax), sim_sem = 0 y
                         IoU -> 1[solapamiento > 0].
Sin forense (A2) se obtiene construyendo los grafos con ``use_forensic=False``.

Las features se calculan SOLO a partir de ``GraphTensors`` (que a su vez solo ve
``evidence_view``): ninguna etiqueta, condición ni identificador entra aquí.
"""
from __future__ import annotations

import numpy as np

from uafeg.models.p1_graph import THESIS_RELATION_INDEX as R
from uafeg.thesis_graph import KIND_INDEX, GraphTensors

CLAIM, VISUAL, FORENSIC, SOURCE = (KIND_INDEX[k] for k in ("claim", "visual", "forensic", "source"))
REL_SUPPORT, REL_CONTRA, REL_UNKNOWN, REL_OVERLAP = R["support"], R["contradiction"], R["unknown"], R["overlap"]

PER_CLAIM = ("S_apoyo", "S_contra", "S_desc", "S_sim", "F_forense")
FEATURE_NAMES = tuple(f"{agg}_{k}" for k in PER_CLAIM for agg in ("max", "mean")) + (
    "n_claims", "n_visual", "n_support_edges", "n_contra_edges", "has_forensic", "has_image",
    "coverage_bar", "quality_mean", "forensic_global", "forensic_area", "g_mean_visual", "g_mean_claim",
    "g_forensic", "frac_denied", "frac_undetermined", "u_mean_claim")


def node_gates(g: GraphTensors) -> np.ndarray:
    c, q, r, u = g.gate[:, 0], g.gate[:, 1], g.gate[:, 2], g.gate[:, 3]
    return np.clip(c * q * r * (1.0 - u), 0.0, 1.0)


def claim_edge_table(g: GraphTensors, gate: bool = True, edge_attributes: bool = True) -> list[dict]:
    """Per-claim edge contributions (also used by the human-review report)."""
    gv = node_gates(g) if gate else np.ones(g.n, dtype=np.float32)
    kind = g.kind
    forensic = np.flatnonzero(kind == FORENSIC)
    f_idx = int(forensic[0]) if len(forensic) else None
    p_edit = float(g.x[f_idx, 0]) if f_idx is not None else 0.0
    iou_f = {}
    if f_idx is not None:
        for s, d, rel, a in g.edges:
            if rel == REL_OVERLAP and d == f_idx and kind[s] == VISUAL:
                iou_f[s] = float(a[4])
    rows = []
    for i in np.flatnonzero(kind == CLAIM):
        contrib = []
        for s, d, rel, a in g.edges:
            if s != i or kind[d] != VISUAL:
                continue
            if edge_attributes:
                ps, pc, pu, sim = float(a[0]), float(a[1]), float(a[2]), float(a[3])
                iou = iou_f.get(d, 0.0)
            else:
                ps, pc, pu, sim = float(rel == REL_SUPPORT), float(rel == REL_CONTRA), float(rel == REL_UNKNOWN), 0.0
                iou = float(iou_f.get(d, 0.0) > 0)
            gj = float(gv[d])
            fterm = float(gv[f_idx]) * p_edit * iou if f_idx is not None else 0.0
            contrib.append({"visual_node": int(d), "g": gj, "relation": int(rel), "p_support": ps,
                            "p_contradiction": pc, "p_unknown": pu, "sim_sem": sim, "iou_forensic": iou,
                            "support": gj * ps, "contradiction": gj * pc, "unknown": gj * pu,
                            "forensic": fterm})
        rows.append({"claim_node": int(i), "g": float(gv[i]), "edges": contrib,
                     "S_apoyo": sum(e["support"] for e in contrib),
                     "S_contra": sum(e["contradiction"] for e in contrib),
                     "S_desc": sum(e["unknown"] for e in contrib),
                     "S_sim": sum(e["g"] * e["sim_sem"] for e in contrib),
                     "F_forense": max([e["forensic"] for e in contrib], default=0.0)})
    return rows


def claim_features(g: GraphTensors, gate: bool = True, edge_attributes: bool = True) -> np.ndarray:
    gv = node_gates(g) if gate else np.ones(g.n, dtype=np.float32)
    kind = g.kind
    rows = claim_edge_table(g, gate, edge_attributes)
    feats = []
    for k in PER_CLAIM:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        feats += [float(vals.max()) if len(vals) else 0.0, float(vals.mean()) if len(vals) else 0.0]
    vis = np.flatnonzero(kind == VISUAL)
    cl = np.flatnonzero(kind == CLAIM)
    fo = np.flatnonzero(kind == FORENSIC)
    rel = [e[2] for e in g.edges if kind[e[0]] == CLAIM and kind[e[1]] == VISUAL]
    evidence_nodes = np.flatnonzero(kind != CLAIM)
    f_idx = int(fo[0]) if len(fo) else None
    feats += [
        float(len(cl)), float(len(vis)), float(sum(r == REL_SUPPORT for r in rel)),
        float(sum(r == REL_CONTRA for r in rel)), float(f_idx is not None),
        float(np.any(kind == SOURCE)), float(g.coverage_bar),
        float(g.gate[evidence_nodes, 1].mean()) if len(evidence_nodes) else 0.0,
        float(gv[f_idx] * g.x[f_idx, 0]) if f_idx is not None else 0.0,
        float(g.x[f_idx, 2]) if f_idx is not None else 0.0,
        float(gv[vis].mean()) if len(vis) else 0.0,
        float(gv[cl].mean()) if len(cl) else 0.0,
        float(gv[f_idx]) if f_idx is not None else 0.0,
        float(g.x[cl, 12].mean()) if len(cl) else 0.0,   # polarity one-hot: 11 affirmed, 12 denied, 13 undetermined
        float(g.x[cl, 13].mean()) if len(cl) else 0.0,
        float(g.gate[cl, 3].mean()) if len(cl) else 0.0,
    ]
    return np.asarray(feats, dtype=np.float32)


def feature_matrix(graphs: list[GraphTensors], gate: bool = True, edge_attributes: bool = True) -> np.ndarray:
    return np.stack([claim_features(g, gate, edge_attributes) for g in graphs])
