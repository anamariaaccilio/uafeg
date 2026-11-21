"""Estadística descriptiva de los grafos de evidencia finales (sección 4).

    python -m experiments.graph_stats

Reporta (por split y por condición): nodos y aristas promedio por reclamo, % de aristas
afirmación–visual de tipo apoya / contradice / desconocido, distribución de g por tipo de
especialista y condición (p. ej. g visual en C9 frente a C0) y una figura de ejemplo de un
grafo real (de validación). Es descriptivo: no usa ningún modelo ni etiqueta.
Salida: results/final/graph_stats.json, tabla_grafo.tex, fig_g_distribucion.pdf, fig_grafo_ejemplo.pdf
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from experiments.thesis_data import REPO_B, load_all
from uafeg.aggregator import node_gates
from uafeg.models.p1_graph import THESIS_RELATION_TYPES
from uafeg.thesis_graph import NODE_KINDS

OUT = REPO_B / "results/final"
REL_CV = ("support", "contradiction", "unknown")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    stats = {"per_split": {}, "per_condition": {}, "g_by_kind_condition": {}}
    acc = defaultdict(lambda: defaultdict(list))
    gk = defaultdict(lambda: defaultdict(list))
    for split, d in data.items():
        nn, ne = [], []
        for g, m in zip(d.graphs, d.meta):
            c = m["condition"]
            nn.append(g.n); ne.append(len(g.edges))
            acc[c]["nodes"].append(g.n); acc[c]["edges"].append(len(g.edges))
            rel = [THESIS_RELATION_TYPES[e[2]] for e in g.edges
                   if NODE_KINDS[g.kind[e[0]]] == "claim" and NODE_KINDS[g.kind[e[1]]] == "visual"]
            for r in REL_CV:
                acc[c][r].append(sum(x == r for x in rel))
            acc[c]["n_visual"].append(int(np.sum(g.kind == NODE_KINDS.index("visual"))))
            acc[c]["has_forensic"].append(int(np.any(g.kind == NODE_KINDS.index("forensic"))))
            gates = node_gates(g)
            for i in range(g.n):
                gk[NODE_KINDS[g.kind[i]]][c].append(float(gates[i]))
        stats["per_split"][split] = {"claims": len(d.graphs), "nodes_mean": float(np.mean(nn)), "edges_mean": float(np.mean(ne))}
    for c in sorted(acc, key=lambda s: int(s[1:])):
        a = acc[c]
        tot = sum(sum(a[r]) for r in REL_CV)
        stats["per_condition"][c] = {
            "claims": len(a["nodes"]), "nodes_mean": float(np.mean(a["nodes"])), "edges_mean": float(np.mean(a["edges"])),
            "visual_nodes_mean": float(np.mean(a["n_visual"])), "frac_with_forensic": float(np.mean(a["has_forensic"])),
            "claim_visual_edges": int(tot),
            **{f"pct_{r}": (100.0 * sum(a[r]) / tot if tot else None) for r in REL_CV}}
    for k in gk:
        stats["g_by_kind_condition"][k] = {c: {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
                                               "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}
                                           for c, v in sorted(gk[k].items(), key=lambda t: int(t[0][1:]))}
    (OUT / "graph_stats.json").write_text(json.dumps(stats, indent=1) + "\n")

    def f(x, d=2):
        return "--" if x is None else f"{x:.{d}f}"
    lines = [r"\begin{table}[H]\centering\scriptsize", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"Cond. & Reclamos & Nodos & Aristas & Visuales & \% forense & \% apoya & \% contradice & $\bar g$ visual \\",
             r"\midrule"]
    for c, r in stats["per_condition"].items():
        gv = stats["g_by_kind_condition"].get("visual", {}).get(c, {}).get("mean")
        lines.append(f"{c} & {r['claims']} & {f(r['nodes_mean'], 1)} & {f(r['edges_mean'], 1)} & {f(r['visual_nodes_mean'], 1)} & "
                     f"{f(100 * r['frac_with_forensic'], 0)} & {f(r['pct_support'], 1)} & {f(r['pct_contradiction'], 1)} & {f(gv)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Grafos de evidencia finales (train+val+test) por condición: tamaño medio, porcentaje de aristas "
              r"afirmación--daño de tipo apoya/contradice y compuerta media $g$ de los nodos visuales. Fuente: "
              r"\texttt{results/final/graph\_stats.json}.}", r"\label{tab:grafo}", r"\end{table}"]
    (OUT / "tabla_grafo.tex").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = sorted({c for k in gk for c in gk[k]}, key=lambda s: int(s[1:]))
    kinds = [k for k in ("claim", "visual", "forensic", "source") if k in gk]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.2 * len(kinds), 3), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), kinds):
        cs = [c for c in conds if gk[k].get(c)]
        ax.boxplot([gk[k][c] for c in cs], showfliers=False)
        ax.set_xticks(range(1, len(cs) + 1)); ax.set_xticklabels(cs, rotation=90, fontsize=7)
        ax.set_title(f"g ({k})", fontsize=9); ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("g = clip(c·q·r·(1-u))")
    fig.tight_layout(); fig.savefig(OUT / "fig_g_distribucion.pdf"); plt.close(fig)

    # example: first validation C3 claim with >=2 visual nodes and a forensic node
    import networkx as nx
    val = data["validation"]
    pick = next(i for i, (g, m) in enumerate(zip(val.graphs, val.meta)) if m["condition"] == "C3"
                and np.sum(g.kind == 1) >= 2 and np.any(g.kind == 2))
    g = val.graphs[pick]
    gates = node_gates(g)
    G = nx.DiGraph()
    colors = {"claim": "#4c72b0", "visual": "#55a868", "forensic": "#c44e52", "source": "#8172b2"}
    for i in range(g.n):
        G.add_node(i, kind=NODE_KINDS[g.kind[i]])
    ecol = {"support": "#2a9d3a", "contradiction": "#d62728", "unknown": "#999999", "overlap": "#ff7f0e",
            "evidence_for": "#cccccc"}
    for s, d, r, a in g.edges:
        if s < d or THESIS_RELATION_TYPES[r] == "evidence_for":
            G.add_edge(s, d, rel=THESIS_RELATION_TYPES[r])
    pos = nx.spring_layout(G, seed=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    nx.draw_networkx_nodes(G, pos, node_color=[colors[G.nodes[n]["kind"]] for n in G], ax=ax,
                           node_size=[300 + 900 * gates[n] for n in G])
    nx.draw_networkx_labels(G, pos, {n: f"{G.nodes[n]['kind'][0]}{n}\ng={gates[n]:.2f}" for n in G}, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=[ecol[G.edges[e]["rel"]] for e in G.edges], ax=ax, arrows=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=c, label=k) for k, c in ecol.items()], fontsize=7, loc="lower right")
    ax.set_title(f"Grafo real (validación, {val.meta[pick]['condition']}); tamaño del nodo ∝ g", fontsize=9)
    ax.axis("off"); fig.tight_layout(); fig.savefig(OUT / "fig_grafo_ejemplo.pdf"); plt.close(fig)
    stats["example_graph_claim_id"] = val.meta[pick]["claim_id"]
    (OUT / "graph_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats["per_split"], indent=1))


if __name__ == "__main__":
    main()
