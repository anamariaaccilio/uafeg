"""Figuras para la presentación (PNG 300 dpi + SVG) en presentacion/figuras/.

    python -m experiments.presentation_figures            # todas las disponibles
    python -m experiments.presentation_figures 2 3 7      # solo algunas

Todos los números se leen de archivos de resultados (nunca a mano):
  results/evaluation.json, artifacts/thesis/eval_arrays.npz, results/final/specialists.json,
  results/final/graph_stats.json, results/posthoc/*.json, review/cases/*.json.
Cada figura dice en el pie de qué archivo sale y si es OFICIAL o POST HOC.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from experiments.labels_es import CLASES, CONDICIONES, SISTEMAS  # noqa: E402
from experiments.thesis_data import ART, REPO_B  # noqa: E402

OUT = REPO_B / "presentacion/figuras"
R = REPO_B / "results"
COL = {"narrativa": "#4A7AB5", "visual": "#0F8B9E", "forense": "#6A5AA6", "fuente": "#5F6B7A",
       "uafeg": "#1E3A5F", "SUPPORTED": "#2F8F57", "SUSPICIOUS": "#B8453B", "INSUFFICIENT": "#B9831A"}
INK, INK2, MUTED, GRID = "#1F2328", "#4B5563", "#8A939E", "#E5E7EB"
NEUTRO = "#A7B0BA"  # otros sistemas
OFICIAL = "Resultado OFICIAL (test evaluado una sola vez)"
POSTHOC = "ANÁLISIS ADICIONAL (post hoc, test ya consultado)"
CLS = ["SUPPORTED", "SUSPICIOUS", "INSUFFICIENT"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "svg.fonttype": "none", "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK,
})


def frame(fig, title, subtitle, source, kind):
    H = fig.get_figheight()
    fig.text(0.012, 1 - 0.09 / H, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.42 / H, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)
    fig.text(0.012, 0.07 / H, f"Fuente: {source} · {kind}", ha="left", va="bottom", fontsize=7.5, color=MUTED)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    print("ok", name)


def hgrid(ax):
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ev():
    return json.loads((R / "evaluation.json").read_text())


def arrays():
    return np.load(ART / "eval_arrays.npz")


def short(s):
    return SISTEMAS[s].split(" · ")[1] if " · " in SISTEMAS[s] else SISTEMAS[s]


# ---------------------------------------------------------------- 1 pipeline
def fig01():
    fig = plt.figure(figsize=(13, 6.2))
    frame(fig, "Cómo decide UAFEG", "Cada evidencia pasa por un especialista, se pondera con la compuerta g "
          "y un agregador interpretable decide o deriva el caso a una persona. ★ = aporte de la tesis.",
          "diseño final del sistema (README, docs/decisiones_cierre.md); GATv2 queda fuera del alcance final",
          "Diagrama del sistema")
    ax = fig.add_axes([0.01, 0.06, 0.98, 0.82])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, text, color, fill=None, star=False, fs=9.2, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fill or "white", ec=color, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text + ("  ★" if star else ""), ha="center", va="center", fontsize=fs,
                color=tc, wrap=True)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3))

    box(0.5, 38, 12, 9, "Relato del\nreclamo", COL["narrativa"])
    box(0.5, 13, 12, 9, "Foto del\nvehículo", COL["visual"])
    box(0.5, 50.5, 12, 7, "Benchmark controlado\nC0–C10 (807 fotos)", INK2, fs=8, star=True)
    spec = [(46, "Narrativo\nmDeBERTa (NLI)", COL["narrativa"]), (32, "Visual\nYOLOv8n-seg", COL["visual"]),
            (18, "Forense\nTruFor", COL["forense"]), (4, "Fuente y calidad\nde imagen", COL["fuente"])]
    for y, t, c in spec:
        box(17, y, 15, 8.5, t, c, fill=c + "14")
    arrow(12.9, 42.5, 16.6, 50)
    for y in (36, 22, 8):
        arrow(12.9, 17.5, 16.6, y + 4)
    box(37, 18, 14, 26, "Paquete de\nevidencias\npor reclamo\n\n(afirmaciones,\ndetecciones,\nmapa forense,\ncalidad)",
        INK2, fs=8.6)
    for y, _, _ in spec:
        arrow(32.4, y + 4.2, 36.6, 31)
    box(55, 30, 16, 14, "Grafo de evidencias\ncon compuerta\ng = c·q·r·(1−u)", COL["uafeg"], fill="#1E3A5F10",
        star=True)
    box(55, 12, 16, 13, "Agregador\ninterpretable\nG1 (G0: g = 1)", COL["uafeg"], fill="#1E3A5F10", star=True)
    arrow(51.4, 31, 54.6, 37)
    arrow(63, 29.6, 63, 25.4)
    box(75, 30, 11, 14, "Calibración T*\n+ ensamble\n5 semillas", INK2, fs=8.6)
    box(75, 12, 11, 13, "Abstención\n(umbrales τU, τc\nfijados en\nvalidación)", COL["uafeg"], star=True, fs=8.4)
    arrow(71.4, 18.5, 74.6, 37)
    arrow(80.5, 29.6, 80.5, 25.4)
    box(89.5, 40, 10, 6, "Respaldado", COL["SUPPORTED"], fill=COL["SUPPORTED"] + "18", fs=8.6)
    box(89.5, 31, 10, 6, "Sospechoso", COL["SUSPICIOUS"], fill=COL["SUSPICIOUS"] + "18", fs=8.6)
    box(89.5, 20, 10, 7, "Insuficiente /\nrevisión humana", COL["INSUFFICIENT"], fill=COL["INSUFFICIENT"] + "18",
        fs=8.2)
    arrow(86.4, 20, 89.1, 43)
    arrow(86.4, 19, 89.1, 34)
    arrow(86.4, 18, 89.1, 23.5)
    box(72, 1, 27.5, 7.5, "Reporte explicable por afirmación\n+ paquete de revisión humana (HTML/CSV)",
        COL["uafeg"], star=True, fs=8.6)
    arrow(94.5, 19.6, 90, 8.9)
    ax.text(37, 52, "Lo que cambia entre G0 y G1 es solo la compuerta g:\nG0 confía igual en todas las evidencias;\n"
            "G1 baja el peso de las poco confiables.", fontsize=8.6, color=INK2, va="center")
    save(fig, "01_pipeline")


# ---------------------------------------------------------------- 2 comparación general
MAIN = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "G0", "G1"]


def fig02():
    e = ev()
    best = e["hypotheses"]["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    rows = sorted(MAIN, key=lambda s: e["systems"][s]["test"]["pr_auc_suspicious"]["ensemble"])
    fig = plt.figure(figsize=(11, 6.6))
    g1, bb = e["systems"]["G1"]["test"]["pr_auc_suspicious"], e["systems"][best]["test"]["pr_auc_suspicious"]
    overlap = "se solapa" if g1["ci_high"] >= bb["ci_low"] else "no se solapa"
    frame(fig, "Qué tan bien detecta cada sistema los casos sospechosos",
          f"UAFEG (G1) queda por debajo de la fusión tardía simple ({best}); su intervalo {overlap} con el de {best}.",
          "results/evaluation.json (systems.*.test.pr_auc_suspicious; IC 95 % bootstrap agrupado, 2 000 réplicas)",
          OFICIAL)
    ax = fig.add_axes([0.33, 0.1, 0.6, 0.76])
    for i, s in enumerate(rows):
        m = e["systems"][s]["test"]["pr_auc_suspicious"]
        c = COL["uafeg"] if s == "G1" else ("#5F6B7A" if s == best else NEUTRO)
        ax.barh(i, m["ensemble"], height=0.62, color=c)
        ax.errorbar(m["ensemble"], i, xerr=[[m["ensemble"] - m["ci_low"]], [m["ci_high"] - m["ensemble"]]],
                    fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.text(m["ci_high"] + 0.008, i, f"{m['ensemble']:.3f}", va="center", fontsize=9,
                color=INK, fontweight="bold" if s in ("G1", best) else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([SISTEMAS[s] + ("\n(mejor rival, elegido en validación)" if s == best else "")
                        for s in rows], fontsize=9)
    for t, s in zip(ax.get_yticklabels(), rows):
        if s in ("G1", best):
            t.set_fontweight("bold")
            t.set_color(COL["uafeg"] if s == "G1" else INK)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Qué tan bien detecta los casos sospechosos (PR-AUC de Suspicious; 1 = perfecto)")
    hgrid(ax)
    save(fig, "02_comparacion_general")


# ---------------------------------------------------------------- 3 forest H1-H3
def fig03():
    e = ev()["hypotheses"]
    best = e["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    items = [("H1", f"H1 · UAFEG frente a {best} (mejor sin grafo)\nPR-AUC de Suspicious en estrés (C3–C5 vs C8–C9)"),
             ("H2", "H2 · UAFEG (con g) frente a G0 (sin g)\nerror de las probabilidades (Brier, invertido)"),
             ("H3", "H3 · UAFEG con forense frente a sin forense (A2)\nPR-AUC de Suspicious en todo test")]
    fig = plt.figure(figsize=(12, 5.8))
    frame(fig, "Las tres hipótesis: solo el aporte del forense se sostiene",
          "Diferencia orientada a favor de UAFEG (a la derecha de 0 = UAFEG mejor), con IC 95 % y Holm; cada fila tiene su propia escala.",
          "results/evaluation.json (hypotheses.H1–H3; bootstrap pareado agrupado por imagen, 2 000 réplicas)", OFICIAL)
    for k, (h, lab) in enumerate(items):
        x = e[h]
        ax = fig.add_axes([0.36, 0.66 - k * 0.255, 0.38, 0.17])
        ok = x["reject_H0_at_0.05"] and x["point_diff"] > 0
        c = COL["SUPPORTED"] if ok else COL["SUSPICIOUS"]
        lim = max(abs(x["ci_low"]), abs(x["ci_high"])) * 1.35
        ax.axvline(0, color=INK2, lw=1)
        ax.plot([x["ci_low"], x["ci_high"]], [0, 0], color=c, lw=3, solid_capstyle="round")
        ax.plot(x["point_diff"], 0, "o", ms=10, color=c, mec="white", mew=2)
        ax.text(x["point_diff"], 0.42, f"{x['point_diff']:+.4f}  [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]",
                ha="center", fontsize=8.6, color=INK)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, 0.9)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.text(-lim * 0.98, -0.55, "← gana el rival", fontsize=7.5, color=MUTED, va="bottom")
        ax.text(lim * 0.98, -0.55, "gana UAFEG →", fontsize=7.5, color=MUTED, va="bottom", ha="right")
        fig.text(0.012, 0.745 - k * 0.255, lab, fontsize=9.2, color=INK, va="center")
        p = x["p_holm"]
        ptxt = f"p de Holm {'< 0.0005' if p == 0 else f'= {p:.3f}'}"
        verdict = ("ACEPTADA: el forense aporta" if ok else
                   "RECHAZADA: el rival es mejor" if x["reject_H0_at_0.05"] else "NO sostenida")
        fig.text(0.77, 0.75 - k * 0.255, f"{'✔' if ok else '✘'} {verdict}\n{ptxt}", fontsize=10, color=c,
                 va="center", fontweight="bold")
    save(fig, "03_hipotesis_forest")


# ---------------------------------------------------------------- 4 oráculo
def fig04():
    e = ev()
    best = e["hypotheses"]["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    order = [("G1", "Percepción real (YOLO, NLI, TruFor)"), ("G1_oracleN", "Relato perfecto"),
             ("G1_oracleVF", "Imagen y forense perfectos"), ("G1_oracleNVF", "Todo perfecto (techo)")]
    fig = plt.figure(figsize=(11.5, 5.8))
    g1o = e["systems"]["G1_oracleNVF"]["test"]["pr_auc_suspicious"]["ensemble"]
    g0o = e["systems"]["G0_oracleNVF"]["test"]["pr_auc_suspicious"]["ensemble"]
    frame(fig, "Con percepción perfecta, el agregador supera al mejor rival real",
          f"G1 pasa de {e['systems']['G1']['test']['pr_auc_suspicious']['ensemble']:.3f} a {g1o:.3f}: el cuello de "
          f"botella está en los especialistas. Con percepción perfecta la compuerta tampoco aporta (G0 {g0o:.3f}).",
          "results/evaluation.json (systems.G1*, G0*; test.pr_auc_suspicious, IC 95 %)", OFICIAL + " · oráculo = diagnóstico")
    ax = fig.add_axes([0.07, 0.13, 0.55, 0.7])
    vals = [e["systems"][s]["test"]["pr_auc_suspicious"] for s, _ in order]
    for i, m in enumerate(vals):
        c = COL["uafeg"] if i in (0, 3) else "#6B87A8"
        ax.bar(i, m["ensemble"], width=0.6, color=c)
        ax.errorbar(i, m["ensemble"], yerr=[[m["ensemble"] - m["ci_low"]], [m["ci_high"] - m["ensemble"]]],
                    fmt="none", ecolor=INK, lw=1, capsize=3)
        ax.text(i + 0.32, m["ensemble"] - 0.03, f"{m['ensemble']:.3f}", ha="left", fontsize=10, fontweight="bold")
    b = e["systems"][best]["test"]["pr_auc_suspicious"]["ensemble"]
    ax.axhline(b, color=COL["SUSPICIOUS"], lw=1.4, ls="--")
    ax.text(3.45, b + 0.008, f"{best} (mejor rival real) = {b:.3f}", ha="right", fontsize=8.6, color=COL["SUSPICIOUS"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(["\n".join(textwrap.wrap(t, 16)) for _, t in order], fontsize=9)
    ax.set_ylim(0.5, 1.03)
    ax.set_ylabel("PR-AUC de Suspicious (eje desde 0.5)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("G1 (UAFEG) según qué percepción recibe", loc="left", fontsize=10.5)
    ax2 = fig.add_axes([0.71, 0.13, 0.26, 0.7])
    pairs = [("G0", "G0 real"), ("G0_oracleNVF", "G0 todo\nperfecto"), ("G1", "G1 real"), ("G1_oracleNVF", "G1 todo\nperfecto")]
    for i, (s, t) in enumerate(pairs):
        m = e["systems"][s]["test"]["pr_auc_suspicious"]
        ax2.bar(i, m["ensemble"], width=0.6, color=COL["uafeg"] if s.startswith("G1") else NEUTRO)
        ax2.text(i, m["ensemble"] + 0.01, f"{m['ensemble']:.3f}", ha="center", fontsize=9)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([t for _, t in pairs], fontsize=8.5)
    ax2.set_ylim(0.5, 1.03)
    ax2.yaxis.grid(True, color=GRID)
    ax2.set_axisbelow(True)
    ax2.set_title("Con y sin compuerta g", loc="left", fontsize=10.5)
    save(fig, "04_oraculo")


# ---------------------------------------------------------------- 5 percepción progresiva
def fig05():
    path = R / "posthoc/percepcion_progresiva.json"
    if not path.exists():
        print("skip 05: falta", path)
        return
    d = json.loads(path.read_text())
    rows = d["rows"]
    x = [100 * r["level"] for r in rows]
    fig = plt.figure(figsize=(12, 5.6))
    cross = [r["level"] for r in rows if r["diff_G1_minus_B5"]["pr_auc_suspicious_test"]["point_diff"] > 0]
    msg = (f"G1 supera a B5 desde el {100 * cross[0]:.0f} % de percepción visual correcta (en todo test)."
           if cross else
           f"Aun con el 100 % de las fotos con anotación real, G1 pasa solo de "
           f"{rows[0]['systems']['G1']['pr_auc_suspicious_test']['value']:.3f} a "
           f"{rows[-1]['systems']['G1']['pr_auc_suspicious_test']['value']:.3f} y sigue debajo de B5 "
           f"({rows[-1]['systems']['B5']['pr_auc_suspicious_test']['value']:.3f}): el detector visual solo no explica la brecha.")
    frame(fig, "¿Qué pasa si el detector visual mejora?", msg,
          "results/posthoc/percepcion_progresiva.json (G1 y B5 reentrenados por nivel; IC 95 % bootstrap agrupado)",
          POSTHOC + " · análisis de sensibilidad declarado, no confirmatorio")
    for k, (key, ttl) in enumerate((("pr_auc_suspicious_test", "Todo test"),
                                    ("pr_auc_suspicious_stress_H1", "Subconjunto de estrés de H1 (C3–C5 vs C8–C9)"))):
        ax = fig.add_axes([0.07 + k * 0.48, 0.13, 0.4, 0.7])
        for s, c, mk in (("G1", COL["uafeg"], "o"), ("B5", "#5F6B7A", "s")):
            v = [r["systems"][s][key]["value"] for r in rows]
            lo = [r["systems"][s][key]["ci_low"] for r in rows]
            hi = [r["systems"][s][key]["ci_high"] for r in rows]
            ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0)
            ax.plot(x, v, color=c, lw=2, marker=mk, ms=8, mec="white", mew=1.5)
            ax.text(x[-1] + 3, v[-1], f"{'UAFEG (G1)' if s == 'G1' else 'B5'} {v[-1]:.3f}", va="center",
                    fontsize=9, color=c, fontweight="bold")
            ax.text(x[0] - 3, v[0], f"{v[0]:.3f}", va="center", ha="right", fontsize=8.5, color=c)
        ax.set_xlim(-14, 128)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v:.0f} %" for v in x])
        ax.set_xlabel("Imágenes con percepción visual correcta (anotación real en lugar de YOLO)")
        if k == 0:
            ax.set_ylabel("PR-AUC de Suspicious (más alto es mejor)")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_title(ttl, loc="left", fontsize=10.5)
    save(fig, "05_percepcion_progresiva")


# ---------------------------------------------------------------- 6 oficial vs percepción mejorada
def fig06():
    path = R / "posthoc/percepcion_mejorada.json"
    if not path.exists():
        print("skip 06: no existe", path)
        return
    d = json.loads(path.read_text())
    e = ev()
    systems = [s for s in ("B5", "B6", "G0", "G1", "A2") if s in d["systems"]]
    fig = plt.figure(figsize=(11, 5.6))
    g1a = e["systems"]["G1"]["test"]["pr_auc_suspicious"]["ensemble"]
    g1b = d["systems"]["G1"]["test"]["pr_auc_suspicious"]["ensemble"]
    frame(fig, "Oficial frente a percepción visual mejorada",
          f"Con {d['detector']} (mAP50 de máscara en validación {d['val_mask_map50']:.3f} frente a 0.118 de v2), "
          + (f"G1 se queda en {g1a:.3f}" if f"{g1a:.3f}" == f"{g1b:.3f}" else f"G1 pasa de {g1a:.3f} a {g1b:.3f}")
          + f"; B5 sigue arriba "
          f"({d['systems']['B5']['test']['pr_auc_suspicious']['ensemble']:.3f}).",
          "results/evaluation.json (oficial) y results/posthoc/percepcion_mejorada.json", POSTHOC)
    ax = fig.add_axes([0.08, 0.19, 0.88, 0.64])
    w = 0.38
    for i, s in enumerate(systems):
        for j, (src, c, lab) in enumerate(((e["systems"][s]["test"]["pr_auc_suspicious"], NEUTRO, "Oficial"),
                                           (d["systems"][s]["test"]["pr_auc_suspicious"],
                                            COL["uafeg"] if s == "G1" else "#5F6B7A", "Percepción mejorada"))):
            xx = i + (j - 0.5) * w
            ax.bar(xx, src["ensemble"], width=w - 0.04, color=c, label=lab if i == 0 else None)
            ax.errorbar(xx, src["ensemble"], yerr=[[src["ensemble"] - src["ci_low"]], [src["ci_high"] - src["ensemble"]]],
                        fmt="none", ecolor=INK, lw=1, capsize=2)
            ax.text(xx, src["ci_high"] + 0.01, f"{src['ensemble']:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([f"{s}\n{CORTO[s]}" for s in systems], fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("PR-AUC de Suspicious (eje desde 0.5)")
    ax.legend(frameon=False, loc="upper left")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    save(fig, "06_oficial_vs_mejorada")


# ---------------------------------------------------------------- 7 heatmap condiciones
def fig07():
    e = ev()
    systems = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "G0", "G1", "A2"]
    conds = [f"C{i}" for i in range(10)]
    M = np.array([[e["systems"][s]["per_condition"][c]["accuracy"] for s in systems] for c in conds])
    fig = plt.figure(figsize=(13, 7.2))
    frame(fig, "Dónde acierta y dónde falla cada sistema",
          "Para los sistemas de fusión, la foto de otro vehículo (C2) y el inpainting (C4) son lo más difícil; "
          "G1 acierta más que B5 en las fotos originales (C0).",
          "results/evaluation.json (systems.*.per_condition.*.accuracy; decisión final con abstención)", OFICIAL)
    ax = fig.add_axes([0.25, 0.13, 0.68, 0.72])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("uafeg", ["#F3F6FA", "#9DB2CC", "#1E3A5F"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(conds)):
        for j in range(len(systems)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.2,
                    color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([f"{s}\n{CORTO[s]}" for s in systems], fontsize=8.2)
    for t, s in zip(ax.get_xticklabels(), systems):
        if s == "G1":
            t.set_color(COL["uafeg"])
            t.set_fontweight("bold")
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([f"{CONDICIONES[c]} ({c})" for c in conds], fontsize=9)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.add_patch(plt.Rectangle((systems.index("G1") - 0.5, -0.5), 1, len(conds), fill=False, ec=COL["uafeg"], lw=2))
    fig.text(0.25, 0.065, "Color y número = fracción de reclamos con la decisión correcta (0 a 1). "
             "C10 (edición nunca vista) no tiene etiqueta y se reporta aparte.", fontsize=8.5, color=INK2)
    save(fig, "07_heatmap_condiciones")


CORTO = {"B1": "texto", "B2": "imagen", "B3": "forense", "B4": "concat.", "B5": "fusión\ntardía",
         "B6": "DeepSets", "B7": "DBF", "B8": "Qwen2-VL", "G0": "sin g", "G1": "UAFEG", "A2": "sin\nforense"}


# ---------------------------------------------------------------- 8 compuerta g
def _test_graphs():
    from experiments.thesis_data import load_split
    return load_split("test")


def fig08():
    from uafeg.aggregator import node_gates
    from uafeg.thesis_graph import NODE_KINDS
    test = _test_graphs()
    conds = [m["condition"] for m in test.meta]
    by = {k: [] for k in ("claim", "visual", "forensic", "source")}
    byc = {c: {k: [] for k in by} for c in ("C0", "C9")}
    for g, c in zip(test.graphs, conds):
        gates = node_gates(g)
        for i in range(g.n):
            k = NODE_KINDS[g.kind[i]]
            if k in by:
                by[k].append(float(gates[i]))
                if c in byc:
                    byc[c][k].append(float(gates[i]))
    out = {"tipo": "descriptivo sobre los grafos de test (los mismos que usó la evaluación oficial)",
           "por_especialista": {k: {"n": len(v), "media": float(np.mean(v)), "mediana": float(np.median(v))}
                                for k, v in by.items()},
           "C0_vs_C9": {c: {k: {"n": len(v), "media": float(np.mean(v)) if v else None} for k, v in d.items()}
                        for c, d in byc.items()}}
    (REPO_B / "presentacion/datos").mkdir(parents=True, exist_ok=True)
    (REPO_B / "presentacion/datos/compuerta_g_test.json").write_text(json.dumps(out, indent=1) + "\n")
    names = {"claim": "Relato", "visual": "Imagen (detector)", "forensic": "Forense (TruFor)", "source": "Fuente / calidad"}
    cols = {"claim": COL["narrativa"], "visual": COL["visual"], "forensic": COL["forense"], "source": COL["fuente"]}
    fig = plt.figure(figsize=(13, 5.8))
    vm = out["por_especialista"]["visual"]["media"]
    frame(fig, "La compuerta g decide cuánto confiar en cada evidencia",
          f"Las detecciones visuales reciben muy poco peso (g medio {vm:.2f} en test); en fotos muy borrosas (C9) "
          "el peso del forense y de la fuente cae casi a 0.",
          "presentacion/datos/compuerta_g_test.json (calculado con uafeg.aggregator.node_gates sobre los grafos de test)",
          OFICIAL + " · descriptivo")
    ax = fig.add_axes([0.06, 0.14, 0.5, 0.68])
    kinds = list(by)
    parts = ax.violinplot([by[k] for k in kinds], positions=range(4), widths=0.75, showmedians=True, showextrema=False)
    for b, k in zip(parts["bodies"], kinds):
        b.set_facecolor(cols[k])
        b.set_edgecolor(cols[k])
        b.set_alpha(0.55)
    parts["cmedians"].set_color(INK)
    for i, k in enumerate(kinds):
        ax.text(i, 1.06, f"media {out['por_especialista'][k]['media']:.2f}\n(n = {len(by[k])})", ha="center",
                fontsize=8.3, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([names[k] for k in kinds])
    ax.set_ylim(-0.03, 1.18)
    ax.set_ylabel("Compuerta g (0 = se ignora, 1 = confianza plena)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Distribución de g por especialista (todo test)", loc="left", fontsize=10.5)
    ax2 = fig.add_axes([0.64, 0.14, 0.33, 0.68])
    w = 0.38
    for i, k in enumerate(kinds):
        for j, c in enumerate(("C0", "C9")):
            v = out["C0_vs_C9"][c][k]["media"]
            if v is None:
                ax2.text(i + (j - 0.5) * w, 0.015, "sin\ndetec-\nciones", ha="center", fontsize=6.8, color=MUTED)
                continue
            ax2.bar(i + (j - 0.5) * w, v, width=w - 0.04, color=cols[k], alpha=1.0 if c == "C0" else 0.45,
                    hatch=None if c == "C0" else "///", edgecolor="white")
            ax2.text(i + (j - 0.5) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([names[k].replace(" (", "\n(") for k in kinds], fontsize=8.5)
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.grid(True, color=GRID)
    ax2.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(fc=NEUTRO, label="Foto original (C0)"),
                        Patch(fc=NEUTRO, alpha=0.45, hatch="///", ec="white", label="Foto muy borrosa (C9)")],
               frameon=False, fontsize=8.5, loc="upper right")
    ax2.set_title("g medio: foto nítida frente a borrosa", loc="left", fontsize=10.5)
    save(fig, "08_compuerta_g")


# ---------------------------------------------------------------- 9 aporte forense
def fig09():
    e = ev()
    h3 = e["hypotheses"]["H3"]
    conds = [f"C{i}" for i in range(10)]
    fig = plt.figure(figsize=(12, 6.4))
    frame(fig, "Qué aporta el forense (TruFor) a UAFEG",
          f"Con forense, PR-AUC de Suspicious sube {h3['point_diff']:+.3f} (IC 95 % [{h3['ci_low']:+.3f}, "
          f"{h3['ci_high']:+.3f}]); la ganancia está sobre todo en las fotos editadas por copiar y pegar (C3, C5).",
          "results/evaluation.json (hypotheses.H3; systems.G1/A2.per_condition.*.accuracy)", OFICIAL)
    ax = fig.add_axes([0.27, 0.12, 0.66, 0.72])
    w = 0.38
    for i, c in enumerate(conds):
        for j, (s, col, lab) in enumerate((("A2", NEUTRO, "UAFEG sin forense (A2)"), ("G1", COL["forense"], "UAFEG con forense (G1)"))):
            v = e["systems"][s]["per_condition"][c]["accuracy"]
            ax.barh(i + (0.5 - j) * w, v, height=w - 0.05, color=col, label=lab if i == 0 else None)
            ax.text(v + 0.01, i + (0.5 - j) * w, f"{v:.2f}", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([f"{CONDICIONES[c]} ({c})" for c in conds], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Fracción de decisiones correctas")
    hgrid(ax)
    hs, ls = ax.get_legend_handles_labels()
    ax.legend(hs[::-1], ls[::-1], frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    save(fig, "09_aporte_forense")


# ---------------------------------------------------------------- 10 calibración
def fig10():
    e = ev()
    a = arrays()
    y = a["labels"]
    lab = y >= 0
    systems = ["G1", "G0", "B5"]
    fig = plt.figure(figsize=(13, 5.4))
    frame(fig, "¿Las probabilidades dicen la verdad?",
          "Cuanto más cerca de la diagonal, mejor calibrado (dice 80 % y acierta ~80 %). El tamaño del punto indica cuántos casos hay.",
          "artifacts/thesis/eval_arrays.npz (probabilidades calibradas de test) y results/evaluation.json (Brier, ECE)",
          OFICIAL)
    for k, s in enumerate(systems):
        ax = fig.add_axes([0.06 + k * 0.32, 0.13, 0.25, 0.66])
        p = a[f"{s}_probs"][lab]
        conf = p.max(1)
        corr = p.argmax(1) == y[lab]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = idx == b
            if m.sum():
                xs.append(conf[m].mean())
                ys.append(corr[m].mean())
                ns.append(int(m.sum()))
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        c = COL["uafeg"] if s == "G1" else ("#5F6B7A" if s == "B5" else NEUTRO)
        ax.plot(xs, ys, color=c, lw=1.5, zorder=2)
        ax.scatter(xs, ys, s=[12 + 2.2 * np.sqrt(n_) * 3 for n_ in ns], color=c, edgecolor="white", lw=1.2, zorder=3)
        for x_, y_, n_ in zip(xs, ys, ns):
            if n_ < 30:
                ax.annotate(f"n={n_}", (x_, y_), xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=MUTED)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_aspect("auto")
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        t = e["systems"][s]["test"]
        ax.set_title(SISTEMAS[s], loc="left", fontsize=9.5, color=c)
        ax.text(0.32, 0.93, f"Brier {t['brier']['ensemble']:.3f}\nECE {t['ece']['ensemble']:.3f}", fontsize=9,
                color=INK, va="top")
        ax.set_xlabel("Confianza del sistema")
        if k == 0:
            ax.set_ylabel("Acierto real")
    save(fig, "10_calibracion")


# ---------------------------------------------------------------- 11 riesgo-cobertura
def fig11():
    from src.abstention.abstention import risk_coverage_curve
    e = ev()
    a = arrays()
    y = a["labels"]
    lab = y >= 0
    fig = plt.figure(figsize=(11, 6))
    g1 = e["systems"]["G1"]
    frame(fig, "Responder menos casos para equivocarse menos",
          f"Derivar los casos más inciertos baja el error; G1 deriva el {100 * g1['abstention_rate']:.1f} % de test "
          f"(objetivo 20 %), pero ordena peor que B5 y G0 (AURC más alto)." if g1["test"]["aurc"]["ensemble"] > max(
              e["systems"][k]["test"]["aurc"]["ensemble"] for k in ("B5", "G0")) else
          f"Derivar los casos más inciertos baja el error; G1 deriva el {100 * g1['abstention_rate']:.1f} % de test.",
          "artifacts/thesis/eval_arrays.npz (U y probabilidades de test) y results/evaluation.json (AURC, abstención)",
          OFICIAL)
    ax = fig.add_axes([0.08, 0.13, 0.62, 0.7])
    for s, c, ls in (("G1", COL["uafeg"], "-"), ("G0", NEUTRO, "-"), ("B5", "#5F6B7A", "--")):
        p = a[f"{s}_probs"][lab]
        U = a[f"{s}_U"][lab]
        rc = risk_coverage_curve(U, p.argmax(1) == y[lab])
        ax.plot(rc["coverage"], rc["risk"], color=c, lw=2, ls=ls,
                label=f"{SISTEMAS[s]}  ·  AURC {e['systems'][s]['test']['aurc']['ensemble']:.3f}")
        if s == "G1":
            k = int(round(0.8 * len(U))) - 1
            ax.plot(rc["coverage"][k], rc["risk"][k], "o", ms=10, color=c, mec="white", mew=2, zorder=5)
            ax.annotate(f"cobertura 0.80\nerror {rc['risk'][k]:.3f}", (rc["coverage"][k], rc["risk"][k]),
                        xytext=(-110, 40), textcoords="offset points", fontsize=9, color=c,
                        arrowprops=dict(arrowstyle="-", color=c, lw=1))
    ax.axvline(0.8, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("Cobertura: fracción de casos que el sistema responde (el resto va a revisión)")
    ax.set_ylabel("Riesgo: fracción de errores entre los respondidos")
    ax.set_xlim(0, 1)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.72, 0.4, "Orden: de más seguro a menos\nseguro según la incertidumbre U\n(entropía de la predicción).\n"
             "AURC = área bajo la curva:\nmás bajo es mejor.", fontsize=8.8, color=INK2)
    save(fig, "11_riesgo_cobertura")


# ---------------------------------------------------------------- 12 regresión
def _visual_hit_per_image():
    """Acierto visual por imagen de test (C0, foto original): F1 entre las clases de daño detectadas por
    YOLO (packets reales) y las clases anotadas en HITL."""
    from experiments.thesis_data import _read_packets
    from uafeg.thesis_oracle import _sources
    src = _sources()
    out = {}
    for p in _read_packets("test"):
        c = p.claim
        if c.condition != "C0":
            continue
        gt = {a["damage_class"] for a in src[c.source_image_id][1] if a.get("polygon")}
        pr = {d.damage_class for d in (p.visual.detections if p.visual else [])}
        tp = len(gt & pr)
        f1 = 2 * tp / (len(gt) + len(pr)) if (gt or pr) else 1.0
        out[c.source_image_id] = f1
    return out


def fig12():
    from sklearn.linear_model import LogisticRegression
    from experiments.thesis_data import load_split
    a = arrays()
    test = load_split("test")
    y = a["labels"]
    lab = y >= 0
    corr = a["G1_probs"].argmax(1) == y
    groups = np.array([m["source_image_id"] for m in test.meta])
    vh = _visual_hit_per_image()
    ids = sorted(vh)
    xi = np.array([vh[i] for i in ids])
    yi = np.array([corr[(groups == i) & lab].mean() for i in ids])
    slope, inter = np.polyfit(xi, yi, 1)
    r2 = 1 - np.sum((yi - (slope * xi + inter)) ** 2) / np.sum((yi - yi.mean()) ** 2)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        k = rng.integers(0, len(xi), len(xi))
        if np.ptp(xi[k]) > 0:
            boots.append(np.polyfit(xi[k], yi[k], 1)[0])
    s_lo, s_hi = np.percentile(boots, [2.5, 97.5])
    # logistic regression: acierto de G1 ~ c̄ + U + q̄ (estandarizadas), IC por bootstrap agrupado
    feats = np.load(ART / "runs/G1/seed0/features.npz")["test"]
    from uafeg.aggregator import FEATURE_NAMES
    cbar = np.array([g.coverage_bar for g in test.graphs])
    q = feats[:, FEATURE_NAMES.index("quality_mean")]
    U = a["G1_U"]
    X = np.stack([cbar, U, q], 1)[lab]
    Z = (X - X.mean(0)) / X.std(0)
    yy = corr[lab].astype(int)
    gg = groups[lab]

    def fit(idx):
        return LogisticRegression(C=1e6, max_iter=2000).fit(Z[idx], yy[idx]).coef_[0]
    coef = fit(np.arange(len(yy)))
    ug = np.unique(gg)
    pos = {u: np.flatnonzero(gg == u) for u in ug}
    bc = []
    for _ in range(1000):
        pick = rng.choice(ug, len(ug))
        bc.append(fit(np.concatenate([pos[u] for u in pick])))
    bc = np.array(bc)
    lo, hi = np.percentile(bc, [2.5, 97.5], axis=0)
    res = {"tipo": POSTHOC + " (descriptivo, test)",
           "regresion_lineal_por_imagen": {"x": "F1 de clases de daño YOLO vs anotación HITL (C0, foto original)",
                                           "y": "fracción de reclamos de esa imagen que G1 acierta (clase más probable)",
                                           "n_imagenes": len(ids), "pendiente": float(slope), "intercepto": float(inter),
                                           "r2": float(r2), "pendiente_ic95_bootstrap": [float(s_lo), float(s_hi)],
                                           "n_bootstrap": 2000},
           "regresion_logistica": {"y": "G1 acierta (clase más probable) en test etiquetado",
                                   "x_estandarizadas": ["cobertura c̄", "incertidumbre U", "calidad q̄"],
                                   "coef": coef.tolist(), "ic95_bootstrap_agrupado": [lo.tolist(), hi.tolist()],
                                   "odds_ratio_por_desviacion": np.exp(coef).tolist(),
                                   "n": int(len(yy)), "n_bootstrap": 1000}}
    (REPO_B / "presentacion/datos").mkdir(parents=True, exist_ok=True)
    (REPO_B / "presentacion/datos/regresiones.json").write_text(json.dumps(res, indent=1) + "\n")
    fig = plt.figure(figsize=(13, 6))
    sig = s_lo > 0 or s_hi < 0
    frame(fig, "Cuando el detector acierta, UAFEG acierta más" if sig and slope > 0 else
          "El acierto de UAFEG casi no depende de lo que vea el detector",
          f"Por imagen: pendiente {slope:+.2f} (IC 95 % [{s_lo:+.2f}, {s_hi:+.2f}]), R² = {r2:.2f}, coherente con g visual ≈ 0. "
          "Lo que sí anticipa el error es la incertidumbre U.",
          "presentacion/datos/regresiones.json (calculado sobre test con eval_arrays.npz y los packets reales)",
          POSTHOC + " · descriptivo, no causal")
    ax = fig.add_axes([0.06, 0.13, 0.42, 0.68])
    jit = rng.uniform(-0.012, 0.012, len(xi))
    ax.scatter(xi + jit, yi, s=26, color=COL["visual"], alpha=0.7, edgecolor="white", lw=0.6)
    xx = np.linspace(0, 1, 50)
    ax.plot(xx, slope * xx + inter, color=COL["uafeg"], lw=2)
    ax.set_xlabel("Acierto del detector en esa foto (F1 de tipos de daño)")
    ax.set_ylabel("Fracción de reclamos que G1 acierta")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"Cada punto es una imagen de test (n = {len(ids)})", loc="left", fontsize=10.5)
    ax2 = fig.add_axes([0.62, 0.13, 0.34, 0.68])
    names = ["Más cobertura de la\nevidencia (c̄)", "Más incertidumbre\ndel modelo (U)", "Mejor calidad\nde la foto (q̄)"]
    orr, olo, ohi = np.exp(coef), np.exp(lo), np.exp(hi)
    for i in range(3):
        c = COL["SUPPORTED"] if olo[i] > 1 else (COL["SUSPICIOUS"] if ohi[i] < 1 else MUTED)
        ax2.plot([olo[i], ohi[i]], [i, i], color=c, lw=3, solid_capstyle="round")
        ax2.plot(orr[i], i, "o", ms=10, color=c, mec="white", mew=2)
        ax2.text(orr[i], i + 0.25, f"×{orr[i]:.2f}  [{olo[i]:.2f}, {ohi[i]:.2f}]", ha="center", fontsize=8.5)
    ax2.axvline(1, color=INK2, lw=1)
    ax2.set_xscale("log")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_ylim(-0.6, 2.7)
    ax2.invert_yaxis()
    ax2.set_xlabel("Las chances de acierto se multiplican por… (por cada desviación estándar)")
    ax2.set_title("Regresión logística del acierto de G1", loc="left", fontsize=10.5)
    save(fig, "12_regresion")


# ---------------------------------------------------------------- 13 ablaciones
def fig13():
    e = ev()
    items = [("G1", "UAFEG completo (G1)"), ("G0", "A1 · sin compuerta (g = 1) = G0"),
             ("A2", "A2 · sin forense"), ("B6", "A3 · DeepSets en vez del agregador = B6"),
             ("A4", "A4 · aristas sin atributos"), ("A5", "A5 · sin controles benignos en entrenamiento")]
    fig = plt.figure(figsize=(13, 5.8))
    frame(fig, "Qué pasa al quitar cada pieza de UAFEG",
          "Quitar el forense (A2) es lo que más daña; sin compuerta (A1) o con DeepSets (A3) mejora un poco; "
          "sin controles benignos (A5) empeora el Brier.",
          "results/evaluation.json (systems.G1, G0, A2, B6, A4, A5; IC 95 %)", OFICIAL)
    for k, (m, ttl, lim) in enumerate((("pr_auc_suspicious", "PR-AUC de Suspicious (más alto es mejor; eje desde 0.6)", (0.6, 0.95)),
                                       ("brier", "Brier (más bajo es mejor; eje desde 0.2)", (0.2, 0.36)))):
        ax = fig.add_axes([0.27 + k * 0.38, 0.12, 0.3, 0.72])
        for i, (s, _) in enumerate(items):
            t = e["systems"][s]["test"][m]
            ax.barh(i, t["ensemble"], height=0.6, color=COL["uafeg"] if s == "G1" else NEUTRO)
            ax.errorbar(t["ensemble"], i, xerr=[[t["ensemble"] - t["ci_low"]], [t["ci_high"] - t["ensemble"]]],
                        fmt="none", ecolor=INK, lw=1, capsize=2)
            ax.text(t["ci_high"] + 0.003, i, f"{t['ensemble']:.3f}", va="center", fontsize=8.5)
        ax.set_xlim(*lim)
        ax.invert_yaxis()
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([t for _, t in items] if k == 0 else [], fontsize=9)
        ax.set_title(ttl, loc="left", fontsize=9.5)
        hgrid(ax)
    save(fig, "13_ablaciones")


# ---------------------------------------------------------------- 14 especialistas
def fig14():
    sp = json.loads((R / "final/specialists.json").read_text())
    v, f, n = sp["visual"], sp["forensic"], sp["narrative"]
    he, ex = v["hitl_eval"], v["external_cardd"]
    fig = plt.figure(figsize=(13.5, 5.6))
    frame(fig, "Los especialistas: el detector visual es el eslabón débil",
          f"YOLO encuentra pocos daños (máscara mAP50 {he['val']['mask_map50']:.3f} en validación); TruFor separa "
          f"bien el copiar-pegar (AUROC {f['auroc_C3_vs_C0']:.2f}) pero no el inpainting ({f['auroc_C4_vs_C0']:.2f}).",
          "results/final/specialists.json", OFICIAL + " · especialistas medidos")
    ax = fig.add_axes([0.065, 0.16, 0.26, 0.64])
    bars = [("HITL val\nmáscara", he["val"]["mask_map50"]), ("HITL test\nmáscara", he["test"]["mask_map50"]),
            ("HITL test\ncaja", he["test"]["box_map50"]), ("CarDD\ncaja (clases)", ex["box_map50_mapped"]),
            ("CarDD caja\n(solo «daño»)", ex["box_map50_agnostic"])]
    for i, (t, val) in enumerate(bars):
        ax.bar(i, val, width=0.62, color=COL["visual"] if i < 3 else "#7CC3CD")
        ax.text(i, val + 0.004, f"{val:.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([t for t, _ in bars], fontsize=7.6)
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("mAP50 (1 = perfecto)")
    ax.set_title("Visual · YOLOv8n-seg", loc="left", fontsize=10.5, color=COL["visual"])
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax2 = fig.add_axes([0.39, 0.16, 0.27, 0.64])
    tb = [("Copiar y\npegar (C3)", f["auroc_C3_vs_C0"]), ("Inpainting\n(C4)", f["auroc_C4_vs_C0"]),
          ("Editada +\nrelato (C5)", f["auroc_C5_vs_C0"]), ("C3–C5\njuntas", f["auroc_C3C4C5_vs_C0"])]
    for i, (t, val) in enumerate(tb):
        ax2.bar(i, val, width=0.62, color=COL["forense"])
        ax2.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=8.5)
    ax2.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax2.text(3.4, 0.51, "azar", fontsize=8, color=MUTED, ha="right")
    ax2.set_xticks(range(len(tb)))
    ax2.set_xticklabels([t for t, _ in tb], fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("AUROC frente a la foto original (validación)")
    ax2.set_title("Forense · TruFor", loc="left", fontsize=10.5, color=COL["forense"])
    ax2.yaxis.grid(True, color=GRID)
    ax2.set_axisbelow(True)
    ax3 = fig.add_axes([0.73, 0.16, 0.24, 0.64])
    nb = [("Tipo de daño\nexacto", n["damage_type_exact_acc"]), ("Familia\nde daño", n["damage_type_family_acc"]),
          ("Afirma o\nniega", n["polarity_acc"])]
    for i, (t, val) in enumerate(nb):
        ax3.bar(i, val, width=0.62, color=COL["narrativa"])
        ax3.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=8.5)
    ax3.set_xticks(range(len(nb)))
    ax3.set_xticklabels([t for t, _ in nb], fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel(f"Acierto (validación, {n['n_validation_claims_with_assertion']} reclamos)")
    ax3.set_title("Narrativo · mDeBERTa NLI", loc="left", fontsize=10.5, color=COL["narrativa"])
    ax3.yaxis.grid(True, color=GRID)
    ax3.set_axisbelow(True)
    save(fig, "14_especialistas")


# ---------------------------------------------------------------- 15 caso paso a paso
def _packet(claim_id):
    from experiments.thesis_data import packets_dir
    for line in (packets_dir() / "test.jsonl").read_text().splitlines():
        if claim_id in line:
            d = json.loads(line)
            if d["claim"]["claim_id"] == claim_id:
                return d
    raise KeyError(claim_id)


def _pick_cases():
    """Regla fija (no elegida por resultado agregado): recorre review/seleccion.json en su orden y toma
    (a) el primer caso SUSPICIOUS que G1 decidió correctamente sin abstenerse y (b) la primera abstención.
    (En el paquete no hay ningún C3 decidido sin abstención; el primer acierto Sospechoso es un C5.)"""
    ids = json.loads((REPO_B / "review/seleccion.json").read_text())["ids"]
    ok = ab = None
    for cid in ids:
        r = json.loads((REPO_B / f"review/cases/{cid}.json").read_text())
        d = r["decision"]
        if ok is None and r["expected_label"] == "SUSPICIOUS" and not d["abstained"] and d["label"] == r["expected_label"]:
            ok = r
        if ab is None and d["abstained"]:
            ab = r
    return ok, ab


def _case_fig(rep, name, title, subtitle):
    from PIL import Image
    from uafeg.benchmark import canonical_repository
    from uafeg.thesis_graph import decode_mask
    pk = _packet(rep["claim_id"])
    repo = canonical_repository()
    fig = plt.figure(figsize=(14, 7.6))
    frame(fig, title, subtitle, f"review/cases/{rep['claim_id']}.json y el EvidencePacket real del Repo A",
          OFICIAL + " · caso real del paquete de revisión")
    with Image.open(repo / pk["claim"]["image_path"]) as im:
        im = im.convert("RGB")
        im.thumbnail((640, 640))
        img = np.asarray(im)
    h, w = img.shape[:2]
    ax = fig.add_axes([0.01, 0.48, 0.24, 0.38])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("1 · Foto recibida", loc="left", fontsize=10)
    ax = fig.add_axes([0.255, 0.48, 0.24, 0.38])
    ov = img.astype(np.float32).copy()
    dets = (pk.get("visual") or {}).get("detections") or []
    for d in dets:
        m = decode_mask(d.get("mask_64x48"))
        if m is None:
            continue
        mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 0
        ov[mm] = ov[mm] * 0.45 + np.array([15, 139, 158], np.float32) * 0.55
    ax.imshow(ov.clip(0, 255).astype(np.uint8))
    ax.axis("off")
    ax.set_title(f"2 · Daños que ve YOLO ({len(dets)})", loc="left", fontsize=10, color=COL["visual"])
    for d in dets[:3]:
        x1, y1 = d["bbox_norm"][0] * w, d["bbox_norm"][1] * h
        ax.text(x1, y1, f"{d['damage_class']} {d['confidence']:.2f}", fontsize=7, color="white",
                bbox=dict(fc=COL["visual"], ec="none", pad=1))
    ax = fig.add_axes([0.50, 0.48, 0.24, 0.38])
    sha = pk["claim"]["transformation"]["image_sha256"]
    npz = repo / "artifacts/specialists/forensic_maps" / f"{sha}.npz"
    ax.imshow(img, alpha=0.55)
    if npz.exists():
        loc = np.load(npz)["localization"].astype(np.float32)
        loc = np.asarray(Image.fromarray(loc).resize((w, h), Image.BILINEAR))
        ax.imshow(np.clip(loc, 0, 1), cmap="Purples", alpha=0.6, vmin=0, vmax=1)
    fo = pk.get("forensic") or {}
    ax.axis("off")
    ax.set_title(f"3 · Mapa forense TruFor (puntaje {fo.get('score', float('nan')):.2f})", loc="left", fontsize=10,
                 color=COL["forense"])
    # graph
    ax = fig.add_axes([0.755, 0.48, 0.235, 0.38])
    ax.axis("off")
    ax.set_title("4 · Grafo de evidencias y su g", loc="left", fontsize=10, color=COL["uafeg"])
    ax.text(0, -0.04, "tamaño e intensidad del nodo = g; grosor de la arista = peso", fontsize=7, color=MUTED)
    nodes = rep["graph"]["nodes"]
    kc = {"claim": COL["narrativa"], "visual": COL["visual"], "forensic": COL["forense"], "source": COL["fuente"]}
    kn = {"claim": "afirmación", "visual": "daño visual", "forensic": "forense", "source": "fuente"}
    groups = {k: [n for n in nodes if n["kind"] == k] for k in kc}
    pos = {}
    colx = {"claim": 0.12, "visual": 0.62, "forensic": 0.62, "source": 0.12}
    yv = 0.9
    for n in groups["visual"][:5]:
        pos[n["id"]] = (colx["visual"], yv)
        yv -= 0.16
    for n in groups["forensic"]:
        pos[n["id"]] = (colx["forensic"], min(yv, 0.12))
    for i, n in enumerate(groups["claim"]):
        pos[n["id"]] = (colx["claim"], 0.75 - 0.2 * i)
    for n in groups["source"]:
        pos[n["id"]] = (colx["source"], 0.3)
    for e_ in rep["graph"]["edges"]:
        if e_["src"] in pos and e_["dst"] in pos:
            (x1, y1), (x2, y2) = pos[e_["src"]], pos[e_["dst"]]
            ax.plot([x1, x2], [y1, y2], color=MUTED, lw=0.6 + 4 * min(1, e_["weight"]), alpha=0.7, zorder=1)
    for n in nodes:
        if n["id"] not in pos:
            continue
        x, y = pos[n["id"]]
        ax.scatter(x, y, s=180 + 700 * n["g"], color=kc[n["kind"]], alpha=0.25 + 0.75 * n["g"], edgecolor="white",
                   zorder=2)
        if x < 0.5:
            ax.text(x, y - 0.09, f"{kn[n['kind']]}\ng={n['g']:.2f}", fontsize=7.6, va="top", ha="center", color=INK)
        else:
            ax.text(x + 0.09, y, f"{kn[n['kind']]}  g={n['g']:.2f}", fontsize=7.6, va="center", color=INK)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1)
    # text panel
    d = rep["decision"]
    ax = fig.add_axes([0.01, 0.06, 0.45, 0.36])
    ax.axis("off")
    ax.set_title("5 · Relato y afirmaciones extraídas", loc="left", fontsize=10, color=COL["narrativa"])
    lines = textwrap.wrap(f"«{rep['narrative_text']}»", 85)
    for a_ in rep["assertions"][:3]:
        lines.append(f"• tipo de daño leído: {a_['damage_type']} · {'afirma' if a_['polarity'] == 'affirmed' else 'niega'}"
                     f" · incertidumbre u = {a_['u']:.2f}")
    lines.append("")
    lines.append(f"Condición real: {CONDICIONES[rep['condition']]} ({rep['condition']}) · "
                 f"etiqueta esperada: {CLASES[rep['expected_label']]}")
    ax.text(0, 0.92, "\n".join(lines), fontsize=8.8, va="top", color=INK, linespacing=1.5)
    ax = fig.add_axes([0.53, 0.1, 0.18, 0.27])
    pr = d["calibrated_probabilities"]
    for i, c in enumerate(CLS):
        ax.barh(i, pr[c], color=COL[c], height=0.6)
        ax.text(pr[c] + 0.02, i, f"{pr[c]:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels([CLASES[c] for c in CLS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.set_title("6 · Probabilidades calibradas", loc="left", fontsize=10)
    ax = fig.add_axes([0.74, 0.06, 0.25, 0.36])
    ax.axis("off")
    ax.set_title("7 · Decisión y explicación", loc="left", fontsize=10, color=COL["uafeg"])
    lab_ = CLASES[d["label"]] + (" → revisión humana" if d["abstained"] else "")
    txt = [f"Decisión: {lab_}"]
    if d["abstained"]:
        txt.append(f"Motivo: {d['abstention_reason']}")
    txt.append(f"U = {d['U']:.3f} (umbral τU = {d['tau_U']:.3f})")
    txt.append(f"Cobertura c̄ = {d['coverage_bar']:.2f}")
    txt.append("")
    txt.append("Evidencia con más peso:")
    for t_ in rep["top_evidence"][:2]:
        txt += textwrap.wrap("• " + t_["detail"], 52)
    ax.text(0, 0.92, "\n".join(txt), fontsize=8.2, va="top", color=INK, linespacing=1.45)
    save(fig, name)


def fig15():
    ok, ab = _pick_cases()
    if ok:
        _case_fig(ok, "15a_caso_correcto", "Un caso real, paso a paso",
                  f"{CONDICIONES[ok['condition']]} ({ok['condition']}): UAFEG lo decide como "
                  f"{CLASES[ok['decision']['label']]}, que es la etiqueta correcta.")
    if ab:
        _case_fig(ab, "15b_caso_abstencion", "Un caso que UAFEG deriva a una persona",
                  "La incertidumbre supera el umbral fijado en validación: el sistema no decide solo y explica por qué.")


# ---------------------------------------------------------------- 16 tablas
def _table_png(name, title, subtitle, source, kind, header, rows, highlight=None, widths=None):
    fig = plt.figure(figsize=(14, 1.6 + 0.42 * len(rows)))
    frame(fig, title, subtitle, source, kind)
    top = 1 - 0.95 / fig.get_figheight()
    ax = fig.add_axes([0.012, 0.35 / fig.get_figheight(), 0.976, top - 0.35 / fig.get_figheight()])
    ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=header, cellLoc="center", colWidths=widths, bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.6)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif highlight and rows[r - 1][0] in highlight:
            cell.set_facecolor("#E8EEF5")
            cell.set_text_props(fontweight="bold")
        if c == 0 and r > 0:
            cell.set_text_props(ha="left")
            cell._loc = "left"
            cell.PAD = 0.02
    save(fig, name)


def fig16():
    e = ev()
    mets = [("pr_auc_suspicious", "PR-AUC Susp. ↑"), ("macro_f1", "Macro-F1 ↑"), ("brier", "Brier ↓"),
            ("ece", "ECE ↓"), ("aurc", "AURC ↓")]

    def rows_for(systems):
        out = []
        for s in systems:
            t = e["systems"][s]["test"]
            out.append([SISTEMAS[s]] + [f"{t[m]['ensemble']:.3f} [{t[m]['ci_low']:.3f}, {t[m]['ci_high']:.3f}]"
                                        for m, _ in mets] + [f"{e['systems'][s]['abstention_rate']:.3f}"])
        return out
    header = ["Sistema"] + [n for _, n in mets] + ["Derivados"]
    widths = [0.27] + [0.13] * 5 + [0.08]
    _table_png("16a_tabla_principal", "Tabla principal (test)", "Ensamble calibrado de 5 semillas; entre corchetes, "
               "IC 95 % por bootstrap agrupado por imagen (2 000 réplicas). ↑ más alto es mejor, ↓ más bajo es mejor.",
               "results/evaluation.json", OFICIAL, header, rows_for(MAIN), {SISTEMAS["G1"], SISTEMAS["B5"]}, widths)
    _table_png("16b_tabla_ablaciones", "Ablaciones (test)", "Cada fila quita una pieza de UAFEG (A1 = G0, A3 = B6).",
               "results/evaluation.json", OFICIAL, header, rows_for(["G1", "G0", "A2", "B6", "A4", "A5"]),
               {SISTEMAS["G1"]}, widths)
    _table_png("16c_tabla_oraculo", "Oráculo (test)", "Percepción perfecta derivada de la procedencia del benchmark: "
               "diagnóstico del techo, no resultado principal.", "results/evaluation.json", OFICIAL, header,
               rows_for(["G1", "G1_oracleN", "G1_oracleVF", "G1_oracleNVF", "G0", "G0_oracleNVF"]), {SISTEMAS["G1"]}, widths)
    h = e["hypotheses"]
    best = h["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    hr = []
    for k, desc in (("H1", f"G1 vs {best}, PR-AUC Susp. en estrés"), ("H2", "G1 vs G0, Brier (G0 − G1)"),
                    ("H3", "G1 vs A2, PR-AUC Susp.")):
        x = h[k]
        ok = x["reject_H0_at_0.05"] and x["point_diff"] > 0
        hr.append([f"{k} · {desc}", f"{x['point_diff']:+.4f}", f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]",
                   "< 0.0005" if x["p_holm"] == 0 else f"{x['p_holm']:.4f}",
                   "ACEPTADA" if ok else ("RECHAZADA (gana el rival)" if x["reject_H0_at_0.05"] else "no sostenida")])
    _table_png("16d_tabla_hipotesis", "Hipótesis (test, Holm α = 0.05)",
               "Diferencias orientadas a favor de UAFEG (> 0 = UAFEG mejor).", "results/evaluation.json (hypotheses)",
               OFICIAL, ["Hipótesis", "Diferencia", "IC 95 %", "p (Holm)", "Veredicto"], hr, None,
               [0.36, 0.12, 0.22, 0.1, 0.2])


FIGS = {1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06, 7: fig07, 8: fig08, 9: fig09, 10: fig10,
        11: fig11, 12: fig12, 13: fig13, 14: fig14, 15: fig15, 16: fig16}


def main():
    which = [int(a) for a in sys.argv[1:]] or list(FIGS)
    for k in which:
        FIGS[k]()


if __name__ == "__main__":
    main()
