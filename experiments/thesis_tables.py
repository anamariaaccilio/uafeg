"""Tablas LaTeX (listas para el capítulo 5) y figuras desde results/evaluation.json.

    python -m experiments.thesis_tables

Nada se transcribe a mano: cada cifra sale de results/evaluation.json, que a su vez
sale de corridas registradas (results/runs/*.json, artifacts/thesis/runs/*/record.json).
"""
from __future__ import annotations

import json

import numpy as np

from experiments.thesis_data import ART, REPO_B

RESULTS = REPO_B / "results"
TABLES = RESULTS / "final"
FIGS = RESULTS / "final"
NAMES = {"B0": "Reglas explícitas", "B1": "Solo texto", "B2": "Solo daño visual", "B3": "Solo forense",
         "B4": "Concatenación + MLP", "B5": "Fusión tardía calibrada", "B7": "DBF", "B8": "MLLM congelado",
         "B6": "DeepSets", "G0": "Agregador, $g=1$", "G1": r"\modelo{} (agregador con $g$)",
         "A1": "A1 sin compuerta ($g_j=1$)", "A2": "A2 sin forense", "A3": "A3 DeepSets", "A4": "A4 aristas sin atributos continuos",
         "A5": "A5 sin C8/C10 en train",
         "G1_oracleN": "Oráculo narrativa", "G1_oracleVF": "Oráculo visual+forense", "G1_oracleNVF": "Oráculo ambos",
         "G0_oracleNVF": "G0, oráculo ambos"}
COLS = [("pr_auc_suspicious", "PR-AUC", 3), ("macro_f1", "Macro-F1", 3), ("brier", "Brier", 3),
        ("ece", "ECE", 3), ("aurc", "AURC", 3)]


def table(rows, ev, caption, label, key="test", shaded=("B6", "G0", "G1"), notes=""):
    """Two lines per system: seed mean ± sd (each seed calibrated), then ensemble value and its 95 % CI."""
    lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.1}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "c" * len(COLS) + "}", r"\toprule",
             r"\textbf{ID} & \textbf{Sistema} & " + " & ".join(rf"\textbf{{{c[1]}}}" for c in COLS) + r" \\",
             r"\midrule"]
    for sid in rows:
        if sid == "--":
            lines.append(r"\midrule")
            continue
        alias = ev.get("aliases", {}).get(sid)
        source = alias or sid
        prefix = r"\rowcolor{black!6} " if sid in shaded else ""
        label_id = sid.replace("_", r"\_")
        name = NAMES.get(sid, sid) + (f" (= {alias})" if alias else "")
        if source in ev["systems"]:
            m = ev["systems"][source][key]
            l1 = " & ".join(f"{m[c[0]]['seed_mean']:.{c[2]}f} $\\pm$ {m[c[0]]['seed_sd']:.{c[2]}f}" for c in COLS)
            l2 = " & ".join(f"{{\\scriptsize {m[c[0]]['ensemble']:.{c[2]}f} [{m[c[0]]['ci_low']:.{c[2]}f}, {m[c[0]]['ci_high']:.{c[2]}f}]}}" for c in COLS)
            lines.append(f"{prefix}{label_id} & {name} & {l1} \\\\")
            lines.append(f"{prefix} & & {l2} \\\\")
        else:
            reason = ev.get("not_run", {}).get(sid, "NOT_RUN")
            lines.append(f"{prefix}{label_id} & {name} & \\multicolumn{{{len(COLS)}}}{{c}}{{\\textit{{NOT\\_RUN}}: {reason}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    if notes:
        lines.insert(-1, notes)
    return "\n".join(lines) + "\n"


def condition_table(ev, systems=("B6", "G0", "G1")):
    best = ev["hypotheses"]["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    systems = tuple(dict.fromkeys(systems + (best,)))
    conds = [c for c in ev["systems"]["G1"]["per_condition"] if c != "C10"]
    head = r"\begin{table}[H]\centering\footnotesize" + "\n" + r"\begin{tabular}{l" + "r" * len(conds) + "}\n\\toprule\n"
    head += "Sistema & " + " & ".join(conds) + r" \\" + "\n\\midrule\n"
    body = ""
    for s in systems:
        if s not in ev["systems"]:
            continue
        pc = ev["systems"][s]["per_condition"]
        body += s.replace("_", r"\_") + " & " + " & ".join(f"{pc[c]['delta_M']:+.3f}" for c in conds) + r" \\" + "\n"
    tail = ("\\bottomrule\n\\end{tabular}\n\\caption[Degradación por condición]{Degradación $\\Delta M_s = M_{C0} - M_s$ "
            "con $M$ = exactitud de la decisión final (ensamble calibrado, con abstención), en test. "
            "C10 se reporta aparte como estrés OOD.}\n\\label{tab:delta_m}\n\\end{table}\n")
    return head + body + tail


def hypotheses_table(ev):
    h = ev["hypotheses"]
    best = h["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    text = {"H1": f"G1 vs {best} (mejor sin grafo en validación); PR-AUC SUSPICIOUS en C3--C5 vs C8--C9; G1 $-$ {best}",
            "H2": "G1 vs G0; Brier; Brier(G0) $-$ Brier(G1)",
            "H3": "G1 vs A2 (sin forense); PR-AUC SUSPICIOUS; G1 $-$ A2"}

    def pf(p, n):
        return f"$<${1 / n:.4f}" if p == 0 else f"{p:.4f}"
    lines = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabularx}{\textwidth}{lXccccc}", r"\toprule",
             r"\textbf{H} & \textbf{Contraste (efecto $>0$ favorece a G1)} & \textbf{Efecto} & \textbf{IC 95\%} & $p$ & "
             r"$p_{\mathrm{Holm}}$ & \textbf{Veredicto} \\", r"\midrule"]
    for k in ("H1", "H2", "H3"):
        x = h[k]
        if x.get("status") == "NOT_EVALUATED":
            lines.append(f"{k} & NOT\\_EVALUATED & -- & -- & -- & -- & -- \\\\")
            continue
        verdict = ("aceptada" if x["point_diff"] > 0 else "rechazada") if x["reject_H0_at_0.05"] else "no sostenida"
        n = x["n_resamples"]
        lines.append(f"{k} & {text[k]} & {x['point_diff']:+.4f} & [{x['ci_low']:+.4f}, {x['ci_high']:+.4f}] & "
                     f"{pf(x['p_value_approx'], n)} & {pf(x['p_holm'], n)} & {verdict} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\caption[Contraste de hipótesis]{Contraste de H1--H3 con bootstrap pareado agrupado por "
              r"\texttt{source\_image\_id} (2\,000 réplicas) y corrección de Holm ($\alpha=0{,}05$, $m=3$). "
              r"$p<0.0005$: ninguna de las 2\,000 réplicas tuvo el signo contrario.}",
              r"\label{tab:hipotesis}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def figures(ev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.abstention.abstention import risk_coverage_curve
    arr = np.load(ART / "eval_arrays.npz")
    y = arr["labels"]
    lab = y >= 0
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for s in ("G1", "G0", "B6", ev["hypotheses"]["H1"]["comparison"].split(" vs ")[1].split(" ")[0]):
        if f"{s}_probs" not in arr:
            continue
        p, U = arr[f"{s}_probs"][lab], arr[f"{s}_U"][lab]
        rc = risk_coverage_curve(U, p.argmax(1) == y[lab])
        ax.plot(rc["coverage"], rc["risk"], label=f"{s} (AURC {rc['aurc']:.3f})")
    ax.set_xlabel("Cobertura"); ax.set_ylabel("Riesgo (error)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "riesgo_cobertura.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    for s in ("G1", "G0", "B6"):
        p = arr[f"{s}_probs"][lab]
        conf, correct = p.max(1), p.argmax(1) == y[lab]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [correct[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, "o-", label=s)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Confianza"); ax.set_ylabel("Exactitud")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "calibracion.pdf"); plt.close(fig)

    conds = [c for c in ev["systems"]["G1"]["per_condition"] if c != "C10"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.27
    for i, s in enumerate(("B6", "G0", "G1")):
        d = [ev["systems"][s]["per_condition"][c]["delta_M"] for c in conds]
        ax.bar(np.arange(len(conds)) + (i - 1) * width, d, width, label=s)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_ylabel(r"$\Delta M_s$ (exactitud)"); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "delta_m_condicion.pdf"); plt.close(fig)


def main() -> int:
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    ev["aliases"] = {"A1": "G0", "A3": "B6", "G1_oracle_none": "G1"}
    TABLES.mkdir(parents=True, exist_ok=True)
    main_rows = ["B0", "B1", "B2", "B3", "--", "B4", "B5", "B7", "B8", "--", "B6", "G0", "G1"]
    (TABLES / "tab_main.tex").write_text(table(
        main_rows, ev, r"Resultados principales en test (evaluado una sola vez). Primera línea: media $\pm$ desviación entre 5 semillas "
        r"(cada una calibrada con su $T^\ast$); segunda línea: ensamble calibrado e IC 95\% del bootstrap agrupado por "
        r"\texttt{source\_image\_id} (2\,000 réplicas) sobre el ensamble calibrado. Sin GATv2: G0/G1 son el agregador "
        r"interpretable sobre el grafo.", "tab:resultados_main"))
    (TABLES / "tab_ablaciones.tex").write_text(table(
        ["G1", "--", "A1", "A2", "A3", "A4", "A5"], ev,
        "Ablaciones A1--A5 de G1, mismo protocolo que la tabla principal.", "tab:ablaciones", shaded=("G1",)))
    (TABLES / "tab_oraculo.tex").write_text(table(
        ["G1", "G1_oracleN", "G1_oracleVF", "G1_oracleNVF", "--", "G0", "G0_oracleNVF"], ev,
        "Protocolo oráculo (diagnóstico): percepción predicha u oráculo para la narrativa y para visual+forense.",
        "tab:oraculo", shaded=("G1",)))
    (TABLES / "tab_sensibilidad.tex").write_text(table(
        ["B6", "G0", "G1"], ev, "Análisis de sensibilidad C2: test sin los casos AMBIGUOS (criterios C2--C5). "
        "Compárese con la tabla principal (con ambiguos).",
        "tab:sensibilidad", key="test_without_ambiguous"))
    (TABLES / "tab_delta_m.tex").write_text(condition_table(ev))
    (TABLES / "tab_hipotesis.tex").write_text(hypotheses_table(ev))
    figures(ev)
    print("tables:", sorted(p.name for p in TABLES.iterdir()), "figures:", sorted(p.name for p in FIGS.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
