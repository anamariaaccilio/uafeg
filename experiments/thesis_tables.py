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
TABLES = RESULTS / "tables"
FIGS = RESULTS / "figures"
NAMES = {"B0": "Reglas explícitas", "B1": "Solo texto", "B2": "Solo daño visual", "B3": "Solo forense",
         "B4": "Concatenación + MLP", "B5": "Fusión tardía calibrada", "B7": "DBF", "B8": "MLLM congelado",
         "B6": "DeepSets", "P1": "Grafo sin compuerta", "P2": r"\modelo{}",
         "A1": "A1 sin compuerta ($g_j=1$)", "A2": "A2 sin forense", "A3": "A3 DeepSets", "A4": "A4 sin atributos de arista",
         "A5": "A5 sin controles benignos en train", "P2_legacy": "P2, taxonomía previa",
         "P2_oracleN": "Oráculo narrativa", "P2_oracleVF": "Oráculo visual+forense", "P2_oracleNVF": "Oráculo ambos"}
COLS = [("pr_auc_suspicious", "PR-AUC", 3), ("macro_f1", "Macro-F1", 3), ("brier", "Brier", 3),
        ("ece", "ECE", 3), ("aurc", "AURC", 3)]


def cell(m, digits, with_ci=True):
    s = f"{m['seed_mean']:.{digits}f} $\\pm$ {m['seed_sd']:.{digits}f}"
    if with_ci:
        s += f" \\newline {{\\tiny [{m['ci_low']:.{digits}f}, {m['ci_high']:.{digits}f}]}}"
    return s


def table(rows, ev, caption, label, key="test", shaded=("B6", "P1", "P2"), notes=""):
    lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\renewcommand{\arraystretch}{1.2}",
             r"\begin{tabularx}{\textwidth}{lX" + "c" * len(COLS) + "}", r"\toprule",
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
        if source in ev["systems"]:
            m = ev["systems"][source][key]
            vals = " & ".join(cell(m[c[0]], c[2]) for c in COLS)
            lines.append(f"{prefix}{label_id} & {NAMES.get(sid, sid)}{' (= ' + alias + ')' if alias else ''} & {vals} \\\\")
        else:
            reason = ev.get("not_run", {}).get(sid, "NOT_RUN")
            lines.append(f"{prefix}{label_id} & {NAMES.get(sid, sid)} & \\multicolumn{{{len(COLS)}}}{{c}}{{\\textit{{NOT\\_RUN}}: {reason}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    if notes:
        lines.insert(-1, notes)
    return "\n".join(lines) + "\n"


def condition_table(ev, systems=("B6", "P1", "P2")):
    conds = [c for c in ev["systems"]["P2"]["per_condition"] if c != "C10"]
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
    lines = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabularx}{\textwidth}{lXcccc}", r"\toprule",
             r"\textbf{H} & \textbf{Contraste} & \textbf{Efecto} & \textbf{IC 95\%} & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for k in ("H1", "H2", "H3"):
        x = h[k]
        if x.get("status") == "NOT_EVALUATED":
            lines.append(f"{k} & {x['reason'][:120]}... & -- & -- & -- & {x['p_holm']:.3f} \\\\")
        else:
            lines.append(f"{k} & {x['comparison']}; {x['statistic']} & {x['point_diff']:+.4f} & "
                         f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}] & {x['p_value_approx']:.3f} & {x['p_holm']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\caption[Contraste de hipótesis]{Contraste de H1--H3 con bootstrap pareado agrupado por "
              r"\texttt{source\_image\_id} y corrección de Holm ($\alpha=0{,}05$, $m=3$).}",
              r"\label{tab:hipotesis}", r"\end{table}"]
    return "\n".join(l.replace("_", r"\_") if "texttt" not in l and "label" not in l and "$" not in l else l
                     for l in lines) + "\n"


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
    for s in ("P2", "P1", "B6", ev["hypotheses"]["H1"]["comparison"].split(" vs ")[1].split(" ")[0]):
        if f"{s}_probs" not in arr:
            continue
        p, U = arr[f"{s}_probs"][lab], arr[f"{s}_U"][lab]
        rc = risk_coverage_curve(U, p.argmax(1) == y[lab])
        ax.plot(rc["coverage"], rc["risk"], label=f"{s} (AURC {rc['aurc']:.3f})")
    ax.set_xlabel("Cobertura"); ax.set_ylabel("Riesgo (error)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "riesgo_cobertura.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    for s in ("P2", "P1"):
        p = arr[f"{s}_probs"][lab]
        conf, correct = p.max(1), p.argmax(1) == y[lab]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(conf, bins) - 1, 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [correct[idx == b].mean() for b in range(10) if (idx == b).any()]
        ax.plot(xs, ys, "o-", label=s)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("Confianza"); ax.set_ylabel("Exactitud")
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "calibracion.pdf"); plt.close(fig)

    conds = [c for c in ev["systems"]["P2"]["per_condition"] if c != "C10"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.27
    for i, s in enumerate(("B6", "P1", "P2")):
        d = [ev["systems"][s]["per_condition"][c]["delta_M"] for c in conds]
        ax.bar(np.arange(len(conds)) + (i - 1) * width, d, width, label=s)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_ylabel(r"$\Delta M_s$ (exactitud)"); ax.legend(); fig.tight_layout()
    fig.savefig(FIGS / "delta_m_condicion.pdf"); plt.close(fig)


def main() -> int:
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    ev["aliases"] = {"A1": "P1", "A3": "B6", **({"A2": "P2"} if ev["hypotheses"]["H3"].get("status") == "NOT_EVALUATED" else {})}
    TABLES.mkdir(parents=True, exist_ok=True)
    main_rows = ["B0", "B1", "B2", "B3", "--", "B4", "B5", "B7", "B8", "--", "B6", "P1", "P2"]
    (TABLES / "tab_main.tex").write_text(table(
        main_rows, ev, r"Resultados principales en test (escala reducida). Media $\pm$ desviación entre 5 semillas "
        r"(por semilla, cada una calibrada con su $T^\ast$) y, debajo, IC 95\% del bootstrap agrupado por "
        r"\texttt{source\_image\_id} (2\,000 réplicas) sobre el ensamble $M=5$ calibrado.", "tab:plantilla_main"))
    (TABLES / "tab_ablaciones.tex").write_text(table(
        ["P2", "--", "A1", "A2", "A3", "A4", "A5", "--", "P2_legacy"], ev,
        "Ablaciones A1--A5 y taxonomía previa, mismo protocolo que la tabla principal.", "tab:ablaciones", shaded=("P2",)))
    (TABLES / "tab_oraculo.tex").write_text(table(
        ["P2", "P2_oracleN", "P2_oracleVF", "P2_oracleNVF"], ev,
        "Protocolo oráculo (diagnóstico): percepción predicha u oráculo para la narrativa y para visual+forense.",
        "tab:oraculo", shaded=("P2",)))
    (TABLES / "tab_sensibilidad.tex").write_text(table(
        ["B6", "P1", "P2"], ev, "Análisis de sensibilidad: test sin los casos AMBIGUOS (criterios C2--C5).",
        "tab:sensibilidad", key="test_without_ambiguous"))
    (TABLES / "tab_delta_m.tex").write_text(condition_table(ev))
    (TABLES / "tab_hipotesis.tex").write_text(hypotheses_table(ev))
    figures(ev)
    print("tables:", sorted(p.name for p in TABLES.iterdir()), "figures:", sorted(p.name for p in FIGS.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
