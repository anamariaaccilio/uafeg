"""Reporte PRELIMINAR en validación (especialistas incompletos). No lee test.

    python -m experiments.thesis_val_report

Usa artifacts/thesis/runs_validation_prelim/*/seed*/logits.npz (solo contienen 'val').
Métricas SIN calibrar (T* se ajusta en validación; calibrar y medir sobre el mismo
conjunto sería circular): PR-AUC SUSPICIOUS, Brier, NLL y AURC por U = H(p). Media ±
desviación entre 5 semillas y valor del ensamble M=5. Todo se marca
"validación, especialistas incompletos".
"""
from __future__ import annotations

import json

import numpy as np

from experiments.thesis_data import ART, REPO_B, load_split
from src.abstention.abstention import risk_coverage_curve
from src.evaluation.metrics import CLASS_ORDER, brier_score_multiclass, nll, pr_auc_suspicious

OUT = REPO_B / "results/validation_prelim"
RUNS = ART / "runs_validation_prelim"
ORDER = ["B1", "B2", "B3", "B4", "B5", "B7", "B6", "P1", "P2", "A1", "A3", "A4", "A5"]
ALIASES = {"A1": "P1", "A3": "B6"}
WARNING = "validación, especialistas incompletos (narrativo + calidad; visual y forense = null)"


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def metrics(y, p):
    lab = [CLASS_ORDER[i] for i in y]
    U = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
    return {"pr_auc_suspicious": pr_auc_suspicious(lab, p), "brier": brier_score_multiclass(lab, p),
            "nll": nll(lab, p), "aurc": risk_coverage_curve(U, p.argmax(1) == y)["aurc"]}


def main() -> int:
    val = load_split("validation")
    keep = val.labels >= 0
    y = val.labels[keep]
    probs = {}
    for s in ORDER:
        src = ALIASES.get(s, s)
        files = sorted((RUNS / src).glob("seed*/logits.npz"))
        if files:
            probs[s] = [softmax(np.load(f)["val"][keep]) for f in files]
    if "B1" in probs and "B2" in probs:  # B5: late fusion of the unimodal members, equal weights, seed-paired
        probs["B5"] = [softmax(np.log(a) + np.log(b)) for a, b in zip(probs["B1"], probs["B2"])]
    rows = {}
    for s in ORDER:
        if s not in probs:
            continue
        per = [metrics(y, p) for p in probs[s]]
        ens = metrics(y, np.mean(probs[s], 0))
        rows[s] = {k: {"mean": float(np.mean([m[k] for m in per])), "sd": float(np.std([m[k] for m in per], ddof=1)),
                       "ensemble": ens[k]} for k in ens}
        rows[s]["n_seeds"] = len(per)
        rows[s]["alias_of"] = ALIASES.get(s)
    not_run = {"B3": "forense = null en estos packets", "B8": "sin MLLM abierto reproducible en CPU 4 núcleos/7.6 GB",
               "A2": "idéntico a P2 mientras forense = null"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validation_prelim.json").write_text(json.dumps(
        {"warning": WARNING, "n_validation": int(keep.sum()), "systems": rows, "not_run": not_run}, indent=1) + "\n")

    md = [f"*{WARNING}; n = {int(keep.sum())} reclamos de validación, 110 fuentes. Sin calibrar. Media ± d.e. en 5 semillas (ensamble M=5 entre paréntesis).*", "",
          "| Sistema | PR-AUC SUSPICIOUS ↑ | Brier ↓ | AURC ↓ |", "|---|---|---|---|"]
    tex = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabular}{lccc}", r"\toprule",
           r"Sistema & PR-AUC SUSP. & Brier & AURC \\", r"\midrule"]
    for s, r in rows.items():
        name = s + (f" (= {r['alias_of']})" if r["alias_of"] else "")
        f = lambda k: f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f} ({r[k]['ensemble']:.3f})"
        md.append(f"| {name} | {f('pr_auc_suspicious')} | {f('brier')} | {f('aurc')} |")
        tex.append(f"{name} & " + " & ".join(f"{r[k]['mean']:.3f} $\\pm$ {r[k]['sd']:.3f}"
                                            for k in ("pr_auc_suspicious", "brier", "aurc")) + r" \\")
    for s, why in not_run.items():
        md.append(f"| {s} | NOT_RUN: {why} | | |")
    tex += [r"\bottomrule", r"\end{tabular}", rf"\caption{{Resultados preliminares: {WARNING}.}}",
            r"\label{tab:val_prelim}", r"\end{table}"]
    (OUT / "tabla_validacion.md").write_text("\n".join(md) + "\n")
    (OUT / "tabla_validacion.tex").write_text("\n".join(tex) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    names = list(rows)
    for ax, k, lbl in ((axes[0], "pr_auc_suspicious", "PR-AUC SUSPICIOUS"), (axes[1], "brier", "Brier")):
        ax.bar(names, [rows[s][k]["mean"] for s in names], yerr=[rows[s][k]["sd"] for s in names], capsize=3,
               color=["#2b6cb0" if s in ("B6", "P1", "P2") else "#a0aec0" for s in names])
        ax.set_title(lbl, fontsize=10); ax.tick_params(axis="x", rotation=60, labelsize=8)
    fig.suptitle(WARNING, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "fig_val_prauc_brier.png", dpi=150); fig.savefig(OUT / "fig_val_prauc_brier.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    for s in [x for x in ("B1", "B4", "B7", "B6", "P1", "P2") if x in probs]:
        p = np.mean(probs[s], 0)
        rc = risk_coverage_curve(-(p * np.log(np.clip(p, 1e-12, 1))).sum(1), p.argmax(1) == y)
        ax.plot(rc["coverage"], rc["risk"], label=f"{s} (AURC {rc['aurc']:.3f})")
    ax.set_xlabel("Cobertura"); ax.set_ylabel("Riesgo (error)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title(WARNING, fontsize=7); fig.tight_layout()
    fig.savefig(OUT / "fig_val_riesgo_cobertura.png", dpi=150); fig.savefig(OUT / "fig_val_riesgo_cobertura.pdf"); plt.close(fig)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
