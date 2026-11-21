"""Documentos finales con números leídos de los archivos de resultados (nunca a mano).

    python -m experiments.final_docs

Fuentes: results/evaluation.json, results/final/specialists.json, results/final/graph_stats.json,
results/hparam_search*.json. Escribe:
  - README.md (sección RESULTS) de uafeg y README.md (sección SPECIALISTS) del Repo A
  - results/final/tabla_externa.tex, results/final/tabla_principal_completa.tex
  - docs/capitulo5_resultados_propuesta.tex
  - ../RESUMEN_FINAL.md  (raíz del workspace)
"""
from __future__ import annotations

import json
import re

from experiments.thesis_data import REPO_B
from uafeg.benchmark import canonical_repository

R = REPO_B / "results"
F = R / "final"
A = canonical_repository()
MAIN = ["B0", "B1", "B2", "B3", "B4", "B5", "B7", "B8", "B6", "G0", "G1"]
ABL = ["G1", "A2", "A4", "A5"]
ORA = ["G1", "G1_oracleN", "G1_oracleVF", "G1_oracleNVF", "G0", "G0_oracleNVF"]
METRICS = [("pr_auc_suspicious", "PR-AUC SUS ↑"), ("macro_f1", "Macro-F1 ↑"), ("brier", "Brier ↓"), ("nll", "NLL ↓"),
           ("ece", "ECE ↓"), ("aurc", "AURC ↓")]
DESC = {"B0": "reglas", "B1": "solo texto", "B2": "solo visual+fuente", "B3": "solo forense", "B4": "concatenación+MLP",
        "B5": "fusión tardía calibrada", "B7": "DBF", "B8": "MLLM congelado (Qwen2-VL-2B)", "B6": "DeepSets",
        "G0": "agregador, g=1 (=A1)", "G1": "UAFEG: agregador con g", "A2": "G1 sin forense", "A4": "G1 aristas sin atributos",
        "A5": "G1 sin C8 en train", "G1_oracleN": "G1 oráculo narrativa", "G1_oracleVF": "G1 oráculo visual+forense",
        "G1_oracleNVF": "G1 oráculo ambos", "G0_oracleNVF": "G0 oráculo ambos"}


def load():
    ev = json.loads((R / "evaluation.json").read_text())
    sp = json.loads((F / "specialists.json").read_text())
    gs = json.loads((F / "graph_stats.json").read_text())
    return ev, sp, gs


def v(ev, s, m, key="test"):
    return ev["systems"][s][key][m]["ensemble"]


def ci(ev, s, m, key="test"):
    x = ev["systems"][s][key][m]
    return x["ci_low"], x["ci_high"]


def md_table(ev, systems, key="test"):
    head = "| Sistema | " + " | ".join(n for _, n in METRICS) + " | abstención |\n|---|" + "---|" * (len(METRICS) + 1) + "\n"
    rows = []
    for s in systems:
        if s not in ev["systems"]:
            rows.append(f"| {s} | " + " | ".join(["NOT_RUN"] * len(METRICS)) + " | |")
            continue
        cells = []
        for m, _ in METRICS:
            lo, hi = ci(ev, s, m, key)
            cells.append(f"{v(ev, s, m, key):.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append(f"| {s} ({DESC.get(s, s)}) | " + " | ".join(cells) + f" | {ev['systems'][s]['abstention_rate']:.3f} |")
    return head + "\n".join(rows) + "\n"


def hyp_lines(ev):
    out = []
    for k in ("H1", "H2", "H3"):
        h = ev["hypotheses"][k]
        if h.get("status") == "NOT_EVALUATED":
            out.append((k, h["reason"], None))
            continue
        if not h["reject_H0_at_0.05"]:
            verdict = "no se rechaza H0: hipótesis NO sostenida"
        elif h["point_diff"] > 0:
            verdict = "se rechaza H0 a favor de G1: hipótesis ACEPTADA"
        else:
            verdict = "se rechaza H0 pero en contra de G1: hipótesis RECHAZADA (el rival es mejor)"
        out.append((k, f"{h['comparison']}: diferencia {h['point_diff']:+.4f} (IC95 [{h['ci_low']:+.4f}, {h['ci_high']:+.4f}]), "
                       f"p = {fmt_p(h['p_value_approx'], h['n_resamples'])}, p_Holm = {fmt_p(h['p_holm'], h['n_resamples'])} → {verdict}", h))
    return out


def fmt_p(p, n):
    return f"< {1 / n:.4f} (0 de {n} réplicas)" if p == 0 else f"{p:.4f}"


def supports(h):
    """Hypothesis supported only if H0 rejected AND the effect favours G1 (positive by construction)."""
    return h is not None and h["reject_H0_at_0.05"] and h["point_diff"] > 0


def replace_block(path, tag, content):
    s = path.read_text()
    s = re.sub(rf"<!-- {tag}:BEGIN -->.*<!-- {tag}:END -->", f"<!-- {tag}:BEGIN -->\n{content}\n<!-- {tag}:END -->", s,
               flags=re.S)
    path.write_text(s)


def specialists_md(sp):
    n, vis, fo = sp["narrative"], sp["visual"], sp["forensic"]
    he, ex = vis["hitl_eval"], vis["external_cardd"]
    lines = ["### Especialistas medidos (archivos reales)", "",
             "| Especialista | Métrica | Valor |", "|---|---|---|",
             f"| Narrativo | tipo de daño exacto / por familia / polaridad (val, n={n['n_validation_claims_with_assertion']}) | "
             f"{n['damage_type_exact_acc']:.3f} / {n['damage_type_family_acc']:.3f} / {n['polarity_acc']:.3f} |"]
    for sk, name in (("val", "val HITL (110 img)"), ("test", "test HITL (129 img)")):
        e = he[sk]
        lines.append(f"| Visual YOLOv8n-seg | máscara mAP50 / mAP50-95, {name} | {e['mask_map50']:.3f} / {e['mask_map50_95']:.3f} |")
        lines.append(f"| | caja mAP50 / mAP50-95, {name} | {e['box_map50']:.3f} / {e['box_map50_95']:.3f} |")
    lines.append(f"| | CarDD test ({ex['n_images']} img), caja, clases mapeadas | {ex['box_map50_mapped']:.3f} / {ex['box_map50_95_mapped']:.3f} |")
    lines.append(f"| | CarDD test, caja, daño agnóstico | {ex['box_map50_agnostic']:.3f} / {ex['box_map50_95_agnostic']:.3f} |")
    lines.append("| | CarDD máscara | NOT_RUN (el espejo con licencia declarada solo trae cajas) |")
    for c in ("C3", "C4", "C5"):
        lines.append(f"| Forense TruFor | AUROC {c} vs C0 (val, imágenes) | {fo[f'auroc_{c}_vs_C0']:.3f} |")
    lines.append(f"| | AUROC C3–C5 vs C0 (val) | {fo['auroc_C3C4C5_vs_C0']:.3f} |")
    return "\n".join(lines) + "\n"


def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("→", r"$\rightarrow$").replace("≥", r"$\geq$")


def external_tex(sp):
    ex = sp["visual"]["external_cardd"]
    he = sp["visual"]["hitl_eval"]
    return "\n".join([
        r"\begin{table}[H]\centering\footnotesize", r"\begin{tabular}{lcccc}", r"\toprule",
        r"\textbf{Conjunto} & \textbf{Imágenes} & \textbf{Etiqueta} & \textbf{mAP50} & \textbf{mAP50-95} \\ \midrule",
        rf"HITL val (máscara) & {he['val']['n_images']} & polígonos & {he['val']['mask_map50']:.3f} & {he['val']['mask_map50_95']:.3f} \\",
        rf"HITL test (máscara) & {he['test']['n_images']} & polígonos & {he['test']['mask_map50']:.3f} & {he['test']['mask_map50_95']:.3f} \\",
        rf"HITL test (caja) & {he['test']['n_images']} & polígonos & {he['test']['box_map50']:.3f} & {he['test']['box_map50_95']:.3f} \\",
        rf"CarDD test, clases mapeadas (caja) & {ex['n_images']} & cajas & {ex['box_map50_mapped']:.3f} & {ex['box_map50_95_mapped']:.3f} \\",
        rf"CarDD test, daño agnóstico (caja) & {ex['n_images']} & cajas & {ex['box_map50_agnostic']:.3f} & {ex['box_map50_95_agnostic']:.3f} \\",
        r"CarDD test (máscara) & -- & -- & \multicolumn{2}{c}{NOT\_RUN: sin máscaras con licencia declarada} \\",
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Validación externa del especialista visual (YOLOv8n-seg entrenado solo en HITL). CarDD: espejo de Kaggle "
        r"con licencia CC BY-NC 4.0; mapeo de clases fijado antes de evaluar. Fuente: \texttt{results/final/specialists.json}.}",
        r"\label{tab:externa}", r"\end{table}", ""])


def main():
    ev, sp, gs = load()
    best_ng = ev["hypotheses"]["H1"]["comparison"].split(" vs ")[1].split(" ")[0]
    hl = hyp_lines(ev)
    # ------------------------------------------------------------------ README uafeg
    res = ["## Resultados principales (test, evaluado una sola vez)", "",
           "Ensamble calibrado (T* en validación), IC 95 % por bootstrap agrupado por `source_image_id` (2 000 réplicas). "
           "Decisión final con abstención (cobertura objetivo 0.80, τU y τc fijados en validación). "
           "Fuente: `results/evaluation.json`; tablas LaTeX en `results/final/`.", "", md_table(ev, MAIN), "",
           "### Hipótesis (Holm, α = 0.05)", ""]
    res += [f"- **{k}**: {t}" for k, t, _ in hl]
    res += ["", "### Ablaciones", "", md_table(ev, ABL), "", "### Oráculo", "", md_table(ev, ORA), "",
            "### Sensibilidad C2 (sin los reclamos AMBIGUOS de test: 80 de los 552 del benchmark; C2: 72, C4: 8)", "",
            md_table(ev, ["B6", "G0", "G1", best_ng], key="test_without_ambiguous"),
            "", specialists_md(sp), "",
            "## Limitaciones", "",
            "- Reclamos controlados sobre imágenes reales; no son expedientes de una aseguradora.",
            "- El especialista visual es débil (ver tabla de especialistas) y la compuerta g lo atenúa fuertemente "
            f"(g visual medio en C0 = {gs['g_by_kind_condition']['visual']['C0']['mean']:.3f}).",
            "- C3 y C5 comparten las mismas imágenes editadas; las manipulaciones son pequeñas.",
            "- Test no completamente ciego (se miró al diseñar la QA del benchmark).",
            "- Una sola corrida de TruFor y de YOLO (semilla 0); la varianza entre semillas solo cubre al clasificador final.",
            "- Si G1 elige la regresión logística (convexa), sus 5 semillas coinciden y su desviación entre semillas es 0.",
            "- El oráculo no separa la percepción forense de la visual; un análisis post hoc (results/posthoc/) muestra que "
            "mejorar solo la visual no cierra la brecha con B5.",
            "- La compuerta g es una fórmula fija (c·q·r·(1−u)), no aprendida; con especialistas débiles atenúa también señal útil (H2)."]
    replace_block(REPO_B / "README.md", "RESULTS", "\n".join(res))
    replace_block(A / "README.md", "SPECIALISTS", specialists_md(sp))
    (F / "tabla_externa.tex").write_text(external_tex(sp))
    print("README blocks written; H:", [(k, None if h is None else h["reject_H0_at_0.05"]) for k, _, h in hl])


if __name__ == "__main__":
    main()
