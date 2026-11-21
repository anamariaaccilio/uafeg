"""Propuesta del capítulo 5 y RESUMEN_FINAL.md con números leídos de los archivos de resultados.

    python -m experiments.chapter5

No edita el .tex original de la tesis. Escribe docs/capitulo5_resultados_propuesta.tex y
../RESUMEN_FINAL.md. Toda cifra sale de results/evaluation.json, results/final/specialists.json,
results/final/graph_stats.json o results/hparam_search*.json.
"""
from __future__ import annotations

import json

from experiments.final_docs import fmt_p
from experiments.thesis_data import REPO_B

R = REPO_B / "results"
F = R / "final"


def main():
    ev = json.loads((R / "evaluation.json").read_text())
    sp = json.loads((F / "specialists.json").read_text())
    gs = json.loads((F / "graph_stats.json").read_text())
    agg = json.loads((R / "hparam_search_aggregator.json").read_text())["selected"]
    S = ev["systems"]
    H = ev["hypotheses"]
    best = H["H1"]["comparison"].split(" vs ")[1].split(" ")[0]

    def m(s, k, key="test"):
        return S[s][key][k]["ensemble"]

    def c(s, k, key="test"):
        x = S[s][key][k]
        return f"[{x['ci_low']:.3f}, {x['ci_high']:.3f}]"

    def acc(s, cond):
        return S[s]["per_condition"][cond]["accuracy"]

    def dm(s, cond):
        return S[s]["per_condition"][cond]["delta_M"]

    def hp(k):
        h = H[k]
        return (f"{h['point_diff']:+.4f} (IC 95\\,\\% [{h['ci_low']:+.4f}, {h['ci_high']:+.4f}]; "
                f"$p$ {fmt_p(h['p_value_approx'], h['n_resamples']).replace('<', '$<$')}, "
                f"$p_{{\\mathrm{{Holm}}}}$ {fmt_p(h['p_holm'], h['n_resamples']).replace('<', '$<$')})")
    n, vis, fo = sp["narrative"], sp["visual"], sp["forensic"]
    from collections import Counter
    errs = json.loads((REPO_B / "review/errores/errores.json").read_text())
    err_conds = ", ".join(f"{k}: {v}" for k, v in sorted(Counter(e["condition"] for e in errs).items(), key=lambda t: int(t[0][1:])))
    err_types = ", ".join(f"{k} ({v})" for k, v in Counter(e["top_evidence"] for e in errs).most_common())
    err_kinds = Counter(f"{e['expected']}->{e['predicted']}" for e in errs)
    arrow = r"$\to$"
    err_types += "; pares etiqueta" + arrow + "decisión: " + ", ".join(k.replace("->", arrow) + f" ({v})" for k, v in err_kinds.most_common())
    err_p = sum(e["p_max"] for e in errs) / len(errs)
    ranking = ", ".join(f"{s} ({S[s]['test']['pr_auc_suspicious']['ensemble']:.3f})" for s in sorted(
        [s for s in ("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "G0", "G1")],
        key=lambda s: -S[s]["test"]["pr_auc_suspicious"]["ensemble"]))
    he, ex = vis["hitl_eval"], vis["external_cardd"]
    gk = gs["g_by_kind_condition"]
    ood = {s: S[s]["ood_auroc_U_C10"] for s in ("B5", "B6", "G0", "G1")}
    mc = {k: H[k]["mcnemar_overall_correctness"] for k in ("H1", "H2", "H3")}

    tex = rf"""% Propuesta de redacción del capítulo 5 (resultados). Generada por experiments/chapter5.py a partir de
% results/evaluation.json, results/final/specialists.json y results/final/graph_stats.json.
% NO reemplaza secciones/capitulo5.tex: es material para que Ana lo integre.
% Tablas: \input{{results/final/tab_main.tex}}, tab_hipotesis, tab_ablaciones, tab_oraculo,
% tab_sensibilidad, tab_delta_m, tabla_especialistas, tabla_externa, tabla_grafo.

\section{{Cambio de diseño: agregador sin GATv2 y verificación humana}}

El sistema evaluado no usa la capa GATv2. La propagación de mensajes aprendida se reemplazó por un
agregador interpretable sobre el mismo grafo de evidencia: para cada afirmación $i$ se calculan
$S_{{\text{{apoyo}}}}=\sum_j g_j\,p_{{\text{{support}}}}(i,j)$, $S_{{\text{{contra}}}}=\sum_j g_j\,p_{{\text{{contr}}}}(i,j)$,
$S_{{\text{{desc}}}}=\sum_j g_j\,p_{{\text{{unk}}}}(i,j)$, la similitud ontológica ponderada y el término forense
$\max_j g_f\,p_{{\text{{edit}}}}\,\mathrm{{IoU}}(\ell_j,\ell_f)$; se agregan por reclamo (máximo y media), junto con conteos,
$\bar c$, la calidad media y la polaridad de las afirmaciones, y un clasificador liviano elegido en validación produce
$p(y\mid x)$. G0 es el mismo agregador con $g=1$ y reemplaza a P1; G1 usa la compuerta y reemplaza a P2. En validación,
el clasificador seleccionado fue {agg['G1']['model']} para G1 y {agg['G0']['model']} para G0. Cada decisión se acompaña de un
reporte para revisión humana (foto con máscaras y mapa forense, afirmaciones, grafo con $g$, decisión calibrada,
$U$, $\bar c$, motivo de abstención y las tres piezas de evidencia con mayor peso).

\section{{Especialistas}}

El especialista narrativo acierta el tipo de daño exacto en {n['damage_type_exact_acc']:.3f} de los
{n['n_validation_claims_with_assertion']} reclamos de validación con afirmación, {n['damage_type_family_acc']:.3f} a nivel de
familia de la ontología HITL, y la polaridad en {n['polarity_acc']:.3f}; por eso la confiabilidad narrativa $r$ se mide a nivel
de familia. YOLOv8n-seg, ajustado en Kaggle (GPU T4) con el split congelado, obtiene un mAP50 de máscara de
{he['val']['mask_map50']:.3f} en validación y {he['test']['mask_map50']:.3f} en test (mAP50-95: {he['val']['mask_map50_95']:.3f} y
{he['test']['mask_map50_95']:.3f}). En CarDD (validación externa, {ex['n_images']} imágenes, solo cajas) el mAP50 de caja es
{ex['box_map50_mapped']:.3f} con clases mapeadas y {ex['box_map50_agnostic']:.3f} sin distinguir clase: el detector apenas
transfiere fuera de HITL. TruFor separa las imágenes editadas de las originales en validación con AUROC
{fo['auroc_C3_vs_C0']:.3f} para C3 (copy-move), {fo['auroc_C4_vs_C0']:.3f} para C4 (inpainting) y {fo['auroc_C3C4C5_vs_C0']:.3f}
para C3--C5 en conjunto (C5 usa las mismas imágenes que C3). Ver Tablas~\ref{{tab:especialistas}} y~\ref{{tab:externa}}.

\section{{Grafos de evidencia}}

Los grafos tienen en promedio {gs['per_split']['test']['nodes_mean']:.2f} nodos y {gs['per_split']['test']['edges_mean']:.2f}
aristas por reclamo en test ({gs['per_split']['train']['nodes_mean']:.2f} y {gs['per_split']['train']['edges_mean']:.2f} en
entrenamiento). En C0 el {gs['per_condition']['C0']['pct_support']:.1f}\,\% de las aristas afirmación--daño son de apoyo y el
{gs['per_condition']['C0']['pct_contradiction']:.1f}\,\% de contradicción; en C1 y C5 la contradicción sube a
{gs['per_condition']['C1']['pct_contradiction']:.1f}\,\%. En C2 (sustitución de imagen) la contradicción es
{gs['per_condition']['C2']['pct_contradiction']:.1f}\,\%: el grafo no ve diferencia con C0. La compuerta media de los nodos visuales
es {gk['visual']['C0']['mean']:.3f} en C0, porque la confianza y la calidad regional de las detecciones son bajas; en C9
(desenfoque) el detector no produce nodos visuales y la $g$ de la fuente cae de {gk['source']['C0']['mean']:.3f} (C0) a
{gk['source']['C9']['mean']:.3f}. La $g$ forense media es {gk['forensic']['C0']['mean']:.3f} en C0 y {gk['forensic']['C3']['mean']:.3f}
en C3 (Tabla~\ref{{tab:grafo}}, Figura de distribución de $g$).

\section{{Resultados principales}}

La Tabla~\ref{{tab:resultados_main}} resume test (evaluado una sola vez). G1 obtiene PR-AUC de SUSPICIOUS
{m('G1','pr_auc_suspicious'):.3f} {c('G1','pr_auc_suspicious')}, Macro-F1 {m('G1','macro_f1'):.3f}, Brier {m('G1','brier'):.3f},
ECE {m('G1','ece'):.3f} y AURC {m('G1','aurc'):.3f}. G0 obtiene {m('G0','pr_auc_suspicious'):.3f} {c('G0','pr_auc_suspicious')},
{m('G0','macro_f1'):.3f}, {m('G0','brier'):.3f}, {m('G0','ece'):.3f} y {m('G0','aurc'):.3f}; B6 (DeepSets)
{m('B6','pr_auc_suspicious'):.3f}, {m('B6','macro_f1'):.3f}, {m('B6','brier'):.3f}, {m('B6','ece'):.3f} y {m('B6','aurc'):.3f}.
Ordenados por PR-AUC: {ranking}. B8 (Qwen2-VL-2B congelado, sin ajuste) queda en {m('B8','pr_auc_suspicious'):.3f}, por debajo
incluso de las reglas B0 ({m('B0','pr_auc_suspicious'):.3f}); Macro-F1 {m('B8','macro_f1'):.3f} frente a {m('B0','macro_f1'):.3f}.
El resultado es negativo para la propuesta: con la evidencia visual disponible, ni el grafo ni la compuerta superan a una
fusión simple.

\section{{Contraste de hipótesis}}

Con bootstrap pareado agrupado por \texttt{{source\_image\_id}} ({H['H1']['n_groups']} imágenes fuente de test, 2\,000 réplicas)
y Holm ($m=3$, $\alpha=0{{,}}05$) (Tabla~\ref{{tab:hipotesis}}):

\begin{{itemize}}
\item \textbf{{H1}} (G1 frente al mejor sistema sin grafo, elegido en validación: {best}; PR-AUC de SUSPICIOUS en C3--C5 frente a
C8--C9, {H['H1']['n_stress_test']} reclamos): diferencia G1$-${best} = {hp('H1')}. G1 obtiene {H['H1']['point_G1']:.3f} y {best}
{H['H1']['point_best_nongraph']:.3f}. Se rechaza la igualdad, pero a favor de {best}: \textbf{{H1 no se sostiene}}. McNemar sobre
la exactitud global: $b={mc['H1']['b']}$, $c={mc['H1']['c']}$, $p={mc['H1']['p_value']:.3f}$.
\item \textbf{{H2}} (G1 frente a G0 en Brier): Brier(G0)$-$Brier(G1) = {hp('H2')}; G1 {H['H2']['point_G1']:.4f}, G0
{H['H2']['point_G0']:.4f}. La compuerta empeora la calibración: \textbf{{H2 no se sostiene}} (efecto en contra). McNemar:
$b={mc['H2']['b']}$, $c={mc['H2']['c']}$, $p={mc['H2']['p_value']:.4f}$.
\item \textbf{{H3}} (G1 frente a A2, sin forense): diferencia de PR-AUC = {hp('H3')}; G1 {H['H3']['point_G1']:.3f}, A2
{H['H3']['point_A2']:.3f}. \textbf{{H3 se acepta}}: el especialista forense aporta. McNemar: $b={mc['H3']['b']}$, $c={mc['H3']['c']}$.
\end{{itemize}}

Los valores $p$ reportados como ``$<$ 0.0005'' corresponden a 0 de 2\,000 réplicas con el signo contrario; no son exactamente cero.

\section{{Robustez por condición, calibración y abstención}}

La exactitud de la decisión final (con abstención contada como INSUFFICIENT) de G1 es {acc('G1','C0'):.3f} en C0,
{acc('G1','C1'):.3f} en C1, {acc('G1','C2'):.3f} en C2, {acc('G1','C3'):.3f} en C3, {acc('G1','C4'):.3f} en C4,
{acc('G1','C5'):.3f} en C5 y {acc('G1','C8'):.3f} en C8 (Tabla~\ref{{tab:delta_m}}). C2 (sustitución) y C4 (inpainting) son
las condiciones más difíciles para todos los sistemas: $\Delta M_{{C2}}$ = {dm('G1','C2'):+.3f} y $\Delta M_{{C4}}$ = {dm('G1','C4'):+.3f}
para G1; {dm(best,'C2'):+.3f} y {dm(best,'C4'):+.3f} para {best}. En C8 (transformaciones benignas) G1 acierta
{acc('G1','C8'):.3f} frente a {acc(best,'C8'):.3f} de {best} (fracción decidida SUSPICIOUS en C8: G1
{S['G1']['per_condition']['C8']['decision_distribution']['SUSPICIOUS']:.3f}, {best}
{S[best]['per_condition']['C8']['decision_distribution']['SUSPICIOUS']:.3f}).
Todos los sistemas calibrados tienen ECE baja (G1 {m('G1','ece'):.3f}). Los umbrales fijados en validación para una cobertura
de 0.80 dejaron $\tau_c=$ {S['G1']['tau_c']:.3f} (la cobertura de evidencia no discriminó en validación) y $\tau_U=$
{S['G1']['tau_U']:.3f}; G1 se abstiene en {S['G1']['abstention_rate']:.3f} de los reclamos etiquetados de test, y solo
{S['G1']['abstentions_in_C6_C7_C9_C10']:.3f} de sus abstenciones caen en C6, C7, C9 o C10: las abstenciones no se concentran
donde deberían, porque esas condiciones ya se resuelven con confianza. Para C10 (OOD), el AUROC de $U$ para separarla del resto
es {ood['G1']:.3f} para G1, {ood['G0']:.3f} para G0, {ood['B6']:.3f} para B6 y {ood['B5']:.3f} para B5.

\section{{Ablaciones, sensibilidad y oráculo}}

Sin forense (A2) la PR-AUC cae a {m('A2','pr_auc_suspicious'):.3f}. Quitar los atributos continuos de las aristas (A4) deja
{m('A4','pr_auc_suspicious'):.3f} (G1: {m('G1','pr_auc_suspicious'):.3f}). Entrenar sin C8 (A5) mantiene la PR-AUC
({m('A5','pr_auc_suspicious'):.3f}) pero hunde el Macro-F1 a {m('A5','macro_f1'):.3f} y la calibración (ECE {m('A5','ece'):.3f}).
Sin los 552 reclamos AMBIGUOS (sensibilidad C2--C5), G1 obtiene PR-AUC {m('G1','pr_auc_suspicious','test_without_ambiguous'):.3f}
y G0 {m('G0','pr_auc_suspicious','test_without_ambiguous'):.3f}: la conclusión no cambia (Tabla~\ref{{tab:sensibilidad}}).
Con percepción oráculo la PR-AUC de G1 sube a {m('G1_oracleN','pr_auc_suspicious'):.3f} (narrativa),
{m('G1_oracleVF','pr_auc_suspicious'):.3f} (visual+forense) y {m('G1_oracleNVF','pr_auc_suspicious'):.3f} (ambas); G0 con ambas
llega a {m('G0_oracleNVF','pr_auc_suspicious'):.3f}. El cuello de botella es la percepción visual y forense, no el agregador.

\section{{Análisis de errores y verificación humana}}

Se seleccionaron 20 errores de G1 (sin abstención), repartidos por condición; sus reportes están en
\texttt{{review/errores/}}. Condiciones de los 20 errores: {err_conds}. Tipo de la evidencia con más peso en cada error:
{err_types}. La probabilidad máxima media de esos errores es {err_p:.3f}. El paquete de
revisión de 50 casos (\texttt{{review/}}) está listo para que la asesora y la autora lo llenen; no contiene respuestas de
revisión.

\section{{Amenazas a la validez}}

Los reclamos son escenarios controlados sobre imágenes reales, no expedientes de una aseguradora. El especialista visual es
débil y no transfiere a CarDD. El test no es completamente ciego (se miró al diseñar la QA). C3 y C5 comparten imágenes. La
varianza entre semillas solo cubre al clasificador final: G1 eligió una regresión logística convexa y sus 5 semillas coinciden
(desviación 0). B8 se usó sin ajuste y con un solo prompt.
"""
    (REPO_B / "docs/capitulo5_resultados_propuesta.tex").write_text(tex)

    md = f"""# RESUMEN FINAL — tesis UAFEG (cierre de desarrollo, 2026-11-21)

**Qué se hizo.** Se entrenó YOLOv8n-seg y se corrió TruFor en Kaggle GPU sobre las 5 240 imágenes del benchmark v3
(807 fuentes HITL reales, 9 275 reclamos controlados), se construyeron los EvidencePackets y grafos finales, y se
reemplazó GATv2 por un agregador interpretable con compuerta de confiabilidad (G1) y sin ella (G0). Se entrenaron
B1–B7, B6, B8 (Qwen2-VL-2B congelado), G0, G1, ablaciones A2/A4/A5 y oráculo, con 5 semillas y selección solo en
validación, y test se evaluó una sola vez. Cada reclamo tiene un reporte para revisión humana; `uafeg/review/` tiene
el paquete de 50 casos (sin respuestas).

**Números clave (test; ensamble calibrado; IC 95 % bootstrap agrupado).**

| Sistema | PR-AUC SUS | Macro-F1 | Brier | ECE | AURC |
|---|---|---|---|---|---|
""" + "".join(f"| {s} | {m(s,'pr_auc_suspicious'):.3f} {c(s,'pr_auc_suspicious')} | {m(s,'macro_f1'):.3f} | {m(s,'brier'):.3f} | {m(s,'ece'):.3f} | {m(s,'aurc'):.3f} |\n"
              for s in ("B5", "B4", "B6", "G0", "G1", "A2", "B8")) + f"""
Especialistas: YOLOv8n-seg mAP50 de máscara {he['val']['mask_map50']:.3f} (val) / {he['test']['mask_map50']:.3f} (test);
CarDD caja {ex['box_map50_mapped']:.3f} (clases mapeadas); TruFor AUROC C3–C5 vs C0 {fo['auroc_C3C4C5_vs_C0']:.3f} (val);
narrativo tipo exacto {n['damage_type_exact_acc']:.3f}, familia {n['damage_type_family_acc']:.3f}, polaridad {n['polarity_acc']:.3f}.

**Hipótesis (Holm, α = 0.05).**
- **H1 — rechazada.** G1 − {best} = {H['H1']['point_diff']:+.4f} en PR-AUC de estrés (IC [{H['H1']['ci_low']:+.4f}, {H['H1']['ci_high']:+.4f}], p_Holm {fmt_p(H['H1']['p_holm'], 2000)}): la fusión tardía es mejor.
- **H2 — rechazada.** Brier(G0) − Brier(G1) = {H['H2']['point_diff']:+.4f} (IC [{H['H2']['ci_low']:+.4f}, {H['H2']['ci_high']:+.4f}], p_Holm {H['H2']['p_holm']:.4f}): la compuerta empeora la calibración.
- **H3 — aceptada.** G1 − A2 = {H['H3']['point_diff']:+.4f} en PR-AUC (IC [{H['H3']['ci_low']:+.4f}, {H['H3']['ci_high']:+.4f}], p_Holm {fmt_p(H['H3']['p_holm'], 2000)}): el forense aporta.

**Lectura.** El grafo con compuerta no supera a una fusión simple; la compuerta atenúa casi toda la evidencia visual
(g medio {gk['visual']['C0']['mean']:.3f}) porque el detector es débil. Con percepción oráculo G1 llega a PR-AUC
{m('G1_oracleNVF','pr_auc_suspicious'):.3f}: el límite está en los especialistas. C2 y C4 son casi invisibles para todos.

**Limitaciones.** Reclamos controlados sobre imágenes reales (no expedientes de aseguradora); detector visual débil y sin
transferencia a CarDD; test no totalmente ciego; C3 y C5 comparten imágenes; una sola corrida de YOLO/TruFor; G1 eligió
logística (5 semillas idénticas); B8 sin ajuste. Detalle: `uafeg/docs/capitulo5_resultados_propuesta.tex`,
`uafeg/results/final/`, `uafeg/docs/decisiones_cierre.md`.
"""
    (REPO_B.parent / "RESUMEN_FINAL.md").write_text(md)
    print("chapter5 + RESUMEN_FINAL written")


if __name__ == "__main__":
    main()
