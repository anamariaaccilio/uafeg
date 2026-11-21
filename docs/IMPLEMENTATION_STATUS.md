# UAFEG (Repo B) — Implementation Status

Última actualización: 2026-10-17 (pasada de CIERRE Y CONSISTENCIA). Fuente: código real +
tests ejecutados + comparación línea por línea contra el `.tex` real de la tesis, localizado en
esta pasada en el `.tex` de la tesis (archivo local de la autora) (leído, no modificado).

Distinción obligatoria entre tres estados independientes: **Implementation** (¿existe código
funcional?), **Tests** (¿hay tests automatizados reales?), **Scientific evaluation** (¿se
entrenó/evaluó sobre el benchmark real?). Nunca confundir `IMPLEMENTED` con `TRAINED` con
`EVALUATED`.

## Estado declarado explícitamente (instrucción de cierre)

> Actualización 2026-11-06 (rama `infra/complete-pipeline`): se implementaron los atributos
> continuos de arista y la capa GATv2 de la tesis; A4 pasa a implementada. La taxonomía de nodos
> sigue siendo **PENDING_ADVISOR_DECISION**. Nada de esto se entrenó ni evaluó.

- **B6 = IMPLEMENTED_CODE_NOT_TRAINED**
- **P1 = IMPLEMENTED_CODE_NOT_TRAINED**
- **P2 = IMPLEMENTED_CODE_NOT_TRAINED**
- **READY_FOR_MODELS = NO** (benchmark v2 con specialists null; v3 `PENDING_SCIENTIFIC_QA`)
- **H1 / H2 / H3 = NOT_EVALUATED**

Toda métrica obtenida por `experiments.train`/`experiments.evaluate` en esta sesión usa
`--data smoke` (fixtures sintéticas) y se marca `result_type: SMOKE_RESULT` en su manifest —
equivalente a **SMOKE_ONLY**: nunca debe leerse como resultado científico (p.ej. el macro-F1≈0.16
obtenido en un smoke run no es evidencia de que el modelo funcione o no funcione).

## Contraste controlado B6 / P1 / P2

Verificado en la pasada de CIERRE (2026-10-17) leyendo el código de los tres modelos línea por
línea para confirmar que no hay diferencias accidentales entre ellos más allá de las dos
variables que se quieren aislar (estructura de grafo, reliability gate).

| PROPERTY | B6 | P1 | P2 |
|---|---|---|---|
| Input | `EvidencePacket` vía `packet_to_modality_features` | idéntico | idéntico |
| Encoder | `EvidenceEncoderBank` (shared_dim=32, hidden_dim=64) | idéntico (misma clase, mismos defaults) | idéntico |
| Graph | ninguno (conjunto plano de embeddings) | `EvidenceGraph` (`build_graph`) | idéntico a P1 |
| Explicit relations | N/A | `evidence_for` + narrativa↔visual (support/contradiction/unknown) | idéntico a P1 |
| Reliability gate | N/A | ninguno (`message_scale` = 1.0 siempre) | `g_j=clip(c·q·r·(1-u),0,1)` (`message_scale` sobreescrito) |
| Pooling | **media** sobre embeddings de modalidades presentes | **media** sobre nodos del grafo tras GATv2 (incluye nodo `claim`) | idéntico a P1 |
| Classifier | `Linear(32,64)→ReLU→Linear(64,3)` | idéntico (misma arquitectura) | idéntico |
| Label space | `CLASS_ORDER` de `src.evaluation.metrics` (SUPPORTED/SUSPICIOUS/INSUFFICIENT) | idéntico | idéntico |
| Split | (ninguno real ejecutado aún; mismo `benchmark.lock.json`/manifest previstos para los tres cuando se entrene) | idéntico | idéntico |
| Training protocol | `experiments.train`/`uafeg.models.training.train` (Adam, CrossEntropyLoss, mismo manejo de seed) | idéntico | idéntico |

**Hallazgo corregido en esta pasada**: B6 usaba pooling por **suma** mientras P1/P2 ya usaban
pooling por **media**. Con suma, la magnitud del vector agrupado de B6 escalaba con la cantidad
de modalidades presentes (1 a 4 por claim), lo cual habría confundido "aporta el grafo" (B6 vs
P1) con "aporta tener más evidencia presente" — una diferencia accidental ajena a la estructura
relacional que la comparación debe aislar. Corregido a media en
`src/uafeg/models/b6_deepsets.py` (un único cambio de línea + docstring); no se reentrenó nada
porque no había entrenamiento real previo que invalidar, solo los smoke runs de la sesión
anterior, que ya estaban marcados `SMOKE_RESULT`.

## Tabla de implementación

| Component | Status | Tests | Scientific evaluation |
|---|---|---|---|
| EvidencePacket → EvidenceGraph | IMPLEMENTED | 24 tests | N/A |
| Relation builder narrativa↔visual | IMPLEMENTED | 3 tests | N/A |
| B6 (DeepSets) | IMPLEMENTED_CODE_NOT_TRAINED | 11 tests | NOT_RUN (solo SMOKE_ONLY) |
| P1 (GATv2 relacional, sin gate) | IMPLEMENTED_CODE_NOT_TRAINED | 7 tests | NOT_RUN (solo SMOKE_ONLY) |
| P2 (P1 + reliability gate) | IMPLEMENTED_CODE_NOT_TRAINED | 21 tests | NOT_RUN (solo SMOKE_ONLY) |
| Ablaciones A1/A2/A3 (no_gate, no_forensic, graph_to_deepsets) | IMPLEMENTED_CODE_NOT_TRAINED | 8 tests | NOT_RUN |
| Ablación A4 (no_edge_attributes) | IMPLEMENTED_CODE_NOT_TRAINED | 2 tests | NOT_RUN |
| Ablación A5 (no_benign_no_ood_training) | NOT_IMPLEMENTED (`NotImplementedError` explícito, no fingido) | — | — |
| Capa GATv2 de la tesis (W_r por relación, atención sobre a_ij, residual + LayerNorm) | IMPLEMENTED_CODE_NOT_TRAINED | 14 tests | NOT_RUN |
| Atributos continuos de arista a_ij + máscara | IMPLEMENTED | 11 tests | N/A |
| Ablaciones extra (no_uncertainty, no_explicit_relations) | IMPLEMENTED, fuera del protocolo de la tesis | incluidos en los 8 tests | NOT_RUN |
| Protocolo oracle (4 combinaciones) | IMPLEMENTED (corregido esta pasada, ver tabla de abajo) | 6 tests | N/A (diagnóstico) |
| Train/eval runner | IMPLEMENTED | validado por 7 ejecuciones reales (`--data smoke`) | SMOKE_ONLY |
| Calibration (temperature scaling) | IMPLEMENTED | 2 tests | NOT_RUN |
| Uncertainty (entropy ensemble) | IMPLEMENTED | 6 tests | NOT_RUN |
| Abstention (τ_U/τ_c) | IMPLEMENTED (corregido esta pasada: τ_c usa evidence_coverage, no confidence) | 6 tests | NOT_RUN |
| Grouped bootstrap + Holm | IMPLEMENTED (Holm añadido esta pasada) | 8 tests | NOT_RUN |
| H1/H2/H3 runners dedicados | NOT_IMPLEMENTED | — | — |

Total de tests: **Repo A 158 passed / 5 skipped. Repo B 103 passed / 0 failed** (76 antes de la pasada del 2026-11-06). `git diff --check`
limpio en ambos.

## Tesis vs. código — tabla de verificación

| ELEMENTO | TESIS | CÓDIGO | MATCH/MISMATCH | ACCIÓN |
|---|---|---|---|---|
| Reliability gate | `g_j=clip(c_j q_j r_j (1-u_j),0,1)` (eq:gate, capitulo2.tex) | Idéntico en `gate.py:reliability_gate` | **MATCH** | ninguna |
| H1 (robustez) | P2 vs **mejor sistema sin grafo** (B0–B8) bajo **condiciones de estrés**, PR-AUC(SUSPICIOUS) | No implementado (runner H1 no existe) | MATCH conceptual, pendiente de implementar | Corregida la descripción en `docs/TRAINING.md` (antes decía "B6 vs P1"); runner queda `NOT_IMPLEMENTED` |
| H2 (calibración) | P2 calibrado vs P1 calibrado, Brier | No implementado | MATCH conceptual, pendiente | sin cambios de código |
| H3 (forensia) | P2 con forense vs P2 sin forense, subconjunto con manipulación | `no_forensic` ablation ya implementada y testeada | MATCH | sin cambios |
| Ablación A1 (no gate) | `g_j=1` para todos los nodos | `no_gate` ≡ ejecutar P1 (probado `torch.allclose` con P2 gate=1) | **MATCH** | metadato `thesis_id: A1` añadido |
| Ablación A2 (sin forense) | quitar nodos forensic | `strip_forensic` | **MATCH** | metadato `thesis_id: A2` añadido |
| Ablación A3 (GATv2→DeepSets) | "mismos EvidencePacket, sin estructura" | no existía mapeo explícito a B6 | MISMATCH (solo de nomenclatura) | **CORREGIDO**: nuevo `graph_to_deepsets` ablation ≡ ejecutar B6, `configs/ablations/graph_to_deepsets.yaml` |
| Ablación A4 (sin atributos de arista continuos) | aristas con `[p_support,p_contradiction,p_unknown,sim_sem,overlap_spatial]` (eq:aristas), A4 quita solo los puntajes | `Edge.attributes` lleva el vector continuo + máscara (`uafeg.relations.EdgeAttributes`), consumido por `ThesisGATv2Layer`; A4 = `use_edge_attributes=False` conserva el tipo y enmascara a_ij | **MATCH** | **CORREGIDO** (pasada 2026-11-06): A4 pasa a `IMPLEMENTED_CODE_NOT_TRAINED` |
| Ablación A5 (sin C8/C10 en train) | filtrar split de entrenamiento por condición | no existe condición por EvidencePacket en el runner | **MISMATCH real** | NO corregido (requiere nueva funcionalidad: propagar condición C0–C10 hasta `experiments.train`); declarado `NOT_IMPLEMENTED` explícito |
| "no_uncertainty" (ablación previa de esta sesión) | no existe en la tesis | implementada | **EXTRA, no es del protocolo** | mantenida, pero `thesis_id: null` explícito en config y en `ABLATIONS` |
| "no_explicit_relations" (ablación previa) | no existe en la tesis (A4 es más suave: conserva tipo, quita solo puntajes) | implementada (quita la arista completa) | **EXTRA, no es del protocolo** | mantenida, `thesis_id: null` explícito |
| Protocolo oracle | 4 combinaciones: narrativa × (**visual+forense togglados JUNTOS**) | versión previa togglaba narrativa × visual **independiente**, nunca tocaba forensic | **MISMATCH real** | **CORREGIDO**: `build_oracle_packet(oracle_narrative, oracle_visual_forensic)` + nueva `oracle_forensic_evidence()` (deriva de `claim.transformation`) |
| Calibración (temperature scaling, NLL, solo validation) | eq. de capitulo2.tex, Guo et al. | `TemperatureScaler.fit` sobre `val_logits/val_labels` únicamente | **MATCH** | ninguna |
| Incertidumbre `U(x)=H(mean_prob)`, ensemble M=5 | eq:ensamble, eq:entropia | `predictive_uncertainty` implementa la fórmula; orquestación de M=5 seeds no existe (correcto no ejecutarla, ver "No hacer") | **MATCH en fórmula**, orquestación pendiente | sin cambios (orquestar 5 seeds es "entrenar benchmark real", prohibido en esta pasada) |
| Abstención: `U(x)>τ_U` **o** `c̄(x)<τ_c` | eq:abstencion — τ_c compara contra **cobertura de evidencia**, no confianza | antes usaba `confidence` (max-probabilidad) | **MISMATCH real** | **CORREGIDO**: renombrado a `evidence_coverage` en `abstention.py` y tests |
| Corrección de comparaciones múltiples | **Holm**, α=0.05 global, aplicado a H1/H2/H3 | antes solo existía `bonferroni_correction` (marcado explícitamente "no verificado") | **MISMATCH real** | **CORREGIDO**: `holm_correction` añadido en `src/evaluation/statistics.py`, Bonferroni se conserva como alternativa documentada |
| Bootstrap agrupado por `source_image_id`, 2.000–5.000 réplicas | capitulo4.tex, "Diseño estadístico" | `grouped_bootstrap_ci(..., n_resamples=2000, ...)` | **MATCH** (2000 está dentro del rango 2000–5000) | ninguna |
| Métricas principales: PR-AUC(SUSPICIOUS) principal, macro-F1/precision/recall/AUROC secundarias | sec:metricas | `compute_all_metrics` reporta `pr_auc_suspicious` primero | **MATCH** | ninguna |
| ECE | `ECE=Σ(|B_b|/N)|acc-conf|` (eq:ece) | idéntico, bins por confianza top-1 | **MATCH** | ninguna |
| AURC / risk-coverage | "clasificación selectiva" (Geifman & El-Yaniv) | implementación estándar equivalente | **MATCH** | ninguna |
| Nodos del grafo | 4 tipos: `claim` (afirmaciones del relato, no un hub único), `visual`, `forensic`, `source` | 5 tipos: `claim` (hub único del caso), `narrative`, `visual`, `image_quality`, `forensic`; no existe nodo `source` ni múltiples nodos `claim` por afirmación | **MISMATCH real** (diseño conceptual distinto) | **PENDING_ADVISOR_DECISION**: redefinir el esquema de nodos es un cambio de arquitectura mayor (afecta B6/P1/P2 ya construidos encima) y cambia qué representa una unidad de evidencia. La decisión es de la asesora, no del código. |
| Atributos de arista | vector continuo `a_ij=[p_support,p_contradiction,p_unknown,sim_sem,overlap_spatial]` (eq:aristas) | `Edge.attributes` transporta los cinco valores con una máscara de observabilidad; `sim_sem` queda enmascarado si el especialista visual no reporta confianzas y `overlap_spatial` siempre, porque ninguna punta de la relación narrativa-visual expone una región comparable | **MATCH, con dos atributos declarados no observables hoy** | **CORREGIDO** (pasada 2026-11-06). Un 0.0 enmascarado es un ausente declarado, nunca un valor medido; `overlap_spatial` se activará cuando `ForensicResult` publique localización |
| Mensaje GATv2 con compuerta | `m=g_j·α_ij,r·W_r·h_j` (pesos **W_r por tipo de relación**), update con **residual + LayerNorm** (eq:mensaje, eq:update) | `ThesisGATv2Layer`: `W_r` por tipo de relación (self-loop incluido, con peso propio), atención que consume `a_ij` y su máscara, y `h_i ← LayerNorm(h_i + σ(Σ m_ij))`. `g_j` escala el mensaje del vecino, no el `h_i` del residual | **MATCH** | **CORREGIDO** (pasada 2026-11-06). `model.gat_variant: thesis` por defecto; `legacy` conserva la capa previa solo para comparar contra ella |

### Resumen de la comparación

- **Matches confirmados**: gate, calibración, incertidumbre (fórmula), ECE, AURC, bootstrap
  agrupado, métricas principales, H2, H3, A1, A2.
- **Mismatches corregidos** (código real, source de verdad = tesis, cambio contenido y de bajo
  riesgo): A3 (mapeo a B6), protocolo oracle (visual+forense bundle), abstención (evidence_coverage
  en vez de confidence), corrección de comparaciones múltiples (Holm en vez de solo Bonferroni),
  descripción de H1 en la documentación.
- **Mismatches cerrados el 2026-11-06**: atributos continuos de arista `a_ij` (con máscara de
  observabilidad), A4, y la capa de mensaje de la tesis (`W_r` por relación + atención sobre
  `a_ij` + residual y LayerNorm).
- **Mismatches pendientes**: A5 (filtrar el entrenamiento por condición C0–C10; requiere que
  `experiments.train` conozca la condición de cada packet) y la taxonomía de nodos del grafo
  (`source`, `claim` multi-afirmación), que queda **PENDING_ADVISOR_DECISION**.

## Bloqueadores que impiden "Scientific evaluation"

1. `benchmark.lock.json` (ahora correctamente re-pinneado contra el commit actual de Repo A,
   `bd4b3aa...`) sigue fijado a v0.2.0 con specialists null.
2. v3 candidato sigue `PENDING_SCIENTIFIC_QA` / `READY_FOR_MODELS=NO`.
3. Especialista forense sin código real; especialista visual sin pesos.
4. Tras la pasada del 2026-11-06, P1/P2 siguen "IMPLEMENTED_CODE_NOT_TRAINED" pero ya no son
   una simplificación en atributos de arista ni en la forma del mensaje: la variante por defecto
   `gat_variant: thesis` implementa eq:aristas, eq:mensaje y eq:update. Queda **una** divergencia
   de diseño, la taxonomía de nodos (`source`, `claim` multi-afirmación), en
   PENDING_ADVISOR_DECISION, más dos atributos (`sim_sem` sin confianzas visuales y
   `overlap_spatial`) declarados no observables con el contrato actual. Cualquier resultado futuro
   debe citar esto como limitación.
