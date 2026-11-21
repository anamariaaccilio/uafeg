# Figuras para la presentación

Todas están en `presentacion/figuras/`, en PNG a 300 dpi y en SVG, con fondo blanco y la paleta de la tesis.
Se regeneran con `python -m experiments.presentation_figures` (o, por ejemplo, `... 2 3 7` para algunas).
Ningún número está escrito a mano: cada figura lo lee del archivo que figura en su pie. El pie también dice si el
resultado es **OFICIAL** (test evaluado una sola vez) o un **ANÁLISIS ADICIONAL (post hoc, test ya consultado)**.

Nota de color: la paleta de la tesis tiene pares poco distinguibles para daltonismo (narrativa/visual y
Respaldado/Sospechoso). Por eso cada color va siempre acompañado de una etiqueta escrita, una trama o su posición.

| Figura | Mensaje en una frase | Diapositiva sugerida | Números de |
|---|---|---|---|
| `01_pipeline` | UAFEG pasa cada evidencia por un especialista, la pondera con la compuerta g y decide o deriva a una persona; ★ marca los aportes. | Propuesta / metodología | diseño final (README, `docs/decisiones_cierre.md`) |
| `02_comparacion_general` | G1 (0.851) queda por debajo de la fusión tardía B5 (0.878) en detectar casos sospechosos. | Resultados: visión general | `results/evaluation.json` |
| `03_hipotesis_forest` | H1 y H2 se rechazan (gana el rival); H3 se acepta (el forense aporta). | Resultados: hipótesis | `results/evaluation.json` (`hypotheses`) |
| `04_oraculo` | Con percepción perfecta G1 sube de 0.851 a 0.965: el agregador funciona y el límite está en los especialistas. | Discusión: techo del método | `results/evaluation.json` (sistemas `*_oracle*`) |
| `05_percepcion_progresiva` | Aun con el 100 % de las fotos con anotación real, G1 pasa solo de 0.851 a 0.859 y sigue debajo de B5 (0.881): el detector visual solo no explica la brecha (post hoc). | Discusión: ¿y si mejora el detector? | `results/posthoc/percepcion_progresiva.json` |
| `06_oficial_vs_mejorada` | Con un detector mejor en validación (yolo11m-seg, 0.141 frente a 0.118), G1 se queda en 0.851 y B5 sigue arriba (0.879) (post hoc). | Discusión: ¿y si mejora el detector? | `results/posthoc/percepcion_mejorada.json` |
| `07_heatmap_condiciones` | La foto de otro vehículo (C2) y el inpainting (C4) son lo más difícil para los sistemas de fusión. | Resultados: por condición | `results/evaluation.json` (`per_condition`) |
| `08_compuerta_g` | La compuerta da muy poco peso a las detecciones visuales y apaga el forense y la fuente en fotos borrosas. | Propuesta: la compuerta en acción | `presentacion/datos/compuerta_g_test.json` |
| `09_aporte_forense` | TruFor sube la PR-AUC de Suspicious en +0.062, sobre todo en copiar y pegar (C3, C5). | Resultados: H3 | `results/evaluation.json` |
| `10_calibracion` | Las probabilidades de G1, G0 y B5 están bien calibradas (ECE ≤ 0.016). | Resultados: confiabilidad | `artifacts/thesis/eval_arrays.npz`, `results/evaluation.json` |
| `11_riesgo_cobertura` | Derivar los casos inciertos baja el error, aunque G1 ordena peor que B5 y G0 (AURC mayor). | Resultados: abstención | `artifacts/thesis/eval_arrays.npz`, `results/evaluation.json` |
| `12_regresion` | El acierto de G1 por imagen casi no depende del detector (R² 0.02); la incertidumbre U sí anticipa el error. | Discusión: por qué pasa | `presentacion/datos/regresiones.json` (post hoc) |
| `13_ablaciones` | Quitar el forense es lo que más daña; quitar la compuerta no empeora. | Resultados: ablaciones | `results/evaluation.json` |
| `14_especialistas` | YOLO es el eslabón débil; TruFor detecta copiar y pegar pero no inpainting. | Especialistas | `results/final/specialists.json` |
| `15a_caso_correcto` | Un caso real que UAFEG decide bien, con foto, máscaras, mapa forense, grafo y explicación. | Demostración | `review/cases/<id>.json` + EvidencePacket |
| `15b_caso_abstencion` | Un caso que UAFEG deriva a revisión porque su incertidumbre supera el umbral. | Demostración: revisión humana | `review/cases/<id>.json` + EvidencePacket |
| `16a_tabla_principal` … `16d_tabla_hipotesis` | Tablas principal, de ablaciones, de oráculo y de hipótesis como imagen. | Respaldo / anexos | `results/evaluation.json` |

## Detalles que conviene saber antes de presentar

- **Ejes truncados**: las figuras 4 y 13 no empiezan en 0 (lo dice el eje). Las diferencias son reales, pero se ven
  más grandes que en un eje desde 0.
- **Figura 3**: cada hipótesis tiene su propia escala horizontal (H2 está en unidades de Brier).
- **Figura 10**: el tamaño del punto indica cuántos casos tiene ese tramo; los tramos con menos de 30 casos llevan su n.
- **Figura 12** es descriptiva y post hoc: muestra asociación, no causalidad. El acierto de G1 se mide con la clase
  más probable (antes de la abstención). La recta usa 129 imágenes de test (foto original, C0).
- **Figura 15**: los casos se eligen con una regla fija, no por cómo salen. Se recorre `review/seleccion.json` en
  orden y se toma el primer caso Sospechoso que G1 acierta sin abstenerse (es un C5: en el paquete no hay ningún C3
  decidido sin abstención) y la primera abstención (un C3). Los textos de la explicación son los del reporte
  real. Hay errores visibles del especialista narrativo: por ejemplo, «corrosión» se lee como *crack*. Conviene
  mencionarlo, porque muestra lo que la revisión humana debe poder detectar.
