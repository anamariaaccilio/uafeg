# Mensajes clave de la tesis (para la defensa)

Cada número sale de un archivo de resultados; entre paréntesis va la figura de `presentacion/figuras/`.
**Oficial** = test evaluado una sola vez (`results/evaluation.json`). **Post hoc** = análisis adicional hecho
después de consultar test (`results/posthoc/`, `presentacion/datos/`). Lo post hoc nunca cambia el veredicto de
H1–H3.

## El resultado en una frase

UAFEG no supera a una fusión tardía simple: su PR-AUC de Suspicious es 0.851 frente a 0.878 de B5 (H1 rechazada),
y la compuerta empeora un poco la calibración (H2 rechazada). Lo que sí se sostiene es que el forense aporta,
que el agregador tiene un techo alto si la percepción mejora, que la compuerta reconoce la evidencia poco confiable
y que el sistema explica cada decisión y sabe derivar casos a una persona.

## Los 4 aportes que se sostienen con datos

### 1. El análisis forense aporta de forma significativa (H3, oficial)
- Con TruFor, la PR-AUC de Suspicious sube **+0.062** (IC 95 % [+0.050, +0.073], p de Holm < 0.0005) frente al mismo
  UAFEG sin forense (A2).
- La ganancia se concentra en las fotos editadas: en copiar y pegar (C3) G1 acierta 0.65 y A2 0.00; en foto editada +
  relato que niega (C5), 0.95 frente a 0.84.
- Figuras: `03_hipotesis_forest`, `09_aporte_forense`, `13_ablaciones`.

### 2. El agregador tiene un techo alto: el límite está en los especialistas (oráculo, oficial y diagnóstico)
- Con percepción perfecta (relato, imagen y forense), G1 sube de **0.851 a 0.965**. Con imagen y forense perfectos,
  0.943; con solo el relato perfecto, 0.866.
- Matiz honesto: con percepción perfecta, el agregador sin compuerta (G0) llega a 0.970, así que el techo es del
  agregador, no de la compuerta. Además, B5 no se evaluó con oráculo.
- Figura: `04_oraculo`.

### 3. La compuerta g reconoce la evidencia poco confiable (descriptivo, oficial)
- En test, el peso medio de las detecciones visuales es **0.02**: la compuerta casi las ignora, coherente con un
  detector débil.
- En fotos muy borrosas (C9) el peso del forense cae de 0.20 a 0.002 y el de la fuente de 0.80 a 0.007 respecto de
  la foto original (C0).
- La incertidumbre de G1 separa la edición nunca vista (C10) del resto algo mejor que la de los rivales: AUROC 0.646,
  frente a 0.597 de G0 y 0.571 de B5. Es descriptivo: no tiene intervalo ni prueba.
- Lo que **no** se sostiene: que esa ponderación mejore la decisión. H2 se rechaza: con g el Brier empeora en 0.010.
- Figura: `08_compuerta_g`.

### 4. Explicabilidad, revisión humana y abstención (oficial)
- Hay un reporte por cada reclamo de test (1 591): afirmaciones, detecciones, mapa forense, grafo con su g,
  probabilidades y la evidencia con más peso.
- El paquete de revisión tiene 50 casos estratificados, una página HTML que exporta a CSV y las 316 abstenciones.
  Ninguna respuesta de revisión está inventada: la planilla está vacía.
- G1 deriva el **18.4 %** de test a revisión (objetivo 20 %, umbral fijado en validación).
- Entre el 80 % de casos más seguros, el error de la clase más probable es 0.172, frente a 0.222 si responde todo.
  En esta métrica B5 ordena un poco mejor (0.165 a la misma cobertura; AURC 0.071 frente a 0.080 de G1).
- Figuras: `11_riesgo_cobertura`, `15a_caso_correcto`, `15b_caso_abstencion`.

## El resultado negativo, explicado sin esconderlo

**Qué pasó.** B5 gana a G1 en todo test (0.878 frente a 0.851) y en el subconjunto de estrés de H1 (diferencia
−0.039, IC 95 % [−0.053, −0.025]). También ganan G0 (0.871), B6 (0.870) y B4 (0.868). Por condición, B5 acierta más
en las fotos editadas: copiar y pegar (C3) 0.86 frente a 0.65; inpainting (C4) 0.26 frente a 0.01; foto de otro
vehículo (C2) 0.16 frente a 0.00. G1 acierta más en las fotos originales (C0), 0.67 frente a 0.37, y en los cambios
benignos (C8), 0.71 frente a 0.58. (`07_heatmap_condiciones`)

**Por qué.**
1. **Los especialistas son débiles.**
   - El detector visual: mAP50 de máscara 0.118 en validación y 0.096 en test; en CarDD, 0.054 de caja.
   - El relato: el tipo de daño exacto acierta 0.50.
   - TruFor no distingue el inpainting (AUROC 0.544).

   (`14_especialistas`)
2. **La compuerta multiplica esa debilidad en lugar de corregirla.** Al bajar casi a cero el peso de las
   detecciones visuales, y bastante el del forense, G1 pierde señal que B5 sí aprovecha, porque B5 combina las
   salidas sin atenuarlas. Por eso G0, sin compuerta, supera a G1 (H2).
3. **Mejorar solo el detector visual no alcanza (post hoc).**
   - En la curva de percepción progresiva, aun con el 100 % de las fotos con anotación real, G1 pasa de 0.851 a
     0.859 y B5 sigue en 0.881. No cruzan en ningún nivel (`05_percepcion_progresiva`).
   - Además, el acierto de G1 por imagen casi no depende del acierto del detector: R² = 0.02, pendiente +0.03 con
     IC 95 % [−0.01, +0.07] (`12_regresion`).
   - Con un detector real mejor (yolo11m-seg, mAP50 de máscara en validación 0.141 frente a 0.118), reentrenando
     todo con el mismo protocolo, G1 se queda en 0.851 y B5 en 0.879. La diferencia tipo H1 sigue siendo −0.042,
     con IC 95 % [−0.055, −0.028] (`06_oficial_vs_mejorada`).
   - El salto del oráculo, en cambio, aparece cuando son perfectos a la vez la imagen y el forense (0.943).
   - Conclusión defendible: la brecha se explica por la percepción en conjunto, sobre todo la forense (inpainting) y
     el cruce relato–imagen (sustitución). No se explica solo por YOLO. No se midió un oráculo solo forense, así que
     no se puede separar más.

**Cómo decirlo en la defensa.** «La hipótesis principal no se cumple. Con los especialistas actuales, una fusión
tardía calibrada es mejor. El diseño de UAFEG no es el problema: con percepción perfecta llega a 0.965. El problema es
lo que le entra. Y la compuerta, que debía protegerlo de la evidencia mala, termina descartando también señal útil.»

## 5 preguntas probables del jurado

1. **«Si B5 es mejor, ¿para qué sirve UAFEG?»**
   B5 da una probabilidad; UAFEG da además la razón, afirmación por afirmación, y sabe derivar el caso a una persona.
   Su detección es algo peor (0.851 frente a 0.878), pero el forense le aporta de forma significativa (+0.062) y su
   techo con mejor percepción es 0.965. En un proceso de aseguradora, un error explicado y revisable vale distinto
   que uno opaco. Igual, no lo vendo como mejor detector: no lo es.
2. **«¿No ajustaste nada mirando test?»**
   No. Test se evaluó una sola vez y el acceso quedó registrado (`results/test_access_log.json`). Hiperparámetros,
   umbrales de abstención, temperatura y el rival de H1 se eligieron en validación.
   Todo lo posterior está marcado como post hoc y no cambia H1–H3. La curva de percepción, por ejemplo, reproduce
   exactamente H1 en su nivel 0 %.
3. **«¿Por qué la compuerta empeora el Brier si debía ayudar?»**
   Porque multiplica cuatro factores (cobertura, calidad, confiabilidad e incertidumbre). Con especialistas débiles
   el resultado es casi cero para las detecciones visuales (g medio 0.02), y eso apaga señal que, aunque ruidosa,
   ayudaba. Lo correcto es aprender g en vez de fijarla por fórmula. Lo dejo como trabajo futuro.
4. **«¿Por qué no usaste un modelo multimodal grande?»**
   Se probó: Qwen2-VL-2B congelado (B8) obtuvo la peor PR-AUC, 0.438. Sin ajuste fino, un modelo general no
   reconoce estas manipulaciones.
5. **«Los datos son sintéticos, ¿sirve para casos reales?»**
   Las fotos son reales (HITL, CC0). Los relatos y las manipulaciones son controlados, para tener etiqueta
   conocida. Es una limitación declarada: el siguiente paso es validar con expedientes reales de una aseguradora.
   En CarDD, el detector visual transfiere poco (mAP50 de caja 0.054), otra señal de que hace falta más datos.

## Trabajo futuro concreto

- **Mejor detector visual**, entrenado con más datos de daño vehicular con licencia académica (CarDD con máscaras
  bajo licencia clara, por ejemplo). Ojo: la curva post hoc y la prueba con yolo11m-seg indican que esto solo no
  cierra la brecha.
- **Forense específico para inpainting**: TruFor está en 0.544 en C4. Un detector entrenado con
  inpainting de daño vehicular es la mejora con mayor potencial según el oráculo.
- **Reentrenar el especialista narrativo (NLI)** con relatos de reclamos en español. Hoy acierta 0.50 en el tipo
  de daño exacto y confunde clases cercanas (por ejemplo, «corrosión» leída como *crack*).
- **Compuerta aprendida**: en lugar de g = c·q·r·(1−u) fija, aprender cuánto pesa cada factor, con la restricción
  de seguir siendo interpretable.
- **Datos reales de aseguradora** para validar externamente todo el sistema y la revisión humana (llenar
  `review/revision_humana.csv` con revisores reales).
