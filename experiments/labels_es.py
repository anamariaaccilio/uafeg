"""Etiquetas legibles en español para sistemas, condiciones y métricas (figuras, tablas y JSON)."""
from __future__ import annotations

SISTEMAS = {
    "B0": "B0 · Reglas fijas",
    "B1": "B1 · Solo texto",
    "B2": "B2 · Solo imagen (detector + calidad)",
    "B3": "B3 · Solo forense (TruFor)",
    "B4": "B4 · Todo concatenado + red neuronal",
    "B5": "B5 · Fusión tardía calibrada",
    "B6": "B6 · DeepSets sobre las evidencias",
    "B7": "B7 · Fusión con incertidumbre (DBF)",
    "B8": "B8 · Modelo multimodal grande (Qwen2-VL-2B)",
    "G0": "G0 · Agregador sin compuerta (g = 1)",
    "G1": "G1 · UAFEG (agregador con compuerta g)",
    "A2": "A2 · UAFEG sin forense",
    "A4": "A4 · UAFEG con aristas sin atributos",
    "A5": "A5 · UAFEG sin controles benignos en entrenamiento",
    "G1_oracleN": "G1 con relato perfecto",
    "G1_oracleVF": "G1 con imagen y forense perfectos",
    "G1_oracleNVF": "G1 con percepción perfecta",
    "G0_oracleNVF": "G0 con percepción perfecta",
}
SISTEMAS_CORTO = {k: v.split(" · ")[0] if " · " in v else v for k, v in SISTEMAS.items()}

CONDICIONES = {
    "C0": "Caso original (sin cambios)",
    "C1": "El relato niega un daño real",
    "C2": "Foto de otro vehículo (sustitución)",
    "C3": "Foto editada (copiar y pegar)",
    "C4": "Foto editada (inpainting)",
    "C5": "Foto editada + relato que niega",
    "C6": "Falta el relato",
    "C7": "Falta la foto",
    "C8": "Cambio benigno (JPEG, recorte)",
    "C9": "Foto muy borrosa",
    "C10": "Edición nunca vista (inversión de color)",
}

METRICAS = {
    "pr_auc_suspicious": "Qué tan bien detecta los casos sospechosos (PR-AUC de Suspicious, más alto es mejor)",
    "macro_f1": "Acierto equilibrado entre las tres clases (Macro-F1, más alto es mejor)",
    "brier": "Error de las probabilidades (Brier, más bajo es mejor)",
    "nll": "Sorpresa ante la respuesta correcta (NLL, más bajo es mejor)",
    "ece": "Desajuste entre confianza y acierto (ECE, más bajo es mejor)",
    "aurc": "Riesgo promedio al ir respondiendo más casos (AURC, más bajo es mejor)",
    "abstention_rate": "Fracción de casos derivados a revisión humana",
    "delta_M": "Caída de acierto respecto del caso original (ΔM)",
}

CLASES = {"SUPPORTED": "Respaldado", "SUSPICIOUS": "Sospechoso", "INSUFFICIENT": "Insuficiente"}

ESPECIALISTAS = {"claim": "Relato", "narrative": "Relato", "visual": "Imagen (detector)",
                 "forensic": "Forense (TruFor)", "source": "Fuente / calidad"}
