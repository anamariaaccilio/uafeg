"""Agrega campos "descripcion" en español a los JSON de resultados, sin cambiar ningún número.

    python -m experiments.add_descriptions

Verifica que, quitando las claves "descripcion"/"descripciones", el JSON sea idéntico al original.
"""
from __future__ import annotations

import copy
import json

from experiments.labels_es import CLASES, CONDICIONES, METRICAS, SISTEMAS
from experiments.thesis_data import REPO_B

R = REPO_B / "results"


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in ("descripcion", "descripciones")}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def evaluation():
    path = R / "evaluation.json"
    ev = json.loads(path.read_text())
    before = copy.deepcopy(strip(ev))
    ev["descripciones"] = {"sistemas": SISTEMAS, "condiciones": CONDICIONES, "metricas": METRICAS, "clases": CLASES,
                           "nota": "Campos solo descriptivos agregados el 2026-11-21; los números no cambian."}
    for s, row in ev["systems"].items():
        row["descripcion"] = SISTEMAS.get(s, s)
        for c, pc in row["per_condition"].items():
            pc["descripcion"] = CONDICIONES.get(c, c)
    for k in ("H1", "H2", "H3"):
        ev["hypotheses"][k]["descripcion"] = {
            "H1": "UAFEG (G1) frente al mejor sistema sin grafo elegido en validación, en casos de estrés",
            "H2": "UAFEG con compuerta (G1) frente al mismo agregador sin compuerta (G0), en Brier",
            "H3": "UAFEG con forense (G1) frente a UAFEG sin forense (A2)"}[k]
    assert strip(ev) == before, "descriptions changed a number"
    path.write_text(json.dumps(ev, indent=1, sort_keys=True, default=float) + "\n")


def specialists():
    path = R / "final/specialists.json"
    sp = json.loads(path.read_text())
    before = copy.deepcopy(strip(sp))
    sp["narrative"]["descripcion"] = "Especialista del relato (mDeBERTa NLI): acierto en el tipo de daño y en si afirma o niega"
    sp["visual"]["descripcion"] = "Especialista visual (YOLOv8n-seg v2): mAP50 de máscara y caja en HITL y validación externa CarDD"
    sp["forensic"]["descripcion"] = "Especialista forense (TruFor): AUROC para separar fotos editadas de la original"
    assert strip(sp) == before
    path.write_text(json.dumps(sp, indent=1, sort_keys=True, default=float) + "\n")


def main():
    evaluation()
    specialists()
    print("ok")


if __name__ == "__main__":
    main()
