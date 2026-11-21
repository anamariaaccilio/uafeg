"""Reporte de especialistas (todo medido en archivos reales; validación salvo que se diga).

    python -m experiments.specialist_report

- Narrativo: acierto de tipo de daño de la afirmación principal (exacto y por familia de la
  ontología HITL) y acierto de polaridad (negación), en validación.
- Visual: mAP50 / mAP50-95 de máscara y caja en val y test HITL y caja en CarDD externo
  (artifacts/damage/yolo_record.json, producido por el kernel de Kaggle).
- Forense: AUROC del puntaje TruFor para separar imágenes editadas de C3/C4/C5 frente a las
  originales C0, en validación (imagen única = unidad).
- Confiabilidad r por especialista (data/specialists/reliability_v3.json).
Salida: results/final/specialists.json y results/final/tabla_especialistas.tex
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.thesis_data import REPO_B
from uafeg.benchmark import canonical_repository
from uafeg.thesis_graph import DAMAGE_FAMILIES

A = canonical_repository()
OUT = REPO_B / "results/final"


def family(c):
    return next((k for k, v in DAMAGE_FAMILIES.items() if c in v), c)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    claims = [json.loads(l) for l in (A / "data/manifests/benchmark_v3/manifest.jsonl").read_text().splitlines()]
    val = [c for c in claims if c["split"] == "validation"]
    nar = {r["text_sha256"]: r for r in map(json.loads, (A / "artifacts/specialists/narrative.jsonl").read_text().splitlines())}
    exact = fam = pol = n = 0
    for c in val:
        if not c.get("narrative_text"):
            continue
        row = nar[hashlib.sha256(c["narrative_text"].encode()).hexdigest()]
        if not row["assertions"]:
            continue
        a = max(row["assertions"], key=lambda a: a["p"][0] + a["p"][2])
        asserted = set(c["metadata"].get("asserted_damage_classes", []))
        expected = "denied" if c["condition"] in {"C1", "C5"} else "affirmed"
        n += 1
        exact += a["damage_type"] in asserted
        fam += family(a["damage_type"]) in {family(x) for x in asserted}
        pol += a["polarity"] == expected
    narrative = {"n_validation_claims_with_assertion": n, "damage_type_exact_acc": exact / n,
                 "damage_type_family_acc": fam / n, "polarity_acc": pol / n}

    rec_path = A / "artifacts/damage/yolo_record.json"
    visual = json.loads(rec_path.read_text()) if rec_path.exists() else {"status": "NOT_RUN"}

    forensic = {"status": "NOT_RUN"}
    fpath = A / "artifacts/specialists/forensic.jsonl"
    if fpath.exists():
        rows = {r["image_sha256"]: r for r in map(json.loads, fpath.read_text().splitlines())}
        per = defaultdict(dict)
        for c in val:
            if c["condition"] in ("C0", "C3", "C4", "C5") and c.get("image_path"):
                sha = c["transformation"]["image_sha256"]
                if sha in rows:
                    per[c["condition"]][sha] = rows[sha]["score"]
        forensic = {"status": "MEASURED", "n_images": {k: len(v) for k, v in per.items()}}
        neg = list(per["C0"].values())
        for cond in ("C3", "C4", "C5"):
            pos = list(per[cond].values())
            forensic[f"auroc_{cond}_vs_C0"] = float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))
        pos = [v for k in ("C3", "C4", "C5") for v in per[k].values()]
        forensic["auroc_C3C4C5_vs_C0"] = float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))
        forensic["provenance"] = json.loads((A / "artifacts/specialists/forensic.provenance.json").read_text())
    rel = json.loads((A / "data/specialists/reliability_v3.json").read_text())
    reliability = {"narrative_r": rel["narrative"]["r"], "visual_r": rel["visual"].get("r"),
                   "forensic": {k: rel["forensic"].get(k) for k in ("status", "normalization", "n_validation_images",
                                                                    "validation_auroc_score_local_edit")}}
    out = {"narrative": narrative, "visual": visual, "forensic": forensic, "reliability": reliability}
    (OUT / "specialists.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    def f(x):
        return "NOT\\_RUN" if x is None else f"{x:.3f}"
    lines = [r"\begin{table}[H]\centering\footnotesize", r"\begin{tabular}{llr}", r"\toprule",
             r"\textbf{Especialista} & \textbf{Métrica} & \textbf{Valor} \\ \midrule",
             rf"Narrativo (mDeBERTa NLI) & tipo de daño exacto (val, n={n}) & {f(narrative['damage_type_exact_acc'])} \\",
             rf" & tipo de daño por familia de la ontología & {f(narrative['damage_type_family_acc'])} \\",
             rf" & polaridad (negación) & {f(narrative['polarity_acc'])} \\"]
    he = visual.get("hitl_eval", {})
    for split, name in (("val", "val HITL"), ("test", "test HITL")):
        if split in he:
            lines.append(rf"Visual (YOLOv8n-seg) & mAP50 / mAP50-95 máscara, {name} & {f(he[split]['mask_map50'])} / {f(he[split]['mask_map50_95'])} \\")
            lines.append(rf" & mAP50 / mAP50-95 caja, {name} & {f(he[split]['box_map50'])} / {f(he[split]['box_map50_95'])} \\")
    ext = visual.get("external_cardd", {})
    if ext.get("status") == "MEASURED":
        lines.append(rf" & CarDD test, caja, clases mapeadas (n={ext['n_images']}) & {f(ext['box_map50_mapped'])} / {f(ext['box_map50_95_mapped'])} \\")
        lines.append(rf" & CarDD test, caja, daño agnóstico & {f(ext['box_map50_agnostic'])} / {f(ext['box_map50_95_agnostic'])} \\")
        lines.append(r" & CarDD test, máscara & NOT\_RUN (solo cajas) \\")
    else:
        lines.append(rf" & CarDD externo & {ext.get('status', 'NOT_RUN').replace('_', chr(92) + '_')} \\")
    if forensic.get("status") == "MEASURED":
        for cond in ("C3", "C4", "C5"):
            lines.append(rf"Forense (TruFor) & AUROC {cond} vs C0 (val) & {f(forensic[f'auroc_{cond}_vs_C0'])} \\")
        lines.append(rf" & AUROC C3--C5 vs C0 (val) & {f(forensic['auroc_C3C4C5_vs_C0'])} \\")
    else:
        lines.append(r"Forense (TruFor) & AUROC & NOT\_RUN \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Especialistas medidos sobre archivos reales. Fuente: \texttt{results/final/specialists.json}.}",
              r"\label{tab:especialistas}", r"\end{table}"]
    (OUT / "tabla_especialistas.tex").write_text("\n".join(lines) + "\n")
    print(json.dumps({"narrative": narrative, "forensic": {k: v for k, v in forensic.items() if k != "provenance"}},
                     indent=1))


if __name__ == "__main__":
    main()
