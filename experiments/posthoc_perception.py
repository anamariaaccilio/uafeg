"""ANÁLISIS ADICIONAL (POST HOC, test ya consultado): curva de percepción progresiva.

    python -m experiments.posthoc_perception

Pregunta: ¿cuánto mejora UAFEG (G1) frente a la fusión tardía B5 cuando mejora la
percepción VISUAL? No es un resultado confirmatorio y no toca nada de results/final/.

Protocolo (fijado antes de ejecutar, anotado en docs/decisiones_cierre.md):
- Niveles p ∈ {0, 0.25, 0.5, 0.75, 1.0}: fracción de source_image_id cuya evidencia visual se
  reemplaza por la visual oráculo (uafeg.thesis_oracle.oracle_visual_forensic, SOLO la parte
  visual; la narrativa y TruFor quedan como fueron predichas). Muestreo por source_image_id con
  una permutación fija (semilla 0) de todos los ids, así que los conjuntos están anidados.
- La mezcla se aplica igual en train, validación y test (simula un detector de esa calidad).
- Hiperparámetros: los ya elegidos en validación para el sistema oficial (sin búsqueda nueva):
  G1 = results/hparam_search_aggregator.json; B1, B2, B3 = results/hparam_search.json.
  5 semillas; B5 = fusión tardía calibrada de B1, B2, B3 (pesos en validación), como en lo oficial.
- T*, τU y τc en validación (experiments.thesis_evaluate.evaluate_system, sin cambios).
- Métricas en test fijadas de antemano: PR-AUC de SUSPICIOUS en todo test etiquetado (la de la
  figura principal) y en el subconjunto de estrés de H1 (C3, C4, C5, C8, C9); Brier.
  Diferencia G1 − B5 con bootstrap pareado agrupado por source_image_id (2 000 réplicas).
Salida: results/posthoc/percepcion_progresiva.json
"""
from __future__ import annotations

import gc
import json
import time
from datetime import date, datetime, timezone

import numpy as np
import yaml

from experiments import thesis_aggregator as TA
from experiments import thesis_train as TT
from experiments.thesis_data import LABEL_INDEX, REPO_B, SPLITS, Split, _read_packets, packets_dir, verify_inputs
from experiments.thesis_evaluate import CONFIG, b5_logits, evaluate_system
from src.evaluation.metrics import CLASS_ORDER, pr_auc_suspicious
from src.evaluation.statistics import grouped_bootstrap_ci, paired_grouped_bootstrap_diff
from uafeg.thesis_graph import build_thesis_graph, evidence_view
from uafeg.thesis_oracle import oracle_visual_forensic

OUT = REPO_B / "results/posthoc"
LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2, 3, 4)
SAMPLING_SEED = 0
N_BOOT = 2000


def oracle_ids() -> list[str]:
    ids = set()
    for s in SPLITS:
        for line in (packets_dir() / f"{s}.jsonl").read_text().splitlines():
            ids.add(json.loads(line)["claim"]["source_image_id"])
    ids = sorted(ids)
    perm = np.random.default_rng(SAMPLING_SEED).permutation(len(ids))
    return [ids[i] for i in perm]


def build_split(split: str, chosen: set[str]) -> tuple[Split, int]:
    ambiguous = set(json.loads((packets_dir() / "ambiguous_claim_ids.json").read_text()))
    graphs, labels, meta, n_or = [], [], [], 0
    for packet in _read_packets(split):
        c = packet.claim
        if c.source_image_id in chosen:
            vis, _ = oracle_visual_forensic(packet)
            if vis is not None:
                packet = packet.model_copy(update={"visual": vis})
                n_or += 1
        graphs.append(build_thesis_graph(evidence_view(packet)))
        labels.append(LABEL_INDEX.get(c.expected_label.value if c.expected_label else None, -1))
        meta.append({"claim_id": c.claim_id, "source_image_id": c.source_image_id, "condition": c.condition,
                     "label": c.expected_label.value, "ambiguous": c.claim_id in ambiguous})
    return Split(split, graphs, np.array(labels), meta), n_or


def main() -> int:
    import torch
    torch.set_num_threads(1)  # same as the official training (experiments.thesis_train.main)
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = verify_inputs()
    cfg = yaml.safe_load(CONFIG.read_text())
    sel_flat = json.loads((REPO_B / "results/hparam_search.json").read_text())["selected"]
    sel_agg = json.loads((REPO_B / "results/hparam_search_aggregator.json").read_text())["selected"]
    stress = cfg["hypotheses"]["H1"]["stress_conditions"]
    order = oracle_ids()
    rows = []
    for p in LEVELS:
        t0 = time.time()
        chosen = set(order[:int(round(p * len(order)))])
        data, n_or = {}, {}
        for s in SPLITS:
            data[s], n_or[s] = build_split(s, chosen)
        # the training code reads the default (non-oracle) data through these caches
        flat_cache = {("thesis", (False, False), True): data}
        agg_cache = {((False, False), True): data}
        runs = {"B1": [], "B2": [], "B3": [], "G1": []}
        for seed in SEEDS:
            for b in ("B1", "B2", "B3"):
                runs[b].append(TT.run_one(b, sel_flat[b], seed, flat_cache)["logits"])
        X, y_tr, y_val, vm = TA.data_for(TA.SYSTEMS["G1"], agg_cache, with_test=True)
        for seed in SEEDS:
            out, _, _ = TA.fit_predict(sel_agg["G1"], X, y_tr, y_val, vm, seed)
            runs["G1"].append({k: np.log(np.clip(v, 1e-12, 1)) for k, v in out.items()})
        val, test = data["validation"], data["test"]
        val_mask = val.labels >= 0
        y_val_l = val.labels[val_mask]
        labelled = test.labels >= 0
        y_test = test.labels[labelled]
        # B1-B3 logits for val come from training_view(validation) == labelled rows; pad to full val
        def full_val(r):
            z = np.zeros((len(val.labels), 3), dtype=np.float64)
            z[val_mask] = r["val"]
            return {"val": z, "test": r["test"]}
        parts = {b: [full_val(r) for r in runs[b]] for b in ("B1", "B2", "B3")}
        b5_runs, b5_w = b5_logits(parts, y_val_l, val_mask)
        res = {"G1": evaluate_system("G1", runs["G1"], data, cfg, val_mask, y_val_l, labelled, y_test),
               "B5": evaluate_system("B5", b5_runs, data, cfg, val_mask, y_val_l, labelled, y_test)}
        conds = np.array([m["condition"] for m in test.meta])
        groups = np.array([m["source_image_id"] for m in test.meta])
        lab = np.flatnonzero(labelled)
        st = np.flatnonzero(np.isin(conds, stress) & labelled)

        def pr_fn(name, idx_base):
            pr = res[name]["probs"][idx_base]
            yy = test.labels[idx_base]
            return lambda idx: pr_auc_suspicious([CLASS_ORDER[i] for i in yy[idx]], pr[idx])

        def brier_fn(name):
            pr = res[name]["probs"][lab]
            yy = test.labels[lab]
            return lambda idx: float(np.mean(np.sum((pr[idx] - np.eye(3)[yy[idx]]) ** 2, 1)))
        row = {"level": p, "n_source_ids_oracle": len(chosen), "n_claims_with_oracle_visual": n_or,
               "b5_weights": b5_w, "systems": {}, "diff_G1_minus_B5": {}}
        for name in ("G1", "B5"):
            r = {"tau_U": res[name]["tau_U"], "tau_c": res[name]["tau_c"], "T_star": res[name]["T_star"],
                 "abstention_rate": float(res[name]["abstain"][labelled].mean())}
            for key, base in (("pr_auc_suspicious_test", lab), ("pr_auc_suspicious_stress_H1", st)):
                fn = pr_fn(name, base)
                ci = grouped_bootstrap_ci(groups[base], fn, n_resamples=N_BOOT)
                r[key] = {"value": fn(np.arange(len(base))), "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
            r["brier_test"] = brier_fn(name)(np.arange(len(lab)))
            r["accuracy_decision_test"] = float(np.mean(res[name]["pred"][lab] == test.labels[lab]))
            row["systems"][name] = r
        for key, base in (("pr_auc_suspicious_test", lab), ("pr_auc_suspicious_stress_H1", st)):
            d = paired_grouped_bootstrap_diff(groups[base], pr_fn("G1", base), pr_fn("B5", base), n_resamples=N_BOOT)
            row["diff_G1_minus_B5"][key] = {k: d[k] for k in ("point_diff", "ci_low", "ci_high", "p_value_approx")}
        row["wall_seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(json.dumps({"level": p, "G1": row["systems"]["G1"]["pr_auc_suspicious_test"]["value"],
                          "B5": row["systems"]["B5"]["pr_auc_suspicious_test"]["value"],
                          "secs": row["wall_seconds"]}), flush=True)
        del data, flat_cache, agg_cache, X, runs, res
        gc.collect()
    report = {
        "tipo": "ANÁLISIS ADICIONAL (POST HOC, test ya consultado). Análisis de sensibilidad declarado; no confirmatorio.",
        "descripcion": "Curva de percepción progresiva: G1 y B5 reentrenados con una fracción creciente de imágenes "
                       "(por source_image_id) cuya evidencia visual es la anotación real en vez de la predicha por YOLO.",
        "date": str(date.today()), "inputs": inputs, "levels": list(LEVELS), "seeds": list(SEEDS),
        "sampling": {"unit": "source_image_id", "seed": SAMPLING_SEED, "nested": True, "n_ids": len(order),
                     "splits_mixed": list(SPLITS), "oracle_part": "visual only (narrative and TruFor as predicted)"},
        "configs": {"G1": sel_agg["G1"], **{b: sel_flat[b] for b in ("B1", "B2", "B3")}},
        "metrics_fixed_in_advance": ["pr_auc_suspicious_test", "pr_auc_suspicious_stress_H1", "brier_test"],
        "n_bootstrap": N_BOOT, "rows": rows, "wall_seconds_total": round(time.time() - t_start, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "percepcion_progresiva.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
