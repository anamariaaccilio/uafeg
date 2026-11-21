"""ANÁLISIS ADICIONAL (POST HOC, test ya consultado): UAFEG con percepción visual mejorada.

    UAFEG_PACKETS_DIR=<packets post hoc> python -m experiments.posthoc_improved --detector yolo11m-seg

Solo se ejecuta si un detector nuevo superó a v2 (0.1177) en mAP50 de máscara en VALIDACIÓN.
Protocolo (el mismo de lo oficial, sin mirar test para elegir nada):
- Búsqueda en validación, mismo presupuesto y criterio que lo oficial (NLL de validación, semilla 0):
  G1, G0 y A2 con la rejilla del agregador (7 configuraciones); B6 con SEARCH_GRID (8); B1, B2 y B3 con
  FLAT_GRID (8). B5 = fusión tardía calibrada de B1–B3 con pesos en validación.
- 5 semillas por sistema; T*, τU y τc en validación (experiments.thesis_evaluate.evaluate_system).
- Test: PR-AUC SUS, Macro-F1, Brier, NLL, ECE y AURC con IC 95 % por bootstrap agrupado por source_image_id
  (2 000 réplicas, paralelizado en 4 núcleos con joblib), y las tres comparaciones análogas a H1–H3.
  El rival de la comparación tipo H1 se elige en validación entre los sistemas sin grafo reentrenados (B5, B6).
Nada de esto se escribe en results/final, results/evaluation.json ni artifacts/thesis/runs.
Salida: results/posthoc/percepcion_mejorada.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from datetime import date

import numpy as np
import yaml
from joblib import Parallel, delayed

from experiments import thesis_aggregator as TA
from experiments import thesis_train as TT
from experiments.thesis_data import REPO_B, packets_dir, verify_inputs
from experiments.thesis_evaluate import CONFIG, b5_logits, evaluate_system, metric_block, per_condition
from src.evaluation.metrics import CLASS_ORDER, pr_auc_suspicious
from src.evaluation.statistics import grouped_bootstrap_ci, holm_correction, paired_grouped_bootstrap_diff
from uafeg.benchmark import canonical_repository

OUT = REPO_B / "results/posthoc"
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = int(os.environ.get("UAFEG_SMOKE_NBOOT", 2000))  # smoke test only; real runs use 2 000
METRICS = ("pr_auc_suspicious", "macro_f1", "brier", "nll", "ece", "aurc")


def _ci_task(name, key, y, probs, pred, U, groups):
    def fn(idx):
        return metric_block(y[idx], probs[idx], pred[idx], U[idx], None)[key]
    ci = grouped_bootstrap_ci(groups, fn, n_resamples=N_BOOT)
    return name, key, ci["ci_low"], ci["ci_high"]


def _diff_task(tag, groups, pa, pb, y, kind):
    if kind == "prauc":
        fa = lambda idx: pr_auc_suspicious([CLASS_ORDER[i] for i in y[idx]], pa[idx])  # noqa: E731
        fb = lambda idx: pr_auc_suspicious([CLASS_ORDER[i] for i in y[idx]], pb[idx])  # noqa: E731
    else:  # brier: diff = Brier(b) - Brier(a)  (>0 favours a)
        eye = np.eye(3)
        fa = lambda idx: -float(np.mean(np.sum((pa[idx] - eye[y[idx]]) ** 2, 1)))  # noqa: E731
        fb = lambda idx: -float(np.mean(np.sum((pb[idx] - eye[y[idx]]) ** 2, 1)))  # noqa: E731
    return tag, paired_grouped_bootstrap_diff(groups, fa, fb, n_resamples=N_BOOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--val-mask-map50", type=float, required=True)
    args = ap.parse_args()
    official = canonical_repository() / "artifacts/packets/v3"
    if "UAFEG_PACKETS_DIR" not in os.environ or packets_dir().resolve() == official.resolve():
        raise SystemExit("Set UAFEG_PACKETS_DIR to the post hoc packets (never the official v3 packets).")
    if args.val_mask_map50 <= 0.1177:
        raise SystemExit("The new detector does not beat v2 on validation (0.1177): Part 1.2 must not run.")
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    inputs = verify_inputs()
    cfg = yaml.safe_load(CONFIG.read_text())
    # ---------------------------------------------------------------- search (validation only)
    agg_cache, flat_cache, search = {}, {}, {}
    for s in ("G1", "G0", "A2"):
        X, y_tr, y_val, vm = TA.data_for(TA.SYSTEMS[s], agg_cache, with_test=False)
        rows = [{"config": c, "best_val_nll": TA.fit_predict(c, X, y_tr, y_val, vm, 0)[1]["best_val_nll"]}
                for c in TA.GRID]
        search[s] = rows
        print(s, min(r["best_val_nll"] for r in rows), flush=True)
    flat_cache[("thesis", (False, False), True)] = agg_cache[((False, False), True)]  # same data, loaded once
    rows = []
    for c in TT.SEARCH_GRID:
        out = TT.train_graph("B6", c, 0, flat_cache)
        rows.append({"config": c, "best_val_nll": out["info"]["best_val_nll"]})
    search["B6"] = rows
    print("B6", min(r["best_val_nll"] for r in rows), flush=True)
    for s in ("B1", "B2", "B3"):
        search[s] = [{"config": c, "best_val_nll": TT.run_one(s, c, 0, flat_cache)["info"]["best_val_nll"]}
                     for c in TT.FLAT_GRID]
        print(s, min(r["best_val_nll"] for r in search[s]), flush=True)
    selected = {s: min(r, key=lambda x: x["best_val_nll"])["config"] for s, r in search.items()}
    # ---------------------------------------------------------------- 5 seeds
    runs = {s: [] for s in ("G1", "G0", "A2", "B6", "B1", "B2", "B3")}
    for s in ("G1", "G0", "A2"):
        X, y_tr, y_val, vm = TA.data_for(TA.SYSTEMS[s], agg_cache, with_test=True)
        for seed in SEEDS:
            out, _, _ = TA.fit_predict(selected[s], X, y_tr, y_val, vm, seed)
            runs[s].append({k: np.log(np.clip(v, 1e-12, 1)) for k, v in out.items()})
    data = agg_cache[((False, False), True)]
    val, test = data["validation"], data["test"]
    val_mask = val.labels >= 0
    for seed in SEEDS:
        runs["B6"].append(TT.train_graph("B6", selected["B6"], seed, flat_cache)["logits"])
        for s in ("B1", "B2", "B3"):
            runs[s].append(TT.run_one(s, selected[s], seed, flat_cache)["logits"])

    def full_val(r):  # training_view(validation) == labelled rows (C10 is test-only)
        z = np.zeros((len(val.labels), 3))
        z[val_mask] = r["val"]
        return {"val": z, "test": r["test"]}
    for s in ("B6", "B1", "B2", "B3"):
        runs[s] = [full_val(r) for r in runs[s]]
    y_val = val.labels[val_mask]
    labelled = test.labels >= 0
    y_test = test.labels[labelled]
    b5_runs, b5_w = b5_logits({k: runs[k] for k in ("B1", "B2", "B3")}, y_val, val_mask)
    systems = {"G1": runs["G1"], "G0": runs["G0"], "A2": runs["A2"], "B6": runs["B6"], "B5": b5_runs}
    res = {n: evaluate_system(n, r, data, cfg, val_mask, y_val, labelled, y_test) for n, r in systems.items()}
    print("trained+evaluated", round(time.time() - t0), "s", flush=True)
    # ---------------------------------------------------------------- bootstrap (4 cores)
    conds = np.array([m["condition"] for m in test.meta])
    groups = np.array([m["source_image_id"] for m in test.meta])
    lab = np.flatnonzero(labelled)
    tasks = [delayed(_ci_task)(n, k, test.labels[lab], res[n]["probs"][lab], res[n]["pred"][lab], res[n]["U"][lab],
                               groups[lab]) for n in res for k in METRICS]
    stress = cfg["hypotheses"]["H1"]["stress_conditions"]
    st = np.flatnonzero(np.isin(conds, stress) & labelled)
    vconds = np.array([m["condition"] for m in val.meta])
    vst = np.isin(vconds, stress) & val_mask
    val_scores = {s: pr_auc_suspicious([CLASS_ORDER[i] for i in val.labels[vst]], res[s]["val_probs"][vst])
                  for s in ("B5", "B6")}
    rival = max(val_scores, key=val_scores.get)
    tasks += [delayed(_diff_task)("H1", groups[st], res["G1"]["probs"][st], res[rival]["probs"][st],
                                  test.labels[st], "prauc"),
              delayed(_diff_task)("H2", groups[lab], res["G1"]["probs"][lab], res["G0"]["probs"][lab],
                                  test.labels[lab], "brier"),
              delayed(_diff_task)("H3", groups[lab], res["G1"]["probs"][lab], res["A2"]["probs"][lab],
                                  test.labels[lab], "prauc")]
    out = Parallel(n_jobs=4, verbose=5)(tasks)
    report = {
        "tipo": "ANÁLISIS ADICIONAL (POST HOC, test ya consultado). No reemplaza ni reinterpreta H1–H3 oficiales.",
        "descripcion": f"Mismo protocolo que lo oficial con el detector {args.detector} (elegido por mAP50 de máscara "
                       f"en validación = {args.val_mask_map50:.4f} > 0.1177 de v2).",
        "detector": args.detector, "val_mask_map50": args.val_mask_map50, "date": str(date.today()),
        "inputs": inputs, "packets_dir": str(packets_dir()), "search": search, "selected": selected,
        "b5_weights": b5_w, "n_bootstrap": N_BOOT, "systems": {}, "comparisons": {},
        "rival_H1_like": {"chosen": rival, "validation_stress_pr_auc": val_scores}}
    for n, r in res.items():
        point = metric_block(test.labels[lab], r["probs"][lab], r["pred"][lab], r["U"][lab], None)
        report["systems"][n] = {"T_star": r["T_star"], "tau_U": r["tau_U"], "tau_c": r["tau_c"],
                                "abstention_rate": float(r["abstain"][labelled].mean()), "test": {},
                                "per_condition": per_condition(r, list(conds), test.labels, labelled)}
        for k in METRICS:
            report["systems"][n]["test"][k] = {"ensemble": point[k]}
    for item in out:
        if len(item) == 4:
            n, k, lo, hi = item
            report["systems"][n]["test"][k].update({"ci_low": lo, "ci_high": hi})
        else:
            tag, d = item
            report["comparisons"][tag] = d
    report["comparisons"]["H1"]["comparison"] = f"G1 vs {rival}, PR-AUC SUS en estrés (G1 − rival)"
    report["comparisons"]["H2"]["comparison"] = "G1 vs G0, Brier(G0) − Brier(G1)"
    report["comparisons"]["H3"]["comparison"] = "G1 vs A2, PR-AUC SUS (G1 − A2)"
    adj = holm_correction([report["comparisons"][h]["p_value_approx"] for h in ("H1", "H2", "H3")])
    for h, a in zip(("H1", "H2", "H3"), adj):
        report["comparisons"][h]["p_holm"] = a
    report["wall_seconds"] = round(time.time() - t0, 1)
    out_dir = Path(os.environ["UAFEG_SMOKE_OUT"]) if "UAFEG_SMOKE_OUT" in os.environ else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "percepcion_mejorada.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps({n: report["systems"][n]["test"]["pr_auc_suspicious"] for n in res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
