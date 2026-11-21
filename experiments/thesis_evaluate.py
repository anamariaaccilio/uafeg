"""Evaluación final: calibración, abstención, test una sola vez, bootstrap, H1-H3 y tablas.

    python -m experiments.thesis_evaluate            # refuses to run twice (test once)

Protocolo (fijado en configs/evaluation.yaml ANTES de leer test):
- T* por sistema y por semilla, y para el ensamble M=5, ajustado en validación (NLL).
- tau_U y tau_c en validación con fit_thresholds_on_validation para la cobertura
  operativa objetivo; abstenerse = derivar a revisión (INSUFFICIENT).
- Test: PR-AUC SUSPICIOUS, Macro-F1, Brier, NLL, ECE, AURC; por condición y ΔM_s;
  C10 aparte (OOD: AUROC de U para separar C10 del resto, y destino de sus decisiones).
- Bootstrap agrupado por source_image_id; diferencias pareadas; McNemar; Holm alpha=0.05.
- Sensibilidad: todo lo anterior con y sin los casos AMBIGUOUS.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from experiments.thesis_data import ART, LABEL_INDEX, REPO_B, load_all, training_view, verify_inputs
from src.abstention.abstention import fit_thresholds_on_validation, risk_coverage_curve
from src.calibration.calibrate import TemperatureScaler
from src.evaluation.metrics import CLASS_ORDER, compute_all_metrics, expected_calibration_error
from src.evaluation.statistics import grouped_bootstrap_ci, holm_correction, paired_grouped_bootstrap_diff

RESULTS = REPO_B / "results"
CONFIG = REPO_B / "configs/evaluation.yaml"
SUS = CLASS_ORDER.index("SUSPICIOUS")
INS = CLASS_ORDER.index("INSUFFICIENT")
NONGRAPH = ("B0", "B1", "B2", "B3", "B4", "B5", "B7", "B6", "B8")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def calibrate(val_logits, y_val, logits):
    scaler = TemperatureScaler().fit(val_logits, y_val)
    return scaler.temperature, scaler.calibrate_probabilities(logits)


def load_logits(system):
    runs = sorted((ART / "runs" / system).glob("seed*/logits.npz"))
    return [dict(np.load(r)) for r in runs]


# ------------------------------------------------------------------ B0 and B5
def b0_logits(data):
    """Reglas B0 (src.fusion.rules, sin cambios) sobre los EvidencePackets reales. P(y | regla)
    se estima en validación (conteos con Laplace) para tener probabilidades."""
    from experiments.thesis_data import _read_packets
    from src.fusion.features import build_feature_vector
    from src.fusion.rules import evaluate
    from types import SimpleNamespace

    def rule_ids(split):
        out = []
        for p in _read_packets(split):
            nr = None
            if p.narrative is not None:
                top = max(p.narrative.raw_scores, key=p.narrative.raw_scores.get)
                nr = SimpleNamespace(top_damage=top, top_confidence=p.narrative.raw_scores[top],
                                     normalized_scores=p.narrative.normalized_scores)
            fv = build_feature_vector(nr, p.visual, p.image_quality, visual_model_available=p.visual is not None)
            out.append(evaluate(fv).rule_id)
        return out
    ids = {s: rule_ids(s) for s in ("validation", "test")}
    val_keep = data["validation"].labels >= 0
    counts = {}
    for rid, y in zip(np.array(ids["validation"])[val_keep], data["validation"].labels[val_keep]):
        counts.setdefault(rid, np.ones(3))[y] += 1
    table = {k: v / v.sum() for k, v in counts.items()}
    prior = np.bincount(data["validation"].labels[val_keep], minlength=3) + 1.0
    prior = prior / prior.sum()
    to_logits = lambda rids: np.log(np.stack([table.get(r, prior) for r in rids]))
    return [{"val": to_logits(ids["validation"]), "test": to_logits(ids["test"])}], {
        k: v.tolist() for k, v in table.items()}


def b5_logits(parts, y_val, val_mask):
    """Fusión tardía calibrada: cada unimodal calibrado con su T en validación; log p
    combinado con pesos elegidos en validación (rejilla 0..1 paso 0.25, mínima NLL)."""
    seeds = min(len(p) for p in parts.values())
    out, chosen = [], []
    grid = [w for w in np.linspace(0, 1, 5)]
    for k in range(seeds):
        cal = {}
        for name, runs in parts.items():
            lv = runs[k]["val"][val_mask]
            t = TemperatureScaler().fit(lv, y_val).temperature
            cal[name] = {s: np.log(softmax(runs[k][s] / t)) for s in ("val", "test")}
        names = list(cal)
        best, best_w = math.inf, None
        for ws in np.array(np.meshgrid(*[grid] * len(names))).T.reshape(-1, len(names)):
            if ws.sum() == 0:
                continue
            z = sum(w * cal[n]["val"][val_mask] for w, n in zip(ws, names))
            nll = -np.log(softmax(z)[np.arange(len(y_val)), y_val] + 1e-12).mean()
            if nll < best:
                best, best_w = nll, ws
        chosen.append(dict(zip(names, best_w.tolist())))
        out.append({s: sum(w * cal[n][s] for w, n in zip(best_w, names)) for s in ("val", "test")})
    return out, chosen


# ------------------------------------------------------------------ metrics
def metric_block(y, probs, pred, U, groups):
    labels = [CLASS_ORDER[i] for i in y]
    m = compute_all_metrics(labels, [CLASS_ORDER[i] for i in pred], probs)
    rc = risk_coverage_curve(U, probs.argmax(1) == y)
    return {"pr_auc_suspicious": m["pr_auc_suspicious"], "macro_f1": m["macro_f1"], "brier": m["brier"],
            "nll": m["nll"], "ece": m["ece"], "aurc": rc["aurc"], "auroc_macro": m.get("auroc_ovr_macro")}


def evaluate_system(name, runs, data, cfg, val_mask, y_val, test_mask, y_test):
    target = cfg["abstention"]["target_coverage"]
    cbar = {s: np.array([g.coverage_bar for g in data[s].graphs]) for s in ("validation", "test")}
    per_seed = []
    for r in runs:
        t, p_test = calibrate(r["val"][val_mask], y_val, r["test"])
        _, p_val = calibrate(r["val"][val_mask], y_val, r["val"])
        th = fit_thresholds_on_validation(entropy(p_val[val_mask]), cbar["validation"][val_mask], target)
        U = entropy(p_test)
        abst = th.decide(U, cbar["test"])
        pred = np.where(abst, INS, p_test.argmax(1))
        per_seed.append({"T": t, **metric_block(y_test, p_test[test_mask], pred[test_mask], U[test_mask], None)})
    # ensemble M = len(runs): mean of member probabilities, then T* on validation
    mean_val = np.mean([softmax(r["val"]) for r in runs], 0)
    mean_test = np.mean([softmax(r["test"]) for r in runs], 0)
    t_star, p_test = calibrate(np.log(mean_val[val_mask] + 1e-12), y_val, np.log(mean_test + 1e-12))
    _, p_val = calibrate(np.log(mean_val[val_mask] + 1e-12), y_val, np.log(mean_val + 1e-12))
    th = fit_thresholds_on_validation(entropy(p_val[val_mask]), cbar["validation"][val_mask], target)
    U = entropy(p_test)
    abst = th.decide(U, cbar["test"])
    pred = np.where(abst, INS, p_test.argmax(1))
    return {"name": name, "M": len(runs), "T_star": t_star, "tau_U": th.tau_u, "tau_c": th.tau_c,
            "per_seed": per_seed, "probs": p_test, "U": U, "pred": pred, "abstain": abst,
            "val_probs": p_val, "val_pr_auc": compute_all_metrics(
                [CLASS_ORDER[i] for i in y_val], [CLASS_ORDER[i] for i in p_val[val_mask].argmax(1)],
                p_val[val_mask])["pr_auc_suspicious"]}


def summarize(res, y, mask, groups, n_boot, seed=42):
    probs, pred, U = res["probs"][mask], res["pred"][mask], res["U"][mask]
    yy = y
    point = metric_block(yy, probs, pred, U, None)
    g = np.asarray(groups)
    out = {}
    for key in ("pr_auc_suspicious", "macro_f1", "brier", "nll", "ece", "aurc"):
        def fn(idx, key=key):
            return metric_block(yy[idx], probs[idx], pred[idx], U[idx], None)[key]
        ci = grouped_bootstrap_ci(g, fn, n_resamples=n_boot, seed=seed)
        seeds = [s[key] for s in res["per_seed"]]
        out[key] = {"ensemble": point[key], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                    "seed_mean": float(np.mean(seeds)), "seed_sd": float(np.std(seeds, ddof=1)) if len(seeds) > 1 else 0.0}
    return out


def per_condition(res, conds, y_all, labels_mask):
    out = {}
    for c in sorted(set(conds), key=lambda s: int(s[1:])):
        idx = np.array([k for k, cc in enumerate(conds) if cc == c])
        dist = {CLASS_ORDER[k]: float(np.mean(res["pred"][idx] == k)) for k in range(3)}
        row = {"n": int(len(idx)), "decision_distribution": dist,
               "abstention_rate": float(res["abstain"][idx].mean())}
        if c != "C10":
            row["accuracy"] = float(np.mean(res["pred"][idx] == y_all[idx]))
        out[c] = row
    clean = out["C0"]["accuracy"]
    for c, row in out.items():
        if "accuracy" in row:
            row["delta_M"] = clean - row["accuracy"]
    return out


def delta_stress(res, conds, y_all, stress, idx=None):
    conds = np.asarray(conds)
    idx = np.arange(len(conds)) if idx is None else idx
    c, yy, pp = conds[idx], y_all[idx], res["pred"][idx]
    def acc(cond):
        m = c == cond
        return float(np.mean(pp[m] == yy[m])) if m.any() else float("nan")
    clean = acc("C0")
    return float(np.nanmean([clean - acc(s) for s in stress]))


def mcnemar(correct_a, correct_b):
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    return {"b": b, "c": c, "p_value": float(binomtest(b, b + c, 0.5).pvalue)}


def cohens_h(p1, p2):
    return float(2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-evaluate (logged as a repeated test access)")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    log = RESULTS / "test_access_log.json"
    accesses = json.loads(log.read_text()) if log.exists() else []
    if accesses and not args.force:
        raise SystemExit("Test was already evaluated once; use --force to repeat (it will be logged).")
    inputs = verify_inputs()
    data = load_all()
    val = data["validation"]
    val_mask = val.labels >= 0
    y_val = val.labels[val_mask]
    test = data["test"]
    conds = [m["condition"] for m in test.meta]
    groups_all = [m["source_image_id"] for m in test.meta]
    labelled = test.labels >= 0
    y_test = test.labels[labelled]
    ambiguous = np.array([m["ambiguous"] for m in test.meta])
    n_boot = cfg["bootstrap"]["n_resamples"]

    systems = {}
    b0_runs, b0_table = b0_logits(data)
    systems["B0"] = b0_runs
    for s in ("B1", "B2", "B3", "B4", "B7", "B6", "B8", "G0", "G1", "A2", "A4", "A5",
              "G1_oracleN", "G1_oracleVF", "G1_oracleNVF", "G0_oracleNVF"):
        runs = load_logits(s)
        if runs:
            systems[s] = runs
    b5_parts = {k: systems[k] for k in ("B1", "B2", "B3") if k in systems}
    systems["B5"], b5_weights = b5_logits(b5_parts, y_val, val_mask)

    evaluated = {}
    for name, runs in systems.items():
        res = evaluate_system(name, runs, data, cfg, val_mask, y_val, labelled, y_test)
        evaluated[name] = res

    report = {"date": str(date.today()), "inputs": inputs, "config": cfg, "b0_rule_table": b0_table,
              "b5_weights": b5_weights, "systems": {}, "not_run": json.loads(
                  (RESULTS / "not_run.json").read_text()) if (RESULTS / "not_run.json").exists() else {}}
    for name, res in evaluated.items():
        groups = [g for g, keep in zip(groups_all, labelled) if keep]
        main_all = summarize(res, y_test, labelled, groups, n_boot)
        keep_na = labelled & ~ambiguous
        groups_na = [g for g, keep in zip(groups_all, keep_na) if keep]
        main_na = summarize(res, test.labels[keep_na], keep_na, groups_na, max(200, n_boot // 10))
        ood = None
        from sklearn.metrics import roc_auc_score
        is_ood = np.array([c == "C10" for c in conds])
        ood = float(roc_auc_score(is_ood, res["U"])) if is_ood.any() else None
        abst_conc = float(np.mean(np.isin(np.array(conds)[res["abstain"]], ["C6", "C7", "C9", "C10"]))) if res["abstain"].any() else None
        report["systems"][name] = {
            "M": res["M"], "T_star": res["T_star"], "tau_U": res["tau_U"], "tau_c": res["tau_c"],
            "val_pr_auc_suspicious": res["val_pr_auc"],
            "test": main_all, "test_without_ambiguous": main_na,
            "per_condition": per_condition(res, conds, test.labels, labelled),
            "ood_auroc_U_C10": ood, "abstention_rate": float(res["abstain"][labelled].mean()),
            "abstentions_in_C6_C7_C9_C10": abst_conc,
            "per_seed": res["per_seed"],
        }

    # ------------------------------------------------------------ hypotheses
    hyp = cfg["hypotheses"]
    stress = hyp["H1"]["stress_conditions"]
    from src.evaluation.metrics import pr_auc_suspicious
    val_conds = np.array([m["condition"] for m in val.meta])
    val_stress = np.isin(val_conds, stress) & val_mask
    test_stress = np.isin(np.array(conds), stress) & labelled

    def stress_prauc(res, split_mask, labels, probs_key):
        p = res[probs_key][split_mask]
        return pr_auc_suspicious([CLASS_ORDER[i] for i in labels[split_mask]], p)
    candidates = [s for s in hyp["H1"]["nongraph_candidates"] if s in evaluated]
    val_scores = {s: stress_prauc(evaluated[s], val_stress, val.labels, "val_probs") for s in candidates}
    best_ng = max(val_scores, key=val_scores.get)
    groups_all_arr = np.array(groups_all)
    st_idx = np.flatnonzero(test_stress)
    y_st = test.labels[st_idx]

    def prauc_fn(name):
        p = evaluated[name]["probs"][st_idx]
        return lambda idx: pr_auc_suspicious([CLASS_ORDER[i] for i in y_st[idx]], p[idx])
    h1 = paired_grouped_bootstrap_diff(groups_all_arr[st_idx], prauc_fn("G1"), prauc_fn(best_ng), n_resamples=n_boot)
    h1.update({"comparison": f"G1 vs {best_ng} (best non-graph system by validation PR-AUC on the stress subset)",
               "statistic": f"PR-AUC SUSPICIOUS on test restricted to {stress}; diff = G1 - best (>0 favours G1)",
               "point_G1": prauc_fn("G1")(np.arange(len(st_idx))), "point_best_nongraph": prauc_fn(best_ng)(np.arange(len(st_idx))),
               "validation_stress_pr_auc": val_scores, "n_stress_test": int(len(st_idx)),
               "secondary_delta_M": {n: {c: r.get("delta_M") for c, r in report["systems"][n]["per_condition"].items()}
                                     for n in ("G1", best_ng)}})
    lab_idx = np.flatnonzero(labelled)

    def brier_fn(name):
        p = evaluated[name]["probs"]
        return lambda idx: float(np.mean(np.sum((p[lab_idx][idx] - np.eye(3)[y_test[idx]]) ** 2, 1)))
    y_all = test.labels
    h2 = paired_grouped_bootstrap_diff(groups_all_arr[labelled], brier_fn("G0"), brier_fn("G1"), n_resamples=n_boot)
    h2.update({"comparison": "G1 vs G0, Brier (calibrated ensembles)", "statistic": "Brier(G0) - Brier(G1) (>0 favours G1)",
               "point_G1": brier_fn("G1")(np.arange(int(labelled.sum()))), "point_G0": brier_fn("G0")(np.arange(int(labelled.sum())))})
    if "A2" in evaluated:
        lab = np.flatnonzero(labelled)
        y_lab = test.labels[lab]

        def full_prauc(name):
            p = evaluated[name]["probs"][lab]
            return lambda idx: pr_auc_suspicious([CLASS_ORDER[i] for i in y_lab[idx]], p[idx])
        h3 = paired_grouped_bootstrap_diff(groups_all_arr[lab], full_prauc("G1"), full_prauc("A2"), n_resamples=n_boot)
        h3.update({"status": "EVALUATED", "comparison": "UAFEG (G1) vs A2 (G1 without forensic nodes)",
                   "statistic": "PR-AUC SUSPICIOUS on the full labelled test set; diff = G1 - A2 (>0 favours G1)",
                   "point_G1": full_prauc("G1")(np.arange(len(lab))), "point_A2": full_prauc("A2")(np.arange(len(lab)))})
    else:
        h3 = {"status": "NOT_EVALUATED", "p_value_approx": 1.0,
              "reason": "forensic specialist NOT_RUN: A2 (P2 without forensic nodes) is identical to P2, so "
                        "H3 (UAFEG vs A2, PR-AUC SUSPICIOUS) cannot be tested; p=1 is used in Holm (conservative)."}
    correct = {n: (evaluated[n]["pred"] == y_all) for n in ("G1", "G0", best_ng) + (("A2",) if "A2" in evaluated else ())}
    h1["mcnemar_overall_correctness"] = mcnemar(correct["G1"][labelled], correct[best_ng][labelled])
    h2["mcnemar_overall_correctness"] = mcnemar(correct["G1"][labelled], correct["G0"][labelled])
    if "A2" in evaluated:
        h3["mcnemar_overall_correctness"] = mcnemar(correct["G1"][labelled], correct["A2"][labelled])
    pvals = [h1["p_value_approx"], h2["p_value_approx"], h3["p_value_approx"]]
    adj = holm_correction(pvals)
    for h, a in zip((h1, h2, h3), adj):
        h["p_holm"] = a
        h["reject_H0_at_0.05"] = bool(a < cfg["hypotheses"]["alpha"]) if h.get("status") != "NOT_EVALUATED" else None
    report["hypotheses"] = {"H1": h1, "H2": h2, "H3": h3, "alpha": cfg["hypotheses"]["alpha"], "holm_m": 3}

    accesses.append({"date": str(date.today()), "packets_manifest_sha256": inputs["packets_manifest_sha256"],
                     "forced": bool(args.force)})
    log.write_text(json.dumps(accesses, indent=1) + "\n")
    (RESULTS / "evaluation.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    np.savez(ART / "eval_arrays.npz", **{f"{n}_{k}": evaluated[n][k] for n in evaluated
                                         for k in ("probs", "U", "pred", "abstain")},
             labels=test.labels, conditions=np.array(conds), claim_ids=np.array([m["claim_id"] for m in test.meta]),
             val_labels=val.labels, val_conditions=np.array([m["condition"] for m in val.meta]),
             **{f"{n}_val_probs": evaluated[n]["val_probs"] for n in evaluated},
             **{f"{n}_thresholds": np.array([evaluated[n]["tau_U"], evaluated[n]["tau_c"], evaluated[n]["T_star"]]) for n in evaluated})
    print(json.dumps({"H1": {k: h1[k] for k in ("point_diff", "p_value_approx", "p_holm")},
                      "H2": {k: h2[k] for k in ("point_diff", "p_value_approx", "p_holm")},
                      "H3": h3["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
