"""Entrenamiento del agregador sin GATv2: G1 (UAFEG final), G0 (= A1, g = 1) y ablaciones.

    python -m experiments.thesis_aggregator search     # <= 8 configs por sistema, solo validación
    python -m experiments.thesis_aggregator train --seeds 0 1 2 3 4

Sistemas:
  G1   agregador con compuerta g                       (reemplaza a P2)
  G0   el mismo agregador con g = 1 (= A1)             (reemplaza a P1)
  A2   G1 sin nodos forenses
  A4   G1 con aristas sin atributos continuos
  A5   G1 entrenado sin C8 (C10 nunca está en train)
  G1_oracleN / G1_oracleVF / G1_oracleNVF   percepción oráculo (narrativa, visual+forense, ambas)

Búsqueda (mismo presupuesto, 7 configuraciones por sistema, criterio = NLL de validación):
regresión logística multinomial (C ∈ {0.1, 1, 10}, features estandarizadas con media/desv. de
train) y gradient boosting (learning_rate ∈ {0.05, 0.1} × max_depth ∈ {2, 3}, subsample 0.8,
número de árboles ≤ 400 elegido por NLL de validación = early stopping). Semillas: random_state
del boosting (subsample); la regresión logística es convexa y determinista, así que sus 5
semillas coinciden (se reporta así, sin fingir varianza).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from experiments.thesis_data import ART, REPO_B, load_all, training_view, verify_inputs
from uafeg.aggregator import FEATURE_NAMES, feature_matrix

RESULTS = REPO_B / "results"
RUNS_DIR = ART / "runs"
SUMMARY_DIR = RESULTS / "runs"
SYSTEMS = {
    "G1": {"gate": True},
    "G0": {"gate": False},
    "A2": {"gate": True, "use_forensic": False},
    "A4": {"gate": True, "edge_attributes": False},
    "A5": {"gate": True, "exclude_train_conditions": ["C8"]},
    "G1_oracleN": {"gate": True, "oracle": [True, False]},
    "G1_oracleVF": {"gate": True, "oracle": [False, True]},
    "G1_oracleNVF": {"gate": True, "oracle": [True, True]},
    "G0_oracleNVF": {"gate": False, "oracle": [True, True]},
}
GRID = ([{"model": "logreg", "C": c} for c in (0.1, 1.0, 10.0)]
        + [{"model": "gboost", "learning_rate": lr, "max_depth": d, "subsample": 0.8}
           for lr in (0.05, 0.1) for d in (2, 3)])
MAX_TREES = 400


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())


def data_for(spec, cache, with_test=True):
    key = (tuple(spec.get("oracle", [False, False])), spec.get("use_forensic", True))
    if key not in cache:
        cache[key] = load_all(oracle=key[0], use_forensic=key[1])
    d = cache[key]
    train = training_view(d["train"], tuple(spec.get("exclude_train_conditions", [])))
    val = training_view(d["validation"])
    feats = {}
    for name, split in (("train", train), ("val", d["validation"])) + ((("test", d["test"]),) if with_test else ()):
        feats[name] = feature_matrix(split.graphs, gate=spec.get("gate", True),
                                     edge_attributes=spec.get("edge_attributes", True))
    val_mask = d["validation"].labels >= 0
    return feats, train.labels, d["validation"].labels, val_mask


def fit_predict(cfg, X, y_tr, y_val, val_mask, seed):
    """Returns log-probabilities for every split in X (except train) and training info."""
    if cfg["model"] == "logreg":
        mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6
        Z = {k: (v - mu) / sd for k, v in X.items()}
        clf = LogisticRegression(C=cfg["C"], max_iter=5000, random_state=seed)
        clf.fit(Z["train"], y_tr)
        out = {k: clf.predict_proba(Z[k]) for k in X if k != "train"}
        info = {"best_val_nll": nll(out["val"][val_mask], y_val[val_mask]), "n_iter": int(clf.n_iter_.max()),
                "coef": clf.coef_.tolist(), "standardization": {"mu": mu.tolist(), "sd": sd.tolist()}}
        return out, info, clf
    clf = GradientBoostingClassifier(n_estimators=MAX_TREES, learning_rate=cfg["learning_rate"],
                                     max_depth=cfg["max_depth"], subsample=cfg["subsample"], random_state=seed)
    clf.fit(X["train"], y_tr)
    curve = [nll(p[val_mask], y_val[val_mask]) for p in clf.staged_predict_proba(X["val"])]
    best_n = int(np.argmin(curve)) + 1
    out = {}
    for k in X:
        if k == "train":
            continue
        for i, p in enumerate(clf.staged_predict_proba(X[k])):
            if i + 1 == best_n:
                out[k] = p
                break
    info = {"best_val_nll": curve[best_n - 1], "n_trees": best_n, "val_nll_curve_every10": curve[::10],
            "feature_importances": dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))}
    return out, info, clf


def git_commit(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def cmd_search(args):
    verify_inputs()
    cache, results, selected = {}, {}, {}
    for system in args.systems:
        X, y_tr, y_val, vm = data_for(SYSTEMS[system], cache, with_test=False)
        rows = []
        for cfg in GRID:
            t0 = time.time()
            _, info, _ = fit_predict(cfg, X, y_tr, y_val, vm, seed=0)
            rows.append({"config": cfg, "best_val_nll": info["best_val_nll"]})
            print(system, cfg, round(info["best_val_nll"], 4), f"{time.time() - t0:.0f}s", flush=True)
        results[system] = rows
        selected[system] = min(rows, key=lambda r: r["best_val_nll"])["config"]
    path = RESULTS / "hparam_search_aggregator.json"
    old = json.loads(path.read_text()) if path.exists() else {"all": {}, "selected": {}}
    old["all"].update(results)
    old["selected"].update(selected)
    old.update({"criterion": "minimum validation NLL (seed 0)", "budget_configs_per_system": len(GRID),
                "grid": GRID})
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")
    print(json.dumps(selected, indent=1))


def cmd_train(args):
    sel = json.loads((RESULTS / "hparam_search_aggregator.json").read_text())["selected"]
    inputs = verify_inputs()
    cache = {}
    import sklearn
    from uafeg.benchmark import canonical_repository
    for system in args.systems:
        cfg = sel.get(system) or sel[args.config_from.get(system, "G1")]
        X, y_tr, y_val, vm = data_for(SYSTEMS[system], cache, with_test=True)
        for seed in args.seeds:
            run_dir = RUNS_DIR / system / f"seed{seed}"
            if (run_dir / "record.json").exists() and not args.force:
                print("skip", system, seed)
                continue
            t0 = time.time()
            out, info, _ = fit_predict(cfg, X, y_tr, y_val, vm, seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez(run_dir / "logits.npz", **{k: np.log(np.clip(v, 1e-12, 1)) for k, v in out.items()})
            np.savez(run_dir / "features.npz", **X)
            record = {"system": system, "seed": seed, "config": cfg, "spec": SYSTEMS[system],
                      "feature_names": list(FEATURE_NAMES), "training": info, "n_train": int(len(y_tr)),
                      "early_stopping": "gboost: number of trees chosen by validation NLL; logreg: convex, fit to convergence",
                      "git_commit_uafeg": git_commit(REPO_B), "git_commit_canonical": git_commit(canonical_repository()),
                      "inputs": inputs,
                      "outputs_sha256": {"logits.npz": hashlib.sha256((run_dir / "logits.npz").read_bytes()).hexdigest()},
                      "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                   "sklearn": sklearn.__version__},
                      "wall_seconds": round(time.time() - t0, 1), "result_type": "FINAL_TRAINING",
                      "test_logits_note": "computed once; read only by experiments.thesis_evaluate"}
            (run_dir / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
            SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
            small = {k: record[k] for k in ("system", "seed", "config", "n_train", "git_commit_uafeg", "outputs_sha256",
                                            "wall_seconds")}
            small["best_val_nll"] = info["best_val_nll"]
            small["packets_manifest_sha256"] = inputs["packets_manifest_sha256"]
            (SUMMARY_DIR / f"{system}_seed{seed}.json").write_text(json.dumps(small, indent=1, sort_keys=True) + "\n")
            print(system, seed, round(info["best_val_nll"], 4), f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--systems", nargs="+", default=["G1", "G0", "A2", "A4", "A5"])
    t = sub.add_parser("train")
    t.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    t.add_argument("--force", action="store_true")
    args = ap.parse_args()
    # oracle variants reuse the configuration selected for the non-oracle system
    args.config_from = {k: ("G0" if k.startswith("G0") else "G1") for k in SYSTEMS}
    {"search": cmd_search, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
