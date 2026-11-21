"""Entrenamiento a escala reducida (CPU) de B1-B7, B6/P1/P2 y ablaciones.

    python -m experiments.thesis_train search            # <=8 configs en validación, mismo presupuesto B6/P1/P2
    python -m experiments.thesis_train train --systems P2 P1 B6 --seeds 0 1 2 3 4

Optimización (igual para todos): AdamW (weight decay 1e-4), ReduceLROnPlateau sobre la
NLL de validación, early stopping por NLL de validación (patience 5, máx. 40 épocas),
gradient clipping 1.0, lotes de 64. La selección de hiperparámetros usa solo
validación. Las probabilidades de test se calculan UNA vez al final de cada corrida
final (sin mirarlas: se guardan en artifacts/ y solo las lee experiments.thesis_evaluate).

Cada corrida guarda config, semilla, commit, hashes de entrada y salida, versiones de
librerías y checkpoint (artifacts/thesis/runs/<system>/seed<k>/), más un resumen JSON
pequeño versionado en results/runs/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.thesis_data import ART, REPO_B, flat_matrix, load_all, training_view, verify_inputs
from uafeg.models.batched import DBF, MLP, EvidenceModel, collate
from uafeg.thesis_graph import LEGACY_KINDS, NODE_KINDS

RESULTS = REPO_B / "results"
WITH_TEST = True
RUNS_DIR = ART / "runs"
SUMMARY_DIR = RESULTS / "runs"
"""False in --validation-only mode: test is never collated, inferred or saved."""
MAX_EPOCHS, PATIENCE, BATCH, WD, CLIP = 40, 5, 64, 1e-4, 1.0
SEARCH_GRID = [{"dropout": d, "heads": h, "pooling": p, "lr": 2e-3, "dim": 64, "hidden": 64, "layers": 2}
               for d, h, p in itertools.product((0.1, 0.3), (2, 4), ("mean", "attention"))]
FLAT_GRID = [{"dropout": d, "lr": 2e-3, "hidden": 64} for d in (0.1, 0.3)]

# system -> how to build it. Aliases are documented, never silently re-run.
GRAPH_SYSTEMS = {
    "B6": {"arch": "B6"}, "P1": {"arch": "P1"}, "P2": {"arch": "P2"},
    "A2": {"arch": "P2", "use_forensic": False},
    "A4": {"arch": "P2", "edge_attributes": False},
    "A5": {"arch": "P2", "exclude_train_conditions": ["C8"]},
    "P2_legacy": {"arch": "P2", "taxonomy": "legacy"},
    "P2_oracleN": {"arch": "P2", "oracle": [True, False]},
    "P2_oracleVF": {"arch": "P2", "oracle": [False, True]},
    "P2_oracleNVF": {"arch": "P2", "oracle": [True, True]},
}
ALIASES = {"A1": ("P1", "P2 con g_j = 1 es por construcción P1 (test_p2_with_unit_gate_equals_p1)"),
           "A3": ("B6", "A3 (GATv2 -> DeepSets con los mismos EvidencePacket) coincide con B6")}
FLAT_SYSTEMS = {"B1": ("text",), "B2": ("visual",), "B3": ("forensic",), "B4": ("text", "visual", "forensic")}


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def dataset_for(spec: dict, cache: dict) -> dict:
    key = (spec.get("taxonomy", "thesis"), tuple(spec.get("oracle", [False, False])), spec.get("use_forensic", True))
    if key not in cache:
        cache[key] = load_all(taxonomy=key[0], oracle=key[1], use_forensic=key[2])
    return cache[key]


def nll_of(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, y).item())


def fit(model: nn.Module, forward, n_train: int, y_train: torch.Tensor, val_forward, y_val: torch.Tensor,
        lr: float, seed: int, extra_loss=None) -> tuple[nn.Module, dict]:
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    best, best_state, bad, history = float("inf"), None, 0, []
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n_train, generator=gen)
        for start in range(0, n_train, BATCH):
            idx = perm[start:start + BATCH]
            opt.zero_grad()
            out = forward(idx)
            loss = extra_loss(out, y_train[idx]) if extra_loss else F.cross_entropy(out, y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nll_of(val_forward(), y_val)
        sched.step(v)
        history.append(round(v, 5))
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, {"val_nll_history": history, "best_val_nll": best, "epochs_run": len(history)}


def index_batch(full: dict, idx) -> dict:
    return {k: v[idx] for k, v in full.items()}


def train_graph(system: str, cfg: dict, seed: int, cache: dict) -> dict:
    spec = GRAPH_SYSTEMS[system]
    data = dataset_for(spec, cache)
    train = training_view(data["train"], tuple(spec.get("exclude_train_conditions", [])))
    val = training_view(data["validation"])
    use_attr = spec.get("edge_attributes", True)
    splits = (("train", train), ("val", val)) + ((("test", data["test"]),) if WITH_TEST else ())
    full = {s: collate(d.graphs, use_edge_attributes=use_attr) for s, d in splits}
    n_kinds = len(LEGACY_KINDS if spec.get("taxonomy") == "legacy" else NODE_KINDS)
    set_seed(seed)
    model = EvidenceModel(spec["arch"], n_kinds=n_kinds, dim=cfg["dim"], hidden=cfg["hidden"], heads=cfg["heads"],
                          layers=cfg["layers"], dropout=cfg["dropout"], pooling=cfg["pooling"])
    y_tr, y_val = torch.tensor(train.labels), torch.tensor(val.labels)
    model, info = fit(model, lambda idx: model(index_batch(full["train"], idx)), len(train.labels), y_tr,
                      lambda: model(full["val"]), y_val, cfg["lr"], seed)
    with torch.no_grad():
        model.eval()
        logits = {"val": model(full["val"]).numpy()}
        if WITH_TEST:
            logits["test"] = model(full["test"]).numpy()
    return {"model": model, "info": info, "logits": logits, "n_train": len(train.labels)}


def train_flat(system: str, cfg: dict, seed: int, cache: dict) -> dict:
    data = dataset_for({}, cache)
    train, val, test = training_view(data["train"]), training_view(data["validation"]), data["test"]
    parts = FLAT_SYSTEMS[system]
    X = {s: torch.tensor(flat_matrix(d, parts)) for s, d in
         (("train", train), ("val", val)) + ((("test", test),) if WITH_TEST else ())}
    mu, sd = X["train"].mean(0), X["train"].std(0) + 1e-6  # standardization fitted on train only
    X = {k: (v - mu) / sd for k, v in X.items()}
    set_seed(seed)
    model = MLP(X["train"].shape[1], cfg["hidden"], cfg["dropout"])
    model, info = fit(model, lambda idx: model(X["train"][idx]), len(train.labels), torch.tensor(train.labels),
                      lambda: model(X["val"]), torch.tensor(val.labels), cfg["lr"], seed)
    with torch.no_grad():
        model.eval()
        logits = {k: model(X[k]).numpy() for k in X if k != "train"}
    return {"model": model, "info": info, "logits": logits, "n_train": len(train.labels),
            "standardization": {"mu": mu.tolist(), "sd": sd.tolist()}}


def train_dbf(cfg: dict, seed: int, cache: dict, modalities=("text", "visual")) -> dict:
    data = dataset_for({}, cache)
    train, val, test = training_view(data["train"]), training_view(data["validation"]), data["test"]
    Xs = {}
    for s, d in (("train", train), ("val", val)) + ((("test", test),) if WITH_TEST else ()):
        Xs[s] = [torch.tensor(flat_matrix(d, (m,))) for m in modalities]
    stats = [(x.mean(0), x.std(0) + 1e-6) for x in Xs["train"]]
    Xs = {k: [(x - mu) / sd for x, (mu, sd) in zip(v, stats)] for k, v in Xs.items()}
    set_seed(seed)
    model = DBF([x.shape[1] for x in Xs["train"]], cfg["hidden"], cfg["dropout"])

    def loss_fn(out, y):
        logp, ops = out
        fused = F.nll_loss(logp, y)
        return fused + sum(DBF.dirichlet_loss(alpha, y) for _, _, alpha in ops) / len(ops)

    model, info = fit(model, lambda idx: model([x[idx] for x in Xs["train"]]), len(train.labels),
                      torch.tensor(train.labels), lambda: model(Xs["val"])[0], torch.tensor(val.labels),
                      cfg["lr"], seed, extra_loss=loss_fn)
    with torch.no_grad():
        model.eval()
        logits = {k: model(Xs[k])[0].numpy() for k in Xs if k != "train"}
    return {"model": model, "info": info, "logits": logits, "n_train": len(train.labels)}


def forensic_available(cache: dict) -> bool:
    data = dataset_for({}, cache)
    return bool(flat_matrix(data["train"], ("forensic",))[:, 0].any())


def save_run(system: str, seed: int, cfg: dict, out: dict, inputs: dict, wall: float) -> None:
    run_dir = RUNS_DIR / system / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(out["model"].state_dict(), run_dir / "checkpoint.pt")
    np.savez(run_dir / "logits.npz", **out["logits"])
    import sklearn
    from uafeg.benchmark import canonical_repository
    record = {
        "system": system, "seed": seed, "config": cfg, "spec": GRAPH_SYSTEMS.get(system, {"flat": system}),
        "optimizer": {"name": "AdamW", "weight_decay": WD, "scheduler": "ReduceLROnPlateau(0.5, 2)",
                      "early_stopping": f"val NLL, patience {PATIENCE}", "max_epochs": MAX_EPOCHS,
                      "grad_clip": CLIP, "batch": BATCH},
        "training": out["info"], "n_train": out["n_train"],
        "git_commit_uafeg": git_commit(REPO_B), "git_commit_canonical": git_commit(canonical_repository()),
        "inputs": inputs,
        "outputs_sha256": {"checkpoint.pt": hashlib.sha256((run_dir / "checkpoint.pt").read_bytes()).hexdigest(),
                           "logits.npz": hashlib.sha256((run_dir / "logits.npz").read_bytes()).hexdigest()},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
        "wall_seconds": round(wall, 1), "result_type": "REDUCED_SCALE_TRAINING",
        "test_logits_note": "computed once at the end of this run; read only by experiments.thesis_evaluate",
    }
    (run_dir / "record.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    (SUMMARY_DIR).mkdir(parents=True, exist_ok=True)
    small = {k: record[k] for k in ("system", "seed", "config", "n_train", "git_commit_uafeg", "outputs_sha256",
                                    "wall_seconds")}
    small["best_val_nll"] = out["info"]["best_val_nll"]
    small["epochs_run"] = out["info"]["epochs_run"]
    small["packets_manifest_sha256"] = inputs["packets_manifest_sha256"]
    (SUMMARY_DIR / f"{system}_seed{seed}.json").write_text(json.dumps(small, indent=1, sort_keys=True) + "\n")


def run_one(system: str, cfg: dict, seed: int, cache: dict) -> dict:
    if system in GRAPH_SYSTEMS:
        return train_graph(system, cfg, seed, cache)
    if system in FLAT_SYSTEMS:
        return train_flat(system, cfg, seed, cache)
    if system == "B7":
        mods = ("text", "visual", "forensic") if forensic_available(cache) else ("text", "visual")
        return train_dbf(cfg, seed, cache, mods)
    raise ValueError(system)


def cmd_search(args) -> None:
    cache: dict = {}
    verify_inputs()
    results = {}
    for system in ("B6",):  # GATv2 systems (P1/P2) are out of the final scope (2026-11-20)
        for i, cfg in enumerate(SEARCH_GRID):
            t0 = time.time()
            out = train_graph(system, cfg, seed=0, cache=cache)
            results.setdefault(system, []).append({"config": cfg, "best_val_nll": out["info"]["best_val_nll"],
                                                   "epochs": out["info"]["epochs_run"]})
            print(system, i, cfg, round(out["info"]["best_val_nll"], 4), f"{time.time() - t0:.0f}s", flush=True)
    for system in ("B1", "B2", "B4", "B7"):
        for cfg in FLAT_GRID:
            out = run_one(system, cfg, 0, cache)
            results.setdefault(system, []).append({"config": cfg, "best_val_nll": out["info"]["best_val_nll"]})
            print(system, cfg, round(out["info"]["best_val_nll"], 4), flush=True)
    # pooling chosen jointly (same for B6, P1, P2): lowest mean of per-system best val NLL
    pooling_score = {p: np.mean([min(r["best_val_nll"] for r in results[s] if r["config"]["pooling"] == p)
                                 for s in ("B6",)]) for p in ("mean", "attention")}
    pooling = min(pooling_score, key=pooling_score.get)
    selected = {}
    for s in ("B6",):
        cands = [r for r in results[s] if r["config"]["pooling"] == pooling]
        selected[s] = min(cands, key=lambda r: r["best_val_nll"])["config"]
    for s in ("B1", "B2", "B4", "B7"):
        selected[s] = min(results[s], key=lambda r: r["best_val_nll"])["config"]
    selected["B3"] = selected["B1"]
    summary = {"criterion": "minimum validation NLL, seed 0 (B6: 8 configs; flat: 2 configs)",
               "budget": {"graph_systems": len(SEARCH_GRID), "flat_systems": len(FLAT_GRID)},
               "pooling_scores": pooling_score, "pooling_selected": pooling, "selected": selected,
               "all": results}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "hparam_search.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pooling": pooling, "selected": selected}, indent=1))


PRELIM_CONFIG = {"dropout": 0.1, "heads": 2, "pooling": "attention", "lr": 2e-3, "dim": 64, "hidden": 64, "layers": 2}


def cmd_train(args) -> None:
    global WITH_TEST, RUNS_DIR, SUMMARY_DIR, MAX_EPOCHS, PATIENCE
    if args.validation_only:
        # Preliminary mode: fixed thesis-default config (no search), short epochs, NO test.
        WITH_TEST = False
        RUNS_DIR = ART / "runs_validation_prelim"
        SUMMARY_DIR = RESULTS / "validation_prelim" / "runs"
        MAX_EPOCHS, PATIENCE = args.max_epochs, 3
        selected = {s: (PRELIM_CONFIG if s in GRAPH_SYSTEMS else {"dropout": 0.1, "lr": 2e-3, "hidden": 64})
                    for s in list(GRAPH_SYSTEMS) + list(FLAT_SYSTEMS) + ["B7"]}
    else:
        selected = json.loads((RESULTS / "hparam_search.json").read_text())["selected"]
    inputs = verify_inputs()
    cache: dict = {}
    for system in args.systems:
        if system == "B3" and not forensic_available(cache):
            note = {"system": "B3", "status": "NOT_RUN", "reason": "forensic specialist NOT_RUN: no forensic evidence"}
            SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
            (SUMMARY_DIR / "B3_NOT_RUN.json").write_text(json.dumps(note, indent=1) + "\n")
            print(note)
            continue
        for seed in args.seeds:
            if (RUNS_DIR / system / f"seed{seed}" / "record.json").exists() and not args.force:
                print("skip existing", system, seed)
                continue
            t0 = time.time()
            out = run_one(system, selected[system], seed, cache)
            save_run(system, seed, selected[system], out, inputs, time.time() - t0)
            print(system, seed, round(out["info"]["best_val_nll"], 4), out["info"]["epochs_run"],
                  f"{time.time() - t0:.0f}s", flush=True)


def main() -> int:
    torch.set_num_threads(1)
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("search")
    t = sub.add_parser("train")
    t.add_argument("--systems", nargs="+", required=True)
    t.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    t.add_argument("--force", action="store_true")
    t.add_argument("--validation-only", action="store_true", help="preliminary: no test at all")
    t.add_argument("--max-epochs", type=int, default=15)
    args = ap.parse_args()
    {"search": cmd_search, "train": cmd_train}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
