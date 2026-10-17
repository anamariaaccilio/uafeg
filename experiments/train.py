"""CLI de entrenamiento comun para B6/P1/P2.

    python -m experiments.train --system B6 --config configs/b6_deepsets.yaml
    python -m experiments.train --system P1 --config configs/p1_graph.yaml
    python -m experiments.train --system P2 --config configs/p2_uafeg.yaml --ablation no_uncertainty

El benchmark real (v2/v3) no esta disponible para entrenamiento (v2 tiene
specialists en null; v3 esta PENDING_SCIENTIFIC_QA, ver
PROJECT_COMPLETION_GAP.md): `--data` solo acepta 'smoke' hoy. Cualquier
metrica de esta corrida se marca result_type=SMOKE_RESULT, nunca
REAL_RESULT.

Train ajusta pesos sobre train; `--val-data` (si se da) SOLO se reporta
para seleccion de modelo (val_loss/val_accuracy por epoca), nunca actualiza
pesos. No existe `--test-data`: test solo se evalua con experiments.evaluate.
"""
from __future__ import annotations

import argparse
import sys

from uafeg.models.training import save_checkpoint, train as train_model

from experiments.ablations import apply_ablation
from experiments.common import build_model, build_run_manifest, load_benchmark_lock, load_config, new_run_id, save_run_manifest
from experiments.smoke_data import make_smoke_dataset


def _load_dataset(spec: str, seed: int):
    if spec == "smoke":
        return make_smoke_dataset(n=40, seed=seed)
    raise NotImplementedError(
        "Carga de benchmark real NO disponible: v2 tiene specialists en null y v3 esta "
        "PENDING_SCIENTIFIC_QA (ver PROJECT_COMPLETION_GAP.md). Use --data smoke."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=["B6", "P1", "P2"])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--ablation",
        default=None,
        choices=[
            "no_gate", "no_forensic", "graph_to_deepsets",  # A1, A2, A3 (tesis)
            "no_edge_attributes", "no_benign_no_ood_training",  # A4, A5 (tesis, NOT_IMPLEMENTED)
            "no_uncertainty", "no_explicit_relations",  # extra, no corresponden a A1-A5
        ],
    )
    parser.add_argument("--data", default="smoke")
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["system"] = args.system
    training_cfg = config.get("training", {})
    seed = args.seed if args.seed is not None else training_cfg.get("seed", 42)
    epochs = args.epochs if args.epochs is not None else training_cfg.get("epochs", 1)
    lr = training_cfg.get("learning_rate", 1e-3)

    train_packets, train_labels = _load_dataset(args.data, seed)
    val_packets = val_labels = None
    if args.val_data:
        val_packets, val_labels = _load_dataset(args.val_data, seed + 1)

    model_kwargs = dict(config.get("model", {}))
    if args.ablation:
        if args.ablation == "no_gate":
            if args.system != "P1":
                raise SystemExit("La ablacion no_gate (A1) equivale a P1: pase --system P1 (ver configs/ablations/no_gate.yaml)")
        elif args.ablation == "graph_to_deepsets":
            if args.system != "B6":
                raise SystemExit("La ablacion graph_to_deepsets (A3) equivale a B6: pase --system B6 (ver configs/ablations/graph_to_deepsets.yaml)")
        else:
            model_kwargs, train_packets = apply_ablation(args.ablation, model_kwargs, train_packets)
            if val_packets is not None:
                _, val_packets = apply_ablation(args.ablation, {}, val_packets)
    config["model"] = model_kwargs

    model = build_model(config)
    history = train_model(model, train_packets, train_labels, val_packets, val_labels, epochs=epochs, lr=lr, seed=seed)

    run_id = new_run_id(args.system)
    checkpoint_path = args.checkpoint_out or f"artifacts/checkpoints/{run_id}.pt"
    metadata = {"system": args.system, "ablation": args.ablation, "seed": seed, "run_id": run_id, "data": args.data}
    save_checkpoint(model, checkpoint_path, metadata)

    try:
        benchmark_lock = load_benchmark_lock()
    except FileNotFoundError:
        benchmark_lock = {}

    result_type = "SMOKE_RESULT" if args.data == "smoke" else "REAL_RESULT"
    manifest = build_run_manifest(
        system=args.system,
        config_path=args.config,
        seed=seed,
        run_id=run_id,
        metrics={
            "train_loss_final": history["train_loss"][-1],
            "val_loss": history.get("val_loss"),
            "val_accuracy": history.get("val_accuracy"),
            "result_type": result_type,
        },
        checkpoint_path=checkpoint_path,
        benchmark_lock=benchmark_lock,
    )
    manifest_path = save_run_manifest(manifest)
    print(f"[experiments.train] run_id={run_id} ablation={args.ablation} checkpoint={checkpoint_path}")
    print(f"[experiments.train] manifest={manifest_path}")
    print(f"[experiments.train] result_type={result_type} (ver PROJECT_COMPLETION_GAP.md: benchmark real aun bloqueado)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
