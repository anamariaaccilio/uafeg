"""CLI de evaluacion comun para B6/P1/P2.

    python -m experiments.evaluate --system B6 --config configs/b6_deepsets.yaml --checkpoint path.pt

SOLO evaluacion final: nunca ajusta thresholds/temperatura aqui (eso se
hace en validation, ver src.calibration / src.abstention). El benchmark
real no esta disponible (ver PROJECT_COMPLETION_GAP.md); `--data` solo
acepta 'smoke' hoy, y el resultado se marca result_type=SMOKE_RESULT.
"""
from __future__ import annotations

import argparse
import json
import sys

from src.evaluation.metrics import CLASS_ORDER, compute_all_metrics
from uafeg.models.training import load_checkpoint, predict_proba

from experiments.common import build_model, load_config
from experiments.smoke_data import make_smoke_dataset


def _load_dataset(spec: str, seed: int):
    if spec == "smoke":
        return make_smoke_dataset(n=40, seed=seed)
    raise NotImplementedError(
        "Evaluacion sobre benchmark real NO disponible todavia (ver PROJECT_COMPLETION_GAP.md). "
        "Use --data smoke."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", required=True, choices=["B6", "P1", "P2"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="smoke")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["system"] = args.system
    model = build_model(config)
    load_checkpoint(model, args.checkpoint)

    packets, labels = _load_dataset(args.data, args.seed)
    probabilities = predict_proba(model, packets)
    predictions = [CLASS_ORDER[i] for i in probabilities.argmax(axis=1)]
    metrics = compute_all_metrics(labels, predictions, probabilities)

    result_type = "SMOKE_RESULT" if args.data == "smoke" else "REAL_RESULT"
    print(f"[experiments.evaluate] system={args.system} checkpoint={args.checkpoint} result_type={result_type}")
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
