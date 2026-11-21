"""Integra B8 (Qwen2-VL-2B-Instruct congelado, zero-shot, Kaggle GPU) como una corrida más.

    python -m experiments.b8_import

Lee artifacts/b8/b8_scores.jsonl del Repo A (log-puntajes de las opciones A/B/C por reclamo,
sin etiquetas), verifica su hash contra la procedencia y escribe
artifacts/thesis/runs/B8/seed0/logits.npz (val, test) en el orden de los splits. B8 no tiene
parámetros entrenables ni semillas: M = 1. La calibración T* y la abstención se ajustan en
validación como para todos los sistemas (experiments.thesis_evaluate).
"""
import hashlib
import json

import numpy as np

from experiments.thesis_data import ART, load_split
from src.evaluation.metrics import CLASS_ORDER
from uafeg.benchmark import canonical_repository

OPTION = {"SUPPORTED": 0, "SUSPICIOUS": 1, "INSUFFICIENT": 2}  # A, B, C in the prompt


def main():
    a = canonical_repository() / "artifacts/b8"
    prov = json.loads((a / "b8.provenance.json").read_text())
    raw = (a / "b8_scores.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == prov["output_sha256"]
    scores = {r["claim_id"]: r["logscores_ABC"] for r in map(json.loads, raw.decode().splitlines())}
    out = {}
    for split, key in (("validation", "val"), ("test", "test")):
        ids = [m["claim_id"] for m in load_split(split).meta]
        out[key] = np.array([[scores[c][OPTION[k]] for k in CLASS_ORDER] for c in ids], dtype=np.float64)
    d = ART / "runs/B8/seed0"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "logits.npz", **out)
    rec = {"system": "B8", "seed": 0, "M": 1, "provenance": prov, "class_order": list(CLASS_ORDER),
           "outputs_sha256": {"logits.npz": hashlib.sha256((d / "logits.npz").read_bytes()).hexdigest()},
           "note": "frozen zero-shot MLLM; no training, no seeds"}
    (d / "record.json").write_text(json.dumps(rec, indent=1))
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
