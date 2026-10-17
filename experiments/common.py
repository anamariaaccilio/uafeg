"""Utilidades compartidas por experiments.train y experiments.evaluate:
registro de sistemas (B6/P1/P2), carga de config, y manifest de corrida
(system, config, seed, run_id, commit, benchmark_version, hashes del
contrato, timestamp, metrics, checkpoint path). No se trackean pesos
grandes en git; solo el manifest JSON."""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from uafeg.models.b6_deepsets import DeepSetsClassifier
from uafeg.models.p1_graph import GraphClassifier
from uafeg.models.p2_uafeg import ReliabilityGatedGraphClassifier

MODEL_REGISTRY = {
    "B6": DeepSetsClassifier,
    "P1": GraphClassifier,
    "P2": ReliabilityGatedGraphClassifier,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_model(config: dict):
    """Construye el modelo indicado en config['system'] con config['model']
    como kwargs. No acepta claves de ablacion (disable_gate_terms,
    include_cross_relations) aqui: ver experiments.ablations.apply_ablation
    para eso, de forma explicita y separada del config base."""
    system = config.get("system")
    if system not in MODEL_REGISTRY:
        raise ValueError(f"Sistema desconocido: {system!r}. Validos: {list(MODEL_REGISTRY)}")
    model_cls = MODEL_REGISTRY[system]
    model_kwargs = dict(config.get("model", {}))
    return model_cls(**model_kwargs)


def current_git_commit(repo_path: Path | None = None) -> str | None:
    repo_path = repo_path or REPO_ROOT
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def load_benchmark_lock(lock_path: Path | None = None) -> dict:
    lock_path = lock_path or (REPO_ROOT / "benchmark.lock.json")
    return json.loads(Path(lock_path).read_text())


def new_run_id(system: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{system}-{timestamp}-{uuid.uuid4().hex[:8]}"


def build_run_manifest(
    system: str,
    config_path: str,
    seed: int,
    run_id: str,
    metrics: dict,
    checkpoint_path: str | None,
    benchmark_lock: dict,
) -> dict:
    return {
        "system": system,
        "config": str(config_path),
        "seed": seed,
        "run_id": run_id,
        "commit": current_git_commit(),
        "benchmark_version": benchmark_lock.get("benchmark_version"),
        "benchmark_schema_version": benchmark_lock.get("schema_version"),
        "benchmark_git_commit": benchmark_lock.get("git_commit"),
        "benchmark_contract_hashes": benchmark_lock.get("contract_files_sha256"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


def save_run_manifest(manifest: dict, runs_dir: Path | None = None) -> Path:
    runs_dir = runs_dir or RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{manifest['run_id']}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path
