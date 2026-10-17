"""Entrenamiento, evaluacion, checkpoint e inferencia genericos, compartidos
por B6/P1/P2 (cualquier modelo que exponga `forward_batch(packets) ->
logits`). Train ajusta pesos; validation es solo para seleccion de modelo/
early stopping (no se usa aqui para calibracion ni abstencion, eso vive en
src.calibration / src.abstention del repositorio canonico). Test nunca debe
pasarse a estas funciones de entrenamiento.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import CLASS_ORDER


def labels_to_indices(labels: Sequence[str]) -> torch.Tensor:
    index = {label: i for i, label in enumerate(CLASS_ORDER)}
    unknown = sorted(set(labels) - set(index))
    if unknown:
        raise ValueError(f"Etiquetas fuera de CLASS_ORDER: {unknown}")
    return torch.tensor([index[label] for label in labels], dtype=torch.long)


def train(
    model: nn.Module,
    train_packets: Sequence,
    train_labels: Sequence[str],
    val_packets: Sequence | None = None,
    val_labels: Sequence[str] | None = None,
    epochs: int = 1,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    """Entrena `model` sobre train_packets/train_labels (fit real de pesos).
    val_packets/val_labels, si se dan, SOLO se usan para reportar
    val_loss/val_accuracy por epoca (seleccion de modelo), nunca para
    actualizar pesos. No tocar test aqui."""
    if len(train_packets) != len(train_labels):
        raise ValueError("train_packets y train_labels deben tener la misma longitud")
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    target = labels_to_indices(train_labels)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model.forward_batch(list(train_packets))
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        history["train_loss"].append(float(loss.item()))

        if val_packets is not None and val_labels is not None and len(val_packets) > 0:
            model.eval()
            with torch.no_grad():
                val_logits = model.forward_batch(list(val_packets))
                val_target = labels_to_indices(val_labels)
                val_loss = loss_fn(val_logits, val_target)
                val_pred = val_logits.argmax(dim=-1)
                val_accuracy = float((val_pred == val_target).float().mean().item())
            history["val_loss"].append(float(val_loss.item()))
            history["val_accuracy"].append(val_accuracy)
    return history


def predict_proba(model: nn.Module, packets: Sequence) -> np.ndarray:
    """Probabilidades softmax (RAW, no calibradas) en orden CLASS_ORDER.
    Para probabilidades calibradas usar src.calibration.calibrate sobre los
    logits de este mismo modelo."""
    model.eval()
    with torch.no_grad():
        logits = model.forward_batch(list(packets))
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.numpy()


def predict_logits(model: nn.Module, packets: Sequence) -> np.ndarray:
    """Logits crudos (pre-softmax), entrada requerida por temperature
    scaling (src.calibration.calibrate)."""
    model.eval()
    with torch.no_grad():
        logits = model.forward_batch(list(packets))
    return logits.numpy()


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict) -> None:
    """Guarda state_dict + metadata (system, config, seed, run_id, commit,
    benchmark_version, benchmark_hashes, timestamp, metrics). No se trackea
    en git: ver .gitignore de checkpoints/."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str))


def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    """Carga state_dict en `model` (in-place) y devuelve la metadata
    guardada junto al checkpoint."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint["metadata"]
