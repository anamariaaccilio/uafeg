"""Ablaciones de UAFEG.

Corregido en la pasada de consistencia de 2026-10-17 contra el .tex real de
la tesis (capitulo4.tex, seccion "Estudios de ablacion"), que define
EXACTAMENTE cinco ablaciones A1-A5:

    A1. Sin reliability gating: se fija g_j=1 en todos los nodos.
    A2. Sin evidencia forense: se quitan los nodos forensic.
    A3. GATv2 reemplazado por DeepSets con los mismos EvidencePacket.
    A4. Sin atributos de arista: las relaciones conservan su tipo pero
        pierden los puntajes continuos.
    A5. Entrenamiento sin controles benignos (C8) ni condiciones OOD (C10).

Mapeo a este modulo:
    - A1 == 'no_gate' (ejecutar system=P1 directamente; probado identico a
      P2 con gate forzado a 1 en tests/test_p2_uafeg.py).
    - A2 == 'no_forensic' (strip_forensic).
    - A3 == 'graph_to_deepsets' (ejecutar system=B6 directamente: "mismos
      EvidencePacket, sin estructura" es, por definicion, B6).
    - A4 y A5 estan marcadas status=NOT_IMPLEMENTED mas abajo: A4
      requeriria que las aristas lleven un vector de atributos continuos
      [p_support, p_contradiction, p_unknown, sim_sem, overlap_spatial]
      (hoy solo existe el tipo categorico de relacion, ver
      src/uafeg/relations.py), y A5 requeriria filtrar el dataset de
      entrenamiento por condicion C0-C10 (hoy experiments.train no conoce
      la condicion de cada EvidencePacket). Ambas son NUEVA funcionalidad,
      fuera de alcance de una pasada de consistencia — no se inventan ni se
      fingen implementadas.

Dos ablaciones EXTRA (no correspondientes a ninguna A1-A5 de la tesis) se
mantienen porque ya estaban implementadas y probadas antes de esta
correccion, y no son incorrectas, solo no son parte del protocolo
preespecificado de la tesis: 'no_uncertainty' (fuerza u_j=0 sin tocar los
demas factores del gate) y 'no_explicit_relations' (elimina las aristas
narrativa-visual por completo, mas agresivo que A4). Se marcan
explicitamente como EXTRA para no confundirlas con el protocolo
confirmatorio H1-H3.

Ninguna ablacion toca expected_label."""
from __future__ import annotations

from src.schemas.packet import EvidencePacket


def strip_forensic(packets: list[EvidencePacket]) -> list[EvidencePacket]:
    """A2 / 'no_forensic': fuerza forensic=None en todos los packets,
    independientemente de si el especialista forense estaba disponible."""
    return [p.model_copy(update={"forensic": None}) for p in packets]


ABLATIONS: dict[str, dict] = {
    "no_gate": {
        "thesis_id": "A1",
        "description": "P2 sin reliability gate es, por construccion, P1 (g_j=1 para todos los nodos). Ejecutar con --system P1.",
        "model_kwargs": {},
        "equivalent_system": "P1",
    },
    "no_forensic": {
        "thesis_id": "A2",
        "description": "Elimina la modalidad forense de todos los EvidencePackets antes de construir el grafo.",
        "model_kwargs": {},
        "data_transform": strip_forensic,
        "equivalent_system": "P2",
    },
    "graph_to_deepsets": {
        "thesis_id": "A3",
        "description": "GATv2 reemplazado por DeepSets con los mismos EvidencePacket. Ejecutar con --system B6.",
        "model_kwargs": {},
        "equivalent_system": "B6",
    },
    "no_edge_attributes": {
        "thesis_id": "A4",
        "status": "NOT_IMPLEMENTED",
        "description": (
            "Sin atributos de arista (conservar tipo, perder puntajes continuos). Requiere que las "
            "aristas lleven [p_support, p_contradiction, p_unknown, sim_sem, overlap_spatial] "
            "(eq:aristas de la tesis); hoy src/uafeg/relations.py solo produce el tipo categorico. "
            "No implementado en esta pasada (requiere nueva funcionalidad)."
        ),
    },
    "no_benign_no_ood_training": {
        "thesis_id": "A5",
        "status": "NOT_IMPLEMENTED",
        "description": (
            "Entrenamiento sin controles benignos (C8) ni condiciones OOD (C10). Requiere que "
            "experiments.train filtre por condicion C0-C10 por EvidencePacket; hoy el dataset de "
            "entrenamiento no transporta la condicion. No implementado en esta pasada (requiere "
            "nueva funcionalidad)."
        ),
    },
    "no_uncertainty": {
        "thesis_id": None,
        "description": (
            "EXTRA (no corresponde a ninguna A1-A5 de la tesis): reliability gate sin el termino "
            "(1 - u_j), fuerza u_j=0 para todos los nodos de evidencia."
        ),
        "model_kwargs": {"disable_gate_terms": frozenset({"uncertainty"})},
        "equivalent_system": "P2",
    },
    "no_explicit_relations": {
        "thesis_id": None,
        "description": (
            "EXTRA (no corresponde a ninguna A1-A5 de la tesis; mas agresiva que A4): elimina por "
            "completo las aristas de relacion narrativa-visual (support/contradiction/unknown), "
            "dejando solo las aristas evidence_for (grafo estrella)."
        ),
        "model_kwargs": {"include_cross_relations": False},
        "equivalent_system": "P2",
    },
}


def apply_ablation(name: str, model_kwargs: dict, packets: list[EvidencePacket]) -> tuple[dict, list[EvidencePacket]]:
    """Devuelve (model_kwargs actualizado, packets transformados) para la
    ablacion `name`. No muta los argumentos de entrada. Lanza
    NotImplementedError explicito para ablaciones con status=NOT_IMPLEMENTED
    (A4, A5) en vez de fingir un resultado."""
    if name not in ABLATIONS:
        raise ValueError(f"Ablacion desconocida: {name!r}. Validas: {list(ABLATIONS)}")
    spec = ABLATIONS[name]
    if spec.get("status") == "NOT_IMPLEMENTED":
        raise NotImplementedError(
            f"Ablacion {name!r} ({spec.get('thesis_id')}) no esta implementada: {spec['description']}"
        )
    updated_kwargs = {**model_kwargs, **spec.get("model_kwargs", {})}
    updated_packets = spec["data_transform"](packets) if "data_transform" in spec else list(packets)
    return updated_kwargs, updated_packets
