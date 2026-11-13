"""Protocolo oráculo (4 combinaciones) para la taxonomía de la tesis.

Diagnóstico, no resultado principal: la percepción oráculo deriva de la
PROCEDENCIA del benchmark (anotaciones HITL de la imagen mostrada, operación y
región de la transformación, clases afirmadas por el generador del relato),
nunca de ``expected_label`` ni de ``condition``.

- Narrativa oráculo: una afirmación por clase de ``metadata.asserted_damage_classes``,
  negada si la transformación niega una clase (``denied_annotated_class``), con
  p one-hot, u = 0, r = 1.
- Visual+forense oráculo (se activan JUNTOS, como pide la tesis): una detección
  por anotación GT de la imagen mostrada (la del donante en una sustitución; sin
  el objeto borrado en un inpainting; coordenadas trasladadas en un recorte), y
  un nodo forense con score 1 y la región editada si la operación es una edición
  localizada, score 0 si no.

Corrige el oráculo forense previo (``uafeg.models.oracle``), que marcaba como
manipulado todo reclamo con ``transformation``; en v3 todos la tienen (C0 usa
``identity``), de modo que nunca discriminaba.
"""
from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.schemas.evidence import (CANONICAL_DAMAGE_CLASSES, DamageAnnotation, ForensicResult,
                                  NarrativeAssertion, NarrativeEvidence, VisualEvidence)
from src.schemas.packet import EvidencePacket
from uafeg.benchmark import canonical_repository

LOCAL_EDITS = {"localized_copy_move", "damage_mask_inpainting", "localized_rgb_inversion"}
GRID = (48, 64)


@lru_cache(maxsize=1)
def _sources() -> dict:
    path = canonical_repository() / "data/processed/damage_specialist/source_manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    out = {}
    for r in rows:
        from PIL import Image
        with Image.open(canonical_repository() / r["path"]) as im:
            out[r["source_image_id"]] = (im.size, r["annotations"])
    return out


def _encode(mask: np.ndarray) -> str:
    return base64.b64encode(np.packbits(mask.astype(np.uint8), axis=None).tobytes()).decode()


def _raster(points_norm) -> np.ndarray:
    import cv2
    m = np.zeros(GRID, dtype=np.uint8)
    pts = np.array([[x * GRID[1], y * GRID[0]] for x, y in points_norm], dtype=np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def oracle_narrative(packet: EvidencePacket) -> NarrativeEvidence | None:
    claim = packet.claim
    if not claim.narrative_text:
        return None
    denied = claim.transformation.parameters.get("denied_annotated_class")
    assertions = []
    for cls in claim.metadata.get("asserted_damage_classes", []):
        p = (0.0, 0.0, 1.0) if cls == denied else (1.0, 0.0, 0.0)
        assertions.append(NarrativeAssertion(
            text="oracle", damage_type=cls, polarity="denied" if cls == denied else "affirmed", p=p, u=0.0,
            coverage=0.25, missing_fields=["part", "severity", "circumstance"], reliability=1.0,
            class_nli={k: (p if k == cls else (0.0, 1.0, 0.0)) for k in CANONICAL_DAMAGE_CLASSES}))
    scores = {k: float(any(a.damage_type == k and a.polarity == "affirmed" for a in assertions))
              for k in CANONICAL_DAMAGE_CLASSES}
    return NarrativeEvidence(raw_scores=scores, normalized_scores=scores, model_name="oracle-provenance",
                             device="cpu", assertions=assertions)


def oracle_visual_forensic(packet: EvidencePacket) -> tuple[VisualEvidence | None, ForensicResult | None]:
    claim = packet.claim
    if not claim.image_path or packet.image_quality is None:
        return None, None
    t = claim.transformation
    params = t.parameters
    shown = t.donor_source_image_ids[0] if t.operation == "same_split_image_substitution" else claim.source_image_id
    (w, h), annotations = _sources()[shown]
    dx = dy = 0.0
    ow, oh = w, h
    if t.operation == "benign_crop":
        dx, dy = params["annotation_translation"]
        ow, oh = params["output_size"]
    removed = params.get("annotation_object_index") if t.operation == "damage_mask_inpainting" else None
    dets = []
    for i, a in enumerate(annotations):
        if i == removed or not a.get("polygon"):
            continue
        pts = [(min(max((x + dx) / ow, 0), 1), min(max((y + dy) / oh, 0), 1)) for x, y in a["polygon"]["points"]]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        dets.append(DamageAnnotation(damage_class=a["damage_class"], confidence=1.0,
                                     bbox_norm=(min(xs), min(ys), max(xs), max(ys)),
                                     area_fraction=min(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys))),
                                     uncertainty=0.0, reliability=1.0, coverage=1.0,
                                     mask_64x48=_encode(_raster(pts))))
    visual = VisualEvidence(detections=dets, model_name="oracle-provenance",
                            image_width=packet.image_quality.width, image_height=packet.image_quality.height)
    if t.operation in LOCAL_EDITS:
        mask = np.zeros(GRID, dtype=bool)
        if "destination_box" in params:
            x1, y1, x2, y2 = params["destination_box"]
            iw, ih = packet.image_quality.width, packet.image_quality.height
            mask[int(y1 / ih * GRID[0]):max(int(y1 / ih * GRID[0]) + 1, int(y2 / ih * GRID[0])),
                 int(x1 / iw * GRID[1]):max(int(x1 / iw * GRID[1]) + 1, int(x2 / iw * GRID[1]))] = True
        elif removed is not None:
            a = annotations[removed]
            mask = _raster([(x / w, y / h) for x, y in a["polygon"]["points"]])
        forensic = ForensicResult(available=True, method="oracle-provenance", score=1.0, reliability=1.0,
                                  uncertainty=0.0, localization_area=float(mask.mean()), mask_64x48=_encode(mask))
    else:
        forensic = ForensicResult(available=True, method="oracle-provenance", score=0.0, reliability=1.0,
                                  uncertainty=0.0, localization_area=0.0)
    return visual, forensic


def build_thesis_oracle_packet(packet: EvidencePacket, oracle_narrative_flag: bool,
                               oracle_visual_forensic_flag: bool) -> EvidencePacket:
    updates = {}
    if oracle_narrative_flag:
        updates["narrative"] = oracle_narrative(packet)
    if oracle_visual_forensic_flag:
        updates["visual"], updates["forensic"] = oracle_visual_forensic(packet)
    return packet.model_copy(update=updates) if updates else packet
