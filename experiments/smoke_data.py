"""Dataset sintetico de fixtures para smoke tests de experiments.train /
experiments.evaluate. NUNCA debe reportarse como benchmark real: el
benchmark real (v2/v3) sigue bloqueado para modelos (ver
PROJECT_COMPLETION_GAP.md). Cualquier metrica calculada sobre este dataset
es SMOKE_RESULT, nunca REAL_RESULT."""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import CLASS_ORDER
from src.schemas.claim import ClaimRecord
from src.schemas.evidence import DamageAnnotation, DecisionLabel, NarrativeEvidence, VisualEvidence
from src.schemas.packet import EvidencePacket


def make_smoke_dataset(n: int = 40, seed: int = 42) -> tuple[list[EvidencePacket], list[str]]:
    """Genera n EvidencePackets sinteticos deterministas (por seed) con
    narrativa+visual (a veces solo narrativa), cubriendo las 3 clases de
    decision. source_image_id agrupa cada 3 claims para poder ejercitar
    bootstrap agrupado en smoke tests."""
    rng = np.random.default_rng(seed)
    packets: list[EvidencePacket] = []
    labels: list[str] = []
    for i in range(n):
        label = CLASS_ORDER[rng.integers(0, len(CLASS_ORDER))]
        damage_class = "dent" if rng.random() > 0.5 else "scratch"
        narrative = NarrativeEvidence(
            raw_scores={damage_class: 0.85},
            normalized_scores={damage_class: 0.85},
            model_name="smoke-fixture",
            device="cpu",
        )
        visual = None
        if rng.random() > 0.2:
            visual_class = damage_class if rng.random() > 0.3 else ("scratch" if damage_class == "dent" else "dent")
            visual = VisualEvidence(
                detections=[DamageAnnotation(damage_class=visual_class, confidence=float(rng.uniform(0.4, 0.95)))],
                model_name="smoke-fixture",
                image_width=64,
                image_height=64,
            )
        packets.append(
            EvidencePacket(
                claim=ClaimRecord(
                    claim_id=f"smoke-{seed}-{i}",
                    source_image_id=f"smoke-source-{seed}-{i // 3}",
                    expected_label=DecisionLabel(label),
                    annotated_damage_classes=[damage_class],
                ),
                narrative=narrative,
                visual=visual,
            )
        )
        labels.append(label)
    return packets, labels
