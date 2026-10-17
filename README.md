# UAFEG

**Uncertainty-Aware Forensic Evidence Graph**: repositorio de la propuesta de tesis sobre
verificación robusta de reclamos de seguros vehiculares. Combina evidencia narrativa, visual y
forense en un grafo, con una compuerta de confiabilidad que modula cuánto pesa cada pieza de
evidencia según su cobertura, calidad, confiabilidad e incertidumbre.

El repositorio hermano `vehicle-claim-verification-baselines` (Repo A) es la única fuente de
`ClaimRecord`, `EvidencePacket`, labels, manifest, splits, condiciones C0–C10 y evaluación
compartida (métricas, calibración, incertidumbre, abstención, bootstrap). Se consume por
instalación editable; no hay schemas duplicados.

## B6 → P1 → P2

Tres sistemas comparten exactamente el mismo encoder de evidencia, el mismo espacio de labels
(SUPPORTED/SUSPICIOUS/INSUFFICIENT) y el mismo protocolo de entrenamiento, para que la
comparación aísle una sola variable cada vez:

- **B6** (`src/uafeg/models/b6_deepsets.py`): evidence encoders → pooling permutation-invariante
  (DeepSets) → clasificador. Sin grafo explícito.
- **P1** (`src/uafeg/models/p1_graph.py`): `EvidenceGraph` → atención relacional GATv2
  (implementación propia, sin `torch_geometric`) → pooling → el mismo clasificador que B6. Sin
  reliability gate. B6 vs P1 aísla el aporte de la estructura de grafo.
- **P2** (`src/uafeg/models/p2_uafeg.py`): reutiliza P1 como base; la única diferencia es el
  reliability gate `g_j = clip(c_j·q_j·r_j·(1-u_j), 0, 1)` (`src/uafeg/models/gate.py`), aplicado
  al mensaje de cada nodo de evidencia antes de la atención. P1 vs P2 aísla el aporte de la
  compuerta.

**Estado: `IMPLEMENTED_CODE_NOT_TRAINED` para los tres.** El código, el forward/train/checkpoint/
inferencia y los tests son reales (ver `uafeg/docs/IMPLEMENTATION_STATUS.md` para el detalle
completo, incluida la comparación línea por línea contra el `.tex` de la tesis). **Ningún
entrenamiento se ha ejecutado sobre el benchmark real.** Las únicas ejecuciones hechas hasta
ahora usan `--data smoke` (fixtures sintéticas, `experiments/smoke_data.py`) y se marcan
`SMOKE_RESULT` explícitamente — nunca deben leerse como resultado científico.

## Por qué no se ha entrenado todavía

`READY_FOR_MODELS = NO`. El benchmark v2 (el que este repo tiene pinneado) tiene los
`EvidencePacket` congelados con los especialistas en `null` (nunca se ejecutaron antes de
congelar). El candidato v3 de Repo A (807 fuentes, 9.275 claims) está `PENDING_SCIENTIFIC_QA`:
la condición C2 (sustitución) necesita adjudicación humana y C3/C4/C5 necesitan revisión de
realismo perceptual. Ver `../PROJECT_COMPLETION_GAP.md` para el detalle y el plan.

## Instalación y tests

```bash
cd '/home/ana/Documentos/tesis 2/uafeg'
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

## Benchmark pinneado

`benchmark.lock.json` fija el commit real de Repo A, `schema_version`, el manifest/splits/
EvidencePackets de `benchmark_v2` y sus SHA256 reales. Se regenera leyendo artefactos reales de
un Repo A limpio y commiteado — nunca se edita a mano:

```bash
env -u PYTHONPATH .venv/bin/python scripts/pin_benchmark.py --write
env -u PYTHONPATH .venv/bin/python scripts/pin_benchmark.py --check
```

Promover a v3 (o a una versión futura) requiere que Repo A marque esa versión
`READY_FOR_MODELS=YES` primero; este repo no promueve versiones por su cuenta.

## Cómo se entrenará (cuando el benchmark esté listo)

```bash
python -m experiments.train --system {B6,P1,P2} --config configs/{b6_deepsets,p1_graph,p2_uafeg}.yaml
python -m experiments.evaluate --system {B6,P1,P2} --config ... --checkpoint ...
```

Hoy `--data` solo acepta `smoke`; pasar cualquier otro valor lanza `NotImplementedError`
explícito en vez de fingir que hay datos reales disponibles. Las ablaciones A1–A3 de la tesis
están implementadas (`--ablation no_gate|no_forensic|graph_to_deepsets`); A4 y A5 lanzan
`NotImplementedError` a propósito (requieren funcionalidad que no existe todavía). Ver
`uafeg/docs/IMPLEMENTATION_STATUS.md`.

## Qué todavía NO debe interpretarse como resultado

- Cualquier métrica marcada `SMOKE_RESULT` en un manifest de `artifacts/runs/` (fixtures
  sintéticas, no el benchmark real).
- El grafo mínimo (`EvidenceGraph`) no hace razonamiento aprendido por sí solo; eso es trabajo de
  B6/P1/P2 montados encima.
- Ningún número de B6/P1/P2 es comparable a una cifra "final" de tesis mientras
  `READY_FOR_MODELS = NO`.

## Ejemplo mínimo (construcción de grafo, sin modelos)

```python
from src.schemas.claim import ClaimRecord
from src.schemas.packet import EvidencePacket
from uafeg import build_graph

packet = EvidencePacket(claim=ClaimRecord(claim_id="example"))
graph = build_graph(packet)  # solo nodo claim: evidencias ausentes
```

`None` representa evidencia ausente; visual presente con detecciones vacías no equivale a
ausencia. Forense `available=False` no crea nodo. Labels, condición experimental y paths no se
copian al grafo ni a los features de B6/P1/P2 (verificado por tests dedicados).

No se versionan raw datasets, weights, `.venv`, `.env` ni caches, ni checkpoints (`artifacts/`
está en `.gitignore`).
