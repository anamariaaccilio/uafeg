# UAFEG

**Uncertainty-Aware Forensic Evidence Graph**: repositorio de la propuesta de tesis sobre verificación robusta de reclamos de seguros vehiculares.

El milestone implementado es únicamente `EvidencePacket → EvidenceGraph → validación`. El grafo mínimo es una estrella determinista: un nodo claim, un nodo por modalidad disponible y aristas dirigidas `evidence_for` hacia claim. Valida tipos, IDs únicos, existencia de nodos, correspondencia de evidencias y aristas completas. No realiza predicciones, entrenamiento, GATv2 ni propagación aprendida.

La hipótesis investigada combina representación de evidencias a nivel de reclamo y propagación consciente de confiabilidad, evaluada bajo contradicción, manipulación, evidencia faltante y degradación. Su novedad y efectividad deben demostrarse experimentalmente; no se afirman resultados.

Son componentes existentes de literatura: GAT/GATv2, DeepSets, NLI, encoders visuales, damage detection/segmentation, modelos forenses (incluido TruFor), concatenación, late fusion, temperature scaling, deep ensembles, calibración estándar, Discounted Belief Fusion y el concepto genérico de grafo multimodal. No se presentan como invenciones de UAFEG.

- **P1**: Evidence Graph + modelo de grafo, sin reliability gate. NOT IMPLEMENTED; solo configuración de intención.
- **P2**: P1 + mecanismo reliability-aware de UAFEG. NOT IMPLEMENTED; solo configuración de intención.
- **MVP disponible**: construcción/validación de grafo sin modelo. No equivale a P1 completo.

El repositorio hermano `vehicle-claim-verification-baselines` es la única fuente de ClaimRecord, EvidencePacket, labels, manifest, splits, condiciones y evaluación. Se consume por instalación editable; no hay schemas duplicados ni submodules. No se crearon carpetas vacías de modelos/reliability/uncertainty.

```bash
cd '/home/ana/Documentos/tesis 2/uafeg'
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
env -u PYTHONPATH .venv/bin/python -m pytest -q
```

Ejemplo mínimo, sin modelos ni datos raw:

```python
from src.schemas.claim import ClaimRecord
from src.schemas.packet import EvidencePacket
from uafeg import build_graph

packet = EvidencePacket(claim=ClaimRecord(claim_id="example"))
graph = build_graph(packet)  # solo nodo claim: evidencias ausentes
```

Las fixtures son sintéticas y compatibles con el schema canónico. `None` representa evidencia ausente; visual presente con detecciones vacías no equivale a ausencia. Forense `available=False` no crea nodo. Labels, condición experimental y paths no se copian al grafo. Los IDs solo son metadata de trazabilidad; futuros modelos no deben usarlos como features. El grafo usa salidas de especialistas existentes, sin fabricar scores ni reliability.

`benchmark.lock.json` fija el commit real de Repo A, schema 0.2.0, manifest de 127 reclamos controlados, splits y EvidencePackets con sus SHA256 reales. El dataset es fotografía real de daño vehicular más escenarios sintéticos, no expedientes de una aseguradora. Las salidas NLI/visual/forense/calidad de los packets son null; no se inventa evidencia para construir grafos.

La generación, taxonomía C0–C10, narrativas, transformaciones, auditoría de duplicados y benchmark pertenecen exclusivamente a Repo A. Esta versión implementa un subconjunto; no equivale a completar C0–C10 ni a validar científicamente modelos. El test de compatibilidad carga los packets reales con el schema de A y comprueba que el grafo no copie labels.

El lock se obtiene mediante lectura de artefactos reales de un Repo A limpio y commiteado:

```bash
env -u PYTHONPATH .venv/bin/python scripts/pin_benchmark.py --write
env -u PYTHONPATH .venv/bin/python scripts/pin_benchmark.py --check
```

No editar hashes a mano. Cambiar el benchmark requiere nueva validación y actualizar el lock explícitamente.

Para reproducir en otra máquina, colocar ambos repos como hermanos y hacer checkout del commit indicado en el lock antes de instalar. Cambiar el contrato requiere actualizar el lock de forma explícita y repetir validación. Los wheels de Repo A no son el mecanismo de distribución soportado para este MVP; su configuración reside en el checkout editable.

Las comparaciones B6 vs P1, P1 vs P2 y P2 vs ablations usarán exactamente el mismo manifest, splits, packets, labels, métricas y test set. Ver el protocolo canónico de Repo A. Ya hay un primer benchmark de escenarios controlados; faltan revisión científica, mayor cobertura, métricas y sistemas de comparación antes de entrenar.

CI define Python 3.11 + instalación editable + pytest sin modelos ni datasets. Se activa manualmente con `workflow_dispatch` indicando el repositorio canónico publicado (`owner/name`), pues aún no hay remotos reales configurados; el commit se obtiene del lock. No se ha ejecutado Actions. No requiere secretos adicionales para repositorios públicos; el acceso remoto privado entre repositorios no está configurado ni validado.

No se versionan raw datasets, weights, `.venv`, `.env` ni caches. No se ha iniciado entrenamiento. La siguiente fase experimental requiere autorización.

El lock actual consume el benchmark completo v0.2.0 (`benchmark_v2`) del repo
canónico, con schema 0.3.0 y las once condiciones. C10 es exclusivamente de test
(etiqueta OOD), separada de las tres clases de decisión. Los 226 EvidencePackets
mantienen especialistas null: el grafo mínimo no recibe etiquetas como features.
No se han implementado P1/P2 ni entrenado modelos.
