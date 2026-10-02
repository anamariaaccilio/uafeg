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

`benchmark.lock.json` registra el commit real y SHA256 de archivos canónicos. La versión 0.1.0 del packet fue introducida explícitamente en esta migración. Manifest, splits persistidos y EvidencePackets congelados son `null / NOT AVAILABLE YET`: no hay dataset real, y la fixture de tests no ocupa su lugar. El test de compatibilidad verifica commit, versión y hashes contra el checkout realmente importado. Esto fija el contrato del MVP, no un benchmark experimental terminado.

Para reproducir en otra máquina, colocar ambos repos como hermanos y hacer checkout del commit indicado en el lock antes de instalar. Cambiar el contrato requiere actualizar el lock de forma explícita y repetir validación. Los wheels de Repo A no son el mecanismo de distribución soportado para este MVP; su configuración reside en el checkout editable.

Las comparaciones B6 vs P1, P1 vs P2 y P2 vs ablations usarán exactamente el mismo manifest, splits, packets, labels, métricas y test set. Ver el protocolo canónico de Repo A. Faltan datos reales, métricas y sistemas de comparación antes de entrenar.

CI define Python 3.11 + instalación editable + pytest sin modelos ni datasets. Se activa manualmente con `workflow_dispatch` indicando el repositorio canónico publicado (`owner/name`), pues aún no hay remotos reales configurados; el commit se obtiene del lock. No se ha ejecutado Actions. No requiere secretos adicionales para repositorios públicos; el acceso remoto privado entre repositorios no está configurado ni validado.

No se versionan raw datasets, weights, `.venv`, `.env` ni caches. No se ha iniciado entrenamiento. La siguiente fase experimental requiere autorización.
