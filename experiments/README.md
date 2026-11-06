# Ejecutores de experimentos

[Instalación y ejemplo completo](../README.md) · [Preparación de datos reales](../docs/TRAINING.md)

| Archivo | Función |
|---|---|
| `train.py` | CLI B6/P1/P2; guarda checkpoint y registro. Solo carga `--data smoke`. |
| `evaluate.py` | Carga checkpoint y calcula métricas sobre datos sintéticos; carga real pendiente. |
| `common.py` | Configuraciones, registro de modelos y manifiesto de corrida. |
| `smoke_data.py` | Paquetes sintéticos para comprobar ejecución. |
| `ablations.py` | Modificaciones controladas; A4/A5 no implementadas fallan explícitamente. |

Ejecutar como módulos desde la raíz de `uafeg`, con `.venv/bin/python -m experiments.train` o `experiments.evaluate`. No lanzar los archivos directamente. Consultar `--help` para argumentos. `artifacts/checkpoints/` y `artifacts/runs/` son salidas locales fuera de Git.

La CLI de evaluación actual no reconstruye automáticamente una ablación desde metadatos del checkpoint. Para una corrida con ablación se necesita garantizar una configuración de evaluación equivalente antes de interpretar sus cifras.
