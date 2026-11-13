# Pipeline completo de la tesis (escala reducida, CPU). Cada paso falla con un mensaje
# claro si falta su prerrequisito. Ejecutar desde uafeg/.
#   make all     verify-lock -> packets -> train -> calibrate/evaluate -> hypotheses -> tables
#   make smoke   versión rápida con fixtures (tests + smoke run sintético)
SHELL := /bin/bash
A := ../vehicle-claim-verification-baselines
PY := env -u PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(abspath .venv/bin/python)
PACKETS := $(A)/artifacts/packets/v3/packets_manifest.json
SEEDS := 0 1 2 3 4
MAIN := B1 B2 B3 B4 B7 B6 P1 P2
ABL := A2 A4 A5 P2_legacy
ORACLE := P2_oracleN P2_oracleVF P2_oracleNVF

.PHONY: all smoke test verify-lock specialists packets search train evaluate hypotheses tables require-%

all: verify-lock packets train evaluate tables

verify-lock:
	$(PY) scripts/pin_benchmark.py --check
	cd $(A) && $(PY) scripts/apply_v3_validity_criteria.py --check

require-yolo:
	@test -f $(A)/artifacts/damage/weights/best.pt || { echo "FALTA artifacts/damage/weights/best.pt: corre 'cd $(A) && python scripts/train_damage_specialist.py'"; exit 1; }

specialists: require-yolo
	cd $(A) && test -f artifacts/specialists/narrative.jsonl || $(PY) scripts/run_specialists.py narrative --threads 3
	cd $(A) && test -f artifacts/specialists/quality.jsonl || $(PY) scripts/run_specialists.py quality
	cd $(A) && test -f artifacts/specialists/visual.jsonl || $(PY) scripts/run_specialists.py visual --threads 3
	@echo "forense: ver src/forensic/trufor_adapter.py (NOT_RUN si no se autoriza ejecutar TruFor)"

packets: specialists
	cd $(A) && $(PY) scripts/build_evidence_packets.py reliability
	cd $(A) && $(PY) scripts/build_evidence_packets.py packets

require-packets:
	@test -f $(PACKETS) || { echo "FALTAN los EvidencePackets reales ($(PACKETS)): corre 'make packets'"; exit 1; }

search: verify-lock require-packets
	test -f results/hparam_search.json || $(PY) -m experiments.thesis_train search

train: search
	$(PY) -m experiments.thesis_train train --systems $(MAIN) $(ABL) $(ORACLE) --seeds $(SEEDS)

require-runs:
	@ls artifacts/thesis/runs/P2/seed4/record.json >/dev/null 2>&1 || { echo "FALTAN corridas entrenadas: corre 'make train'"; exit 1; }

# calibración (T*), umbrales (tau_U, tau_c), test una sola vez, bootstrap y H1-H3 en un paso
evaluate: verify-lock require-runs
	test -f results/evaluation.json || $(PY) -m experiments.thesis_evaluate

hypotheses: evaluate
	$(PY) -c "import json; print(json.dumps(json.load(open('results/evaluation.json'))['hypotheses'], indent=1))"

tables: evaluate
	$(PY) -m experiments.thesis_tables

test:
	$(PY) -m pytest -q
	cd $(A) && $(PY) -m pytest -q -p no:cacheprovider

smoke:
	$(PY) -m pytest -q tests/test_batched_models.py tests/test_thesis_loader.py tests/test_graph.py
	$(PY) -m experiments.train --system P2 --config configs/p2_uafeg.yaml --data smoke --epochs 1 \
	  --checkpoint-out artifacts/checkpoints/p2-smoke.pt
	$(PY) -m experiments.evaluate --system P2 --config configs/p2_uafeg.yaml --data smoke \
	  --checkpoint artifacts/checkpoints/p2-smoke.pt
