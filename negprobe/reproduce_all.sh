#!/usr/bin/env bash
# Re-runs every experiment in the paper from scratch and records the environment.
#
#   bash reproduce_all.sh            # everything the machine can do
#   SKIP_GPU=1 bash reproduce_all.sh               # CPU-only machines
#
# Outputs: results/, results_truth/, results_generate/, figures/, logs/, ENVIRONMENT.txt, CHECKSUMS.txt
set -uo pipefail
cd "$(dirname "$0")"

DEVICE=${DEVICE:-auto}
DTYPE=${DTYPE:-auto}
SKIP_GPU=${SKIP_GPU:-0}
DATA=data/negated_data
VOCAB=data/common_vocab_cased.txt
mkdir -p logs

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a logs/reproduce.log; }

run() {  # run <logname> <command...>
  local name=$1; shift
  log "START $name"
  if "$@" > "logs/$name.log" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name (see logs/$name.log)"
  fi
}

# ---------------------------------------------------------------- environment
{
  echo "date (UTC): $(date -u)"
  echo "host: $(uname -a)"
  echo "python: $(python -V 2>&1)"
  echo
  echo "--- pip freeze ---"
  pip freeze
  echo
  echo "--- torch ---"
  python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("mps available:", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
} > ENVIRONMENT.txt 2>&1
log "wrote ENVIRONMENT.txt"

# ---------------------------------------------------------------- data
if [ ! -d "$DATA" ]; then
  log "downloading probes"
  bash get_data.sh > logs/get_data.log 2>&1
fi
python - <<'PY' | tee -a logs/reproduce.log
from negprobe.data import load_pairs
for sub in ("GoogleRE", "TREx"):
    p = load_pairs("data/negated_data", [sub])
    print(f"{sub}: {len(p)} pairs, {len({x.relation for x in p})} relations")
PY

# ---------------------------------------------------------------- A + B
for M in bert-base-cased bert-large-cased; do
  run "run_$M" python -m negprobe.run --model "$M" --data "$DATA" --vocab "$VOCAB" \
      --subsets GoogleRE TREx --out results
done

# The paper's Pythia and Qwen runs: repaired templates, float32, Colab T4 GPU (28 Sep 2026).
# Pythia-410m also runs on a CPU (about an hour on an Apple M1).
run "run_pythia-410m" python -m negprobe.run \
    --model EleutherAI/pythia-410m --data "$DATA" --vocab "$VOCAB" \
    --subsets GoogleRE TREx --device "$DEVICE" --dtype fp32 --batch-size 64 --out results
if [ "$SKIP_GPU" = "1" ]; then
  log "SKIP  pythia-1.4b (SKIP_GPU=1)"
else
  run "run_pythia-1.4b" python -m negprobe.run \
      --model EleutherAI/pythia-1.4b --data "$DATA" --vocab "$VOCAB" \
      --subsets GoogleRE TREx --device "$DEVICE" --dtype fp32 --out results
fi

# the released-template comparison reported in docs/DATA_ISSUE.md
for M in bert-base-cased bert-large-cased; do
  run "run_${M}_released" env NEGPROBE_RELEASED_TEMPLATES=1 python -m negprobe.run --model "$M" \
      --data "$DATA" --vocab "$VOCAB" --subsets GoogleRE TREx --out results_released
done

# ---------------------------------------------------------------- C + D
if [ "$SKIP_GPU" != "1" ]; then
  run "truth_qwen_instruct" python -m negprobe.truth \
      --model Qwen/Qwen2.5-1.5B-Instruct --data "$DATA" --subsets TREx GoogleRE --per-relation 50 \
      --device "$DEVICE" --dtype fp32 --out results_truth
  # base model, four-example prompt (--chat no: Qwen2.5 base also ships a chat template)
  run "truth_qwen_base_fewshot" python -m negprobe.truth --model Qwen/Qwen2.5-1.5B \
      --data "$DATA" --subsets TREx GoogleRE --per-relation 50 --chat no \
      --device "$DEVICE" --dtype fp32 --out results_truth
  # Experiment D, implemented but not run for the paper:
  run "generate_qwen_instruct" python -m negprobe.generate --model Qwen/Qwen2.5-1.5B-Instruct \
      --data "$DATA" --subsets TREx --per-relation 50 --device "$DEVICE" --dtype "$DTYPE"
fi

# ---------------------------------------------------------------- figures + verification
run "figures" python -m negprobe.figures results --truth results_truth --out figures
run "verify"  python verify_numbers.py

# ---------------------------------------------------------------- checksums
{
  echo "# sha256 of inputs and outputs, $(date -u)"
  shasum -a 256 data/negated_data/relations.jsonl 2>/dev/null || sha256sum data/negated_data/relations.jsonl
  find results results_truth results_generate results_released -name '*.csv' -o -name '*.json' 2>/dev/null \
    | sort | while read -r f; do shasum -a 256 "$f" 2>/dev/null || sha256sum "$f"; done
} > CHECKSUMS.txt 2>/dev/null
log "wrote CHECKSUMS.txt"
log "DONE — see logs/, then compare figures/ and results/ with the committed copies"
