#!/bin/bash
# REGULARIZATION CONTROL on the COPY task (the char-LM control's missing half).
#
# The char-LM control (run_digreg.sh, commit bb79f7f) showed the digital
# baseline was flattered by ~0.31 bpc: training-time state noise sigma=0.02 on
# the digital variant was worth 2.1x the entire analog advantage, retracting
# "analog beats digital". Every copy-task "margin kept" figure was measured
# against the same unregularized digital baseline and has NO such control --
# the ledger currently assumes the bias runs the same direction there (i.e.
# all copy margins are upper bounds).
#
# This row tests that assumption, and it matters more than a routine control:
# the paper's central claim (the datapath-degradation principle) makes the
# OPPOSITE prediction for copy. Copy depends on PRECISE state retention, so
# state noise degrades a part of the datapath the task DOES depend on and
# should HURT the digital baseline, not help it.
#
# Arms (ep30/n80k, the budget of the definitive copy row):
#   n0.02     noise only -- the one mechanism that carried the char-LM effect
#   n0.02_b6  noise + 6-bit quant = the full analog state degradation without
#             send-on-delta gating (quant was worth 0 on char-LM; on copy it
#             is lossy in exactly the way precise recall cannot absorb)
# Loads: M=16 (L=33) and M=32 (L=65), seeds 0/1/2. 12 cells total.
# (wd arm dropped: worth exactly 0 on char-LM. n0.05 dropped: dose saturated.)
#
# PRE-REGISTERED READINGS (written before any cell lands; mirrored in ledger):
#  (A) digital+noise IMPROVES copy acc -> the baseline was under-regularized
#      here too; every margin-kept figure shrinks; copy conclusions get MORE
#      negative for the neuromorphic routes. (The ledger's current assumption.)
#  (B) digital+noise HURTS copy acc -> out-of-sample CONFIRMATION of the
#      datapath-degradation principle (noise degrades a needed part), and the
#      existing copy margins stand as fair comparisons.
#  (C) no measurable effect -> margins stand.
#  The PRINCIPLE predicts (B); analogy-from-char-LM predicts (A). First cell
#  in the project where the two make opposite predictions.
#
# Work-queue design as run_digreg.sh: atomic-mkdir claim, skip any cell whose
# out JSON exists, add workers on any GPU at any time.
# usage: ./run_digreg_copy.sh <gpu>
set -u
GPU="${1:?usage: run_digreg_copy.sh <gpu>}"
cd /work/zeyuwang/neuro_poc || exit 1
PY="$HOME/miniconda3/envs/dino_wm/bin/python"
LOCKS=digreg_copy_locks
LOG=copy_logs/digreg_copy.log
mkdir -p "$LOCKS" copy_logs digreg_copy_logs ssm3way_runs

# primary arm and easier load first, so a partial row is already decisive
CELLS="n0.02:33:--dig_noise 0.02
n0.02:65:--dig_noise 0.02
n0.02_b6:33:--dig_noise 0.02 --dig_bits 6
n0.02_b6:65:--dig_noise 0.02 --dig_bits 6"

echo "$CELLS" | while IFS=: read -r ARM L FLAGS; do
  for SEED in 0 1 2; do
    CELL="digital_copy_L${L}_s${SEED}_reg_${ARM}_ep30"
    OUT="ssm3way_runs/${CELL}.json"
    [ -f "$OUT" ] && continue
    mkdir "$LOCKS/$CELL" 2>/dev/null || continue   # someone else owns it
    echo "start $CELL gpu$GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    # shellcheck disable=SC2086
    "$PY" ssm3way.py --gpu "$GPU" --task copy --L "$L" \
      --epochs 30 --copy_n 80000 \
      --variant digital --seed "$SEED" $FLAGS \
      --out "$OUT" > "digreg_copy_logs/${CELL}.log" 2>&1
    if [ -f "$OUT" ]; then
      echo "cell done $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    else
      echo "cell FAILED $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
      rmdir "$LOCKS/$CELL" 2>/dev/null   # release so a later worker retries
    fi
  done
done
echo "WORKER GPU$GPU DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
