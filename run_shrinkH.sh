#!/bin/bash
# Shrink-H row: the ONLY viable test of the firing-floor bound rho >= H_b^-1(M*log2K / H).
#
# Why this row exists: raising the memory load M pushes the task past what a
# spiking-state net can learn at all (spikestate is exactly at chance by M=32),
# so the M-dependence of the bound is untestable on copy. Shrinking the hidden
# width H makes the bound bind at a load the net can still learn: hold M=16
# (L=33, K=16 => 64 bits to retain) and sweep H.
#   H=64  -> 64/64  = 1.000 -> predicted floor 0.500
#   H=96  -> 64/96  = 0.667 -> predicted floor 0.174
#   H=128 -> 64/128 = 0.500 -> predicted floor 0.042
#   H=256 (already run in the main ep30 row) -> floor 0.0026, non-binding
#
# Work-queue design: every worker walks the same ordered cell list and claims a
# cell with an atomic mkdir. Launch one worker per GPU as GPUs free up; workers
# never duplicate each other and never need to agree on a sharding. A cell whose
# output JSON already exists is skipped outright (idempotent across restarts).
#
# usage: ./run_shrinkH.sh <gpu>
set -u
GPU="${1:?usage: run_shrinkH.sh <gpu>}"
cd /work/zeyuwang/neuro_poc || exit 1
PY="$HOME/miniconda3/envs/dino_wm/bin/python"
LOCKS=shrinkH_locks
LOG=copy_logs/shrinkH.log
mkdir -p "$LOCKS" copy_logs shrinkH_logs ssm3way_runs

for H in 64 96 128; do
  for SEED in 0 1 2; do
    for VAR in digital spikestate; do
      CELL="${VAR}_copy_L33_s${SEED}_H${H}_ep30"
      OUT="ssm3way_runs/${CELL}.json"
      [ -f "$OUT" ] && continue
      mkdir "$LOCKS/$CELL" 2>/dev/null || continue   # someone else owns it
      echo "start $CELL gpu$GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
      "$PY" ssm3way.py --gpu "$GPU" --task copy --L 33 --H "$H" \
        --variant "$VAR" --seed "$SEED" --epochs 30 --copy_n 80000 \
        --out "$OUT" > "shrinkH_logs/${CELL}.log" 2>&1
      if [ -f "$OUT" ]; then
        echo "cell done $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
      else
        echo "cell FAILED $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
        rmdir "$LOCKS/$CELL" 2>/dev/null   # release so a later worker retries
      fi
    done
  done
done
echo "WORKER GPU$GPU DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
