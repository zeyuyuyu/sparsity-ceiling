#!/bin/bash
# char-LM state-fidelity dichotomy at MATCHED EMITTED RATE.
#
# Why this row exists: on copy, all three non-digital variants happen to emit at
# ~0.45-0.50, so the margin-kept spread (continuous 0.92 > lossy-analog 0.53 >
# 1-bit 0.08) is attributable to state fidelity rather than to how much each
# variant communicates. The existing 3-seed char-LM row has no such matching --
# analog emits 0.266 at theta=1.0 while spikestate emits 0.645 and spikeout 0.369
# -- so analog's char-LM advantage is confounded with its operating point.
#
# Measured char-LM emission targets (3 seeds each):
#   spikestate  0.645 +- 0.020   (state spikes; rate_state == rate_emitted)
#   spikeout    0.369 +- 0.037   (output spikes; rate_state is exactly 1.000)
# Seed-0 analog theta curve: 0.10 -> 0.722, 0.25 -> 0.557, 0.50 -> 0.485, 1.0 -> 0.279.
# So theta ~0.15 brackets spikestate's rate and theta ~0.75 brackets spikeout's,
# letting analog be read against BOTH comparators at their own operating points.
#
# theta grid is entirely ABOVE the state quantizer LSB (2*rail/2^bits = 8/64 =
# 0.125); any theta below that is bit-identical and inoperative (copy calibration).
# Budget is IDENTICAL to the existing char-LM row (defaults: 6 epochs, 1.4M chars,
# lam=0) so the new cells drop straight into that table. Out names follow the
# existing convention so already-run cells (s0 at theta 0.25 / 0.5) are skipped.
#
# Work-queue design (same as run_shrinkH.sh): each worker walks the same ordered
# cell list and claims a cell with an atomic mkdir; a cell whose out JSON exists
# is skipped. Workers can be added on any GPU at any time, no sharding agreement,
# no duplicate training. A failed cell releases its lock for a later retry.
#
# usage: ./run_charlm_theta.sh <gpu>
set -u
GPU="${1:?usage: run_charlm_theta.sh <gpu>}"
cd /work/zeyuwang/neuro_poc || exit 1
PY="$HOME/miniconda3/envs/dino_wm/bin/python"
LOCKS=charlm_theta_locks
LOG=copy_logs/charlm_theta.log
mkdir -p "$LOCKS" copy_logs charlm_theta_logs ssm3way_runs

# matched points first (0.15 -> spikestate rate, 0.75 -> spikeout rate),
# then the two curve-filling points, so a partial row is already usable.
for TH in 0.15 0.75 0.25 0.5; do
  for SEED in 0 1 2; do
    CELL="analog_charlm_s${SEED}_theta${TH}"
    OUT="ssm3way_runs/${CELL}.json"
    [ -f "$OUT" ] && continue
    mkdir "$LOCKS/$CELL" 2>/dev/null || continue   # someone else owns it
    echo "start $CELL gpu$GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    "$PY" ssm3way.py --gpu "$GPU" --task charlm \
      --variant analog --seed "$SEED" --theta "$TH" \
      --out "$OUT" > "charlm_theta_logs/${CELL}.log" 2>&1
    if [ -f "$OUT" ]; then
      echo "cell done $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    else
      echo "cell FAILED $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
      rmdir "$LOCKS/$CELL" 2>/dev/null   # release so a later worker retries
    fi
  done
done
echo "WORKER GPU$GPU DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
