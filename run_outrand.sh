#!/bin/bash
# PRE-REGISTERED ABLATION: is the event-readout collapse caused by STALENESS
# ITSELF, or by WHICH units the send-on-delta rule chooses to hold?
#
# The theta-gated readout (run_eventreadout.sh / run_eventro_copy.sh) cost
# ~+1.0 bpc on char-LM and ALL of the above-chance margin on copy, already at
# r_out ~ 1 where essentially nothing is gated -- a step function, not a
# function of the sparsity bought.  Attribution was reasoned from the code, not
# measured.  This row swaps the SELECTION RULE only: --out_prand holds `uref`
# for a RANDOM Bernoulli subset of units at a send rate matched to the theta
# row's MEASURED r_out, so staleness rate is held fixed and only the choice of
# held units differs.
#
# Matched rates (char-LM, seed 0, 6 ep, 1.4M chars -- same budget as published):
#   digital reg n0.02:  theta 0.02/0.1/0.5/1.0 -> r_out 0.901/0.594/0.258/0.084
#   analog  theta 0.15: theta 0.1 /1.0         -> r_out 0.895/0.258
# No-gate references already on disk: digital_charlm_s0_reg_n0.02.json (bpc
# 3.031), analog_charlm_s0_theta0.15*.json (bpc 3.185).
#
# Lock-dir work queue (same as run_shrinkH.sh): atomic mkdir claim, skips any
# cell whose out JSON exists, releases the lock on failure.  Safe to add a
# worker on any GPU at any time.
#
# usage: ./run_outrand.sh <gpu>
set -u
GPU=$1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
D=/work/zeyuwang/neuro_poc
RUNS=$D/ssm3way_runs
LOCKS=$D/outrand_locks
LOGS=$D/outrand_logs
mkdir -p "$LOCKS" "$LOGS" "$D/copy_logs"

# arm|send_probability
CELLS="
digital|0.90
analog|0.90
digital|0.59
analog|0.26
digital|0.26
digital|0.08
"

for cell in $CELLS; do
  ARM=${cell%%|*}; PR=${cell##*|}
  if [ "$ARM" = analog ]; then
    NAME="analog_charlm_s0_theta0.15_pr${PR}"
    EXTRA="--variant analog --theta 0.15"
  else
    NAME="digital_charlm_s0_reg_n0.02_pr${PR}"
    EXTRA="--variant digital --dig_noise 0.02"
  fi
  OUT=$RUNS/$NAME.json
  [ -f "$OUT" ] && continue
  mkdir "$LOCKS/$NAME" 2>/dev/null || continue      # someone else has it
  echo "[gpu$GPU] START $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/outrand.log
  $PY $D/ssm3way.py $EXTRA --task charlm --gpu "$GPU" --seed 0 \
      --epochs 6 --chars 1400000 --out_prand "$PR" \
      --out "$OUT" > "$LOGS/$NAME.log" 2>&1
  if [ -f "$OUT" ]; then
    echo "[gpu$GPU] DONE  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/outrand.log
  else
    echo "[gpu$GPU] FAIL  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/outrand.log
    rmdir "$LOCKS/$NAME"                            # release for retry
  fi
done
echo "[gpu$GPU] CHAIN DONE $(date -u +%H:%M:%SZ)" >> $D/copy_logs/outrand.log
