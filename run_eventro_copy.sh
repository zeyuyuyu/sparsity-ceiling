#!/bin/bash
# EVENT-DRIVEN READOUT on the COPY task (seed 0, ep30/n80k) -- the pre-registered
# follow-up to the char-LM event-readout row (commit a9ff715).
#
# WHY: on char-LM the readout gate cost ~+1.0 bpc at ANY threshold (a step
# function at the moment the hold switches on) -- because char-LM needs an
# exact output distribution.  The datapath-degradation principle predicts the
# SAME gate is CHEAP on precise-recall workloads (where output-spiking was
# already the best neuromorphic variant).  If that holds, digital-state +
# event-readout is the project's first real pJ cut at little quality cost.
#
# Arm: digital + dig_noise 0.02 ONLY -- compared against the existing
# noise-regularized references digital_copy_L{33,65}_s{0,1,2}_reg_n0.02_ep30.json
# (same budget, same regularization; the only delta is out_theta).
#
# Lock-dir work queue (same as run_shrinkH.sh / run_eventreadout.sh): each
# worker claims a cell with an atomic mkdir and skips any cell whose out JSON
# exists.  Safe to add a worker on any GPU at any time; a failed cell releases
# its lock.
#
# usage: ./run_eventro_copy.sh <gpu>
set -u
GPU=$1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
D=/work/zeyuwang/neuro_poc
RUNS=$D/ssm3way_runs
LOCKS=$D/eventro_copy_locks
LOGS=$D/eventro_copy_logs
mkdir -p "$LOCKS" "$LOGS"

# L|out_theta -- L=33 (M=16) first so a full column lands early
CELLS="
33|0.02
33|0.05
33|0.1
33|0.25
33|0.5
33|1.0
33|2.0
65|0.02
65|0.05
65|0.1
65|0.25
65|0.5
65|1.0
65|2.0
"

for cell in $CELLS; do
  L=${cell%%|*}; OT=${cell##*|}
  NAME="digital_copy_L${L}_s0_reg_n0.02_ot${OT}_ep30"
  OUT=$RUNS/$NAME.json
  [ -f "$OUT" ] && continue
  mkdir "$LOCKS/$NAME" 2>/dev/null || continue      # someone else has it
  echo "[gpu$GPU] START $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro_copy.log
  $PY $D/ssm3way.py --variant digital --dig_noise 0.02 \
      --task copy --L "$L" --gpu "$GPU" --seed 0 \
      --epochs 30 --copy_n 80000 --out_theta "$OT" \
      --out "$OUT" > "$LOGS/$NAME.log" 2>&1
  if [ -f "$OUT" ]; then
    echo "[gpu$GPU] DONE  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro_copy.log
  else
    echo "[gpu$GPU] FAIL  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro_copy.log
    rmdir "$LOCKS/$NAME"                            # release for retry
  fi
done
echo "[gpu$GPU] CHAIN DONE $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro_copy.log
