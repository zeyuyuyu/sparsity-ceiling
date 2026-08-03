#!/bin/bash
# EVENT-DRIVEN READOUT calibration row (char-LM, seed 0).
#
# WHY: energy_datapath.py showed W_out carries 75% of the analog variant's
# pJ/token and 47% of digital's, while the recurrence -- the only thing the
# published proxy prices as event-driven -- is 8% / 42%.  So the pJ verdict is
# decided by the READOUT, not by the recurrence.  Projected fully-event-driven
# totals assume r_out = r_z; this row MEASURES r_out and its quality cost.
#
# Lock-dir work queue (same as run_shrinkH.sh): each worker claims a cell with an
# atomic mkdir and skips any cell whose out JSON exists.  Safe to add a worker on
# any GPU at any time; a failed cell releases its lock.
#
# usage: ./run_eventreadout.sh <gpu>
set -u
GPU=$1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
D=/work/zeyuwang/neuro_poc
RUNS=$D/ssm3way_runs
LOCKS=$D/eventro_locks
LOGS=$D/eventro_logs
mkdir -p "$LOCKS" "$LOGS"

# arm|out_theta   -- same budget as the published char-LM row (6 ep, 1.4M chars)
CELLS="
analog|0.02
digital|0.02
analog|0.05
digital|0.05
analog|0.1
digital|0.1
analog|0.25
digital|0.25
analog|0.5
digital|0.5
analog|1.0
digital|1.0
analog|2.0
digital|2.0
"

for cell in $CELLS; do
  ARM=${cell%%|*}; OT=${cell##*|}
  if [ "$ARM" = analog ]; then
    NAME="analog_charlm_s0_theta0.15_ot${OT}"
    EXTRA="--variant analog --theta 0.15"
  else
    NAME="digital_charlm_s0_reg_n0.02_ot${OT}"
    EXTRA="--variant digital --dig_noise 0.02"
  fi
  OUT=$RUNS/$NAME.json
  [ -f "$OUT" ] && continue
  mkdir "$LOCKS/$NAME" 2>/dev/null || continue      # someone else has it
  echo "[gpu$GPU] START $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro.log
  $PY $D/ssm3way.py $EXTRA --task charlm --gpu "$GPU" --seed 0 \
      --epochs 6 --chars 1400000 --out_theta "$OT" \
      --out "$OUT" > "$LOGS/$NAME.log" 2>&1
  if [ -f "$OUT" ]; then
    echo "[gpu$GPU] DONE  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro.log
  else
    echo "[gpu$GPU] FAIL  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro.log
    rmdir "$LOCKS/$NAME"                            # release for retry
  fi
done
echo "[gpu$GPU] CHAIN DONE $(date -u +%H:%M:%SZ)" >> $D/copy_logs/eventro.log
