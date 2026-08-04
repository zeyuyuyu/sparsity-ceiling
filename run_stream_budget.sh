#!/bin/bash
# run_stream_budget.sh <gpu>
#
# BUDGET CHECK for the FashionMNIST row-stream REFUTE verdict (c18c724): the ep10/20k
# row's one live alternative explanation is budget (reference 0.788 vs ~0.91 CNN).
# Rerun the reference arms + the two best analog thetas + spikeout at 30 ep / FULL 60k
# train. Pre-registered criteria live in ssm3way_ledger.md (energy-datapath: stream
# BUDGET CHECK): CLOSE = both analog cells still >5-pt deficit vs the better digital
# arm; REOPEN = either within <=5 pts at r_z <= 0.65.
#
# Same lock-dir work queue as run_stream_calib.sh: atomic mkdir claims a cell, skip if
# the out JSON exists, a failed cell releases its lock for retry.
set -u
GPU=${1:?usage: run_stream_budget.sh <gpu>}
cd /work/zeyuwang/neuro_poc || exit 1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
LOCKS=stream_budget_locks
mkdir -p "$LOCKS" ssm3way_runs stream_logs

EP=30
COMMON="--task stream --epochs $EP --stream_n 60000 --stream_nval 5000 --seed 0"

CELLS=(
  "digital_stream_s0_reg_n0.02::--variant digital --dig_noise 0.02"
  "digital_stream_s0::--variant digital"
  "analog_stream_s0_th0.15::--variant analog --theta 0.15"
  "analog_stream_s0_th0.25::--variant analog --theta 0.25"
  "spikeout_stream_s0::--variant spikeout"
)

for cell in "${CELLS[@]}"; do
  NAME="${cell%%::*}"
  ARGS="${cell#*::}"
  OUT="ssm3way_runs/${NAME}_n60k_ep${EP}.json"
  [ -f "$OUT" ] && continue
  mkdir "$LOCKS/$NAME" 2>/dev/null || continue      # someone else has it
  echo "[gpu$GPU] START $NAME $(date -u +%FT%TZ)" >> stream_logs/budget.log
  $PY ssm3way.py $COMMON $ARGS --gpu "$GPU" --out "$OUT" \
      > "stream_logs/${NAME}_n60k_ep${EP}.log" 2>&1
  if [ -f "$OUT" ]; then
    echo "[gpu$GPU] DONE  $NAME $(date -u +%FT%TZ)" >> stream_logs/budget.log
  else
    echo "[gpu$GPU] FAIL  $NAME $(date -u +%FT%TZ)" >> stream_logs/budget.log
    rmdir "$LOCKS/$NAME" 2>/dev/null                # release for retry
  fi
done
echo "[gpu$GPU] CHAIN DONE $(date -u +%FT%TZ)" >> stream_logs/budget.log
