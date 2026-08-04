#!/bin/bash
# run_stream_calib.sh <gpu>
#
# FashionMNIST ROW-STREAM row (28 steps x 28 px, V=10, V/H=0.039) -- the small-output
# shape where energy_shape.py projects a large analog pJ advantage and where NO quality
# number has ever been measured.  This row is the theta re-calibration + the matched
# 4-variant comparison at one seed.  Input stays DENSE here (in_theta=0) so the state
# theta calibration is isolated; the event-input arm is a separate follow-up.
#
# Lock-dir work queue: every worker walks the same ordered cell list and claims a cell
# with an atomic mkdir, skipping any cell whose out JSON already exists.  So a worker
# can be added on any idle GPU at any time -- no sharding agreement, no duplicate
# training; a failed cell releases its lock for retry.
set -u
GPU=${1:?usage: run_stream_calib.sh <gpu>}
cd /work/zeyuwang/neuro_poc || exit 1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
LOCKS=stream_locks
mkdir -p "$LOCKS" ssm3way_runs stream_logs

EP=10
COMMON="--task stream --epochs $EP --stream_n 20000 --stream_nval 5000 --seed 0"

# NAME::EXTRA_ARGS   (reference first, so comparisons are possible early)
CELLS=(
  "digital_stream_s0_reg_n0.02::--variant digital --dig_noise 0.02"
  "digital_stream_s0::--variant digital"
  "analog_stream_s0_th0.15::--variant analog --theta 0.15"
  "analog_stream_s0_th0.25::--variant analog --theta 0.25"
  "analog_stream_s0_th0.5::--variant analog --theta 0.5"
  "analog_stream_s0_th1.0::--variant analog --theta 1.0"
  "analog_stream_s0_th2.0::--variant analog --theta 2.0"
  "spikeout_stream_s0::--variant spikeout"
  "spikestate_stream_s0::--variant spikestate"
)

for cell in "${CELLS[@]}"; do
  NAME="${cell%%::*}"
  ARGS="${cell#*::}"
  OUT="ssm3way_runs/${NAME}_ep${EP}.json"
  [ -f "$OUT" ] && continue
  mkdir "$LOCKS/$NAME" 2>/dev/null || continue      # someone else has it
  echo "[gpu$GPU] START $NAME $(date -u +%FT%TZ)" >> stream_logs/calib.log
  $PY ssm3way.py $COMMON $ARGS --gpu "$GPU" --out "$OUT" \
      > "stream_logs/${NAME}.log" 2>&1
  if [ -f "$OUT" ]; then
    echo "[gpu$GPU] DONE  $NAME $(date -u +%FT%TZ)" >> stream_logs/calib.log
  else
    echo "[gpu$GPU] FAIL  $NAME $(date -u +%FT%TZ)" >> stream_logs/calib.log
    rmdir "$LOCKS/$NAME" 2>/dev/null                # release for retry
  fi
done
echo "[gpu$GPU] CHAIN DONE $(date -u +%FT%TZ)" >> stream_logs/calib.log
