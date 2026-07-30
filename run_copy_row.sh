#!/bin/bash
# copy-task row for the ssm3way comparison: 4 variants x M in {64,32,16} (via L) x seeds 0-2,
# + analog theta calibration {0.1,0.3} at middle M seed 0.  Idempotent: skips cells whose
# result JSON already exists, so a restart never re-trains finished cells.
cd /work/zeyuwang/neuro_poc || exit 1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
[ -x "$PY" ] || { echo "no env python at $PY"; exit 1; }
mkdir -p ssm3way_runs copy_logs
JOBS=()
for L in 129 65 33; do
  for v in digital spikeout analog spikestate; do
    for s in 0 1 2; do
      x=""
      [ "$v" = analog ] && x="--theta 1.0"
      JOBS+=("--task copy --L $L --variant $v --seed $s $x --out ssm3way_runs/${v}_copy_L${L}_s${s}.json")
    done
  done
done
JOBS+=("--task copy --L 65 --variant analog --seed 0 --theta 0.1 --out ssm3way_runs/analog_copy_L65_s0_theta0.1.json")
JOBS+=("--task copy --L 65 --variant analog --seed 0 --theta 0.3 --out ssm3way_runs/analog_copy_L65_s0_theta0.3.json")
echo "launching ${#JOBS[@]} jobs over 8 GPUs $(date -u +%FT%TZ)" >> copy_logs/driver.log
for g in 0 1 2 3 4 5 6 7; do
  (
    for i in "${!JOBS[@]}"; do
      if [ $((i % 8)) -eq $g ]; then
        out="${JOBS[$i]##*--out }"
        [ -f "$out" ] && { echo "skip existing $out" >> copy_logs/driver.log; continue; }
        $PY ssm3way.py --gpu $g ${JOBS[$i]} > "copy_logs/job${i}_gpu${g}.log" 2>&1
      fi
    done
  ) &
done
wait
echo "COPY ROW DONE $(date -u +%FT%TZ)" >> copy_logs/driver.log
