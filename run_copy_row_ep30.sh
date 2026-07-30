#!/bin/bash
# copy-task row RERUN at the higher budget (--epochs 30 --copy_n 80000), after the baseline-gate
# probe showed the old 6-epoch row was budget-limited rather than task-limited.
# 4 variants x M in {16,32,64} (L = 2M+1) x seeds 0-2 = 36 cells.
# analog uses the ep30-calibrated theta=0.2 (chosen to sit ABOVE the 6-bit quantizer LSB 0.125,
# below which theta is inoperative and bit-identical).
# Jobs are ordered by ascending L so the cheap small-M cells complete first and give a usable
# partial row early.  Idempotent: skips any cell whose result JSON already exists.
cd /work/zeyuwang/neuro_poc || exit 1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
[ -x "$PY" ] || { echo "no env python at $PY"; exit 1; }
mkdir -p ssm3way_runs copy_logs
EP=30
N=80000
THETA=0.2
JOBS=()
for L in 33 65 129; do
  for v in digital spikeout analog spikestate; do
    for s in 0 1 2; do
      x=""
      [ "$v" = analog ] && x="--theta $THETA"
      JOBS+=("--task copy --L $L --variant $v --seed $s --epochs $EP --copy_n $N $x --out ssm3way_runs/${v}_copy_L${L}_s${s}_ep30.json")
    done
  done
done
echo "copy row ep30 launch $(date -u +%FT%TZ): ${#JOBS[@]} cells, theta=$THETA, ep=$EP n=$N, GPUs 0-7" >> copy_logs/row_ep30.log
for g in 0 1 2 3 4 5 6 7; do
  (
    for i in "${!JOBS[@]}"; do
      if [ $((i % 8)) -eq $g ]; then
        out="${JOBS[$i]##*--out }"
        [ -f "$out" ] && { echo "skip existing $out" >> copy_logs/row_ep30.log; continue; }
        $PY ssm3way.py --gpu $g ${JOBS[$i]} > "copy_logs/ep30_job${i}_gpu${g}.log" 2>&1
        echo "cell done $out $(date -u +%FT%TZ)" >> copy_logs/row_ep30.log
      fi
    done
    echo "CHAIN GPU$g DONE $(date -u +%FT%TZ)" >> copy_logs/row_ep30.log
  ) &
done
wait
echo "COPY ROW EP30 DONE $(date -u +%FT%TZ)" >> copy_logs/row_ep30.log
