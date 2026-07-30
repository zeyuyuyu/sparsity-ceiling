#!/bin/bash
# analog theta calibration at the NEW budget (ep30, copy_n 80k), L=65 (M=32), seed 0
# + digital L=65 reference at the same budget. Idempotent: skips cells whose JSON exists.
cd /work/zeyuwang/neuro_poc || exit 1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
run() {  # run <gpu> <variant> <extra-args...> <outname>
  g=$1; v=$2; shift 2
  out="${@: -1}"; args="${@:1:$#-1}"
  [ -f "$out" ] && { echo "skip existing $out" >> copy_logs/calib_ep30.log; return; }
  $PY ssm3way.py --gpu $g --task copy --L 65 --seed 0 --epochs 30 --copy_n 80000 \
    --variant $v $args --out "$out" > "copy_logs/calib_$(basename $out .json).log" 2>&1
}
case $1 in
  0) run 0 analog --theta 0.05 ssm3way_runs/analog_copy_L65_s0_th0.05_ep30.json
     run 0 analog --theta 0.5  ssm3way_runs/analog_copy_L65_s0_th0.5_ep30.json ;;
  1) run 1 analog --theta 0.1  ssm3way_runs/analog_copy_L65_s0_th0.1_ep30.json
     run 1 digital             ssm3way_runs/digital_copy_L65_s0_ep30.json ;;
  2) run 2 analog --theta 0.2  ssm3way_runs/analog_copy_L65_s0_th0.2_ep30.json ;;
  3) run 3 analog --theta 0.3  ssm3way_runs/analog_copy_L65_s0_th0.3_ep30.json ;;
esac
echo "CALIB CHAIN GPU$1 DONE $(date -u +%FT%TZ)" >> copy_logs/calib_ep30.log
