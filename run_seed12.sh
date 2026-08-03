#!/bin/bash
# 3-SEED CONFIRMATION of the n=1 gated-readout cells (seeds 1,2) -- the one
# item owed before any paper use of the energy-datapath rows (73c99e6, 60d627c).
#
# Economy: the s0 rows established the SHAPES (step at gate-on on char-LM,
# total collapse on copy, staleness-not-selection in the outrand ablation).
# This row confirms the CLAIMS at three representative operating points per
# curve -- out_theta in {0.02 (nothing meaningfully gated), 0.1 (mid),
# 1.0 (sparse)} -- NOT the full 7-point curves.  The untested thetas
# {0.05,0.25,0.5,2.0} stay n=1 by design.  outrand: the criterion-(A) cells
# (p=0.90 both arms) plus the low-rate digital cell p=0.08 that carries the
# "delta-rule 0.43 bpc better than random" claim.
#
# 30 cells total:
#   copy   eventro: digital reg n0.02, L in {33,65} x ot {0.02,0.1,1.0} x s {1,2}  (12, ep30/n80k)
#   charlm eventro: {digital reg n0.02, analog theta0.15} x ot {0.02,0.1,1.0} x s {1,2}  (12, 6ep/1.4M)
#   outrand:        digital pr {0.90,0.08}, analog pr 0.90, s {1,2}  (6, 6ep/1.4M)
# Long copy cells are ordered first for packing.
#
# Lock-dir work queue (same as run_shrinkH.sh): atomic mkdir claim, skips any
# cell whose out JSON exists, releases the lock on failure.  Safe to add a
# worker on any GPU at any time.
#
# usage: ./run_seed12.sh <gpu>
set -u
GPU=$1
PY=$HOME/miniconda3/envs/dino_wm/bin/python
D=/work/zeyuwang/neuro_poc
RUNS=$D/ssm3way_runs
LOCKS=$D/seed12_locks
LOGS=$D/seed12_logs
mkdir -p "$LOCKS" "$LOGS" "$D/copy_logs"

# kind|seed|a|b   copy: a=L b=out_theta;  otd/ota: a=out_theta;  prd/pra: a=send_prob
CELLS="
copy|1|33|0.02
copy|1|65|0.02
copy|2|33|0.02
copy|2|65|0.02
copy|1|33|0.1
copy|1|65|0.1
copy|2|33|0.1
copy|2|65|0.1
copy|1|33|1.0
copy|1|65|1.0
copy|2|33|1.0
copy|2|65|1.0
otd|1|0.02
ota|1|0.02
otd|2|0.02
ota|2|0.02
otd|1|0.1
ota|1|0.1
otd|2|0.1
ota|2|0.1
otd|1|1.0
ota|1|1.0
otd|2|1.0
ota|2|1.0
prd|1|0.90
pra|1|0.90
prd|2|0.90
pra|2|0.90
prd|1|0.08
prd|2|0.08
"

for cell in $CELLS; do
  KIND=$(echo "$cell" | cut -d'|' -f1)
  SEED=$(echo "$cell" | cut -d'|' -f2)
  A=$(echo "$cell" | cut -d'|' -f3)
  B=$(echo "$cell" | cut -d'|' -f4)
  case $KIND in
    copy)
      NAME="digital_copy_L${A}_s${SEED}_reg_n0.02_ot${B}_ep30"
      ARGS="--variant digital --dig_noise 0.02 --task copy --L $A --epochs 30 --copy_n 80000 --out_theta $B" ;;
    otd)
      NAME="digital_charlm_s${SEED}_reg_n0.02_ot${A}"
      ARGS="--variant digital --dig_noise 0.02 --task charlm --epochs 6 --chars 1400000 --out_theta $A" ;;
    ota)
      NAME="analog_charlm_s${SEED}_theta0.15_ot${A}"
      ARGS="--variant analog --theta 0.15 --task charlm --epochs 6 --chars 1400000 --out_theta $A" ;;
    prd)
      NAME="digital_charlm_s${SEED}_reg_n0.02_pr${A}"
      ARGS="--variant digital --dig_noise 0.02 --task charlm --epochs 6 --chars 1400000 --out_prand $A" ;;
    pra)
      NAME="analog_charlm_s${SEED}_theta0.15_pr${A}"
      ARGS="--variant analog --theta 0.15 --task charlm --epochs 6 --chars 1400000 --out_prand $A" ;;
    *) continue ;;
  esac
  OUT=$RUNS/$NAME.json
  [ -f "$OUT" ] && continue
  mkdir "$LOCKS/$NAME" 2>/dev/null || continue      # someone else has it
  echo "[gpu$GPU] START $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/seed12.log
  $PY $D/ssm3way.py $ARGS --gpu "$GPU" --seed "$SEED" \
      --out "$OUT" > "$LOGS/$NAME.log" 2>&1
  if [ -f "$OUT" ]; then
    echo "[gpu$GPU] DONE  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/seed12.log
  else
    echo "[gpu$GPU] FAIL  $NAME $(date -u +%H:%M:%SZ)" >> $D/copy_logs/seed12.log
    rmdir "$LOCKS/$NAME"                            # release for retry
  fi
done
echo "[gpu$GPU] CHAIN DONE $(date -u +%H:%M:%SZ)" >> $D/copy_logs/seed12.log
