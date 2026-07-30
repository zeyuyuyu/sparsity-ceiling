#!/bin/bash
# REGULARIZATION CONTROL for the char-LM headline.
#
# The headline: on char-LM the analog-state SSM BEATS a matched digital baseline
# by 0.149 +- 0.014 bpc (3 seeds) while emitting on 61% of steps. The obvious
# reviewer objection -- stated as a caveat in the ledger but never tested -- is
# that this is REGULARIZATION, not superior computation. The analog state carries
# additive noise (sigma=0.02), 6-bit quantization over +-4 rails, and send-on-delta
# suppression, on a 173k-param model at a 6-epoch schedule; the digital baseline
# has no regularization at all. In the vision PoC sparsity RAISED accuracy for
# exactly this reason.
#
# This row gives the DIGITAL variant the same medicine and asks how much of the
# 0.149 bpc it recovers. Three questions, decomposed:
#   arm n0.02_b6  digital + noise 0.02 + 6-bit quant = the ENTIRE analog state
#                 degradation WITHOUT the send-on-delta gating. The key arm: it
#                 separates "lossy state as regularizer" from "event gating".
#   arm n0.02     noise only -- is quantization doing anything, or just noise?
#   arm n0.05     stronger noise -- dose-response. If more noise keeps helping,
#                 the digital baseline was simply under-regularized.
#   arm wd1e-4    a conventional tuned baseline (Adam weight decay), i.e. what a
#                 reviewer means by "did you even try regularizing the baseline".
#
# PRE-REGISTERED READINGS (written before any cell lands):
#  (A) if the full arm (n0.02_b6) recovers most of the 0.149 bpc -> the char-LM
#      win is a REGULARIZATION effect. Honest claim becomes "the analog datapath
#      is not a quality tax and its state noise happens to regularize a small
#      model", and the energy story (1.60x cheaper at equal-or-better quality)
#      survives while "analog computes better" dies. This is the likely outcome.
#  (B) if none of the four arms closes the gap -> the win is NOT explainable as
#      state-degradation-as-regularizer, and the send-on-delta gating itself is
#      doing something. Strengthens the headline considerably.
#  (C) if an arm OVERSHOOTS (digital+reg beats analog) -> the honest headline is
#      that a properly regularized digital baseline wins on quality and analog's
#      only remaining claim is energy at matched quality. Must be reported.
# Either way the resulting number goes in the paper: this is the one control a
# reviewer will demand.
#
# Budget IDENTICAL to the existing char-LM row (defaults: 6 ep, 1.4M chars,
# lam=0, H=256, 173,596 params) so cells drop straight into that table.
# Work-queue design as run_shrinkH.sh / run_charlm_theta.sh: atomic-mkdir claim,
# skip any cell whose out JSON exists, add workers on any GPU at any time.
#
# usage: ./run_digreg.sh <gpu>
set -u
GPU="${1:?usage: run_digreg.sh <gpu>}"
cd /work/zeyuwang/neuro_poc || exit 1
PY="$HOME/miniconda3/envs/dino_wm/bin/python"
LOCKS=digreg_locks
LOG=copy_logs/digreg.log
mkdir -p "$LOCKS" copy_logs digreg_logs ssm3way_runs

# most diagnostic arm first so a partial row is already decisive
ARMS="n0.02_b6:--dig_noise 0.02 --dig_bits 6
n0.02:--dig_noise 0.02
n0.05:--dig_noise 0.05
wd1e-4:--wd 1e-4"

echo "$ARMS" | while IFS=: read -r ARM FLAGS; do
  for SEED in 0 1 2; do
    CELL="digital_charlm_s${SEED}_reg_${ARM}"
    OUT="ssm3way_runs/${CELL}.json"
    [ -f "$OUT" ] && continue
    mkdir "$LOCKS/$CELL" 2>/dev/null || continue   # someone else owns it
    echo "start $CELL gpu$GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    # shellcheck disable=SC2086
    "$PY" ssm3way.py --gpu "$GPU" --task charlm \
      --variant digital --seed "$SEED" $FLAGS \
      --out "$OUT" > "digreg_logs/${CELL}.log" 2>&1
    if [ -f "$OUT" ]; then
      echo "cell done $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
    else
      echo "cell FAILED $CELL $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
      rmdir "$LOCKS/$CELL" 2>/dev/null   # release so a later worker retries
    fi
  done
done
echo "WORKER GPU$GPU DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
