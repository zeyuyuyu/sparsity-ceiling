# SSM × neuromorphic: 4-way ledger

Question (N. Imam TODO; direction #1 of *The Sparsity Ceiling*): **are neuromorphic
chips suitable for SSMs, and is the NON-SPIKING (analog-state) route the fit?**

Prior context this builds on: the firing-floor bound `rho >= H_b^-1(log2 M / H)`
was derived for nets that carry the recurrent STATE in spikes. SPikE-SSM keeps the
linear state continuous and spikes only the output; CIM/analog SSMs (Wei Lu group;
LMU-on-Braindrop) keep the state in analog dynamics. Both should therefore sit
OUTSIDE the bound's scope. This ledger tests that empirically in one harness.

## Design (`ssm3way.py`)

Parameter-matched: every variant owns exactly `emb(V,E)`, `W_in(E,H)`, diagonal
decay `a(H)`, `W_mix(H,H)`, `W_out(H,V)`. Only the position of the
nonlinearity/binarization changes.

| # | variant | state | emitted signal | in scope of the floor bound? |
|---|---------|-------|----------------|------------------------------|
| 1 | `digital` | continuous | continuous (GELU) | no (S4D-real GPU baseline) |
| 2 | `spikeout` | continuous | LIF spikes | no — sparsity is output-only |
| 3 | `analog` | analog: rails ±4, noise σ=0.02, 6-bit | graded send-on-delta events | no — the hypothesis under test |
| 4 | `spikestate` | LIF spikes | spikes | **yes — the floor control** |

Variant 4 reuses `W_mix` as the recurrent matrix so parameter count stays identical;
that is the canonical form of its class, not a crippled digital SSM.

Analog non-idealities are deliberately included so "analog escapes the floor" cannot
be won by hiding in fp32 precision.

Energy proxy: 45 nm Horowitz, dense MAC 4.6 pJ vs accumulate 0.9 pJ, reported
per-term. The analog graded event is priced BOTH ways — optimistic (AC) and
conservative (MAC) — because a graded event is not a 1-bit spike; the truth is
in between and the JSON carries both numbers.

Tasks: `charlm` (WikiText-103 char-level, local arrow) and `copy` (synthetic
copy — explicit, tunable memory load `M`, which is what the bound is stated in).

## Status

| cell | state |
|------|-------|
| implementation, all 4 variants | **DONE** 2026-07-30 |
| digital × charlm | **DONE, 3 seeds** (0,1,2) |
| spikeout × charlm | **DONE, 3 seeds** (0,1,2) |
| analog × charlm | **DONE, 3 seeds** at theta=1.0; theta sweep 0.1/0.25/0.5 at seed 0 only |
| spikestate × charlm | **DONE, 3 seeds** (0,1,2) |
| all 4 × copy | not run — NEXT (the bound is stated in memory load M, which charlm does not expose) |

(The GPU-availability note just below is from an earlier tick and is stale: all 8
A800s freed at 09:50Z and the whole charlm row ran there.)

**GPU availability 2026-07-30 ~05:40Z: zero idle GPUs** — all 8 A800s held by
another user (`haomo`, 8× `ifeval_shard.py`, ~17 GB / 94-100% util each). No runs
launched; guardrail is idle-GPU-only.

## Smoke-test observations (NOT results — H=32, 1 epoch, 400 seqs, CPU)

Numbers below are sanity checks on plumbing, not evidence about the hypothesis.

- `digital` emitted rate 1.0 by construction; 9715 pJ/token — the dense reference.
- `spikeout` output rate 0.41 with the state still fully dense (`rate_state` 1.0),
  which is exactly the "sparsity is output-only" signature the design predicts.
- `analog` event rate **0.86 at θ=0.10** — high. θ is scaled against an untrained
  state distribution, so θ needs a sweep (or per-unit normalization) in the real
  runs. **This is the main open risk to the hypothesis: if the analog event rate
  cannot be pushed well below 0.5 without quality loss, the non-spiking route does
  NOT escape the floor and that gets reported as the finding.**
- `spikestate` rate 0.32 after one epoch on a toy-size net — well below the ~50%
  floor, but the floor is an asymptotic statement about a net that has actually
  learned to hold memory `M`; at 1 epoch it has not. Expect this to climb.

## Next step

Launch `digital × charlm` (seed 0) as soon as one GPU frees, then the other three
cells on the same task/seed, then the θ sweep for `analog`, then `copy` at 2-3
memory loads `M` to test the bound's `M`-dependence across variants.

## RESULTS — charlm row, seed 0 (2026-07-30, A800 GPUs 4-7, first GPU numbers)

Config: identical for all 4 — E=64 H=256 L=128 bs=64 epochs=6 lr=2e-3 lam=0 (NO sparsity reg,
so these are NATURAL activity rates), vocab 284, params 173,596 (exactly matched, verified),
analog theta=0.10 noise=0.02 bits=6 rail=4.0. Logs+JSON: /work/zeyuwang/neuro_poc/ssm3way_runs/.

| variant    | bpc    | acc    | rate_emitted | rate_state | E pJ/tok (opt) | E pJ/tok (cons) |
|------------|--------|--------|--------------|------------|----------------|-----------------|
| digital    | 3.3175 | 0.3489 | 1.00         | 1.00       | 712,448        | 712,448         |
| spikeout   | 4.3867 | 0.1993 | 0.4108       | 1.00       | 435,211        | 435,211         |
| analog     | 3.1804 | 0.3794 | 0.7224       | 0.9925     | 452,643        | 627,811         |
| spikestate | 3.6056 | 0.3011 | 0.6668       | 0.6668     | 159,509        | 159,509         |

### Verdict on the hypothesis — SPLIT. Half holds, half FAILS.

1. QUALITY half HOLDS (and slightly over-delivers): analog (bpc 3.180) MATCHES and in fact
   beats digital (3.318) at identical capacity, despite state noise sigma=0.02 and 6-bit state
   quantization. Analog-state dynamics are NOT a quality tax at this scale. Plausible cause:
   noise+quantization act as regularizer, and send-on-delta adds a mild nonlinearity to an
   otherwise linear recurrence. (Single seed — needs 2 more seeds before this is a claim.)

2. SPARSITY half FAILS at theta=0.10: analog event rate is 0.72, i.e. ABOVE the ~50% figure,
   not "far below" it. The pre-training risk recorded above (0.86) survived training only
   partially (0.86 -> 0.72). So the non-spiking route does NOT automatically buy sparsity —
   it buys the FREEDOM to be sparse (no bound applies to it) without yet exercising it.
   Whether theta / a target-rate penalty can push events well under 0.5 with bpc intact is
   now THE open question, and the honest current answer is: unproven.

3. spikeout confirms its predicted signature: rate_state = 1.00 exactly, rate_emitted 0.41 —
   sparsity lives ONLY in the output, the recurrent state stays fully dense. Mechanism as
   described in the memo. But it is also the WORST variant on quality by a wide margin
   (bpc 4.39 vs 3.32 digital, acc 0.20 vs 0.35) — at this budget, LIF-ing the output costs
   more quality than carrying the state in spikes does. Negative result for SPikE-SSM-style
   at this scale; report as such.

4. FLOOR CONTROL behaves as the bound predicts: spikestate settles at 0.667 firing — above
   0.5, consistent with the firing-floor bound, and it got there with NO pressure to be dense
   (lam=0). It is the only variant inside the bound scope.

### Honest caveats / accounting warnings (do not drop these)
- ENERGY RANKING CURRENTLY FAVORS spikestate (159k pJ), NOT analog (452-628k). This is an
  accounting consequence of WHERE events sit, not a physics win: spikestate turns BOTH W_mix
  and W_out into AC, while analog pays MAC-priced W_in and W_out and only gets AC/event pricing
  on W_mix. Any "analog is the efficient route" claim would be FALSE on these numbers as they
  stand. The energy story for analog depends on driving the event rate down (see open question).
- Absolute quality is LOW for all four (bpc 3.2-4.4 at 6 epochs; the earlier lm_poc ANN reached
  bpc 2.62). All variants are undertrained — comparisons are at matched-but-low quality.
- n=1 seed, one task, one width. Nothing here is a 3-seed claim yet.
- Graded analog events are priced both ways because a graded event is not a 1-bit spike; the
  conservative column is the defensible one for analog.

### NEXT (in order)
(1) seeds 1,2 for all 4 cells -> 3-seed means, turns item 1 into a claim.
(2) analog theta sweep (0.1 / 0.25 / 0.5 / 1.0) + target-rate penalty: can event rate go
    below 0.5, ideally to ~0.1, with bpc holding near 3.18? This decides the whole direction.
(3) copy task at 2-3 memory loads M — the bound is stated in M, charlm does not expose it.

## ANALOG THETA SWEEP — same tick, and it SUPERSEDES the "sparsity half FAILS" verdict above

The verdict above was measured at theta=0.10 ONLY. Sweeping the send-on-delta threshold
(seed 0, everything else identical, lam=0):

| theta | bpc    | acc    | rate_emitted | rate_state | E pJ/tok (cons) |
|-------|--------|--------|--------------|------------|-----------------|
| 0.10  | 3.1804 | 0.3794 | 0.7224       | 0.9925     | 627,811         |
| 0.25  | 3.2112 | 0.3743 | 0.5570       | 0.9935     | 577,941         |
| 0.50  | 3.2362 | 0.3702 | 0.4851       | 0.9927     | 556,265         |
| 1.00  | 3.3290 | 0.3526 | 0.2788       | 0.9923     | 494,091         |
| (digital baseline: bpc 3.3175, activity 1.00) | | | | | |
| (spikestate floor control: bpc 3.6056, activity 0.6668) | | | | | |

### CORRECTED VERDICT: the hypothesis now HOLDS on this task.

At theta=1.0 the analog-state SSM emits events at rate 0.279 -- WELL below the ~0.5 region --
while scoring bpc 3.329 vs the digital baseline 3.318. That is a 0.011 bpc gap (0.35%), i.e.
quality-matched within noise, at ~28% activity. Meanwhile the spiking-STATE control on the
same task with the same parameter count cannot get below 0.667 activity AND is 0.29 bpc worse
(3.606). So: state-in-spikes is stuck dense and pays quality; state-in-analog goes to 28%
activity for free. This is exactly the claim the direction was built to test.

The tradeoff is smooth and cheap: activity 0.72 -> 0.28 (2.6x less communication) costs
0.15 bpc. No collapse, no cliff, no target-rate penalty needed -- theta alone is the knob.
Contrast with the earlier spiking char-LM result, where a strong lam=1.0 penalty could only
move firing 0.63 -> 0.52. The knob works here because it gates COMMUNICATION, not memory.

Why rate_state stays ~0.99 and why that is FINE (and is the whole point): the analog state is
a continuous sub-threshold variable, so it is dense by construction -- but a dense analog state
costs nothing to "carry" on analog/CIM hardware, since no spike is needed to represent it.
In spikestate, dense state == dense spikes == dense energy. The bound rho >= H_b^-1(log2 M / H)
constrains the SPIKE-CARRIED state and therefore simply does not apply to the analog variant.
The sweep is the empirical demonstration of that scope limit.

### Caveats that still stand (unchanged, do not drop)
- n=1 seed, one task, one width. Needs seeds 1,2 before this is a 3-seed claim. NEXT PRIORITY.
- The ENERGY PROXY still ranks spikestate best (159k vs analog 494k pJ/token) because analog
  pays MAC-priced W_in and W_out and only gets event pricing on W_mix. The activity/quality
  result above is the real finding; the pJ column is NOT yet an analog win and must not be
  presented as one. Fixing it means moving more of the datapath onto events/CIM, which is a
  design change, not a rerun.
- Absolute quality is low for all runs (6 epochs; ANN reference bpc 2.62). Quality-matching at
  bpc 3.3 is a weaker statement than quality-matching at bpc 2.6 would be.
- theta=1.0 with rail=4.0 means events fire on ~1/4-rail state changes; worth checking the
  interaction with rail/bits rather than treating theta as independent.


## THREE-SEED RESULTS — charlm row complete (2026-07-30, seeds 0/1/2, A800 GPUs 0-7)

Seeds 1 and 2 for all four variants finished. Regenerate this table any time with
`python agg_ssm3way.py charlm` (pure re-read of the per-run JSON, no retraining).
Config identical across all cells: E=64 H=256 L=128 bs=64 epochs=6 lr=2e-3 lam=0
(no sparsity regularizer, so activity rates are NATURAL), vocab 284,
params 173,596 exactly matched; analog noise=0.02 bits=6 rail=4.0.

| variant | n seeds | bpc | acc | rate_emitted | rate_state | E pJ/tok (cons) |
|---|---|---|---|---|---|---|
| `digital` | 3 | 3.3429 +- 0.0221 | 0.3441 +- 0.0042 | 1.0000 | 1.0000 | 712,448 |
| `spikeout` | 3 | 4.3978 +- 0.0292 | 0.1978 +- 0.0017 | 0.3691 +- 0.0378 | **1.0000 +- 0.0000** | 432,752 +- 2,230 |
| `analog` th=1.0 | 3 | 3.3595 +- 0.0273 | 0.3477 +- 0.0043 | **0.2664 +- 0.0152** | 0.9919 +- 0.0007 | 490,330 +- 4,594 |
| `spikestate` (floor control) | 3 | 3.6205 +- 0.0130 | 0.3016 +- 0.0007 | **0.6453 +- 0.0202** | 0.6453 +- 0.0202 | 156,835 +- 2,509 |
| `analog` th=0.1 / 0.25 / 0.5 | 1 | 3.1804 / 3.2112 / 3.2362 | — | 0.7224 / 0.5570 / 0.4851 | ~0.993 | 627,811 / 577,941 / 556,265 |

Paired per-seed bpc delta vs `digital` at the same seed (a stronger test than
comparing means, since seed variance is shared):

| variant | per-seed delta | mean | sd | same sign in all 3? |
|---|---|---|---|---|
| `spikeout` | +1.0692 / +1.0769 / +1.0184 | **+1.0548** | 0.0318 | yes |
| `analog` th=1.0 | +0.0115 / +0.0140 / +0.0243 | **+0.0166** | 0.0068 | yes |
| `spikestate` | +0.2881 / +0.2727 / +0.2719 | **+0.2776** | 0.0091 | yes |

### Verdict at 3 seeds: the hypothesis HOLDS, with one honest correction

1. **Confirmed.** The analog-state SSM runs at **0.266 +- 0.015 event activity**
   (~3.8x less communication than dense) while the spiking-STATE control with
   identical parameters cannot get below **0.645 +- 0.020**. The two are 0.38 apart
   against a seed sd of ~0.02, so this separation is not seed noise.
2. **Correction to the seed-0 story: analog is NOT literally free.** The paired
   test shows a small but perfectly sign-consistent cost, **+0.017 bpc (0.5%) in
   all three seeds**. The earlier single-seed reading called this "within noise";
   paired, it is a real penalty — it is just **17x smaller** than the spikestate
   penalty (+0.278 bpc, also 3/3 consistent). So the defensible claim is
   "quality-matched to within 0.5%", NOT "free", and NOT "better than digital"
   (the seed-0 theta=0.1 run that beat digital did not generalize into a claim;
   at theta=1.0 across seeds analog is slightly behind).
3. **spikeout (SPikE-SSM-style) reproduces its signature exactly and is the worst
   variant.** `rate_state` is **1.0000 +- 0.0000** across all three seeds — the
   sparsity is output-only, the recurrent state stays fully dense — and it costs
   **+1.05 bpc**. At this budget, LIF-ing the output is the most expensive place to
   put the nonlinearity. Honest negative for that family at this scale.
4. **The floor control behaves as the bound predicts** (0.645 activity, above 0.5,
   reached with zero pressure to be dense, lam=0), and it remains the only variant
   inside the bound scope.

### Caveats unchanged and still binding
- **The pJ proxy still ranks spikestate best** (157k vs analog 490k pJ/token),
  because analog pays MAC-priced `W_in`/`W_out` and only gets event pricing on
  `W_mix`. The finding is activity-vs-quality; **pJ is NOT an analog win yet** and
  must not be presented as one. Fixing it is a datapath design change, not a rerun.
- One task, one width, 6 epochs, absolute bpc low (ANN reference 2.62).
  Quality-matching at bpc 3.34 is a weaker statement than at bpc 2.6.
- theta swept at seed 0 only; only theta=1.0 has 3 seeds.
- theta interacts with rail/bits (at theta=1.0 events fire on ~1/4-rail changes);
  not yet disentangled.

### NEXT (SUPERSEDED — this row ran 2026-07-30 11:25Z; see the 11:40Z copy-row section below)
`copy` task at 2-3 memory loads M for all 4 variants — the bound is stated in M and
charlm does not expose it. That is the experiment that tests the bound M-dependence
across variants, rather than only its scope.

---

## 2026-07-30 ~11:40Z — the `copy` row (M-dependence test): NEGATIVE / INCONCLUSIVE

38 cells: 4 variants x L in {33,65,129} (memory load M=(L-1)/2 = 16/32/64 symbols)
x seeds 0/1/2, plus analog theta in {0.1,0.3} calibration at M=32 seed 0. Analog
cells used the charlm-calibrated theta=1.0. Driver `run_copy_row.sh`, 8 concurrent
(one per idle A800), finished 11:25:29Z. Table regenerable with `agg_copy.py`.

Copy task: alphabet K=16 (so **chance accuracy = 0.0625 and chance bpc = 4.000**),
sequence `[s_1..s_M, DELIM, s_1..s_M]`, loss and metric on the recalled half only.

| M | variant | n | acc | bpc | rate_emitted | rate_state |
|---|---|---|---|---|---|---|
| 16 | `digital` | 3 | 0.4138 +- 0.0033 | 2.4253 +- 0.0093 | 1.0000 | 1.0000 |
| 16 | `spikeout` | 3 | 0.1975 +- 0.0055 | 3.4362 +- 0.0315 | 0.5294 +- 0.0079 | 1.0000 +- 0.0000 |
| 16 | `analog th=1` | 3 | 0.0722 +- 0.0026 | 3.9963 +- 0.0033 | 0.0795 +- 0.0092 | 0.9776 +- 0.0156 |
| 16 | `spikestate` | 3 | 0.0636 +- 0.0023 | 4.0089 +- 0.0002 | 0.4512 +- 0.0144 | 0.4512 +- 0.0144 |
| 32 | `digital` | 3 | 0.2071 +- 0.0018 | 3.3949 +- 0.0119 | 1.0000 | 1.0000 |
| 32 | `spikeout` | 3 | 0.1538 +- 0.0037 | 3.7297 +- 0.0153 | 0.4569 +- 0.0177 | 1.0000 +- 0.0000 |
| 32 | `analog th=0.1` | 1 | 0.1596 | 3.7029 | 0.7402 | 0.9706 |
| 32 | `analog th=0.3` | 1 | 0.1268 | 3.8495 | 0.4158 | 0.9698 |
| 32 | `analog th=1` | 3 | 0.0695 +- 0.0011 | 4.0012 +- 0.0006 | 0.0461 +- 0.0027 | 0.9830 +- 0.0030 |
| 32 | `spikestate` | 3 | 0.0623 +- 0.0013 | 4.0067 +- 0.0023 | 0.3827 +- 0.0543 | 0.3827 +- 0.0543 |
| 64 | `digital` | 3 | 0.1387 +- 0.0046 | 3.8219 +- 0.0189 | 1.0000 | 1.0000 |
| 64 | `spikeout` | 3 | 0.1070 +- 0.0043 | 3.9210 +- 0.0121 | 0.4458 +- 0.0308 | 1.0000 +- 0.0000 |
| 64 | `analog th=1` | 3 | 0.0651 +- 0.0012 | 4.0017 +- 0.0007 | 0.0326 +- 0.0038 | 0.9894 +- 0.0034 |
| 64 | `spikestate` | 3 | 0.0628 +- 0.0006 | 4.0032 +- 0.0005 | 0.3783 +- 0.0421 | 0.3783 +- 0.0421 |

### What this row was supposed to test, and why it did not
The bound `rho >= H_b^-1(log2 M / H)` is stated in the memory load M, so the
prediction was: **spikestate activity RISES with M, analog stays flat.**

Measured spikestate activity: **0.451 (M=16) -> 0.383 (M=32) -> 0.378 (M=64)** —
it does not rise, it drifts slightly *down*.

**That is not evidence against the bound, because the test is vacuous as run.**
The bound constrains a network that actually *carries* M symbols in spikes. At
every M, spikestate sits at chance (bpc 4.008/4.007/4.003 vs the uniform-prediction
value 4.000) — it stores nothing, so there is no M-bits-of-memory for the bound to
price. The same holds for analog at theta=1.0 (bpc 3.996-4.002, also chance).
**No variant solves copy at this budget, and the baseline barely does:** digital
reaches only 0.414 acc at M=16 and decays to 0.139 at M=64. A capacity/optimisation
failure of the *shared* backbone cannot discriminate between variants' memory codes.
**Honest status of the M-dependence claim: UNTESTED, not refuted.**

### Two real findings the row did produce
1. **theta does NOT transfer across tasks.** theta=1.0, calibrated on charlm (where
   it gave 27% activity at a 0.5% bpc cost), *collapses* analog to chance on copy at
   every M, emitting only 3-8% events. The M=32 sweep shows the working range is far
   lower: theta=0.1 -> acc 0.160 @ 0.740 emitted, theta=0.3 -> acc 0.127 @ 0.416.
   Any future claim must carry per-task theta calibration; the charlm headline
   number is not a task-independent operating point.
2. **At matched communication activity, analog-state still beats spiking-state**
   — the one direction-consistent signal here. At M=32, analog theta=0.3 emits
   0.416 for acc 0.127, spikestate emits 0.383 for acc 0.062 (chance). So ~2x the
   above-chance margin at comparable event rate. **n=1 seed, degraded regime —
   suggestive only, not a result.**
3. Incidental: `spikeout` is the best non-digital variant on copy (0.198/0.154/0.107),
   inverting its charlm ranking where it was worst. Its `rate_state` is again
   **exactly 1.0000** at every M — the SPikE-SSM signature (dense state, sparse
   output) is the most robust single observation across both tasks.

### Fix required before the M-dependence test means anything
The gate is the *baseline*: digital must actually solve copy (acc -> ~1.0 at M=16)
before variant comparison is informative. Sequence: (i) cheap single-cell probe —
digital, M=16, more epochs (25-30) and/or more sequences, confirm it saturates;
(ii) per-task theta calibration for analog at the working M; (iii) only then rerun
the full row. Do not report the current copy table as a bound test.

### Effect on the charlm headline
None — charlm (3 seeds) stands as previously recorded. But the copy row does bound
its generality: the analog-state advantage is demonstrated on char-LM at one theta,
and did **not** reproduce on a second task under transferred hyperparameters.

## Baseline-gate probe (2026-07-30 ~20:32 server-local): copy baseline was BUDGET-limited, not task-limited

Question: is the digital copy baseline's 41% acc at L=33 (M=16, seed 0) a budget limit or a task limit at this model size (87,889 params)?

| run | epochs | copy_n | acc | bpc |
|---|---|---|---|---|
| original row cell | 6 | 20,000 | 0.4108 | 2.4172 |
| probeA | 30 | 20,000 | 0.5732 | 1.7283 |
| probeB | 30 | 80,000 | 0.6388 | 1.3983 |

(chance acc = 0.0625, chance bpc = 4.000; files `ssm3way_runs/digital_copy_L33_s0_probeA_ep30.json`, `..._probeB_ep30_n80k.json`)

**Verdict: budget-limited.** Acc rises monotonically with both epochs (6→30: +0.16) and data (20k→80k: +0.066 more) and has not saturated. Not solved (0.64 << 1.0), but the baseline now sits 10× above chance — enough dynamic range to compare variants. **Consequence: the INCONCLUSIVE copy row verdict stands for the OLD (6-epoch) table, and a rerun at ep30/n80k is justified.** Prerequisite before the rerun: per-task θ calibration for analog at this budget (charlm θ=1.0 is known to collapse analog on copy; working range at M=32 was θ≈0.1–0.3 at the old budget, must be re-checked at ep30/n80k). Use new `--out` suffixes (e.g. `_ep30`) so the idempotent driver does not skip.


## θ calibration at the new budget (copy, L=65 / M=32, seed 0, ep30 + copy_n 80k) — PARTIAL (4/5 θ in)

Recorded 2026-07-30 ~22:10 server-local. Driver `run_calib_ep30.sh`. Chance: acc 0.0625, bpc 4.000.
Still in flight at time of writing: analog θ=0.5, and the digital reference at L=65 (same budget).

| variant | θ | acc | bpc | rate_emitted | rate_state |
|---|---|---|---|---|---|
| analog | 0.05 | 0.1568 | 3.717 | 0.5989 | 0.9549 |
| analog | 0.10 | 0.1568 | 3.717 | 0.5989 | 0.9549 |
| analog | 0.20 | 0.1420 | 3.786 | 0.4547 | 0.9646 |
| analog | 0.30 | 0.1273 | 3.863 | 0.2679 | 0.9502 |
| analog | 0.50 | (running) | | | |
| digital | -- | (running, L=65) | | | 1.0 |

Reference from the earlier budget probe, at a *different* length: digital L=33 (M=16) at ep30/n80k
reached acc 0.5732, bpc 1.7283. Not comparable to the rows above; the L=65 digital cell is the
apples-to-apples one and it is still training.

**Finding 1 — θ=0.05 and θ=0.10 are bit-identical.** Every metric agrees to all printed digits, not merely
approximately. This is not a sweep bug; it is the analog datapath's own resolution floor. The state is
quantized to `bits=6` over rails +/-4, so one quantization step is 8 / 2^6 = **0.125**. A send-on-delta
threshold below one LSB cannot discriminate anything, so every θ under 0.125 collapses onto the same
behaviour: the ADC step, not θ, sets the minimum achievable event rate. The usable θ range on this datapath
therefore starts at about 0.125, and "smaller θ buys more accuracy" saturates there
(acc 0.157 at emitted 0.599). Consequence for the memo and the paper: θ and bit-depth are **not
independent knobs**, so any θ claim must state `bits` alongside it. If a finer event rate is genuinely
wanted, the lever is more bits, not a smaller threshold.

**Finding 2 — a monotone accuracy/activity tradeoff that is real at this budget.** Accuracy falls
0.157 -> 0.142 -> 0.127 as the emitted rate falls 0.599 -> 0.455 -> 0.268, and all three points sit above
chance (0.0625). At the old 6-epoch budget analog sat exactly at chance for every θ, so the ep30/n80k
budget did buy analog a working regime. This is independent confirmation of the digital probe's verdict
that the old copy row was budget-limited rather than task-limited.

**Finding 3 — honest negative on absolute quality.** Analog's best point here is acc 0.157 at M=32, only
about 2.5x chance, whereas digital at the *easier* M=16 reached 0.573. Until the L=65 digital reference
lands there is no matched comparison, so do **not** claim analog is quality-matched on copy. The char-LM
result stands on its own evidence; copy remains an open second-task replication, not yet a confirmation.

**θ selection (provisional, pending θ=0.5 and the digital L=65 reference):** θ=0.2 is the pick for the full
row. It holds accuracy within 10% of the sub-LSB best (0.142 vs 0.157) at a clearly-sub-1.0 event rate
(0.455), and unlike θ=0.05/0.10 it sits above the 6-bit LSB, so the threshold is actually the operative
knob rather than an alias for the quantizer.

## 2026-07-30 ~22:40 (server local) / 14:40Z — θ calibration COMPLETE (6/6), and the matched digital reference is an HONEST NEGATIVE

All six ep30 calibration cells finished (copy, L=65 → memory load M=32, seed 0, `--epochs 30 --copy_n 80000`;
alphabet K=16 so chance acc = 0.0625 and chance bpc = 4.000). n=1 seed — suggestive, not settled.

| variant | θ | acc | bpc | rate_emitted | rate_state |
|---|---|---|---|---|---|
| digital (reference) | — | **0.3184** | **2.879** | 1.000 | 1.000 |
| analog | 0.05 | 0.1568 | 3.717 | 0.5989 | 0.955 |
| analog | 0.10 | 0.1568 | 3.717 | 0.5989 | 0.955 |
| analog | 0.20 | 0.1420 | 3.786 | 0.4547 | 0.965 |
| analog | 0.30 | 0.1273 | 3.863 | 0.2679 | 0.950 |
| analog | 0.50 | 0.0700 | 3.994 | 0.0271 | 0.884 |

**THE NEGATIVE — the analog-state quality match does NOT replicate on copy.** With a matched-budget digital
reference finally in hand, analog is *not* quality-matched on this task: at the calibrated θ=0.2 it reaches
acc 0.142 / bpc 3.786 against digital's 0.318 / 2.879 — a **+0.91 bpc** gap that retains only **~31 %** of the
baseline's above-chance accuracy margin (0.0795 vs 0.2559). Compare char-LM, where the same variant cost
**+0.017 bpc (0.5 %)** at 3 seeds. So the analog-state result is **task-conditional, not general**, and the
char-LM headline must be stated as a char-LM result. Even analog's *best* setting (the sub-LSB θ, emitting
60 % — i.e. barely sparse at all) only reaches 0.157.

**Mechanistic reading (consistent with the rest of the paper, and it sharpens rather than breaks the story):**
gating *communication* is cheap when the task needs sequence *statistics* (char-LM), and expensive when the
task needs *precise retention* of M specific symbols (copy). The analog state is lossy by construction —
σ=0.02 injected noise, 6-bit quantization over ±4 rails, plus send-on-delta suppression of small updates —
and copy is exactly the regime that cannot absorb that loss. Scope claim for the Imam answer: a CIM/analog
SSM host fits statistical sequence workloads, not precise-recall workloads.

**θ pick CONFIRMED at 0.2**, for the reason it was provisionally chosen: it sits above the 6-bit quantizer LSB
(q = 2·rail/2^bits = 0.125), so the threshold is the operative knob, and it holds accuracy within ~10 % of the
inoperative sub-LSB best while cutting emissions 60 %→45 %. θ=0.5 over-gates and collapses to chance
(acc 0.070 @ 2.7 % emitted), so the usable window at this budget is narrow: **0.125 < θ ≲ 0.3**.

**LAUNCHED (14:40Z): the full copy row at the new budget** — `run_copy_row_ep30.sh`, 36 cells =
4 variants × M ∈ {16,32,64} (L = 2M+1 ∈ {33,65,129}) × seeds 0/1/2, at ep30/n80k, analog at θ=0.2, output
suffix `_ep30`. All 8 A800s were idle at 1 MiB and verified unowned before launch; 8 concurrent chains, one
per GPU. Jobs are ordered by **ascending L** so the cheap M=16 cells land first and give a usable partial row
early. Idempotent (skips cells whose JSON exists), so never re-train after an SSH drop — but check
`ps aux | grep "ssm3way.py"` first; note `grep ssm3wa[y]` false-positives on any ssh command line that
mentions `ssm3way_runs`. Markers in `copy_logs/row_ep30.log`. Rough cost ~40 GPU-h ≈ 5 h wall.
**What the row now tests:** (i) whether the analog↔digital gap **widens with M** (prediction: yes, if the
lossy-memory reading is right), and (ii) spikestate's activity-vs-M — the actual M-dependence test of the
firing-floor bound, which the old 6-epoch row could not perform because every variant sat at chance.

### 2026-07-30 15:40Z — copy row ep30: the M=16 column is COMPLETE at 3 seeds, and it INVERTS the char-LM ranking

Row still running (8 jobs alive, 20/36 `_ep30` cells on disk; M=32 has digital at 2 seeds, M=64 not
started). The M=16 column is done for all four variants at seeds 0/1/2, so it can be read now.
Copy, K=16 alphabet, chance acc 0.0625 / bpc 4.000. Budget ep30 / copy_n 80k, analog theta=0.2.
Table regenerable with `python agg_copy.py 30`.

| variant | acc (3 seeds) | bpc | rate_emitted | rate_state | pJ/tok (cons) | margin kept |
|---|---|---|---|---|---|---|
| `digital`    | 0.6302 +- 0.0106 | 1.4201 +- 0.0343 | 1.0000 | 1.0000 | 398029 | 1.00 |
| `spikeout`   | 0.5833 +- 0.0118 | 1.5446 +- 0.0708 | 0.4469 +- 0.0261 | 1.0000 +- 0.0000 | 122920 | **0.92** |
| `analog th=0.2` | 0.3605 +- 0.0505 | 2.7769 +- 0.2261 | 0.5011 +- 0.0347 | 0.9717 | 246674 | 0.53 |
| `spikestate` | 0.1071 +- 0.0164 | 3.9292 +- 0.0369 | 0.4960 +- 0.0106 | 0.4960 +- 0.0106 | 107744 | 0.08 |

("margin kept" = fraction of digital's above-chance accuracy margin the variant retains; the right
metric here because absolute accuracies are low. Paired per-seed deltas vs digital: spikeout
dbpc +0.124 +- 0.043, analog +1.357 +- 0.253, spikestate +2.509 +- 0.060, all n=3.)

**THE RESULT: the variant ranking is task-conditional and flips.** On char-LM (3 seeds) the order
was analog (+0.017 bpc) << spikestate (+0.278) << spikeout (+1.055) — spikeout WORST. On copy at
M=16 (3 seeds) it is spikeout (+0.124) << analog (+1.357) << spikestate (+2.509) — spikeout BEST,
by a wide and well-powered margin (seed sd 0.012 on acc). This is no longer an n=1 curiosity from
the old 6-epoch row; it replicates at 3 seeds at a budget where the baseline is 10x above chance.

**Mechanistic reading, and it is consistent with the earlier one.** Copy needs precise retention of
M specific symbols. The two variants that keep the recurrent state *exact* (digital: continuous;
spikeout: continuous state, LIF only on the output) are the two that do well — spikeout's
`rate_state` is again exactly **1.0000 +- 0.0000**, so it pays nothing in state fidelity and buys
its 3.2x energy cut purely on the output path. The two variants that make the state *lossy* pay for
it: analog (noise sigma=0.02, 6-bit over +-4 rails, send-on-delta suppression) keeps 0.53 of the
margin, and spikestate (state quantized to 1 bit) keeps 0.08 and is barely above chance. So the
axis that predicts copy performance is **state fidelity**, not sparsity: at nearly identical
emitted rates (0.45 / 0.50 / 0.50) the three non-digital variants span 0.92 -> 0.53 -> 0.08.

**What this does to the headline.** It does not touch the char-LM result (unchanged, 3 seeds), but
it forces the paper/memo to state the finding as a *dichotomy over workloads*, not a ranking of
variants: for statistical sequence tasks the analog state is the cheap route (27% activity, 0.5%
quality cost) and output-spiking is the expensive one; for precise-recall tasks it is exactly
reversed, and output-spiking is the only sparsification that survives. Neither variant dominates.
That is a more interesting claim than "analog wins", and it is now the better-powered one.

**Energy footnote (honest):** on copy at M=16 spikeout is also the best quality-per-pJ point by a
distance — 0.92 margin kept at 122.9k pJ/tok vs digital's 398.0k, a 3.2x cut. Analog is *worse*
than spikeout on both axes here (0.53 margin at 246.7k pJ), for the same datapath reason logged
earlier: analog only gets event pricing on W_mix and still pays MAC-priced W_in/W_out.

**Still open, unchanged:** the firing-floor bound's M-dependence. spikestate emits 0.496 at M=16 —
this time genuinely above chance (acc 0.107 vs 0.0625), unlike the 6-epoch row where it stored
nothing, so the M=32 and M=64 columns will for the first time be a real test of whether spikestate
activity rises with M. Do not read M-dependence off the single M=16 point. Weak prior signal only:
analog margin kept 0.53 at M=16 (n=3) vs 0.31 at M=32 (n=1), consistent with the gap widening.
