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

### 2026-07-31 (tick) — figure script + a correction to what the copy row can actually test

The ep30 copy row is still running (17/36 main cells done, 8 alive; M=16 complete at 3
seeds, M=32 partial, M=64 just started). No GPU was idle, so this tick did a zero-GPU
step: `plot_ssm3way.py`, the two-panel figure the finished row will need, plus the
derivation check below.

**Correction — the firing-floor bound predicts far more than we had been assuming, because
`M` in `rho >= H_b^-1(log2 M / H)` counts distinguishable memory STATES, not symbols.**
Copying M symbols from a K=16 alphabet requires distinguishing K^M states, so the numerator
is `M * log2 K` bits, not `log2 M`. At H=256:

| M (symbols) | bits to retain | predicted floor rho |
|---|---|---|
| 16 | 64 | 0.042 |
| 32 | 128 | 0.110 |
| 64 | 256 | **0.500** |

Under the old (wrong) symbol reading the predicted floors were 0.0018 / 0.0023 / 0.0028 —
three orders below anything measurable, which would have made the whole row uninformative.
The state reading makes M=64 a genuinely sharp prediction.

**What this means for the pending columns, stated before the data lands so it cannot be
fitted after the fact:**
- Measured spikestate state activity at M=16 is **0.4960 +- 0.0106** (seeds 0/1/2) against a
  predicted floor of 0.042. The bound is a LOWER bound, so this is consistent but **12x
  loose — non-binding, therefore not a test.** Same will be true at M=32 (floor 0.110).
- **M=64 is the only informative cell in the row**: the floor rises to 0.500, i.e. right at
  the activity spikestate already shows. If spikestate retains the sequence there and its
  activity stays near 0.5 or climbs, that is the first real confirmation of M-dependence.
- **But the binding condition is retention, and retention is already failing.** At M=16
  spikestate keeps only 0.08 of the digital baseline's above-chance margin (acc 0.107 vs
  digital 0.630). A net that stores nothing is trivially consistent with any floor. So the
  most likely M=64 outcome is *the bound is satisfied and uninformative*, and the honest
  write-up of that is "untested", not "confirmed".
- Practical consequence for a real test later: the bound only bites when `M*log2 K / H`
  approaches 1, i.e. when the memory demand approaches the width. Testing it properly means
  **shrinking H** (e.g. H=64 at M=16 gives a floor of 0.50 on a task the net can still
  learn), not pushing M up on a net that has already stopped learning. That is a cheap
  follow-up row and is the right way to get a binding test.

`plot_ssm3way.py` (committed) renders both panels from logged JSON only: (a) margin kept vs
M per variant, (b) activity vs M with the bound curve overlaid and shaded. It de-duplicates
the theta-calibration cell against the row cell, excludes non-theta=0.2 analog cells, and
annotates any cell with n<3 rather than silently averaging. Regenerate with
`python plot_ssm3way.py 30` -> `fig_ssm3way.pdf`.

## 2026-07-30 ~16:38Z (server local) — ep30 copy row: the M=32 column is COMPLETE at 3 seeds for digital / spikeout / analog, and the prediction that the gap WIDENS with M is CONFIRMED for both non-exact-state variants

Row still running: 20/36 main cells, 8 jobs alive (3× spikestate L=65 finishing the M=32
column, 3× digital + 2× spikeout at L=129 starting the M=64 column). Numbers below are from
`agg_copy.py 30`, all paired per-seed against the digital cell at the same (L, seed).
Copy, K=16, chance acc 0.0625 / bpc 4.000.

| M  | variant | n | acc | bpc | emitted | rate_state | margin kept |
|----|---------|---|-----|-----|---------|------------|-------------|
| 16 | digital | 3 | 0.6302 ± 0.0106 | 1.4201 | 1.000 | 1.000 | 1.00 |
| 16 | spikeout | 3 | 0.5833 ± 0.0118 | 1.5446 | 0.447 | 1.0000 ± 0.0000 | **0.92** |
| 16 | analog θ=0.2 | 3 | 0.3605 ± 0.0505 | 2.7769 | 0.501 | 0.972 | **0.53** |
| 16 | spikestate | 3 | 0.1071 ± 0.0164 | 3.9292 | 0.496 | 0.496 | **0.08** |
| 32 | digital | 3 | 0.3150 ± 0.0031 | 2.8928 ± 0.0127 | 1.000 | 1.000 | 1.00 |
| 32 | spikeout | 3 | 0.1796 ± 0.0009 | 3.6045 ± 0.0025 | 0.485 | 1.0000 ± 0.0000 | **0.46** |
| 32 | analog θ=0.2 | 3 | 0.1419 ± 0.0003 | 3.7877 ± 0.0019 | 0.449 | 0.964 | **0.31** |
| 32 | spikestate | — | (running) | | | | |

Paired Δbpc vs digital at M=32 (n=3): spikeout **+0.7117 ± 0.0113**, analog θ=0.2
**+0.8950 ± 0.0108**. At M=16 those were +0.1245 ± 0.0426 and +1.3569 ± 0.2530.

**FINDING 1 — the gap widens with memory load, for BOTH non-exact-state variants.**
Margin kept falls 0.92 → 0.46 for spikeout and 0.53 → 0.31 for analog when M goes 16 → 32.
This is the prediction registered before the column landed (lossy state should cost more as
the task demands retaining more), and it holds. Note it is *not* analog-specific: spikeout's
margin drops by more in absolute terms (−0.46 vs −0.22), i.e. even a fully continuous state
loses ground once the *output* is spiked and the sequence to be recalled gets longer. The
seed sd at M=32 is tiny (≤0.003 acc for every variant), so the widening is far outside noise.

**FINDING 2 — the state-fidelity ordering survives the harder column.** At M=32, at nearly
matched emitted rates (0.485 vs 0.449), spikeout keeps 0.46 and analog 0.31 — same order as
M=16, and again ordered by how exact the recurrent state is rather than by how sparse the
communication is. The spikestate cell that would complete the ordering is still training.

**FINDING 3 (honest, and it cuts against a "sparser is the cost" reading) — analog does not
buy its quality loss back with communication.** Its emitted rate actually *falls* slightly
from M=16 to M=32 (0.501 → 0.449) at the same θ=0.2 while its margin kept drops from 0.53 to
0.31. So the extra loss is not the net choosing to send less; the fixed send-on-delta
threshold simply prices a harder task worse. Any energy/quality curve for the analog route
must therefore be drawn per memory load, not once.

**What this does NOT show.** Nothing here tests the firing-floor bound. Under the corrected
reading (`M·log₂K` bits, floors 0.042 / 0.110 / 0.500 at M = 16 / 32 / 64 for H=256), the
M=16 and M=32 columns are non-binding by 4–12×, exactly as pre-registered. Only the M=64
column, now training, is binding — and the retention caveat stands: spikestate keeps 0.08 of
the margin at M=16, so a satisfied bound at M=64 will most likely be *uninformative*. The
recommended real test remains shrinking H (H=64 at M=16 → floor 0.50 on a learnable task).

**Effect on the paper claim: none to the char-LM headline; the workload dichotomy gets
stronger.** char-LM (3 seeds): analog +0.017 bpc ≪ spikestate +0.278 ≪ spikeout +1.055.
Copy (3 seeds, both M): spikeout ≪ analog ≪ spikestate, and the ranking is now shown to be
stable across two memory loads rather than at a single point. Energy footnote unchanged and
still unfavourable to analog: at M=32, spikeout 125.2k pJ/tok at 0.46 margin vs analog 231.0k
at 0.31 — analog is worse on both axes on this task, for the datapath reason already logged
(only W_mix gets event pricing).

### 2026-07-31 ~01:10 (server local) / 2026-07-30 17:10Z — the M=32 FLOOR CONTROL landed, and it KILLS the copy task as a test of the bound's M-dependence

Status: ep30 copy row 25/36 cells, 8 jobs alive on the M=64 (L=129) column, all 8 A800s busy.
New since last tick: `spikestate_copy_L65_s{0,1,2}_ep30.json` (M=32 floor control, 3 seeds) and
`digital_copy_L129_s2_ep30.json` (first M=64 cell).

**spikestate at M=32, 3 seeds: acc 0.0623 +- 0.0012, bpc 4.0050 +- 0.0036 — that is EXACTLY chance**
(chance acc 1/16 = 0.0625, chance bpc = log2 16 = 4.000). Paired vs digital: dacc -0.2527 +- 0.0037,
dbpc +1.1122 +- 0.0132, **margin kept 0.00**. Per-seed rate_state 0.4211 / 0.3711 / 0.4233.

**THE FINDING — the copy task CANNOT test the firing-floor bound's M-dependence, and this is now
settled rather than pending.** Measured spikestate state activity *falls* with load: 0.4960 +- 0.0106
(M=16) -> 0.4052 +- 0.0295 (M=32), the opposite of the bound's "activity rises with M" prediction.
But that is **not** evidence against the bound, for the reason pre-registered before this column
landed: the bound `rho >= H_b^-1(M*log2K / H)` prices a network that actually *retains* the sequence,
and at M=32 spikestate retains **nothing at all** (margin kept 0.00, indistinguishable from chance).
Activity falls because the net has given up on retention, not because it beat a floor. Under the
corrected `M*log2K` reading at H=256 the predicted floors are 0.042 / 0.110 / 0.500 for M = 16/32/64,
so M=32's measured 0.405 is satisfied-but-3.7x-loose — non-binding, exactly as pre-registered.

**Consequence for the running M=64 column: it will not rescue the test either.** spikestate is already
at chance at M=32; it cannot plausibly retain 64 symbols. M=64 is the only load where the bound binds
(floor 0.500), but binding requires a net that stores the information, and this one demonstrably does
not. So the honest verdict is structural, not a matter of waiting: **spikestate's learnability ceiling
on copy (~M=16, acc 0.107 vs chance 0.0625) sits BELOW the load at which the bound starts to bind
(M=64). Raising M can never produce a binding-and-meaningful test on this task.** Report the bound's
M-dependence as **UNTESTED, and untestable by this route** — do not cite any copy column as a bound test.

**This promotes the previously-recommended alternative from "nice follow-up" to "the only viable test":
shrink H, don't raise M.** The bound bites when `M*log2K / H -> 1`. At M=16 — a load spikestate can
still partially learn — H=64 gives `16*4/64 = 1.0`, i.e. a predicted floor of **0.500** on a
still-learnable task, and H=96/128 give floors of 0.174/0.042 as a graded series. That is a cheap row
(smaller nets than the current ones) and it is the experiment that can actually confirm or refute the
bound. Recommended as the next launch once the M=64 column frees the GPUs.

**Third reading — the state-fidelity ordering extends to a zero:** at M=32 the margin-kept series by
state exactness is continuous 0.46 (spikeout) > lossy-analog 0.31 > 1-bit 0.00 (spikestate), at matched
emitted rates ~0.41-0.49. A 1-bit recurrent state does not merely degrade precise-recall performance at
this load, it eliminates it. This sharpens the workload dichotomy: for precise-recall workloads the
state must stay in an exact datapath; spiking the state is not a degraded option, it is a non-option.
Energy footnote: spikestate M=32 is the cheapest cell (102.0k pJ/tok) and buys literally nothing, which
is the cleanest illustration that a pJ number is meaningless without the quality it purchased.

Reproduce: `python agg_copy.py 30` in /work/zeyuwang/neuro_poc (budgets never pooled; ep=6 row remains
labeled INCONCLUSIVE). Configs/seeds in each `ssm3way_runs/*.json`.

## Shrink-H row — the only viable test of the firing-floor bound (LAUNCHED 2026-07-30T17:39Z)

The copy row settled that **M-dependence of the bound is untestable on this task**: a
spiking-state SSM is already at exactly chance by M=32 (margin kept 0.00), so it never
reaches the load M=64 where the bound starts to bind. Binding requires storage; this net
stores nothing at the binding load.

The bound `rho >= H_b^-1(M*log2K / H)` also binds when **H shrinks**, and that route keeps
the task inside the net's learnable range. So: hold M=16 (L=33, K=16 => 64 bits to retain),
sweep the hidden width.

| H | M*log2K / H | predicted floor rho | status |
|---|---|---|---|
| 64  | 1.000 | **0.500** | new — the binding cell |
| 96  | 0.667 | 0.174 | new |
| 128 | 0.500 | 0.042 | new |
| 256 | 0.250 | 0.0026 | already in the ep30 main row (non-binding; measured 0.4960+-0.0106) |

Row: 18 new cells = {digital, spikestate} x H in {64,96,128} x seeds {0,1,2}, copy L=33,
epochs 30, copy_n 80000, out suffix `_H<H>_ep30`. Driver `run_shrinkH.sh <gpu>` is a
lock-dir work queue (atomic `mkdir` claim per cell) so one worker per GPU can be added as
GPUs free, with no duplicated cells and no re-training on restart. Cells are ordered
H-then-seed-then-variant so a matched digital/spikestate pair at the binding H=64 lands
first. First worker started on GPU 4 (the only idle GPU; the other 7 are still on the
ep30 M=64 column).

**Pre-registered reading, written before any cell lands:**
1. The test is only valid if the **digital reference at that H still learns** copy
   meaningfully above chance (acc 0.0625). If digital collapses at H=64, that H is
   capacity-limited and tells us nothing about the bound — report it as such, do not read
   spikestate's activity there as a floor confirmation.
2. **Confirmation** = on the H values where digital still learns and spikestate retains a
   non-trivial margin, spikestate's measured state activity sits at or above the predicted
   floor AND **rises as H shrinks**, tracking 0.0026 -> 0.042 -> 0.174 -> 0.500. The
   measured H=256 point is 0.4960, i.e. already far above its own floor, so the
   discriminating observable is the **trend**, not any single cell.
3. **Refutation** = activity stays flat near ~0.5 across all H (or falls) while the nets
   still learn. That would mean ~0.5 is an optimization artifact of the LIF state, not the
   information-theoretic floor, and the bound's empirical support would be nil.
4. **Most likely outcome, stated up front:** spikestate loses retention as H shrinks (it
   already keeps only 0.08 of digital's margin at H=256/M=16), so the honest result may
   again be "satisfied but uninformative". If retention dies before the floor binds, the
   verdict is that **the bound is not empirically testable with this architecture at this
   scale** — which is a real, reportable negative, not a failed experiment.

No results yet.

### 2026-07-30 ~18:10Z — shrink-H row: the VALIDITY GATE PASSES at the binding H, and it passes surprisingly well

Status: ep30 copy row still running (M=64 column, 6 jobs on GPUs 0/1/3/5/7 + shrink-H on 4).
GPUs 2 and 6 freed (their row chains reported CHAIN DONE) and were handed to two new
shrink-H workers; the lock queue claimed distinct cells (`digital_copy_L33_s1_H64_ep30`
on gpu2, `spikestate_copy_L33_s1_H64_ep30` on gpu6) with no duplicate training.
3 shrink-H workers now live (gpu 4/2/6).

**First shrink-H cell (`digital_copy_L33_s0_H64_ep30`): acc 0.5556 / bpc 1.8478**, vs
chance 0.0625 / 4.000. Pre-registered criterion (i) — "the test only counts at an H where
the *digital* reference still learns copy well above chance" — is therefore **SATISFIED at
H=64, the single binding cell** (predicted floor 0.500 under the corrected `M·log2K/H`
reading). The shrink-H row is a valid experiment, not a capacity-limited one.

**Stronger than the gate required, and worth noting on its own:** H=64 digital uses only
**10,513 params** (vs 87,889 at H=256) — a 8.4x smaller net — yet retains most of the
baseline's skill (0.556 vs 0.6302+-0.0106 at H=256, n=1 vs n=3). So copy at M=16 is not
width-limited in the 64..256 range, which is exactly what makes shrinking H the right lever
for a bound test: it moves `M·log2K/H` from 0.0026 to 1.0 (a 385x change in the predicted
floor) while barely moving what the task demands of the digital reference. Raising M could
not do this — it killed retention (spikestate is at chance by M=32).

Still pending and still the whole point: the matched `spikestate` H=64 cells. Criterion (ii)
confirmation = state activity at/above 0.500 AND rising as H shrinks across
H=256/128/96/64 (measured H=256 = 0.4960+-0.0106); criterion (iii) refutation = flat ~0.5
or falling while nets still learn; criterion (iv) most-likely = spikestate retention dies
before the floor binds -> verdict "not empirically testable with this architecture", a
reportable negative. Nothing about the gate passing changes those criteria.

---

### 2026-07-30 ~18:40Z — SHRINK-H BOUND ROW: the H=64 binding cell has both arms in, and the measured activity is BELOW the predicted floor — which, read correctly, turns the bound into a *capacity certificate* rather than a firing-rate prediction

Cells: copy L=33 (M=16, K=16), ep30/n80k, H=64 (10,513 params). Predicted floor under the
corrected `M·log₂K/H` reading: **0.500** (this is the one binding cell of the row).

| variant | n | acc | bpc | state activity ρ | margin kept | pJ/token |
|---|---|---|---|---|---|---|
| digital    | 3 | 0.5600 ± 0.0095 | 1.8275 ± 0.0563 | 1.000 | 1.00 | 42,982 |
| spikestate | 2 | 0.1614 (0.1576/0.1651) | 3.6499 (3.6485/3.6512) | **0.2562** (0.2786/0.2338) | **0.20** | 20,332 |

(chance acc 0.0625, chance bpc 4.000; seed 2 of spikestate still training)

**Against the pre-registered criteria:**
- (i) validity gate — **PASSES** at 3 seeds now, not 1: digital keeps 0.560 ± 0.010 at H=64,
  i.e. 89% of its H=256 accuracy (0.6302) with 8× fewer params. The task is not width-limited.
- (ii) confirmation (activity at/above floor AND rising as H shrinks) — **NOT MET.** The two
  points now on the trend line go the *wrong* way: ρ = 0.496 ± 0.011 at H=256 (floor 0.0026)
  → ρ = 0.256 at H=64 (floor 0.500). Activity **falls** as H shrinks, and at H=64 it lands
  **below** the predicted floor.
- (iii)/(iv) — the sub-floor measurement is *not* a refutation, because the bound's precondition
  is a net that actually retains M·log₂K bits, and this one does not: margin kept 0.20.

**The reading that makes the sub-floor number informative (and is the actual result of this row).**
A bound violation is impossible for a net that carries the required information, so measuring
ρ < floor is direct evidence the state is *not* carrying it. Inverting the bound gives a usable
capacity certificate: at ρ = 0.2562 and H = 64 the spiking state can encode at most
`H·H_b(ρ) = 64 × 0.8211 = 52.6 bits`, against the `M·log₂K = 64 bits` copy at M=16 demands —
an **11-bit deficit, i.e. capacity for ~13 of the 16 symbols.** The bound therefore *predicts
failure at this width*, and the net does fail (acc 0.161 vs digital 0.560, margin kept 0.20).
This is the first cell in the whole project where the bound makes a checkable prediction that
the data bears out — but note it is a prediction of **incapacity**, not of a firing floor.

**Honest status of the firing-floor claim: still UNCONFIRMED, and now under real pressure.**
Across H = 256 → 64 the LIF state's activity sits near 0.25–0.50 regardless of a floor that
moves 190× (0.0026 → 0.500). That is the signature of an **optimization/architecture artifact
of the LIF state**, not of an information-theoretic floor — exactly criterion (iii)'s shape.
The H=96 (floor 0.174) and H=128 (floor 0.042) cells now running are what discriminate: if ρ
tracks ~0.25–0.5 at those widths too, the flat-artifact reading wins and the paper should say
so plainly.

**Do not write "the bound is confirmed" anywhere.** The defensible sentence today is: *the
firing-floor bound is only usable in its contrapositive — as a certificate that a given (ρ, H)
cannot hold the task's memory — and the measured LIF activity does not track the predicted
floor as width varies.*

Energy footnote, unchanged in direction: spikestate is the cheaper cell (20.3k vs 43.0k pJ/token)
and buys 0.20 of the margin — the same "cheap and useless" pattern as the M=32 floor control.

## 2026-07-30 ~19:35Z — shrink-H row: H=64 (the binding cell) complete at 3 seeds, H=96 partially in. The bound fails its pre-registered test in a *stronger* way than "flat", AND the contrapositive reading from the last tick must be RETRACTED as non-diagnostic.

Setup: copy, L=33 (M=16, K=16 => 64 bits to retain), ep30 / copy_n 80000, chance acc 0.0625 / bpc 4.000.
Predicted floors under the corrected `M*log2(K)/H` reading: H=64 -> 0.500 (binding), H=96 -> 0.174, H=128 -> 0.042, H=256 -> 0.0026.

| H | params | digital acc | digital bpc | spikestate acc | spikestate bpc | spikestate rho (state activity) | predicted floor | margin kept | n |
|---|---|---|---|---|---|---|---|---|---|
| 256 | 87,889 | 0.6302 +- 0.0106 | 1.4201 | 0.1071 +- 0.0164 | 3.9292 | **0.4960 +- 0.0106** | 0.0026 | 0.08 | 3/3 |
| 128 | (running) | (running) | | (running) | | | 0.042 | | 0/0 |
| 96 | 18,289 | 0.5869 +- 0.0078 | 1.6351 | 0.1634 | 3.6635 | **0.3522** | 0.174 | 0.19 | 2 / 1 |
| 64 | 10,513 | 0.5600 +- 0.0095 | 1.8275 | 0.1619 +- 0.0039 | 3.6562 | **0.2496 +- 0.0252** | 0.500 | 0.20 | 3/3 |

**Validity gate (pre-registered criterion (i)): PASSES at every measured H.** The digital reference still learns copy far above chance at all widths (0.560 / 0.587 / 0.630 vs chance 0.0625), and it does so with 8x fewer params at H=64. So copy at M=16 is not width-limited anywhere in H in [64,256], which is exactly what makes this a real bound test rather than a capacity-limited one.

**Criterion (ii) is REFUTED, and not merely by flatness — the trend is ANTI-correlated.** Pre-registration required spikestate's state activity to *rise* as H shrinks (tracking floors 0.0026 -> 0.042 -> 0.174 -> 0.500). Measured rho instead *falls monotonically* across three widths: **0.4960 (H=256) -> 0.3522 (H=96) -> 0.2496 (H=64)**, while the predicted floor rises 190x. The two curves cross between H=96 and H=64. This is criterion (iii)'s shape in its strongest form: measured LIF state activity is set by the network's width/architecture, not by the task's information demand. **The ~0.25-0.50 activity is an optimization/architecture artifact, not an information-theoretic floor.**

**RETRACTION of the previous tick's contrapositive claim.** The 2026-07-30 ~18:40Z entry said the bound "earns its keep in the contrapositive" because at H=64 the certificate `H*H_b(rho)` = 64 x 0.8211 = 52.6 bits < the 64 bits copy demands, and the net indeed fails. With H=96 now in, that reading does not survive:

| H | rho | H*H_b(rho) = certified bits | demand M*log2K | verdict of certificate | actual spikestate acc |
|---|---|---|---|---|---|
| 256 | 0.4960 | 256.0 | 64 | 4.0x SURPLUS -> should succeed | 0.107 (fails) |
| 96 | 0.3522 | 89.9 | 64 | 1.40x SURPLUS -> should succeed | 0.163 (fails) |
| 64 | 0.2496 | 51.9 | 64 | 12-bit DEFICIT -> should fail | 0.162 (fails) |

**spikestate accuracy is FLAT (0.107 / 0.163 / 0.162) across a 5x swing in certified capacity (52 -> 256 bits), and if anything is *worse* at the widest, most over-provisioned width.** So the H=64 "prediction of incapacity borne out" was a coincidence: the net fails identically at H=96 where the certificate says capacity is comfortably adequate. The certificate is therefore **necessary-but-not-sufficient and not diagnostic at this scale** — spikestate's failure on copy is a **trainability/optimization failure, not a capacity failure**, and the bound explains neither the failure nor the observed activity.

**Honest verdict on the bound arm (this is now the reportable outcome, and it is a negative):** the firing-floor bound `rho >= H_b^-1(M*log2K / H)` is **not empirically validated by this architecture at this scale, in either its forward or its contrapositive form**. Forward: activity moves opposite to the floor across a 190x floor swing. Contrapositive: performance is invariant to a 5x capacity swing, so the deficit at H=64 predicts nothing the surplus at H=96 doesn't equally contradict. Never write "the bound is confirmed", and per this entry also do not write "the contrapositive earns its keep". Defensible sentence for the paper: *we could not construct a regime where the firing-floor bound binds on a net that still learns the task — measured LIF state activity is width-determined and anti-correlated with the predicted floor, and spiking-state failure on precise recall is an optimization failure that the bound does not explain.* The bound stays in the paper as **theory with a stated, tested scope limit**, not as a validated prediction.

**What still stands, untouched by this negative:** the 3-seed char-LM headline (analog-state 0.266 activity at +0.017 bpc vs digital) and the copy workload dichotomy (state-fidelity ordering: continuous 0.92/0.46 > lossy-analog 0.53/0.31 > 1-bit 0.08/0.00 margin kept at M=16/32, at matched emitted rates). Those are empirical, do not depend on the bound, and are the paper's actual contribution. The bound arm's outcome is a scope-limit result.

**Incidental:** spikestate acc *improves* slightly as H shrinks 256 -> 96 (0.107 -> 0.163), i.e. the 1-bit-state net is not helped by width on this task at all — further evidence the failure is not about capacity. Energy proxy at H=64: spikestate 20.3k vs digital 43.0k pJ/token, buying 0.20 of the margin (same "cheap and useless" pattern).

**Row state at this tick:** 9/18 shrink-H cells on disk; the main ep30 copy row is **DONE** (43 cells, `COPY ROW EP30 DONE`), which freed GPUs 0/1/3 — three more workers added there, so all 8 A800s now run shrink-H and 17/18 cells are claimed. Only `spikestate_copy_L33_s2_H128_ep30` is unclaimed and will be taken by the first worker to free. The H=128 column (floor 0.042) and the two remaining H=96 spikestate seeds are the last data; per the anti-correlated trend above the expectation is rho in the 0.35-0.45 range at H=128, i.e. the trend continues and the verdict does not change.

## 2026-07-31 (tick) — SHRINK-H BOUND ROW: 16/18 cells in, and the bound arm CLOSES as a negative. Criterion (ii) refuted; new anti-capacity finding.

Copy L=33 (M=16, K=16 → demand `M·log₂K` = 64 bits), ep30/n80k, chance acc 0.0625 / bpc 4.000.
Still training: `spikestate_copy_L33_s{1,2}_H128_ep30` (GPUs 3/7). Everything else on disk.

| H | params | predicted floor `H_b⁻¹(64/H)` | digital acc | spikestate acc | spikestate state activity ρ | margin kept | certificate `H·H_b(ρ)` |
|---|---|---|---|---|---|---|---|
| 256 | 87,889 | 0.0026 | 0.6302±0.0106 (n=3) | 0.1071±0.0164 (n=3) | 0.4960±0.0106 | 0.079 | 255.9 bits |
| 128 | 33,169 | 0.042  | 0.5944±0.0023 (n=3) | 0.1513 (n=1) | 0.3469 (n=1) | 0.167 | 119.2 bits |
| 96  | 18,289 | 0.174  | 0.5889±0.0065 (n=3) | 0.1570±0.0063 (n=3) | 0.3909±0.0336 | 0.180 | 92.7 bits |
| 64  | 10,513 | 0.500  | 0.5600±0.0095 (n=3) | 0.1619±0.0039 (n=3) | 0.2496±0.0252 | 0.200 | 51.9 bits |

**Validity gate (i): PASSES at every width.** Digital learns copy far above chance at all H (0.630/0.594/0.589/0.560) — 9× to 8.9× chance — so copy at M=16 is not width-limited anywhere in H∈[64,256] and every cell is a legitimate bound test, not a capacity-limited one.

**Criterion (ii) — REFUTED.** Confirmation required ρ to sit at/above the floor AND *rise* as H shrinks (0.0026→0.042→0.174→0.500, a 190× swing). Measured ρ instead goes 0.496 → 0.347 → 0.391 → 0.250: it **falls**, and the H=128/H=96 points are statistically indistinguishable (0.347 vs 0.391±0.034). At H=64 ρ lands *below* its own floor. Measured LIF state activity is **width-determined, not information-determined** — criterion (iii)'s shape. The ~0.25–0.50 activity band is an optimization/architecture artifact of the LIF recurrent state, not an information-theoretic floor.

**NEW, and the sharpest evidence yet that this is trainability and not capacity: spikestate gets BETTER as it gets narrower, in exact anti-correlation with its own capacity certificate.** Margin kept rises monotonically 0.079 → 0.167 → 0.180 → 0.200 as H falls 256→64, while the certificate `H·H_b(ρ)` falls 255.9 → 119.2 → 92.7 → 51.9 bits against a fixed 64-bit demand (4.0× surplus → 0.81× deficit). The variant with **4× surplus capacity is the worst** and the one in **12-bit deficit is the best**. An information bottleneck cannot produce that ordering; an optimization pathology can (a wider 1-bit recurrent state is harder to train, not more capacious in practice).

**This independently re-confirms the earlier RETRACTION of the "contrapositive earns its keep" reading** (entry 06860c4, retracted in cef2101). The H=64 "predicted incapacity, observed failure" coincidence is now contradicted at three widths: capacity is ample at H=256/128/96 and the net fails there too — worse, in fact. The certificate is **necessary-but-not-sufficient and non-diagnostic at this scale**.

**Reportable verdict on the bound arm (final unless the two pending cells surprise):** `ρ ≥ H_b⁻¹(M·log₂K/H)` is **not empirically validated by this architecture at this scale, in either direction**. It stays in the paper as theory with a stated, *tested* scope limit. Defensible sentence:

> *We could not construct a regime in which the firing-floor bound binds on a network that still learns the task. Across a 190× swing in the predicted floor (H = 256 → 64), measured LIF state activity is anti-correlated with the floor, and spiking-state accuracy improves as the predicted capacity shrinks — so the failure of a spiking recurrent state on precise recall is an optimization failure the bound does not explain.*

**Never write "the bound is confirmed" or "the contrapositive earns its keep."**

**Unaffected by all of this:** the 3-seed char-LM headline (analog-state 0.266 activity at +0.017 bpc) and the copy state-fidelity dichotomy (margin kept: continuous 0.92/0.46 > lossy-analog 0.53/0.31 > 1-bit 0.08/0.00 at M=16/32, matched emitted rates). Those are the paper's actual contribution and never depended on the bound.

**Pending:** the two `spikestate H=128` seeds only tighten an n=1 cell whose ρ already sits between its neighbours; per the trend, no verdict change expected. Reproduce the table with the JSONs under `ssm3way_runs/*_H*_ep30.json`.

## 2026-07-30 ~20:10Z — SHRINK-H BOUND ROW COMPLETE (18/18 cells, 3 seeds at every width). Verdict unchanged (NEGATIVE); one ledger CORRECTION to the predicted floor values.

All 8 A800s idle, no jobs running, `shrinkH.log` shows every worker at `WORKER GPU<n> DONE`.
Regenerate with `python agg_shrinkH.py`.

### CORRECTION to earlier entries — the H-sweep floors were mislabeled
Earlier entries quoted floors `0.0026 / 0.042 / 0.174 / 0.500` at H = 256/128/96/64. The
first two are wrong: `0.042` and `0.110` are the floors of the *M*-sweep at fixed H=256
(M=16/32/64 → 64/128/256 bits over H=256), and they were carried over onto the H axis by
mistake; `0.0026` is a leftover from the superseded `log2 M` (symbol-count) reading.
Recomputed from the corrected demand `M·log2 K = 16·log2 16 = 64 bits`, floor = `H_b^-1(64/H)`:

| H | 64/H | predicted floor |
|---|---|---|
| 256 | 0.250 | **0.042** |
| 128 | 0.500 | **0.110** |
| 96  | 0.667 | 0.174 |
| 64  | 1.000 | 0.500 |

So the floor swing across the row is **12×** (0.042 → 0.500), not the 190× previously
written. H=96 and H=64 were already right. **The verdict does not change** — only the
magnitude of the swing that the anti-correlation is measured against. Any text quoting
"190× floor swing" must be fixed to 12×.

### Final table — copy, L=33 (M=16, K=16 ⇒ 64-bit demand), ep30 / copy_n 80k, 3 seeds per cell
Chance acc 0.0625, chance bpc 4.000. `ρ` = spikestate recurrent-state activity.
`H·H_b(ρ)` = the capacity certificate in bits, against the fixed 64-bit demand.

| H | params | floor | digital acc | digital bpc | spikestate acc | spikestate bpc | ρ | margin kept | paired Δbpc | H·H_b(ρ) |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | 87,889 | 0.042 | 0.6302±0.0106 | 1.4201±0.0343 | 0.1071±0.0164 | 3.9292±0.0369 | 0.4960±0.0106 | 0.079 | +2.509±0.060 | 256.0 |
| 128 | 28,113 | 0.110 | 0.5944±0.0023 | 1.6205±0.0167 | 0.1522±0.0049 | 3.6995±0.0278 | 0.3915±0.0417 | 0.169 | +2.079±0.016 | 123.6 |
| 96  | 18,289 | 0.174 | 0.5889±0.0065 | 1.6258±0.0421 | 0.1570±0.0063 | 3.6762±0.0193 | 0.3909±0.0336 | 0.179 | +2.050±0.023 | 92.7 |
| 64  | 10,513 | 0.500 | 0.5600±0.0095 | 1.8275±0.0563 | 0.1619±0.0039 | 3.6562±0.0111 | 0.2496±0.0252 | 0.200 | +1.829±0.051 | 51.9 |

(The H=256 digital cell excludes `digital_copy_L33_s0_probeA_ep30.json`, the earlier
baseline-gate probe at a different data budget — pooling it would wrongly give n=4.)

### Validity gate (pre-registered criterion i): PASSES at every width
Digital keeps 0.56–0.63 accuracy — 9–10× chance — at all four widths, on nets spanning
10.5k to 87.9k params. Copy at M=16 is **not width-limited anywhere in H ∈ [64,256]**, so
all four cells are legitimate tests rather than capacity-limited artifacts.

### Criterion (ii): REFUTED at 3 seeds, by anti-correlation
Predicted floor rises 12× (0.042 → 0.500) as H shrinks; measured ρ **falls** 2×
(0.496 → 0.392 → 0.391 → 0.250). H=128 and H=96 are statistically indistinguishable
(0.3915±0.0417 vs 0.3909±0.0336). The measured curve crosses the predicted floor between
H=96 (ρ 2.2× above its floor) and H=64 (ρ 2.0× *below* its floor). LIF state activity is
**width-determined, not information-determined** — criterion (iii)'s shape.

### The anti-capacity finding, now fully powered (3 seeds at all four widths)
Margin kept rises **monotonically** 0.079 → 0.169 → 0.179 → 0.200 as H shrinks 256 → 64,
while the capacity certificate `H·H_b(ρ)` falls 256.0 → 123.6 → 92.7 → 51.9 bits against a
fixed 64-bit demand (4.0× surplus → 0.81× deficit). Paired Δbpc improves in lockstep
(+2.509 → +1.829). **The most over-provisioned net is the worst one.** An information
bottleneck cannot produce that ordering; an optimization pathology can. This is the third
and best-powered confirmation of the `cef2101` retraction — the H=64 "predicted incapacity,
observed failure" coincidence does not survive the full row.

### Final reportable verdict on the bound arm (unchanged, now at full power)
`ρ ≥ H_b^-1(M·log2 K / H)` is **not empirically validated by this architecture at this
scale, in either direction**. It stays in the paper as theory with a *tested* scope limit.
Defensible sentence:

> We could not construct a regime where the firing-floor bound binds on a net that still
> learns the task: across a 12× swing in the predicted floor, measured LIF state activity
> is anti-correlated with the floor, and spiking-state accuracy *improves* as predicted
> capacity shrinks — so spiking-state failure on precise recall is an optimization failure
> the bound does not explain.

Never write "the bound is confirmed" or "the contrapositive earns its keep".

**Unaffected:** the 3-seed char-LM headline and the copy state-fidelity dichotomy
(continuous 0.92/0.46 > lossy-analog 0.53/0.31 > 1-bit 0.08/0.00 margin kept at M=16/32 at
matched emitted rates). Those are the paper's contributions and never depended on the bound.

### Energy footnote
spikestate is cheaper at every width (107.7k / 44.8k / 32.5k / 20.3k pJ/token at
H=256/128/96/64 vs digital 398.0k / 123.6k / 78.6k / 43.0k) and buys 0.08–0.20 of the
margin. Same "cheap and useless" pattern as the M=32 floor control: a pJ number is
meaningless without the quality it purchased.

## 2026-07-30 ~20:15Z — NEXT ROW LAUNCHED: char-LM state-fidelity dichotomy at MATCHED EMITTED RATE

With the bound arm closed and all 8 GPUs idle, the remaining high-value cell is the one the
copy task has and char-LM does not: a **matched-emitted-rate** comparison of the three
non-digital variants. On copy the dichotomy is clean because all three emit at ~0.45–0.50,
so the 0.92 / 0.53 / 0.08 margin spread is attributable to state fidelity and not to how
much each variant communicates. On char-LM the existing 3-seed row compares them at
*unmatched* rates — analog emits 0.266 (θ=1.0) while spikestate sits at 0.645 and spikeout's
state is fully dense at 1.000 — so a critic can say analog only looks good because it is
being read at a different operating point.

Measured char-LM emission targets, pulled from the existing 3-seed row:

| variant | rate_state | rate_emitted | bpc |
|---|---|---|---|
| digital | 1.000 | 1.000 | 3.3429±0.0221 |
| spikestate | 0.6453±0.0202 | **0.6453±0.0202** | 3.6205±0.0130 |
| spikeout | 1.0000±0.0000 | **0.3691±0.0374** | 3.3978±0.0292 (paired +1.055) |

Seed-0 analog θ curve: θ=0.10 → emit 0.722 / bpc 3.1804; θ=0.25 → 0.557 / 3.2112;
θ=0.50 → 0.485 / 3.2362; θ=1.00 → 0.279 / 3.3290. So **θ≈0.15 brackets spikestate's rate and
θ≈0.75 brackets spikeout's** — analog can be read against *both* comparators at their own
operating points, which is better than the single match originally planned.

`run_charlm_theta.sh` (lock-dir work queue, same design as `run_shrinkH.sh`): 12 listed cells
= analog × θ ∈ {0.15, 0.75, 0.25, 0.50} × seeds {0,1,2}, of which 10 need training (s0 at
θ=0.25 and θ=0.50 already exist and are skipped by the idempotence check — out names follow
the existing `analog_charlm_s<seed>_theta<θ>.json` convention deliberately). Matched points
are ordered first so a partial row is already usable. char-LM at the **identical budget to
the existing 3-seed row** (default 6 epochs, 1.4M chars, lam=0), so the new points drop
straight into that table. Every θ in the grid is above the quantizer LSB
`2·rail/2^bits = 8/64 = 0.125` and therefore operative.

**Pre-registered reading, written before any cell lands:**
- *Confirmation* — at θ≈0.15 (matching spikestate's 0.645 emission), analog's paired Δbpc vs
  digital stays well below spikestate's +0.278, reproducing the copy dichotomy's ordering
  (continuous > lossy-analog > 1-bit) on a second task at matched rate.
- *Refutation* — analog's advantage at θ=1.0 was an operating-point artifact: forced up to
  0.645 emission its Δbpc approaches or exceeds spikestate's +0.278. That would restate the
  char-LM headline as "analog reaches a better rate/quality *point*", not "analog is better
  at matched rate" — a materially weaker claim, and one that must be caught before the
  writeup rather than after.
- *Most likely* — analog's Δbpc rises with emission but stays below spikestate's, i.e.
  confirmation with a smaller margin than the θ=1.0 comparison suggests.

**A second, unplanned question this row settles.** Seed-0 analog at θ=0.1 scored bpc
**3.1804 vs digital's 3.3175 — it BEAT the digital baseline.** That was previously dismissed
as a single seed that "did not generalize", but it was never actually tested at 3 seeds: the
existing 3-seed analog column is θ=1.0 only. If the low-θ advantage survives three seeds, the
char-LM claim strengthens from "quality-matched to within 0.5%" to "matches or beats digital
at ~65% emission", which would be a materially stronger headline. Stated up front so the
result counts either way: the honest prior is that it is seed noise (digital's own seed sd is
0.022 bpc, and the θ=1.0 column was consistently *worse* than digital across all 3 seeds).

Markers in `copy_logs/charlm_theta.log`, per-cell logs in `charlm_theta_logs/`. No results yet.

## 2026-07-30 ~20:35Z — char-LM MATCHED-EMITTED-RATE row COMPLETE (10 cells, no failures). The pre-registered CONFIRMATION branch holds at BOTH matched points, and the "second question" resolves POSITIVE: analog beats digital at 3 seeds.

The row launched 20 minutes earlier finished inside the same tick (char-LM cells are ~6 epochs
on 1.4M chars). 10/10 cells trained, 0 failed, 8 workers reached `WORKER GPU<n> DONE`.
Regenerate with `python agg_charlm.py`.

### Full char-LM table, 3 seeds per cell, identical budget (6 ep, 1.4M chars, lam=0, 173,596 params)
`paired Δbpc` = per-seed difference vs the digital baseline at the same seed (negative = BETTER
than digital). Comparator rows are the pre-existing 3-seed cells, unchanged.

| variant | θ | n | bpc | emit | rate_state | paired Δbpc | pJ/token |
|---|---|---|---|---|---|---|---|
| digital | — | 3 | 3.3429±0.0221 | 1.0000 | 1.0000 | ±0 | 712,448 |
| analog | 0.15 | 3 | 3.1936±0.0078 | 0.6122 | 0.9940 | **−0.1493±0.0144** | 446,143 |
| analog | 0.25 | 3 | 3.2092±0.0232 | 0.5500 | 0.9938 | −0.1337±0.0343 | 442,473 |
| analog | 0.50 | 3 | 3.2344±0.0026 | 0.4715 | 0.9930 | −0.1085±0.0236 | 437,848 |
| analog | 0.75 | 3 | 3.2932±0.0109 | 0.3798 | 0.9921 | **−0.0497±0.0199** | 432,440 |
| analog | 1.00 | 3 | 3.3595±0.0273 | 0.2664 | 0.9919 | +0.0166±0.0068 | 425,745 |
| spikestate | — | 3 | 3.6205±0.0130 | **0.6453** | 0.6453 | +0.2776±0.0091 | 156,835 |
| spikeout | — | 3 | 4.3978±0.0292 | **0.3691** | 1.0000 | +1.0548±0.0318 | 432,752 |

The θ=1.0 cell reproduces the previously recorded +0.0166 exactly, so the new cells are
consistent with the existing row rather than a differently-configured re-run.

### Matched-rate comparison — the point of the row. CONFIRMED at both operating points.
- **Against spikestate** (emits 0.6453): analog θ=0.15 emits **0.6122** — slightly *less* — and
  scores paired Δbpc **−0.149 vs spikestate's +0.278**, a **0.427 bpc gap in analog's favour at
  matched communication**. Pre-registered confirmation required only "well below +0.278"; the
  result clears it by beating the *digital* baseline outright.
- **Against spikeout** (emits 0.3691): analog θ=0.75 emits **0.3798** and scores **−0.050 vs
  spikeout's +1.055**, a **1.10 bpc gap at matched communication**.
- So analog's char-LM advantage is **not an operating-point artifact**. The refutation branch is
  dead: forcing analog up to spikestate's emission does not erode its advantage, it *increases*
  it (Δbpc improves monotonically as θ falls: +0.017 → −0.050 → −0.109 → −0.134 → −0.149).

### The "second question" resolves POSITIVE and upgrades the char-LM headline
The seed-0 θ=0.1 run that beat digital was previously dismissed as non-generalizing. **It
generalizes.** At θ=0.15, all three seeds are negative (−0.1327, −0.1577, −0.1575) — analog
beats digital by **0.149 bpc (4.5%)** while emitting on only **61%** of steps and using **1.60×
less** energy by the pJ proxy (446k vs 712k). Analog beats digital at every θ ≤ 0.75, i.e. down
to ~38% emission, and only reaches parity at θ=1.0 / 27% emission.

**The old wording "quality-matched to within 0.5%" was an artifact of reading the θ=1.0 column
only** — that column is the *sparsest* point, not the representative one. Corrected headline:
*on char-LM the analog-state SSM beats a matched digital baseline by 0.149±0.014 bpc while
communicating on 61% of steps (3 seeds, all negative), and remains better than digital down to
38% emission.*

**Honest mechanism and caveat.** The likely cause is **regularization, not superior
computation**: the analog state's injected noise (σ=0.02), 6-bit quantization and send-on-delta
suppression together act as a stochastic regularizer on a small (173k-param, 6-epoch) model —
the same effect already seen in the vision PoC, where target-rate sparsity *raised* accuracy.
This must be stated. The advantage may well shrink or vanish with a longer schedule, a larger
model, or a properly regularized digital baseline, and **the digital baseline here is not
regularization-tuned**. Do not claim analog computation is intrinsically better than digital;
claim that at matched capacity and budget the analog datapath is not a quality tax on char-LM,
and at this scale is a small quality *benefit*.

### The task-conditional inversion survives matched rates — and sharpens the mechanism
Both tasks now have a matched-emitted-rate reading, so neither ordering can be dismissed as an
operating-point artifact:

| task | best non-digital | middle | worst |
|---|---|---|---|
| char-LM (statistics) | **analog** −0.149 @ 0.61 | spikestate +0.278 @ 0.65 | **spikeout** +1.055 @ 0.37 |
| copy (precise recall) | **spikeout** 0.92 margin @ 0.45 | analog 0.53 @ 0.50 | **spikestate** 0.08 @ 0.50 |

spikeout is the *best* variant on copy and the *worst* on char-LM, despite having the most exact
recurrent state (`rate_state` exactly 1.0000 on both tasks). So **"state fidelity" alone does not
explain char-LM** — it explains copy. The unifying statement both rows support:

> A neuromorphic variant is cheap exactly when it degrades the part of the datapath the task
> does not depend on. Precise-recall workloads (copy) need an exact recurrent *state* and can
> tolerate a spiked output. Statistical workloads (char-LM) need an exact graded *output*
> distribution and can tolerate a lossy state. Neither route is universal, and the choice is set
> by the workload, not by the energy budget.

This is a better claim than the previous "state fidelity is the axis", because it predicts both
orderings from one principle instead of describing one and inverting for the other.

### Energy footnote (unchanged direction, but analog now wins one axis outright on char-LM)
At θ=0.15 analog is both **better quality and 1.60× cheaper** than digital (446k vs 712k
pJ/token) — the first cell in the project where analog wins on both axes simultaneously.
It is still **not** the cheapest variant: spikestate is 156.8k pJ/token, 2.8× cheaper than
analog, and pays +0.278 bpc for it. The analog datapath caveat stands and is the reason:
`W_out_MAC` alone is 334k of analog's 446k pJ, because only `W_mix` gets event pricing while
`W_in`/`W_out` stay MAC-priced. Fixing that is a datapath design change, not a rerun.

### Caveat a reviewer will raise, stated here
Matching on *emitted* rate matches communication, not state activity: analog's `rate_state` is
0.994 (a dense sub-threshold analog variable) against spikestate's 0.645. The argument for the
match is that a dense analog state costs no spikes to maintain — but it does cost an analog
storage element and its converter per unit, which the pJ proxy does not price. So "matched rate"
means matched *wire traffic*, and that should be said explicitly rather than implied.

## 2026-07-30 ~20:45Z — REGULARIZATION CONTROL for the char-LM headline: LAUNCHED (pre-registered)

The char-LM headline as of `a134f05` is that the analog-state SSM **beats** a matched
digital baseline by **0.149 ± 0.014 bpc** (3 seeds) while emitting on 61% of steps and
using 1.60× less proxy energy. The ledger already carries the caveat that the likely
mechanism is **regularization, not superior computation** — but it was never tested.
This row tests it. It is the one control a reviewer will demand, and it is the last
open data question before writeup.

**Why the caveat is credible enough to need a control.** The analog state carries
additive noise (σ=0.02), 6-bit quantization over ±4 rails, and send-on-delta
suppression of small updates, on a 173,596-param model at a 6-epoch schedule. The
digital baseline has *no* regularization: no noise, no quantization, no weight decay,
no dropout. In the vision PoC, sparsity *raised* accuracy for exactly this reason. So
"analog beats digital" may just be "a regularized model beats an unregularized one".

**Code changes (both behavior-preserving at their defaults; verified by re-rendering
the existing char-LM table byte-identically before launch).**
- `ssm3way.py`: the state degradation is factored out of `_analog` into `_degrade(h,
  sigma, bits)`, and the **digital** variant can now take it via `--dig_noise` /
  `--dig_bits`; plus `--wd` for Adam weight decay. `_analog` is now a thin call to
  `_degrade` with the analog config, so variants 2–4 are bit-identical to before.
  Defaults are all 0 ⇒ every previously-run cell is unaffected. The control config is
  recorded per-run as `dig_reg` in the output JSON.
- `agg_charlm.py`: **latent-bug fix, found before it could corrupt anything.** The
  aggregator grouped cells by `(variant, theta)` only. The new cells are
  `variant=digital, theta=None` with non-zero `dig_reg`, so they would have landed on
  the *same key* as the unregularized digital baseline and silently overwritten it
  seed-by-seed — corrupting the paired reference and therefore **every** `paired dbpc`
  in the char-LM table. Now the regularization config joins the group key and the
  paired reference is pinned to the unregularized digital cell. (Same class of bug as
  the budget-pooling one already fixed in `agg_copy.py`; worth noting that this
  aggregator family has now produced the same silent-pooling failure twice.)

**Row = 12 cells, 4 arms × seeds {0,1,2}**, all `variant=digital`, char-LM, at the
**identical** budget to the existing row (6 ep, 1.4M chars, lam=0, H=256) so they drop
straight into that table. Driver `run_digreg.sh <gpu>`, same lock-dir work queue as
`run_shrinkH.sh` (atomic-mkdir claim, skips any cell whose out JSON exists, workers
addable on any GPU). Outputs `ssm3way_runs/digital_charlm_s<seed>_reg_<arm>.json`,
markers `copy_logs/digreg.log`, logs `digreg_logs/`. All 8 A800s verified idle (0
compute apps, no `ssm3way.py` process) before launch; 8 workers claimed 8 distinct
cells. ~15 min/cell ⇒ ~30 min wall for 12 cells.

The arms are a **decomposition**, not a sweep:
| arm | flags | question it answers |
|---|---|---|
| `n0.02_b6` | `--dig_noise 0.02 --dig_bits 6` | the **entire** analog state degradation *without* the send-on-delta gating — separates "lossy state as regularizer" from "event gating". The key arm. |
| `n0.02` | `--dig_noise 0.02` | is the quantization doing anything, or is it all just noise? |
| `n0.05` | `--dig_noise 0.05` | dose-response — if more noise keeps helping, the baseline was simply under-regularized. |
| `wd1e-4` | `--wd 1e-4` | a conventional tuned baseline, i.e. what a reviewer means by "did you try regularizing it at all". |

**PRE-REGISTERED READINGS (written before any cell landed, so they cannot be fitted
after the fact):**
- **(A) — the likely outcome.** The full arm `n0.02_b6` recovers most of the 0.149 bpc
  ⇒ the char-LM win is a **regularization effect**. The honest claim becomes *"the
  analog datapath is not a quality tax, and its state noise happens to regularize a
  small model at a short schedule"*. The **energy** story survives intact (equal-or-
  better quality at 1.60× lower proxy energy and 61% emission), but *"analog computes
  better"* dies and must be struck from the headline.
- **(B)** None of the four arms closes the gap ⇒ the win is **not** explainable as
  state-degradation-as-regularizer, and the send-on-delta gating itself is contributing.
  This would strengthen the headline considerably.
- **(C)** An arm **overshoots** (regularized digital beats analog) ⇒ the honest headline
  is that a properly regularized digital baseline wins on quality and analog's only
  remaining claim is **energy at matched quality**. Must be reported as such.

Note that (A) and (C) both *narrow* the headline and (A) is the stated prior — this row
is run expecting to weaken its own project's best result, which is the point of it.
Whatever lands goes in the paper either way.

**No results yet.**

### 2026-07-30 ~21:00Z — RESULT: the control row is COMPLETE (12/12, 3 seeds/arm, 0 failures) and it is the OVERSHOOT branch (C). **The char-LM "analog beats digital" headline is RETRACTED.**

Char-LM, 6 ep / 1.4M chars / lam=0 / H=256 / 173,596 params, paired Δbpc vs the
**unregularized** digital baseline (negative = better than it):

| arm | bpc | acc | paired Δbpc | per-seed |
|---|---|---|---|---|
| digital, no reg (old reference) | 3.3429±0.0221 | 0.3441 | +0.0000 | — |
| **digital + noise σ=0.02** | **3.0338±0.0165** | 0.4044 | **−0.3091±0.0244** | −0.287/−0.335/−0.306 |
| **digital + noise 0.02 + 6-bit** | **3.0321±0.0121** | 0.4041 | **−0.3109±0.0169** | −0.294/−0.328/−0.311 |
| **digital + noise σ=0.05** | **3.0309±0.0104** | 0.4060 | **−0.3120±0.0209** | −0.290/−0.331/−0.315 |
| digital + weight decay 1e-4 | 3.3476±0.0122 | 0.3459 | **+0.0047±0.0110** | ≈0 |
| analog θ=0.15 (the headline cell) | 3.1936±0.0078 | 0.3764 | −0.1493±0.0144 | −0.133/−0.158/−0.158 |

**(C) OVERSHOOT, at 3 seeds and far outside seed noise.** State noise on the digital
baseline is worth **−0.309 bpc**, which is **2.1× the entire analog advantage**
(−0.149). Differencing the two paired estimates against the same seeds, **analog
θ=0.15 is +0.160 bpc WORSE than a noise-regularized digital baseline.** The
`a134f05` headline — "on char-LM the analog-state SSM beats a matched digital
baseline by 0.149±0.014 bpc" — was an artifact of comparing against an
**under-regularized** baseline. It is retracted. The ledger's own stated caveat was
right, and understated: the mechanism is not merely regularization, it is
regularization the analog route captures only **half** of.

**Three further readings, each a real finding rather than a hyperparameter detail:**

1. **Quantization contributes nothing; it is the additive noise alone.** σ=0.02 with
   6-bit quantization (−0.3109) and without it (−0.3091) differ by 0.002 bpc against
   seed sd 0.017–0.024 — indistinguishable. And the dose is already **saturated** at
   σ=0.02: σ=0.05 gives −0.3120, the same. So of the three lossy mechanisms in the
   analog state (noise / 6-bit quantization / send-on-delta), only the noise carries
   the quality effect.
2. **Weight decay does nothing (+0.005±0.011), which is an informative dissociation.**
   The effect is not generic capacity control — a conventional regularizer at a
   conventional strength is worth exactly zero here. It is specifically **state-level
   stochasticity injected into the recurrence**. That rules out "any regularizer would
   have done it" and pins the mechanism precisely.
3. **The decomposition of analog's net effect is now explicit and it is the useful
   result of this row.** Analog's net −0.149 = a **noise benefit of −0.309** plus a
   **send-on-delta gating cost of +0.160** at θ=0.15. The gating cost grows with θ
   exactly as the emitted rate falls: at θ=1.0 the net is +0.017, i.e. a gating cost
   of +0.326 at 27% emission. So the analog datapath does supply a genuinely valuable
   regularizer for free and physically — it simply also charges for the sparsity it
   buys, and the charge is larger than the gift is at every θ.

**What the honest char-LM claim becomes.** The digital noise is **training-time only**
(`if self.training` in `_degrade`), so digital+noise has **identical inference cost** to
plain digital — 712,448 pJ/token. The comparison is therefore: regularized digital
**3.034 bpc at 712k pJ/tok** vs analog θ=0.15 **3.194 bpc at 446k pJ/tok**. Analog buys
a **1.60× proxy-energy reduction for a +0.160 bpc (5.3%) quality cost**. That is a real
and defensible energy/quality tradeoff — it is simply **not** a free lunch and **not** a
quality win. Never write "analog beats digital on char-LM" again.

**What is UNAFFECTED.** The ordering among the neuromorphic variants does not move,
because re-referencing shifts every variant by the same constant. Against the
**regularized** digital baseline the char-LM row reads analog **+0.160** ≪ spikestate
**+0.587** ≪ spikeout **+1.364** — same order, same conclusion that analog-state is by
far the best neuromorphic datapath on char-LM. The **datapath-degradation principle**
(a variant is cheap exactly when it degrades the part of the datapath the task does not
depend on) is likewise untouched, since it is a statement about *which variant wins
where*, and both orderings survive re-referencing. The closed bound arm is unaffected.

**A CONSEQUENCE THAT PROPAGATES, and must be handled in the writeup.** Every char-LM
number in this project was referenced to an under-regularized digital baseline, which
**flattered every neuromorphic variant** by ~0.31 bpc. All char-LM Δbpc figures should
be restated against the regularized baseline. The **copy** row was never given this
control, and there the bias runs the *same* direction — a regularized digital baseline
would be *better*, so every copy "margin kept" figure (spikeout 0.92/0.46, analog
0.53/0.31, spikestate 0.08/0.00) is an **upper bound** on what those variants actually
retain. Copy conclusions therefore get *more* negative for the neuromorphic routes, not
less; the dichotomy's direction is safe but its magnitudes are optimistic. **Running the
same 4-arm control on copy at M=16 is now the highest-value remaining data cell.**

**Reproduce:** `run_digreg.sh <gpu>`; table via `python agg_charlm.py`; 12 run JSONs in
`ssm3way_runs/digital_charlm_s*_reg_*.json`.

---

## 2026-07-31 — LAUNCHED (pre-registered): regularization control on the COPY task

**Why.** The char-LM regularization control (bb79f7f) retracted the "analog beats
digital" headline: training-time state noise sigma=0.02 on the digital baseline was
worth −0.309 bpc, 2.1× the entire analog advantage. Every copy-task "margin kept"
figure (spikeout 0.92/0.46, analog 0.53/0.31, spikestate 0.08/0.00 at M=16/32) was
measured against the same unregularized digital baseline and never got this control.
The ledger currently assumes the bias runs the same direction (margins = upper bounds).

**The interesting part: the paper's central claim predicts the opposite.** The
datapath-degradation principle says a degradation is cheap exactly when it hits a
part of the datapath the task does not depend on. Copy depends on precise state
retention, so state noise should HURT the digital baseline on copy — unlike char-LM,
where it helped. This is the first experiment where the principle and
reasoning-by-analogy from char-LM make opposite predictions.

**Design.** `run_digreg_copy.sh` (lock-dir work queue, idempotent). 12 cells =
2 arms × 2 loads × 3 seeds, all digital variant at ep30/n80k (the definitive copy
budget): arm `n0.02` (noise only — the mechanism that carried the char-LM effect)
and arm `n0.02_b6` (noise + 6-bit quant = full analog state degradation minus
send-on-delta; quant was worth 0 on char-LM but is lossy in exactly the way precise
recall cannot absorb) × L ∈ {33, 65} (M = 16, 32). Dropped arms: `wd` (exactly 0 on
char-LM), `n0.05` (dose saturated). Outputs
`ssm3way_runs/digital_copy_L<L>_s<seed>_reg_<arm>_ep30.json`.

**Pre-registered readings (written before any cell landed):**
- **(A) noise IMPROVES digital on copy** → baseline under-regularized there too; all
  margin-kept figures shrink; copy conclusions get MORE negative for the
  neuromorphic routes. (The ledger's standing assumption.)
- **(B) noise HURTS digital on copy** → out-of-sample confirmation of the
  datapath-degradation principle, and the existing copy margins stand as fair.
- **(C) no effect** → margins stand.
- The principle predicts (B). Analogy from char-LM predicts (A). Either outcome is
  reportable; (B) would be the stronger paper result (the central claim would have
  survived a designed opportunity to fail).

**Aggregator note.** `agg_copy.py` group key extended with the `dig_reg` config
(third instance of the group-key trap, caught BEFORE the cells landed this time);
old JSONs lack `dig_reg` and default to (0,0,0), so existing tables are unchanged.

**A FOURTH group-key bug, found by verifying the patch instead of trusting it.**
`agg_copy.py` globbed every copy JSON but keyed on `(epochs, variant, L, theta)` with
no `H`. The shrink-H row wrote copy **L=33 cells at H = 64/96/128**, which collide
with the main row's H=256 cells at the same `(epochs, variant, L, theta)` and win the
per-seed slot on filename order (`..._s0_H128_ep30.json` sorts before
`..._s0_ep30.json`). So regenerating the table silently reported the **H=128** cell as
the M=16 digital reference — acc **0.5944** instead of the true **0.6302** — and
shifted every M=16 margin-kept figure (spikeout 0.92→0.98, analog 0.53→0.56,
spikestate 0.08→0.17). The published ledger numbers were measured before the
shrink-H row existed and are correct; only regeneration was affected. Fixed: `H` is
in the group key, paired deltas are computed against the digital reference **at the
same width**, and the label carries `H=` when it is not 256. Verified post-fix output
reproduces the published M=16 column (digital 0.6302±0.0106, margins 0.92/0.53/0.08)
and the shrink-H margins (0.17/0.18/0.20 at H=128/96/64, matching `agg_shrinkH.py`).
**Standing lesson, now paid for four times: this aggregator family pools on any
dimension absent from its group key. Adding a cell type means auditing the key AND
re-verifying that a previously published column still renders identically.**

## 2026-07-30 ~21:40Z — COPY-SIDE REGULARIZATION CONTROL, M=16 COLUMN: the pre-registered branch (A) landed (noise HELPS on copy too), and the arm-to-prediction mapping in the pre-registration was WRONG — the real out-of-sample test is the quantization arm, and there the principle HOLDS

Row still running (7/12 cells, 5 on disk; L=65/M=32 cells and `L33_s2_b6` in flight).
The M=16 σ=0.02 arm is COMPLETE at 3 seeds, which is the decision cell.
Copy L=33 (M=16, K=16), ep30/n80k, H=256, 87,889 params, chance acc 0.0625 / bpc 4.000.

| arm (digital variant) | n | acc | bpc | paired Δbpc vs plain digital | rate_state |
|---|---|---|---|---|---|
| plain digital (reference) | 3 | 0.6302±0.0106 | 1.4201±0.0320 | — | 1.0000 |
| + state noise σ=0.02 | 3 | **0.6628±0.0090** | **1.2702±0.0286** | **−0.1499±0.0333** (all 3 seeds negative: −0.112 / −0.173 / −0.165) | 1.0000 |
| + σ=0.02 + 6-bit quant | 2 | 0.5570 | 1.6738 | **+0.2449** (+0.269 / +0.220) | 0.9542 / 0.9794 |

**BRANCH (A) LANDED on the noise arm: state noise HELPS the digital baseline on copy
too (−0.150 bpc, +0.033 acc, all three seeds).** The pre-registration said the
datapath-degradation principle predicts (B) — noise should HURT a precise-recall task
— and reasoning-by-analogy-from-char-LM predicts (A). **Analogy won; the principle's
prediction as written failed.** Record it that way and do not soften it. Consequences
per the pre-registration: the copy digital baseline was under-regularized, so every
published copy "margin kept" figure is an upper bound and shrinks. Recomputed at M=16
against the properly regularized reference (acc 0.6628, digital margin 0.6003):
spikeout **0.92 → 0.87**, analog θ=0.2 **0.53 → 0.50**, spikestate **0.08 → 0.074**;
paired Δbpc vs the regularized reference becomes spikeout **+0.274**, analog
**+1.507**, spikestate **+2.659**. The shrink is modest, so the state-fidelity
ordering and the workload dichotomy are unaffected in direction and in magnitude.

**BUT THE PRE-REGISTRATION MIS-ASSIGNED ITS OWN ARMS, and fixing that is the real
result of this row.** Verified in `ssm3way.py:120-131` rather than assumed:
`_degrade()` gates the additive noise behind `if self.training`, but the 6-bit
quantizer is **NOT** gated — it runs at inference too (the rails clamp is always on in
both arms, so the two arms differ by exactly the quantizer). So:
- **`dig_noise` is a training-time regularizer, NOT a datapath degradation.** At
  inference the digital+noise net has a fully exact state. The principle says nothing
  about it, so it was never a valid test of the principle — (A) landing is a statement
  about baseline tuning, not about datapaths.
- **`dig_bits` IS an inference-time state-precision degradation** — the only arm in
  this row that degrades the datapath the way analog does.
**And in that arm the principle holds out-of-sample:** the same 6-bit quantizer costs
**nothing on char-LM** (σ=0.02 vs σ=0.02+b6 differ by −0.0018 bpc against seed sd
0.017–0.024) and **+0.387 bpc on copy** (+0.381 / +0.393 vs the noise-only arm at
matched σ). A statistical task tolerates a lossy state; a precise-recall task does
not — exactly the prediction. Its footprint is visible in the state itself:
`rate_state` drops 1.0000 → 0.954/0.979 because the quantizer zeroes small-magnitude
units, i.e. the loss is real and not a training artifact.

**HONESTY LABELS, because this matters for how it can be written up.** The
quantization contrast is **post-hoc identified** (n=2 seeds, third still training) —
the pre-registration did not name it as the test, so it must be reported as a
post-hoc-but-mechanistically-motivated confirmation, NOT as a pre-registered win. The
pre-registered test, as written, failed. The correct summary sentence is: *the
principle's designed test was mis-specified because the control arm turned out to be
training-time-only; the arm that does degrade the inference datapath behaves exactly as
the principle predicts, at n=2, and a properly regularized copy baseline shrinks every
neuromorphic margin-kept figure by 3–5 points without changing the ordering.*

One weakly-supporting quantitative detail: the noise benefit is **~2× smaller on copy
than on char-LM** (−0.150±0.033 vs −0.309±0.024). Even as a pure regularizer, state
stochasticity buys less on the task that needs precise retention — the right direction
for the principle, but a regularizer-strength observation, not a datapath one.

**Still pending in this row:** the M=32 (L=65) column for both arms, plus
`L33_s2_b6`, which will take the quantization contrast to 3 seeds and say whether the
noise benefit also attenuates as memory load doubles. Table: `python agg_copy.py 30`.

### 2026-07-30 ~22:10Z — copy-side regularization control: M=32 column complete (3 seeds), and the M=16 quantizer arm reaches 3 seeds

Row state: 9/12 cells on disk; the three `digital_copy_L65_s*_reg_n0.02_b6_ep30` cells (quantizer arm at M=32) are still training on GPUs 0/2/7. Table via `python agg_copy.py 30`.

**Result 1 — pre-registered branch (A) confirmed at a SECOND memory load, and the noise benefit GROWS with load.**
Copy ep30/n80k, H=256, chance acc 0.0625. M=32: plain digital 0.3150±0.0031 / bpc 2.8928±0.0127; **+σ=0.02 → acc 0.3425±0.0027 / bpc 2.6778±0.0159, paired Δbpc −0.2150±0.0106 (all 3 seeds negative)**. Compare M=16: −0.1499±0.0333. So training-time state noise helps *more* at the harder load (−0.215 vs −0.150), and both are still smaller than char-LM's −0.3091±0.0244. Earlier ledger reading "the noise benefit is ~2× smaller on copy than char-LM" must be softened to **1.4× smaller at M=32** — the trend across loads runs *toward* the char-LM value, which weakens (does not kill) that weak supporting argument for the datapath-degradation principle.

**Consequence — copy margins at M=32 recomputed against the regularized reference** (reg digital acc 0.3425 ⇒ above-chance margin 0.2800): spikeout **0.46 → 0.42**, analog θ=0.2 **0.31 → 0.28**, spikestate **0.00 → 0.00**. Same 3–5 point shrink as M=16 (0.92→0.87 / 0.53→0.50 / 0.08→0.074); **the state-fidelity ordering and the workload dichotomy are unaffected in direction and magnitude at both loads.** All published copy margin-kept figures remain upper bounds; the correction is now measured, not assumed, at M=16 and M=32.

**Result 2 — the post-hoc quantizer contrast is now at n=3 at M=16 and it strengthens.** `dig_bits=6` is the only *inference-time* state degradation in this control (verified in `ssm3way.py:120-131`: additive noise is gated by `if self.training`, the quantizer is not). At matched σ=0.02, the quantizer costs **+0.3687 bpc on copy** (arm Δbpc +0.2188±0.0514 vs +σ-only −0.1499±0.0333, n=3 each; the earlier n=2 estimate was +0.387) against **−0.002 bpc on char-LM (nothing)**. Same degradation, ~0.37 bpc apart across the two workloads — the datapath-degradation principle's prediction, out of sample. Its state footprint is visible: `rate_state` 1.0000 → 0.971 at M=16 as the quantizer zeroes small units. Honesty label unchanged: **post-hoc identified, mechanistically motivated, NOT a pre-registered win** — the pre-registration bound its prediction to the wrong arm.

**Still pending:** the M=32 quantizer cells (running). Prediction, recorded now: if the principle holds, the quantizer penalty should be *at least* as large at M=32 as at M=16 (+0.37), since precise retention of 32 symbols is more sensitive to state precision than 16. A smaller penalty at M=32 would be evidence against.

### 2026-07-31 — copy-side regularization control COMPLETE (12/12): the M=32 quantizer arm lands at 3 seeds and the pre-registered prediction is MET

Final three cells (`digital_copy_L65_s{0,1,2}_reg_n0.02_b6_ep30`) finished 22:15–22:17Z; all 8 GPUs idle after. Table via `python agg_copy.py 30`.

**Result — the quantizer penalty does NOT shrink at the harder load; the pre-registered floor is cleared.**
Copy L=65 (M=32), ep30/n80k, H=256, chance acc 0.0625. Digital +σ=0.02+6-bit: acc **0.2604±0.0063** / bpc **3.0630±0.0319**; paired Δbpc vs plain digital **+0.1702±0.0444**, dacc −0.0546±0.0090, margin kept 0.78. At matched σ=0.02, the *quantizer alone* therefore costs **+0.3852 bpc at M=32** (arm Δbpc +0.1702 vs noise-only −0.2150±0.0106), against **+0.3687 at M=16** and **−0.002 on char-LM**. The pre-registered criterion ("≥ +0.37 at M=32; smaller would be evidence against") is met.

**Honest reading of the magnitude:** +0.385 vs +0.369 is a flat-to-slightly-rising trend, statistically indistinguishable given the arm sds (~0.04–0.05) — report as "the penalty does not attenuate with load", NOT "the penalty grows with load". The load-robust statement of the contrast is: **the same inference-time 6-bit state quantization costs ~+0.37–0.39 bpc on precise recall at both memory loads and nothing (−0.002) on char-LM** — the datapath-degradation principle's out-of-sample signature, now at 3 seeds at three task/load points. Label unchanged: post-hoc identified, mechanistically motivated, not a pre-registered win (the original pre-registration bound its prediction to the training-time-only noise arm; only THIS M=32 floor was pre-registered in advance, in the 8d050cb entry).

Quantizer state footprint reproduces at M=32: `rate_state` 1.0000 → 0.9756±0.0049 (0.9812/0.9730/0.9725). Side observation: a digital datapath degraded *only* by 6-bit state quantization still keeps 0.78 of the margin at M=32 — far above spikeout 0.46 / analog 0.31 — i.e. precision loss alone explains part but not most of the neuromorphic routes' copy deficit; the event/spike mechanisms add their own cost on top.

**ROW CLOSED. The empirical arm of the SSM×neuromorphic thread is now COMPLETE:** char-LM 3 seeds incl. matched-rate θ sweep + regularization control; copy M∈{16,32,64} 3 seeds + regularization control at M∈{16,32}; shrink-H bound arm closed as a negative. No further data cells are planned. **The writeup is the bottleneck:** (1) restated char-LM headline (analog = 1.60× proxy-energy cut for +0.160 bpc vs a noise-regularized digital baseline; "analog beats digital" is retracted), (2) the datapath-degradation principle as the central claim with both matched-rate tables + the quantizer contrast as evidence, (3) the bound arm as theory-with-a-tested-scope-limit, (4) the Imam memo update.

## 2026-07-31 tick — WRITEUP STEP 1: Imam memo v3 (empirical arm folded in), no GPU used

State at tick start: all 8 A800s idle (1 MiB, 0% util), zero `ssm3way.py` processes, no cells remaining — the empirical arm was closed in commit 48510c8. Per this ledger's own standing verdict ("next ticks = writeup only"), this tick did a zero-GPU writing step.

**What was stale:** `talk/imam_ssm_memo.md` was still **v2 (2026-07-30, literature-only)**. Its §6 proposed the 3-way comparison as *future work* and stated the hypothesis "(3) matches (1)'s quality at the lowest energy". That hypothesis has since been **partly refuted by our own data** (analog does not match a regularized digital baseline; and on copy analog is worse than output-spiking on *both* quality and pJ), so the memo as it stood would have shown Imam a prediction we have since contradicted.

**Written: `talk/imam_ssm_memo_v3.md` (150 lines).** Keeps v2's literature analysis (§1–3: reading list, LMU/NEF precedent, the (a)/(b)/(c) taxonomy) and replaces the hypothesis with measurements. New sections:
- **§4** the actual 4-variant parameter-matched design table (173,596 params matched on char-LM; only the nonlinearity's position moves).
- **§5** char-LM 3-seed table with the **matched-communication-rate** comparisons (analog 0.612 emission beats spikestate 0.645 by 0.427 bpc; analog 0.380 beats spikeout 0.369 by 1.10 bpc) → the advantage over the spiking routes is not an operating-point artifact. `spikeout`'s `rate_state` = 1.0000 flagged as the study's most reproducible observation.
- **§6** copy inversion + margin-kept table already corrected against the noise-regularized reference (spikeout 0.87/0.42, analog 0.50/0.28, spikestate 0.074/0.00) + the "cheapest cell buys nothing" point.
- **§7** the regularization control, written as *the part a reviewer should see first*: the "analog beats digital" line is **retracted in the memo itself**; the surviving claim is stated as **1.60× proxy-energy cut for +0.160 bpc (5.3%)**; decomposition (−0.309 noise benefit + 0.160 gating cost); weight-decay-is-zero dissociation; and the quantizer contrast (−0.002 char-LM vs +0.369/+0.385 copy) carried with its **post-hoc** label and the "quantization alone keeps 0.78 margin ⇒ precision loss is part but not most of the deficit" corollary.
- **§8** the bound arm as a plain **negative** (12× floor swing vs 2× measured ρ *fall*; margin kept rising 0.079→0.200 as H shrinks ⇒ width-determined, not information-determined; optimization failure the bound does not explain). Both failed routes to a binding test are stated, including why raising M fails structurally.
- **§9** the datapath-degradation principle as the single organizing claim + the three-part silicon recommendation + the **ADC-step-floors-θ** co-design constraint (6-bit over ±4 rails ⇒ LSB 0.125, sub-LSB θ bit-identical) and θ's non-transfer across tasks.
- **§10** caveats to state before anyone quotes it — scale, "the energy proxy is a proxy and there is **no measured pJ win**", simulation-not-silicon, SSM recall weakness.

No numbers were re-derived for this memo; every figure was copied from the published ledger entries above (48510c8 and earlier), so nothing here can drift from the run JSONs. v2 is retained in the repo for history and now carries a superseded pointer.

**Remaining writeup targets, in order:** (1) restated char-LM headline into `paper.tex`, (2) the datapath-degradation principle as the paper's central claim with the two matched-rate tables as evidence, (3) the closed bound arm as theory-with-a-tested-scope-limit, (4) the talk deck / `speaker_notes.md` still carry the pre-retraction framing on the SSM slides and need the same correction pass. **No data cells remain; do not launch training.**

## 2026-07-31 tick — WRITEUP STEP 2: the talk deck's speaker notes are corrected to the post-retraction SSM framing (zero-GPU)

No data cells remain and no jobs were running (no `ssm3way.py` processes; all A800s free of our work), so this was a
writing tick per the standing "writeup is the bottleneck" verdict.

**What was stale — and it was worse than "out of date".** `talk/speaker_notes.md` presented direction #1 (the
analog-state SSM) as a *proposed* experiment: Slide 8 said "this is the cleanest test of the core hypothesis, and I can
start it in simulation today", and the Q&A appendix pre-registered a falsification condition ("if an analog-state
recurrent model still can't sparsify below the RNN floor at matched quality") for an experiment that has since run to
completion. Presenting from these notes would have pitched a finished, three-seed, two-task result as future work — and
would have left the speaker with no prepared answer for the two questions the data now forces (the retraction, and
whether the speaker's own bound survived its test).

**Seven in-place edits, all exact-matched (patch aborts if any anchor is missing) — no numbers re-derived, every figure
copied from published ledger entries / `imam_ssm_memo_v3.md`, so the notes cannot drift from the run JSONs:**
1. A STATUS block at the top flagging that direction #1 has run, that the headline moved twice with one retraction, and
   that the .pptx/.pdf have NOT been regenerated.
2. Slide 8 direction One rewritten from plan to result: the 4-way param-matched design, the matched-communication-rate
   char-LM comparison (analog 0.43 bpc better than spikestate at equal emission, 1.10 better than spikeout), the
   **unprompted self-retraction** of "analog beats digital" with the −0.309 noise / +0.160 gating decomposition and the
   weight-decay-is-zero dissociation, the copy inversion (spikeout 0.87 / analog 0.50 / spikestate exactly chance while
   cheapest), and the datapath-degradation principle as the slide-worthy line.
3. SEQUENCING updated (one is complete; three now has data to anchor it).
4. "Which first?" flipped from #1 to **#2 (measured Loihi/SpiNNaker2 energy)** — the honest answer now, since every pJ
   number in direction #1 is a 45nm proxy in simulation and the analog datapath's storage element + converter are
   unpriced by it. Hardware measurement is the binding constraint on the argument.
5. The Slide 6 "so just build analog neurons?" aside now reports the measured, workload-scoped answer.
6. The "what is YOUR contribution?" answer expanded to four items (adds the completed study + the principle), plus a
   new prepared Q&A entry **"Does your own bound hold up empirically?" answered NO** with the shrink-H numbers (12×
   floor swing vs ρ falling 0.496→0.250; margin kept rising 0.079→0.200 as capacity shrinks ⇒ optimization pathology,
   not information bottleneck), and the explicit separation of the still-standing empirical ~50% char-LM ceiling from
   the bound's failed quantitative prediction.
7. The falsification answer rewritten to report direction #1's qualified-positive-with-a-negative-inside, and to name
   direction #2 as the remaining load-bearing experiment.

Plus a **PPTX EDIT LIST** appended to the notes: the deck itself is untouched and still mislabels direction One as
future work, needs one new results slide, needs the bound annotated as scope-limited, and must never gain an "analog
beats digital" slide.

**Verified, not assumed:** both remaining occurrences of the string "beats digital" in the file sit inside retraction
language (the self-retraction and the never-add-this-slide instruction). Backup at `/tmp/speaker_notes.bak`.

**Remaining writeup targets:** (1) the `.pptx`/`.pdf` regeneration per the embedded edit list; (2) the paper. Note a
finding from this tick: **the arXiv `sparsity_ceiling.tex` (local, `/home/zeyu/neuro_paper/`) contains ZERO SSM content**
— `grep -ci "ssm\|analog"` returns 0. So "restate the char-LM headline in paper.tex" was mis-scoped in earlier ticks:
there is no SSM section to restate. The SSM×neuromorphic arm is a **separate second paper**, not a v2 edit, and that
scoping decision is the next real writeup step.

## 2026-07-31 tick — WRITEUP STEP 3: second paper DECIDED + skeleton drafted (zero-GPU)

No jobs running, all 8 A800s idle, no data cells remain (empirical arm closed at 48510c8).
Per the scoping finding in the previous tick (arXiv paper.tex has zero SSM content), the SSM×neuromorphic
arm is a STANDALONE SECOND PAPER. This tick created `paper2/skeleton.md` in the repo:

- Working title: "Degrade What the Task Ignores: Workload-Conditional Datapaths for State-Space Models
  on Neuromorphic Hardware". Target arXiv (cs.NE x cs.LG), 6-8 pp, single author, cites paper 1.
- Central claim = the datapath-degradation principle; 10-section plan mapping every section to
  published ledger evidence by commit hash (E1-E6), nothing re-derived.
- Includes a CLAIMS LEDGER: forbidden sentences ("analog beats digital", "bound confirmed", "190x")
  and mandatory labels (quantizer contrast = post-hoc; margins = vs regularized reference) are in the
  skeleton itself so later drafting ticks cannot drift.
- Figures/tables plan keyed to agg_charlm.py / agg_copy.py / plot_ssm3way.py outputs.
- Next: draft sections 3+4 (protocol + char-LM), then 5+6, then 7 (reuse memo v3 section 8 language).


## 2026-07-31 tick — paper2 drafting step 1: §3 (protocol) + §4 (char-LM) drafted as LaTeX (zero-GPU)

Writing tick per the standing verdict (empirical arm closed 48510c8; skeleton 56d987d). All 8
A800s verified idle, no ssm3way.py processes.

- New files: `paper2/sec03_protocol.tex`, `paper2/sec04_charlm.tex`. LaTeX section files;
  the build happens locally on /home/zeyu (server has no latex). Assembly (main.tex, refs,
  \citet{paper1} placeholder) deferred to a later tick.
- Discipline: Table 1 numbers were regenerated from `agg_charlm.py` THIS TICK and matched the
  published a134f05/bb79f7f ledger entries exactly before being copied in; nothing re-derived.
  Δbpc-vs-regularized column computed as published Δbpc(plain) + 0.309 shift (same constant for
  all variants), consistent with the bb79f7f re-referencing.
- Claims ledger honored: §4.2 reports the "analog beats digital" reading explicitly AS the
  retracted intermediate finding, with the −0.309 noise / +0.160 gating decomposition; §4.3
  states the surviving claim as the 1.60× proxy-energy-for-+0.160-bpc tradeoff; §3.5 carries
  the proxy-not-silicon + unpriced-converter caveats; emitted-rate-vs-state-activity
  distinction is a named metric in §3.4 (analog state density 0.994 stated in both sections).
- Next drafting steps (unchanged from skeleton): §5+§6 (copy + principle), §7 (bound, reuse
  memo v3 §8), intro/related last; then deck regeneration.


## 2026-07-31 tick — PAPER2 DRAFTING STEP 2: §5 (copy) + §6 (principle) drafted (commit 750b1bb, zero-GPU)

All 8 A800s idle, no jobs; writing tick per the standing "no data cells remain" verdict.
`paper2/sec05_copy.tex` (105 lines) + `paper2/sec06_principle.tex` (95 lines).

Discipline held: the copy master table was regenerated from `agg_copy.py 30` this tick and
verified against a8e7a7c / 8e74f48 / 159e341 / 5bb7bcc / 8d050cb / 48510c8 before any number
was copied in; nothing re-derived by hand.

NEW OBSERVATION surfaced by regenerating the full table (first time the M=64 column was read
against the others): **M=64 is VALIDITY-LIMITED and §5.3 says so explicitly.**
- The digital baseline is near-floor at M=64 (acc 0.1671±0.0010, only 2.7× chance, vs 10× at
  M=16), so every variant compresses toward chance.
- Margin kept vs plain digital is NON-MONOTONE in M for both surviving routes (spikeout
  0.92→0.46→0.67, analog 0.53→0.31→0.46) and paired Δbpc compresses (spikeout +0.71 at M=32 →
  +0.16 at M=64) — a shrinking-reference artifact, NOT a recovery. Do not cite M=64 margins
  as evidence that the gap narrows.
- The regularization control never ran at M=64 (digreg row was L∈{33,65} only), so no
  corrected margin exists there; Table 2 marks the M=64 margins as vs-plain.
- Consequence for claims: the "gap widens with load" line is scoped to M=16→32 in the paper.
- One robust M=64 datapoint that IS quoted: spikestate reproduces exactly-chance at a third
  load (acc 0.0623±0.0005), so "1-bit state eliminates recall" now holds at M=32 AND M=64.

§6 content: principle statement; state-fidelity-alone refuted (spikeout rate_state exactly
1.0000 everywhere, best on copy / worst on char-LM); quantizer contrast table (−0.002 char-LM
vs +0.369/+0.385 copy, footprints 0.971/0.976) carried with the MANDATORY post-hoc label and
the mis-specified-pre-registration history; the pre-registered M=32 floor (≥+0.37, met) noted
as the one genuinely pre-registered element; §6.4 reports the noise-benefit trend
(−0.150→−0.215→−0.309, runs toward char-LM) as weakening the strength-based supporting line.

Remaining writeup steps: §7 (bound — reuse memo v3 §8), intro/related/hardware/limitations,
main.tex assembly + local LaTeX build on /home/zeyu; deck regeneration still open.

## 2026-07-31 tick — PAPER2 DRAFTING STEP 3: §7 (the firing-floor bound) drafted

Zero-GPU writing tick. All 8 A800s idle (1 MiB, 0% util), no `ssm3way.py` processes, no data
cells remain — per the standing "writeup is the bottleneck" verdict.

**Written:** `paper2/sec07_bound.tex` (151 lines), the third drafting step after §3+§4
(be0ab4a) and §5+§6 (750b1bb). Language reuses memo v3 §8; all figures copied from the
published ledger entry **471f113** (final 18/18 shrink-H table, 3 seeds/cell) and **159e341**
(why raising M fails structurally) — nothing re-derived this tick.

**Structure:**
- §7 opener restates the bound with the corrected `M·log₂K` numerator and footnotes the
  superseded `log₂M` symbol reading (which would have made every cell unmeasurable).
- §7.1 *Raising the memory load cannot test it* — spikestate at exactly chance at M=32
  (0.0623±0.0012) and M=64 (0.0623±0.0005) vs chance 0.0625; the learnability ceiling
  (~M=16) sits below the binding load (M=64); the falling ρ with load (0.496→0.405) is
  explicitly NOT read as evidence against the bound, since retention is already violated.
  States plainly: copy columns must never be cited as bound tests.
- §7.2 *Shrinking the width is the valid design* — the four pre-registered criteria (i)–(iv)
  reproduced verbatim in prose, then Table 1 (H=256/128/96/64: floors 0.042/0.110/0.174/0.500,
  digital 0.6302/0.5944/0.5889/0.5600, spikestate 0.1071/0.1522/0.1570/0.1619,
  ρ 0.4960/0.3915/0.3909/0.2496, margin kept 0.079/0.169/0.179/0.200, certificate
  256.0/123.6/92.7/51.9 bits). Validity gate passes at every width (9–10× chance across an
  8.4× param range); criterion (ii) refuted **by anti-correlation, not flatness**.
- §7.3 *The anti-capacity finding* — certificate falls 4.0× surplus → 0.81× deficit while
  margin kept RISES monotonically and paired Δbpc improves in lockstep
  (+2.509→+2.079→+2.050→+1.829). The most over-provisioned net is the worst; an information
  bottleneck cannot produce that ordering. Includes the **retracted contrapositive** history
  in its own labelled paragraph (H=64 match coincidental; H=96 has a 1.4× surplus and fails
  identically, 0.157 vs 0.162).
- §7.4 *Verdict* — not empirically validated in either direction; theory with a tested scope
  limit. Closes by separating two things that survive: Paper 1's empirical ~50% char-LM
  firing ceiling (a measurement, not a prediction of the bound) and the practical
  "don't carry a compressed recurrent state in spikes" rule, which §5 establishes
  empirically instead ("the bound was the reason to expect that; it is not the reason to
  believe it").

**Claims discipline verified, not assumed:** grepped the file for the skeleton's forbidden
strings — the only hit is the header comment that *lists* them; the body contains no
"bound is confirmed", no contrapositive-earns-its-keep claim, and no 190× figure (the floor
swing is stated as 12× throughout, per the 471f113 correction).

**Remaining writeup steps:** §8 hardware implications (incl. ADC-step-floors-θ), then
intro/related/limitations/conclusion, then `main.tex` assembly + a local LaTeX build on
/home/zeyu (the server has no latex). Deck regeneration
(`talk/beyond_attention_paradigms.pptx` per the edit list embedded in `talk/speaker_notes.md`)
is still open. **No data cells remain — do not launch training.**


## 2026-07-31 tick — PAPER2 DRAFTING STEP 4: §8 (hardware implications) drafted

Zero-GPU writing tick (all 8 A800s idle, 0 compute apps, no ssm3way.py procs; no data cells
remain). `paper2/sec08_hardware.tex` added (~110 lines), per the skeleton's §8 plan, reusing
memo v3 §9 language. Three subsections:

- **§8.1 split recommendation** — CIM/analog state for statistical/low-retention workloads
  (char-LM: 1.60× proxy-energy for +0.160 bpc / 5.3% vs the noise-regularized digital
  baseline); exact digital state + spiked output for precise recall (margins 0.87 / 0.42 vs
  the regularized reference at M=16/32); spiking the state is a NON-OPTION for memory-bearing
  workloads (exactly chance at M=32 AND M=64: 0.0623±0.0012 / 0.0623±0.0005 vs 0.0625).
- **§8.2 ADC-floors-θ co-design constraint (E6)** — LSB q = 2r/2^b = 0.125 at 6 bits over
  ±4 rails; every θ < q bit-identical (θ=0.05 vs 0.10); event rate floored by converter
  precision, so sparsity and state precision trade off in hardware. Plus θ non-transfer:
  char-LM θ=1.0 collapses analog on copy (emits 3–8%); usable copy window 0.125 < θ ≲ 0.3;
  θ=0.5 over-gates to chance.
- **§8.3 energy honesty** — proxy licenses ONE claim (char-LM 446k vs 712k pJ/tok) and blocks
  two: analog on copy loses both axes to spikeout (M=16: 0.87 @ 122.9k vs 0.50 @ 246.7k;
  M=32: 0.42 @ 125.2k vs 0.28 @ 231.0k; structural reason = event pricing only on W_mix),
  and cheapness-as-merit (spikestate M=32 = cheapest cell, 102k pJ/tok, retains exactly
  nothing). Closes with the unpriced analog storage element + converter (state density 0.994)
  and points to measured Loihi-2/SpiNNaker-2 energy as the next step.

Numbers verified against published entries 83e110a, a9f36d2, 7a10a0d, bb79f7f, a8e7a7c,
8e74f48, 159e341, 5bb7bcc/8d050cb/48510c8 — nothing re-derived. Claims discipline checked by
grep: no "analog beats digital" outside retraction context, no "bound confirmed", no 190×,
copy margins quoted vs the regularized reference only.

Remaining writeup: intro (§1) / related (§2) / limitations (§9) / conclusion (§10), then
main.tex assembly + LaTeX build LOCALLY on /home/zeyu; deck regeneration
(`talk/beyond_attention_paradigms.pptx` per the edit list in `talk/speaker_notes.md`) still
open. No data cells remain — do not launch training.

## 2026-07-31 tick — PAPER2 DRAFTING STEP 5: §1 (introduction) + §2 (related work) drafted, zero-GPU

State at tick start: GPUs 2/3/5 busy with **another user's** jobs (17.5 GB, ~100% util; no
`ssm3way.py` processes anywhere), GPUs 4/6/7 idle, 0/1 holding another user's idle allocations.
No jobs of ours running; the empirical arm remains CLOSED (no data cells). Writing tick, per the
standing verdict.

**Drafted:** `paper2/sec01_intro.tex` (129 lines) and `paper2/sec02_related.tex` (94 lines),
following the skeleton's §1/§2 plans. Discipline held: every experimental figure quoted is copied
from a published ledger entry (a134f05, bb79f7f, 48510c8/8d050cb/5bb7bcc, 471f113, 83e110a) —
nothing re-derived, no aggregator re-run needed for these sections.

**§1 as drafted:** hook = the linear recurrence is what analog silicon physically is, with
LMU-on-Loihi/Braindrop as the existence proof; the three-way datapath fork (spike the state /
spike the output / hold the state in analog) as the paper's organizing question; the gap = no
controlled comparison at matched parameters or matched communication rate. Then the four
contributions, each pointing at its section. **Both retractions are stated in the introduction,
not buried:** the "analog beats digital" reading is presented as retracted with the control that
killed it (−0.309 bpc noise benefit on the baseline = 2.1× the apparent analog advantage), and
the surviving claim is given as the tradeoff (1.60× proxy energy for +0.160 bpc / 5.3%); the
firing-floor bound is introduced as a pre-registered test whose confirmation criterion was
**refuted** (12× floor swing, ρ falls 0.496→0.250, margin kept rises 0.079→0.200). Also states
up front that spikestate is at exactly chance at M=32 *and* M=64 while being the cheapest
variant, and that the weight-decay dissociation (+0.0047±0.0110, i.e. zero) pins the noise
mechanism to state-level stochasticity — the one point that argues *for* analog silicon on
non-energy grounds. Closing scope paragraph puts the simulation / 45nm-proxy / unpriced-converter
limits in the intro rather than only in §9.

**§2 as drafted:** lifted from memo v3 §1–3 as planned. SSM background (S4/S4D diagonal
recurrence = a bank of leaky integrators, which is why it is physically realizable) + LMU as the
pre-HiPPO linear SSM already deployed on Loihi *and* Braindrop (psMNIST 97.15% vs LSTM 89.86%).
The three routes with their representative work: QS4D for (a); SPikE-SSM for (b), with its
sparsity explicitly labelled as *output* sparsity (LRA ~8%, WikiText 24.5%, ppl 33.2 vs S4's
21.0) — the reason a state-carrying firing-floor argument does not apply to it, and which our
`rate_state` = 1.0000 measurement reproduces; CIM-SSM (+ IMSSA/HPD for analog robustness) for
(c). **CIM-SSM's reported numbers are deliberately NOT quoted** (paywalled, unverified) — the
memo's same discipline. Then the two literature gaps the paper fills (no matched-capacity
comparison; energy claims made at uncontrolled operating points) plus a third we created
ourselves: **the comparator's regularization**, noted as a control we have not seen run in this
literature, whose bias runs in favour of the neuromorphic variant everywhere. Final subsection
positions against paper 1 (its ~50% char-LM ceiling is a *measurement* and stands; the bound was
the prediction about *why*, and that is what fails in §7).

**Claims-discipline check run, not assumed:** grepped both files for the forbidden strings. §2 is
clean; §1's three hits are the header-comment list (2) and one occurrence inside the sentence that
performs the retraction — the same pattern as §7/§8.

**Assembly finding (useful for the next tick):** the entire paper cites exactly **9 keys** —
`paper1`, `s4`, `s4d`, `lmu`, `spikessm`, `cimssm`, `qs4d`, `imssa`, `hpd` — and all of them were
introduced in §1/§2; §3–§8 cite only `paper1`. So `refs.bib` is a 9-entry file, cheap to write.

**Remaining writeup steps:** §9 limitations + §10 conclusion (skeleton has both plans), then
`refs.bib` (9 keys) and `main.tex` assembly, then a LaTeX build **locally on /home/zeyu** (the
server has no latex). Deck regeneration (`talk/beyond_attention_paradigms.pptx` per the edit list
in `talk/speaker_notes.md`) still open. **No data cells remain — do not launch training.**


## 2026-07-31 tick — PAPER2 DRAFTING STEP 6: §9 (limitations) + §10 (conclusion) drafted (zero-GPU)

All 8 A800s idle (1 MiB, 0% util), no `ssm3way.py` procs; empirical arm remains CLOSED — writing tick.

- `paper2/sec09_limitations.tex` (67 lines): consolidates the caveats already stated locally in §3.5/§4.3/§5.3/§6.3/§7.4/§8.3 into one section a reviewer can find — proxy-not-silicon + unpriced analog storage/converter (state density 0.994); small scale/short schedule with the honest note that the −0.309 noise benefit is itself the kind of effect that shrinks at scale, so the surviving +0.160 bpc tradeoff could move in EITHER direction; two tasks with exactly one out-of-sample principle test (quantizer contrast, post-hoc label carried; the genuinely pre-registered element is the M=32 ≥+0.37 floor, met); the σ=0.02 regularized reference is saturated-dose but not exhaustively tuned, and "no result moves in the neuromorphic routes' favour if the baseline improves"; the bound-arm negative scoped as a tested scope limit, not a general refutation; the M=64 validity limit restated.
- `paper2/sec10_conclusion.tex` (42 lines): the fork answer is workload-conditional; the ranking reversal at matched rates/3 seeds; the principle as a designer's rule (spend the exactness budget on the side of the datapath the loss depends on); ADC-floors-θ + θ non-transfer as the analog route's co-design price; the bound closed as "design intuition, not validated theory" reusing the §7 sentence ("the bound was the reason to expect that result; it is not the reason to believe it"); closes on measured-silicon energy as the one missing number.
- Claims discipline verified by grep, not assumed: body of both files has no "analog beats digital", no "bound confirmed"/contrapositive, no "190", copy margins referenced only via the corrected sections.
- Every figure quoted (−0.309, +0.160/5.3%, 1.60×, 0.994, −0.002 vs +0.369/+0.385, ≥+0.37, 12×, 2.7× chance at M=64, +0.005±0.011 wd) copied from published ledger entries bb79f7f / 48510c8 / 471f113 / 750b1bb — nothing re-derived.
- **All 10 sections are now drafted. Remaining: refs.bib (9 keys: paper1, s4, s4d, lmu, spikessm, cimssm, qs4d, imssa, hpd), main.tex assembly, LaTeX build LOCALLY on /home/zeyu (server has no latex); deck regeneration still open. No data cells — do not launch training.**

## 2026-07-31 tick — PAPER2 ASSEMBLED AND IT COMPILES: `main.tex` + `refs.bib` written, full 10-section build verified locally (15 pp, clean)

Zero-GPU tick. All 8 A800s idle (0–4 MiB, 0% util), no `ssm3way.py` processes, no data cells
remain — writeup-only per the standing verdict. Commits **7fc91b2** (assembly) and **1f8eb48**
(build fixes), pushed to `origin/main` (verified: `git status -sb` shows no ahead marker).

**What was written.**
- `paper2/main.tex` (95 lines): `article` 10pt preamble (geometry, lmodern, amsmath, booktabs,
  graphicx, microtype, natbib round, hyperref), title/author (Zeyu Wang, Georgia Tech —
  matching paper 1), a ~330-word abstract, and `\input` of sec01…sec10 in order, then
  `\bibliography{refs}`. The claims-discipline header comment from the section files is repeated
  at the top of `main.tex` so the assembly file carries the same constraints.
- `paper2/refs.bib` (78 lines, exactly the 9 cited keys: `paper1, s4, s4d, lmu, spikessm,
  cimssm, qs4d, imssa, hpd`).

**Abstract content — every figure copied from published ledger entries, nothing re-derived.**
It states, in this order: matched-parameter + matched-communication-rate protocol, 2 tasks,
3 seeds; the ranking **inverts** between tasks; char-LM surviving claim = **1.60× proxy-energy
cut for +0.160 bpc (5.3%)** with the **retraction performed in the abstract itself** (baseline
state noise σ=0.02 is worth −0.309 bpc = 2.1× the apparent advantage); copy margins quoted
**already corrected against the noise-regularized reference** (spikeout 0.87/0.42, analog
0.50/0.28, spikestate exactly chance at M=32 and M=64 while cheapest); the
datapath-degradation principle plus its out-of-sample **post-hoc-labelled** quantizer contrast
(−0.002 char-LM vs +0.369/+0.385 copy); the bound arm as a **negative** with the corrected
**12×** floor swing (ρ 0.496→0.250, margin kept rising) and the "width-determined, not
information-determined" wording; and a closing scope sentence (45 nm proxy from simulation, not
silicon; analog storage element and converter unpriced).

**BIBLIOGRAPHY HONESTY — deliberate, do not "fix" by guessing.** The memo v3 reading list marks
`qs4d` (2507.06079), `imssa` (2412.20215) and `hpd` (2508.11935) as *identified via search, not
read in full*. Their titles/authors are therefore **left unasserted**: each entry carries the
arXiv ID plus an explicit "metadata to be verified before submission" note and a `key` field for
sorting, rather than an invented title/author list. `cimssm` carries its known
title/authors/venue but repeats that its quantitative results are **not quoted** (paywalled).
`paper1` is an arXiv preprint with **no ID yet** (submission `submit/7882862` still unannounced),
so the note gives the repo URL instead of a number — that ID must be filled in before submission.

**Build verified locally on /home/zeyu (server has no LaTeX):** sources pulled with
`ssh … 'tar czf - -C /work/zeyuwang/sparsity-ceiling paper2'` (git clone not needed),
`pdflatex → bibtex → pdflatex ×2`. Result: **exit 0, no undefined citations or references, no
multiply-defined labels, ZERO overfull/underfull box warnings, 15 pages, 520 kB**. Two real build
fixes were needed and are committed: (1) `microtype`'s font expansion aborts fatally on this
box's non-scalable default fonts — `\usepackage{lmodern}` fixes it (first pass died with
"auto expansion is only possible with scalable fonts", no PDF produced); (2) the three
metadata-unasserted bib entries have no author to sort by, so BibTeX warned — added `key` fields.
Artifacts kept at `/home/zeyu/paper2_build/` (all sources + `main.pdf`) for the next tick.

**LENGTH FINDING — the draft is 15 pp single-column against a 6–8 pp target in the skeleton.**
That is a real gap, not a rounding error: the paper carries 3 large tables and 10 sections at
10pt one-column with 1in margins. Options for the next tick, cheapest first: (a) switch to a
two-column style (roughly halves it; also what cs.NE arXiv submissions usually look like),
(b) move the shrink-H table and the copy M=64 column to an appendix, (c) prose trimming. Do
**not** solve it by dropping caveats or retraction language — those are load-bearing.

**Remaining writeup items:** (1) length/format pass per above; (2) fill `paper1`'s arXiv ID once
announced and verify the three TODO bib entries; (3) regenerate
`talk/beyond_attention_paradigms.pptx/.pdf` per the edit list embedded in `talk/speaker_notes.md`
(still pre-retraction). **No data cells remain — do not launch training.**

## 2026-07-31 tick — PAPER2 FORMAT/LENGTH PASS: two-column build is clean, but the length target does NOT fall out of formatting

Zero-GPU tick. Server state: GPU 1 was another user's job (13.7 GB / 100%), GPUs 0,2-7 idle at
1 MiB; zero `ssm3way.py` processes. Empirical arm remains closed — no data cells launched.

**What was done.** Fix (a) from the previous tick's length plan: `paper2/main.tex` switched to
`\documentclass[10pt,twocolumn]`, margins 1in -> 0.85in, `\columnsep` 0.28in, and the
title+abstract wrapped in `\twocolumn[\begin{@twocolumnfalse}...]` so they still span the page.
The three wide tables (Table 1 char-LM, Table 2 copy, Table 3 shrink-H) became `table*` floats at
`\footnotesize`; the shrink-H table also got `\tabcolsep` 4pt. Build verified locally at
`/home/zeyu/paper2_build/`: **exit 0, 13 pp, ZERO overfull boxes, zero undefined citations/refs**.

**HONEST FINDING — the previous tick's estimate that "two-column roughly halves it" was WRONG.
Two-column bought 15 pp -> 13 pp, about 13%, not 50%.** Measured cause, not guessed: the body is
**8,404 words** across sec01-sec10, and the rendered density is ~780 words/page (pages 1-12 carry
671-844 words each; page 13 holds only the bibliography's 80 words). Single-column 10pt at 1in
margins was already dense, so the two-column saving is line-breaking slack only. **The remaining
gap to the skeleton's 6-8 pp target is CONTENT LENGTH, not formatting** — reaching 8 pp means
removing roughly 2,500-3,000 words or ~35% of the body.

**Two overfull boxes existed and both are now fixed** (they only appeared once the text was in
narrow columns, so this pass found them rather than caused them): the sec04 `\underbrace`
decomposition display (-0.309 gift + 0.160->0.326 gating cost) was 17.6pt too wide for a column
and is now stacked with `\substack`; the sec06 quantizer-contrast tabular was 41.6pt too wide and
was compacted (shorter header, `{\to}`/`{\pm}` spacing, `\tabcolsep` 4pt) with the "three seeds,
quantizer-alone at matched sigma=0.02" qualifier moved into the adjoining prose so **no
information was dropped** to make it fit.

**Recommendation for the next writeup tick, stated as a decision rather than a menu.** arXiv has
no page limit and 13 pp two-column is a normal arXiv length for a paper carrying three result
tables plus a closed negative arm, so **the default should be to accept 13 pp for the arXiv
version and drop the 6-8 pp target**, which was set before the section drafts existed. If a
conference version is wanted later, the cut order is: (1) move the shrink-H table and the copy
M=64 column to an appendix (M=64 is already validity-limited and non-citable per 750b1bb), (2)
compress sec07 prose, which is the longest section at 9.8 kB and reports a negative result.
**The claims-discipline rule still binds any cut: caveats, post-hoc labels and retraction
language are never what gets trimmed.**

**Remaining writeup items:** (1) length decision per above, (2) fill paper 1's arXiv ID and verify
the three TODO bib entries (qs4d/imssa/hpd, deliberately unasserted) once available, (3) regenerate
`talk/beyond_attention_paradigms.pptx/.pdf` per the edit list in `talk/speaker_notes.md`, which is
still pre-retraction. No data cells remain — do not launch training.

## 2026-07-31 tick — TALK DECK REGENERATED to the post-retraction SSM framing (zero-GPU)

State at tick start: all 8 A800s idle (1 MiB, 0% util), zero `ssm3way.py` processes, empirical arm
closed, no data cells remain. The deck was the largest open writeup item — it still carried the
**pre-result** SSM framing, i.e. it pitched direction One (analog-state SSM) as future work months
after the study finished, with no retraction language anywhere.

**What was done.** `talk/beyond_attention_paradigms.pptx` patched in place by a committed, re-runnable
script `talk/patch_deck.py` (python-pptx). The script matches every target by its exact existing text
and **aborts if any anchor is missing or ambiguous**, so it can never half-apply; whitespace matching
is normalized because the deck uses U+202F inside "Loihi 2". Pre-patch deck preserved at
`/tmp/deck_pre_patch.pptx` on the server. Deck goes 9 → 10 slides.

1. **New results slide inserted at position 8** ("07 · SSM RESULTS", *Direction One, run: what the
   analog-state SSM actually did*), cloned from the directions slide so the styling is identical.
   Four blocks: (01) analog state buys **energy, not quality** — 1.60× proxy-energy cut (446k vs
   712k pJ/token) for +0.160 bpc, with **"beats digital" stated as retracted** and the killing control
   named (−0.309 bpc baseline noise); (02) the ranking **inverts by workload** at matched
   communication rate, copy margins **0.87 / 0.50 / 0.074 quoted vs the noise-regularized reference**,
   1-bit state at exactly chance and cheapest; (03) the **datapath-degradation principle** with the
   quantizer contrast (−0.002 char-LM vs +0.37 copy) carrying its **post-hoc** label; (04) **"My own
   bound did not survive its test"** — 12× floor swing vs ρ falling 0.496→0.250, accuracy improving as
   capacity shrinks, bound = theory with a tested scope limit.
2. **Directions slide** renumbered to "08 · PROPOSED DIRECTIONS" / page 09, retitled *What I would work
   on next*; direction 01 folded from a proposal into "Analog state — done in simulation, open in
   silicon", pointing at direction 02 (measured Loihi 2 / SpiNNaker 2 energy) as what to run first,
   since every pJ figure is a 45nm proxy and the analog storage element and converter are unpriced.
3. **Slide 6 ("THE BET")** — the line claiming analog state gives "no memory wall, no firing floor" now
   reads "no memory wall; the firing floor is a spiking-state property (the bound itself: theory,
   scope limit tested)", per edit-list item 3.

**Every figure was copied from `imam_ssm_memo_v3.md` / published ledger entries, none re-derived**, so
the deck cannot drift from the run JSONs. Claims discipline holds: no "analog beats digital" except
inside the retraction sentence, no "bound confirmed", floor swing stated as 12×, copy margins vs the
regularized reference, energy labelled a proxy.

**Two things caught by verifying the render instead of trusting the save** (both real, both fixed):
(i) first draft of the new slide's body text was 273–301 chars against the deck's own 139–165-char
blocks and would have overflowed a 4.70×1.00 in box — trimmed to ≤216 chars with no claim or caveat
dropped; (ii) the cloned slide came out **white**, because the deck's dark background is a `p:bg`
element on the slide, not a shape in `spTree` — python-pptx's `add_slide` does not carry it, so the
script now deep-copies `p:bg` into the new slide's `p:cSld`. Both verified by rasterizing the built
PDF and looking at pages 6, 8 and 9.

**PDF rebuilt**: the server has **no LibreOffice**, so `beyond_attention_paradigms.pdf` was regenerated
locally on /home/zeyu (`soffice --headless --convert-to pdf`, LibreOffice 7.3), verified at 10 pages,
and copied back. Build note for future ticks: **pptx → pdf must happen on /home/zeyu, not the server.**

`talk/speaker_notes.md`'s PPTX EDIT LIST is now marked APPLIED with the commit reference, so a later
tick does not redo it.

**Remaining writeup items:** (1) fill paper 1's arXiv ID once `submit/7882862` is announced and verify
the three deliberately-unasserted bib entries (qs4d / imssa / hpd), (2) nothing else is open — paper 2
compiles clean at 13 pp, memo v3 is send-ready, speaker notes and deck are both post-retraction.
**No data cells remain — do not launch training.**


## 2026-07-31 tick — PAPER 1 IS ANNOUNCED ON ARXIV AND PAPER 2'S BIBLIOGRAPHY IS NOW FULLY ASSERTED (zero-GPU)

**Paper 1 has its arXiv ID: `arXiv:2607.26648` (v1)** — "The Sparsity Ceiling: Where Spiking
Networks Can and Cannot Trade Activity for Energy", Zeyu Wang, confirmed via the arXiv API
title search (`ti:"sparsity ceiling"`, single hit). The submit/7882862 queue wait is over.

**The three deliberately-unasserted bib entries are verified against the arXiv API listing**
(`id_list=2507.06079,2412.20215,2508.11935`, fetched 2026-07-31) and filled in:
- `qs4d` = QS4D: Quantization-aware training for efficient hardware deployment of structured
  state-space sequential models — Siegel, Yang, Bouhadjar, Fabre, Neftci, Strachan (2025).
- `imssa` = IMSSA: Deploying modern state-space models on memristive in-memory compute
  hardware — Siegel, Yang, Strachan (2024).
- `hpd` = HPD: Hybrid Projection Decomposition for Robust State Space Models on Analog CIM
  Hardware — Feng, Zhou, Lyu, Liu, Liu, Wong, Kang (2025).
All three titles are consistent with how sec02 classifies them (qs4d → route (a) quantized
digital; imssa/hpd → route (c) analog CIM), so no prose change was needed. The "not read in
full" honesty note stays in refs.bib — verification covers metadata, not content.

**paper1's refs.bib note now carries arXiv:2607.26648** instead of the repo-URL placeholder.

**Rebuild verified locally on /home/zeyu** (server has no latex): full pdflatex+bibtex cycle,
exit 0, ZERO overfull boxes, zero undefined citations/references, 13 pp two-column, 531 kB.
Grep confirmed no tex source contains stale "to be verified"/"unannounced" language.

**With this, the LAST remaining writeup item from the 868f067/e9615bc tick list is closed:**
paper 2 compiles clean with a fully-asserted 9-entry bibliography, memo v3 is send-ready, deck
and speaker notes are post-retraction. Open decisions that are the USER'S, not a tick's:
send memo v3 to Imam; decide whether/when to put paper 2 on arXiv. No data cells — do not
launch training. Infra: arXiv API over plain HTTP returns empty from this box — use HTTPS.

## 2026-08-01 tick — reference verification: QS4D read in full; sec02 route-(a) classification CORRECTED (zero-GPU)

State: GPUs 0-6 all idle (1 MiB, 0-2% util; GPU7 still absent from bus), zero ssm3way procs; empirical arm closed. This tick advanced the one remaining pre-submission item: full-read verification of the deliberately-unasserted bib entries (qs4d/imssa/hpd), one per tick.

**QS4D (arXiv:2507.06079; Siegel, Yang, Bouhadjar, Fabre, Neftci, Strachan) read in full (arXiv full text).** Verified:
- Quantizes **S4D with diagonal A** ("we focus on the S4D model with a diagonalized A matrix"; original S4 in supplementary) — the "diagonal SSMs" half of our old sentence was correct.
- **BUT its deployment target is memristive ANALOG in-memory compute, not digital fabric**: tantalum-oxide resistive devices co-integrated with 180 nm CMOS; QAT is motivated by AIMC and shown to confer robustness to analog noise and enable structural pruning; sCIFAR10 reaches 6-bit homogeneous QAT vs 10-bit post-training quantization at the 1% error threshold. Fully non-spiking. Same group and substrate as IMSSA.
- Therefore sec02's route-(a) sentence ("QS4D studies quantization of diagonal SSMs for exactly this deployment target [digital datapath]") **misclassified it**.

**Fix (this commit):** sec02 route (a) now carries no exemplar citation (it is the reference point and needs none); QS4D moved into the route-(c) CIM-adjacent sentence with the verified description; refs.bib header + qs4d entry comment updated (read-in-full 2026-08-01, correction recorded); memo v3 §3(a), §3(c) and its reading list corrected identically (anchor-checked patch, aborts on miss). Local rebuild verified: exit 0, 0 overfull, 0 undefined cites/refs, 13 pp.

**No experimental number changes anywhere** — the correction is citation placement and description only. Remaining unread refs: **imssa (2412.20215), hpd (2508.11935)** — same treatment, one per tick, before any arXiv submission of paper 2.

- **2026-08-01 tick — the STALE-ARTIFACT item is CLOSED, zero-GPU: `paper2/paper2.pdf` rebuilt locally from the post-QS4D sources and re-committed, so the committed PDF now matches the committed .tex for the first time since the QS4D reclassification.** Server state at tick: **7 GPUs visible (0–6, GPU 7 still absent), all idle (1 MiB, 0–2% util), zero `ssm3way.py` procs**, empirical arm still closed, no data cells.
- **Staleness was VERIFIED, not assumed.** All 11 `.tex` files + `refs.bib` are byte-identical (md5) between `/home/zeyu/paper2_build/` and the repo, so the divergence was PDF-only. `pdftotext` on the committed PDF found **one** "memristive" hit (inside the `imssa` bib title, present pre-QS4D) versus **two** in the rebuild (the new sec02 phrase "tantalum-oxide memristive crossbars" plus the bib one) — direct evidence the committed PDF predated commit 9df405b. Text diff is confined to the QS4D reclassification paragraph (QS4D moved out of route (a) "digital, optionally quantized" and into the route-(c) CIM-adjacent paragraph, with the QAT-confers-noise-robustness detail added) plus downstream pagination reflow; body word count 9,350 → 9,370. **No experimental number, caveat, post-hoc label or retraction sentence changed.**
- **Build result:** full `pdflatex → bibtex → pdflatex ×2` cycle, all four passes **exit 0**, **zero undefined citations/references**, **13 pp / 531,343 bytes**. Log carries **2 Underfull hboxes and ZERO Overfull** — the earlier ledger claim of "zero overfull boxes" reproduces exactly; the two underfulls are badness-10000/3657 lines inside `main.bbl` (bibliography line-breaking) and are cosmetic.
- **Infra note for future ticks:** `scp` of the 531 kB PDF to the A800 exceeded a 120 s foreground window and had to be backgrounded — the standing "scp is flaky, small files ok" warning applies at ~0.5 MB, not just at dataset scale. Budget a background transfer for any PDF-sized artifact. Also reconfirmed: LaTeX exists ONLY on /home/zeyu (server has no latex), so paper2 rebuilds are always local-then-upload.
- **PROJECT STATE unchanged otherwise: the SSM×neuromorphic thread still has NO open tick-work.** Empirical arm closed; paper 2 compiles clean at 13 pp with a fully asserted bibliography and a PDF that now matches its sources; memo v3 send-ready; deck + speaker notes post-retraction. What remains are USER decisions, not tick steps: (1) send `talk/imam_ssm_memo_v3.md` to Imam (outward-facing — do not auto-send), (2) whether/when to arXiv-submit paper 2 (external — user's call), (3) optionally full-read imssa/hpd (qs4d already read in full). Future ticks should idle (status line only) unless the user redirects or paper 1 v2 (the overfull line-98 fix) is requested.


## 2026-08-01 tick — IMSSA (arXiv:2412.20215) READ IN FULL; sec02 attribution corrected (zero-GPU)

Server state: 7 GPUs visible (0–6, GPU 7 still absent), all idle, zero ssm3way.py procs. Empirical arm closed; no data cells. This tick executed the pending "optionally full-read imssa/hpd" item for IMSSA.

**What the full read established (arXiv HTML v1):**
- IMSSA = Siegel, Yang, Strachan (2024) — the same group's earlier work preceding QS4D (2507.06079).
- Contribution is a **hardware deployment demonstration**, not a robustness method: recurrent S4D kernels with A, B and C programmed into a **single 64×64 memristive crossbar**, executed **step-by-step** (explicitly not unrolled to a convolution), QAT extended for analog IMC down to **ternary (2-bit) weights**; stronger quantization measured as MORE robust to write noise (σ > 15 μS sustainable at 2-bit vs 5 μS at 5-bit).
- Task: two-class Heidelberg spoken digits ("zero" vs "one"). **Deployed 81.69% vs 95.06% software reference**; authors attribute the drop to stuck-high devices and estimate 95.61% without them. An on-silicon instance of analog imperfection charging quality — consistent with our simulated lossy-state findings.
- **No energy or area numbers.** Whether the recurrent state stays analog between steps or is digitized each step is **not explicitly stated** (output currents are converted for the next timestep; ADC not detailed) — so IMSSA does not settle the native-analog-state question; CIM-SSM remains the only route-(c) exemplar claiming device-physics state decay.

**Corrections applied (anchor-checked patch, `/tmp/imssa_patch.py` on server):**
1. `paper2/sec02_related.tex` — the sentence "IMSSA and HPD address the robustness side directly" **misattributed IMSSA**; rewritten so IMSSA is described as the deployment demonstration (crossbar, ternary weights, 81.7% vs 95.1%, stuck devices) and only HPD keeps the "addresses robustness directly" clause.
2. `paper2/refs.bib` — imssa comment block upgraded from "not read in full" to read-in-full with the key facts; the honesty note about the unstated state-digitization is in the comment.
3. `talk/imam_ssm_memo_v3.md` — reading-list checkbox ☑ for IMSSA with the same facts; line-45 clause no longer lumps IMSSA under "analog-CIM robustness". HPD stays □ not-read-in-full.

**No experimental number, caveat, post-hoc label or retraction sentence touched.** Consequence: `paper2/paper2.pdf` is again 1 commit stale vs sources (same pattern as the QS4D tick) — rebuild LOCALLY on /home/zeyu before submission or next paper2 edit. Remaining unread bib entry: **hpd (2508.11935) only**.

## 2026-08-01 tick — HPD (arXiv:2508.11935) READ IN FULL; sec02/refs.bib/memo-v3 annotations upgraded — NO misattribution found (contrast IMSSA)

- **Full read (4 pp, cs.AR):** simulation-only (PyTorch 2.2, NVIDIA L20 GPU) study of **WEIGHT-perturbation** robustness for Mamba/Mamba2 on analog CIM. Vulnerability analysis: the **final block and its output projection** are the most noise-sensitive components (other layers ~flat under sigma up to 0.05); smaller models degrade more. Fix = SVD split of the LM-head projection: **U·Sigma stays on the CIM array** (same shape as W_out, no hardware change), **V^T offloaded to digital** for exact correction. Evaluated on Mamba/Mamba2 130M–1.4B, WikiText ppl + ARC-e/PIQA/LAMBADA, Gaussian + lognormal weight noise.
- **sec02's existing sentence ("HPD addresses the robustness side directly") was ACCURATE — nothing to retract.** Sentence expanded with three full-read facts: (i) simulation, not silicon; (ii) the noise model perturbs **weights, not the state** — so HPD does not bear on analog state decay, and CIM-SSM remains the only route-(c) exemplar claiming device-physics state (consistent with the IMSSA entry); (iii) HPD's remedy is itself a **hybrid datapath split allocating exactness to the most loss-sensitive component** — an independent instance of the datapath-degradation principle from an unrelated group, now cited as such in sec02.
- **QUOTE TRAP recorded in refs.bib:** HPD's abstract line "reduces perplexity by up to 99.57%" is a **degradation-removed robustness ratio** (their eq. 16), NOT a perplexity reduction — never quote it as a ppl cut.
- **Reading-list status: CLOSED.** All 9 paper-2 bib keys have verified metadata and qs4d/imssa/hpd are all read in full; refs.bib header comment updated to say so. No experimental number, caveat, post-hoc label or retraction touched.
- **Consequence:** `paper2/paper2.pdf` is now stale vs sources (this commit) — rebuild LOCALLY on /home/zeyu (server has no latex) before submission; that is the next tick's candidate step.
- **Server state this tick:** 7 GPUs visible (0–6, GPU 7 still absent), all idle (1 MiB, 0–2%); zero ssm3way.py procs; empirical arm closed.

## 2026-08-01 tick — paper2.pdf rebuilt from post-IMSSA/HPD sources (stale-artifact fix, zero-GPU)

The committed `paper2/paper2.pdf` was 2 commits stale vs sources (e05324b IMSSA correction + a6d2d87 HPD read). Rebuilt LOCALLY on /home/zeyu (server has no latex) and re-committed.

- **Staleness scope verified, not assumed:** `git diff --name-only e355b71..a6d2d87` touches only `sec02_related.tex` + `refs.bib` among build inputs; those two were fetched and all 12 build inputs confirmed md5-identical to repo HEAD a6d2d87 before building.
- **Build:** pdflatex → bibtex → pdflatex ×2, all exit 0; **zero Overfull boxes, zero undefined cites/refs; 13 pp / 532,933 B** (was 531,343 B).
- **Content check on the rebuilt PDF (pdftotext, local):** IMSSA edit present ("hardware demonstration", "81.7% against a 95.1% software reference", "stuck devices") and HPD edit present ("weights, not the state", hybrid-split sentence). No experimental number, caveat, post-hoc label or retraction sentence touched — this is a pure artifact sync.
- **Transfer integrity:** post-scp md5 `6880c004940d197e6776056322ee2a9b` and byte size match the local build exactly.
- **State:** 7 GPUs visible (0–6, GPU 7 still absent), all idle, zero ssm3way.py procs. Empirical arm closed; no data cells.
- **Committed PDF now matches committed sources.** Remaining items are USER decisions only: send `talk/imam_ssm_memo_v3.md` to Imam; arXiv-submit paper 2.

## ENERGY-DATAPATH PHASE — 2026-08-03

New phase goal: turn the measured event-activity win into a measured pJ win by
removing the MAC-priced input/output layers, or record honestly why it cannot be
done.  Prior state: analog-state SSM reaches ~27-61% event activity at a +0.160
bpc cost vs the noise-regularized digital reference on char-LM, but the pJ proxy
credits event pricing ONLY to the recurrence W_mix, so the published number is
446k vs 712k pJ/token = 1.60x -- an activity win, not a datapath win.

### Step 1 — DECOMPOSITION (`energy_datapath.py`, zero-GPU, no new training)

Per-term shares of the PUBLISHED proxy, char-LM (V=284, E=64, H=256), computed
from the existing run JSONs (no quality number re-derived):

| cell | n | bpc | r_z | pJ/tok | W_in% | W_mix% | W_out% |
|---|---|---|---|---|---|---|---|
| digital (any reg arm) | 3 | 3.0338 (n0.02) | 1.000 | 712448 | 10.6 | 42.3 | **46.9** |
| analog th=0.15 | 3 | 3.1936 | 0.612 | 446142 | 16.9 | 8.1 | **75.0** |
| analog th=1.0 | 3 | 3.3595 | 0.266 | 425746 | 17.7 | 3.7 | **78.6** |
| spikeout | 3 | 4.3978 | 0.369 | 432753 | 17.4 | 5.0 | **77.3** |
| spikestate | 3 | 3.6205 | 0.645 | 156834 | 48.1 | 24.3 | 26.9 |

**THE BLOCKING LAYER IS THE READOUT, NOT THE RECURRENCE.** W_out carries 75% of
the analog variant's pJ/token and 47% of digital's, while W_mix -- the only term
the published proxy prices as event-driven for analog -- carries 8%.  At V=284,
H*V = 72,448 exceeds H*H = 65,536, so the readout is the largest single matrix in
the model.  Every neuromorphic mechanism in this project acts on the recurrence.

**CEILING under the current datapath:** analog with a *free* recurrence
(r_z -> 0) still costs **410,040 pJ/token = 1.74x** vs the regularized digital
reference.  The published 1.60x is therefore already at 92% of everything
recurrence sparsification can ever deliver here.  No threshold, no better event
code, and no additional sparsity can move this -- it is a datapath property.

**PROJECTION (assumption, not measurement: r_in = r_out = the cell's measured
r_z):** re-pricing the readout alone takes analog th=0.15 from 446k (1.6x) to
**152k (4.7x)**; re-pricing input and readout takes it to **85k (8.3x)**, while
the same treatment gives digital only 383k (1.9x) because its readout input
changes every step.  Under a fully event-driven datapath the perverse ranking is
also fixed: spikestate is **no longer cheapest** (91.0k) -- analog at th=0.75
costs 53.1k with bpc 3.293 vs spikestate's 3.621.

**Step 3 of the goal — pricing the unpriced — does NOT change the verdict.**
Sweeping the ADC/DAC conversion cost over four decades (0.01-1.0 pJ per graded
event) and the analog storage element over 0-0.5 pJ per unit per timestep, the
converter+storage terms are **0.00%-0.33%** of the analog total.  At H=256 they
are 2-4 orders of magnitude below the matrix terms.  This is honest in both
directions: it removes "unpriced converters" as an objection to the analog route
at this scale, and it does NOT license claiming converters are free in general --
they would matter at large H once the matrices are cheap.

### Step 2 — EVENT-DRIVEN READOUT implemented and row LAUNCHED

`ssm3way.py` gains `--out_theta` (default 0 => every previously-run cell is
bit-unaffected).  The readout becomes an incremental send-on-delta accumulation:
`d_t = u_t - uref; m_t = |d_t| > out_theta; acc += W_out_w @ (d_t*m_t); uref += d_t*m_t`,
so only emitting units of `u = W_mix(z)` ever touch W_out.  **Verified exact, not
assumed:** at out_theta=1e-12 every unit emits and the incremental logits match
the dense readout to **1.4e-7 relative** (float32 noise).  Energy prices W_out at
`H*V*r_out*E_AC` when the gate is on; `rate_out` is now measured and recorded.

Row (14 cells, char-LM seed 0, 6 ep / 1.4M chars, same budget as the published
row): `{analog th=0.15, digital+noise0.02} x out_theta in {0.02,0.05,0.1,0.25,0.5,1.0,2.0}`.
Workers on GPUs 4-7 only (haomo holds GPU 0 with a sharded job; 1-3 left free as
headroom for their remaining shards).  Driver `run_eventreadout.sh`, lock-dir
work queue, idempotent.

### PRE-REGISTERED READING (written and committed BEFORE any cell landed)

1. **The projection's assumption is what is on trial.** Confirmation = at a
   threshold whose quality cost is small (<=0.05 bpc vs the same arm at
   out_theta=0), analog's measured `r_out` is at or below its `r_z`=0.612, so the
   projected 4.7x/8.3x stand.  Refutation = a low `r_out` is only reachable at a
   large bpc cost, i.e. the readout is a *worse* place to buy sparsity than the
   recurrence.
2. **The fair-comparator test, which matters more than (1).** The digital arm
   gets the same event readout.  If digital captures most of the same benefit,
   then the energy was never in the analog state at all -- the honest verdict
   becomes *"the readout is where the energy is, and an event-driven readout is
   available to a digital SSM too"*, which is a **negative for the analog route
   as such** while being the most useful design finding of this phase.  Analog
   should still win somewhat, because its `u = W_mix(z)` is fed by an
   already-gated `z` and so should change on fewer steps than digital's dense
   GELU output.
3. **Most likely outcome, stated up front:** partial (2) -- both arms benefit
   greatly, analog's `r_out` lands below digital's at matched threshold, so
   analog keeps a real but *smaller* edge than the 4.5x the projection implies,
   and the phase's headline becomes a datapath claim rather than an analog claim.
4. Goal item (1) -- a sparse-input streaming task to make W_in legitimately
   event-priced -- is **deferred and partly blocked**: W_in is only 11-18% of the
   total (the readout is 47-79%), and the local NMNIST copy is an incomplete
   3.1 MB partial `test.zip` with no working outbound proxy.  Low payoff, high
   infra cost; the readout is the right target first.

### Step 2 RESULT — the event-readout row is COMPLETE (14/14) and the phase goal is answered NO: the activity win does NOT convert into a pJ win by this route

Char-LM seed 0, 6 ep / 1.4M chars, reference = **digital + state-noise 0.02**
(the post-retraction correct comparator), bpc 3.0309 @ 712,448 pJ/token.
`pJ_pub` prices W_out at `H*V*r_out*E_AC`; `pJ_fev` additionally event-prices
W_in and charges the analog converter+storage terms.

| arm | out_theta | bpc | Δbpc vs ref | r_out | r_z | pJ_pub | × | pJ_fev | × |
|---|---|---|---|---|---|---|---|---|---|
| analog θ=0.15 | 0 | 3.1848 | +0.154 | 1.000 | 0.618 | 446508 | 1.60 | 111272 | 6.40 |
| analog θ=0.15 | 0.02 | 4.0837 | **+1.053** | **0.979** | 0.615 | 175913 | 4.05 | 109627 | 6.50 |
| analog θ=0.15 | 0.05 | 4.0903 | +1.059 | 0.948 | 0.625 | 174470 | 4.08 | 108331 | 6.58 |
| analog θ=0.15 | 0.1 | 4.1265 | +1.096 | 0.895 | 0.613 | 170331 | 4.18 | 104025 | 6.85 |
| analog θ=0.15 | 0.25 | 4.0976 | +1.067 | 0.741 | 0.637 | 161631 | 4.41 | 95669 | 7.45 |
| analog θ=0.15 | 0.5 | 4.1496 | +1.119 | 0.529 | 0.658 | 149038 | 4.78 | 83385 | 8.54 |
| analog θ=0.15 | 1.0 | 4.1895 | +1.159 | 0.258 | 0.683 | 132789 | 5.37 | 67512 | 10.55 |
| analog θ=0.15 | 2.0 | 4.2257 | +1.195 | 0.100 | 0.664 | 121331 | 5.87 | 55781 | 12.77 |
| digital+n0.02 | 0 | 3.0309 | 0.000 | 1.000 | 1.000 | 712448 | 1.00 | 382822 | 1.86 |
| digital+n0.02 | 0.02 | 4.0564 | **+1.026** | **0.901** | 1.000 | 436959 | 1.63 | 376338 | 1.89 |
| digital+n0.02 | 0.05 | 4.0496 | +1.019 | 0.764 | 1.000 | 427990 | 1.66 | 367367 | 1.94 |
| digital+n0.02 | 0.1 | 4.0667 | +1.036 | 0.594 | 1.000 | 416873 | 1.71 | 356250 | 2.00 |
| digital+n0.02 | 0.25 | 4.0450 | +1.014 | 0.357 | 1.000 | 401377 | 1.78 | 340755 | 2.09 |
| digital+n0.02 | 0.5 | 4.0613 | +1.030 | 0.258 | 1.000 | 394899 | 1.80 | 334277 | 2.13 |
| digital+n0.02 | 1.0 | 4.0364 | +1.006 | 0.084 | 1.000 | 383520 | 1.86 | 322898 | 2.21 |
| digital+n0.02 | 2.0 | 4.1286 | +1.098 | 0.012 | 1.000 | 378765 | 1.88 | 318148 | 2.24 |

**PRE-REGISTERED CRITERION (1) IS REFUTED.** It required a threshold whose
quality cost is ≤0.05 bpc.  The *cheapest* readout gate in the row costs
**+0.90 bpc** (analog, vs its own out_theta=0 cell) and every other threshold
costs more.  The readout is a far worse place to buy sparsity than the
recurrence: gating the recurrence bought 1.60× for **+0.160 bpc**; gating the
readout costs **~+1.0 bpc before it buys any meaningful sparsity at all**.

**THE ACTUAL FINDING, AND IT IS NOT ABOUT SPARSITY.** At `out_theta`=0.02 the
analog arm still emits on **97.9%** of readout units — essentially nothing is
gated — and quality has *already fully collapsed* (+0.90 bpc).  Driving r_out
from 0.979 down to 0.100 (a 10× sparsification) then costs only **+0.14 bpc
more**.  So the penalty is a **step function at the moment the hold is switched
on**, near-flat thereafter, and is therefore a property of the **mechanism**, not
of the sparsity it purchases.  Mechanism, verified in code rather than inferred:
the accumulator is *exactly* `acc_t = W_out·uref_t + b`, i.e. a readout of the
**last-sent (stale) value** of `u = W_mix(z)`, so a per-unit staircase error of
≤`out_theta` is summed across H=256 units into every logit, and the readout
becomes **stateful** — each `u_s` now influences all later logits, changing
credit assignment for the whole sequence.  Small per-unit error, large logit
error, harder optimization.

**PRE-REGISTERED CRITERION (2) — the fair-comparator test — CONFIRMED, and it is
the reason the route fails.** The digital arm gets the same readout gate and the
same ~+1.0 bpc penalty, so **no part of this energy saving belongs to the analog
state**; whatever the readout gate is worth is worth it to a digital SSM too.

**PRE-REGISTERED PREDICTION (3) IS WRONG, in the opposite direction.** It said
analog's `r_out` should fall *below* digital's at matched threshold because its
`z` is already gated.  Measured, at `out_theta`=0.1: analog **0.895** vs digital
**0.594** — analog's readout input is *jumpier*, not smoother, because gating `z`
makes it jump discontinuously whenever a unit emits, which drives *more* readout
events.  Record as a wrong prediction, not a nuance.  Related minor observation:
the recurrence rate `r_z` drifts *up* with the readout gate (0.618 → 0.683), so
the two gates interact rather than compose independently.

**OUT-OF-SAMPLE SUPPORT FOR THE DATAPATH-DEGRADATION PRINCIPLE (the one positive
here).** The principle says char-LM needs an exact graded **output** and
tolerates a lossy **state**.  Against the same regularized reference: degrading
the *state* (analog send-on-delta) costs **+0.160 bpc**, while two mechanically
unrelated degradations of the *output* cost **+1.026/+1.053** (send-on-delta
readout hold) and **+1.367** (spikeout's LIF output) — a **6–9× larger** penalty.
Stated honestly as order-of-magnitude agreement between two unrelated output
degradations; the two numbers are **not** equal (+1.03 vs +1.37) and must not be
presented as a quantitative match.

**VERDICT ON THE PHASE GOAL (item 4): NO — analog does not win on pJ, and the
blocking layer is now precisely located.** Chain of evidence:
1. W_out carries **75%** of analog's pJ and 47% of digital's; the recurrence
   carries **8%**.  The energy was never where the mechanisms were.
2. The current datapath's ceiling is **1.74×** even with a free recurrence, so
   the published 1.60× is already at **92%** of that ceiling.
3. The readout *can* be made event-driven (r_out to 0.012, up to 5.87× published
   / 12.8× full-event), but at **~+1.0 bpc on char-LM regardless of threshold**,
   and the same option is available to digital.
4. The previously-unpriced analog costs are **not** the obstacle: over four
   decades of converter cost and a 25× range of storage cost they are
   **0.00–0.33%** of the analog total at H=256.
So on char-LM the ~27–62% event-activity win is **not convertible** into a pJ win
by removing the MAC-priced layers.  The honest headline for this workload remains
the published energy-for-quality tradeoff (1.60× for +0.160 bpc).

**WHAT THE PRINCIPLE PREDICTS NEXT, and it is the one route left to a real
measured pJ win.** An event readout is an *output* degradation, so it should be
expensive exactly where an exact output matters (char-LM — confirmed above) and
**cheap on precise-recall**, where output-spiking was already the *best*
neuromorphic variant (spikeout kept 0.87/0.42 of the digital margin at M=16/32
while analog kept 0.50/0.28).  **Next tick: run the event-readout sweep on the
copy task** (L=33/65, ep30/n80k, the established budget).  Pre-registered now:
if the readout gate is cheap on copy, that is the first configuration in the
project where a large pJ cut is bought at little quality cost, and it would be a
*digital-state + event-readout* design, not an analog one.  If it is expensive
there too, the honest conclusion is that event-driven readouts do not pay on
either workload and the readout's 75% energy share is simply not recoverable at
this scale.

**Scope/caveats.** n=1 seed (calibration row, per project convention — a
3-seed confirmation is owed before any of this is quoted in the paper); the
+1.0 bpc step is large and consistent across 14 cells and two arms, so seed noise
(digital sd ~0.02 bpc) cannot explain it, but the *mechanism* attribution
(staleness summed over H vs altered credit assignment) is **reasoned from the
code, not experimentally separated** — an ablation that holds `uref` fixed for a
random subset of units at matched r_out would separate them.  All pJ figures
remain a 45 nm Horowitz proxy from simulation, not measured silicon.
Reproduce: `python agg_eventro.py`, `python energy_datapath.py ssm3way_runs`.

## Energy-datapath phase, step 3 — event-driven readout on the COPY task (LAUNCHED 2026-08-03 ~07:15Z, seed 0)

**The one route left to a real measured pJ win, pre-registered last tick (a9ff715):** on char-LM the
readout gate cost ~+1.0 bpc at any threshold — a step function at the moment the hold switches on —
because char-LM needs an exact output distribution. The datapath-degradation principle predicts the
SAME gate is CHEAP on precise-recall workloads, where output-spiking (also an output degradation) was
already the best neuromorphic variant (margin kept 0.87 / 0.42 vs the noise-regularized reference at
M=16/32). If that prediction holds, **digital-state + event-readout** is the project's first large pJ
cut at little quality cost. If it fails here too, the conclusion is that the readout's 75% energy
share is not recoverable at this scale on either workload class.

**Row:** `run_eventro_copy.sh` (lock-dir work queue). 14 cells = digital + dig_noise 0.02,
out_theta ∈ {0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0} × L ∈ {33, 65} (M=16, 32), seed 0, ep30/n80k.
Out: `ssm3way_runs/digital_copy_L<L>_s0_reg_n0.02_ot<θ>_ep30.json`. Reference = the existing
3-seed `digital_copy_L{33,65}_s{0,1,2}_reg_n0.02_ep30.json` cells (identical budget and
regularization; the ONLY delta is the readout gate). Chance acc 0.0625.

**PRE-REGISTERED CRITERIA (written before any cell lands):**
1. **CONFIRMATION (principle holds, pJ win real):** at some out_theta with r_out ≤ 0.5 the gated run
   keeps ≥ 0.85 of the reference's above-chance margin (margin kept = (acc − 0.0625)/(acc_ref − 0.0625),
   per-L). That would be a ≥2× readout-traffic cut at ≤15% quality cost — qualitatively unlike
   char-LM's +1.0 bpc step.
2. **REFUTATION:** the char-LM step function reappears — margin collapses (< 0.5 kept) already at
   r_out ≈ 1 (nothing meaningfully gated), at BOTH L. Then the readout's energy share is not
   recoverable on either workload class and the honest verdict is "the 75% term is stranded".
3. **Partial/boundary outcome to report as-is:** cheap at M=16 but expensive at M=32 (retention demand
   interacts with readout staleness), or a graded (non-step) tradeoff — either is a finding, not a failure.
4. **Most likely, stated up front:** confirmation at M=16, weaker at M=32 — by the same load-dependence
   every neuromorphic mechanism has shown on copy.

**Caveats owed before any paper use:** n=1 seed (3-seed confirmation required, as for the char-LM row);
pJ accounting for the copy datapath must reuse `energy_datapath.py`'s terms, not the headline proxy alone.


### 2026-08-03 ~08:45Z — eventro-copy row, M=16 COLUMN IN (7/14 cells): the pre-registered "most likely" outcome is FALSIFIED — the event readout does not merely degrade copy at M=16, it ELIMINATES it, at every threshold

**Status:** L=33 (M=16) column complete (7 θ cells, seed 0); L=65 (M=32) column training on 7 GPUs
(launched 08:37Z as L=33 workers freed). This entry applies the pre-registered criteria (commit
94a415f) to the half that is in. n=1 seed — same caveat as pre-registered.

**Reference (3 seeds, identical budget+σ=0.02 noise reg, no readout gate):** acc **0.6628±0.0088** /
bpc 1.2702±0.0285. Chance acc 0.0625; above-chance margin 0.6003.

**Gated cells (L=33, s0), acc / margin kept / r_out / bpc:**

| out_theta | acc | margin kept | r_out | bpc |
|---|---|---|---|---|
| 0.02 | 0.0638 | 0.002 | 0.9966 | 73.02 |
| 0.05 | 0.0627 | 0.000 | 0.9990 | 23.38 |
| 0.10 | 0.0625 | 0.000 | 0.9979 | 359.97 |
| 0.25 | 0.0635 | 0.002 | 0.9467 | 4.05 |
| 0.50 | 0.0704 | 0.013 | 0.9941 | 4.36 |
| 1.00 | 0.0640 | 0.002 | 0.9064 | 4.10 |
| 2.00 | 0.0633 | 0.001 | 0.7462 | 1304.85 |

**Verdict on the M=16 half, per the pre-registered criteria:**
- Criterion (1) CONFIRMATION is not met at any θ (no cell even reaches r_out ≤ 0.5, and margin kept
  peaks at 0.013 vs the required ≥ 0.85).
- Criterion (4), the pre-registered MOST-LIKELY branch ("confirmation at M=16, weaker at M=32"),
  is **falsified outright** — M=16 is not the easy case, it is a total collapse.
- Criterion (2) REFUTATION requires collapse "at BOTH L", so the formal refutation verdict WAITS for
  the running L=65 column. But the M=16 half already exceeds the refutation shape: margin < 0.5 kept
  at r_out ≈ 1 — measured ≤ 0.013 at r_out 0.91–1.00.

**The signature is worse than char-LM's step function, and it looks like a TRAINING failure, not an
inference-quality tax.** On char-LM the gate cost a flat ~+1.0 bpc but the model still functioned;
here acc sits at exactly chance at every θ, including θ=0.02 where 99.66% of readout units emit
every step — i.e. the gate destroys copy while gating essentially nothing. And bpc is wildly
non-monotone in θ (73 / 23 / 360 / 4.05 / 4.36 / 4.10 / 1305): the ≈4.0 cells learned a uniform
output (chance bpc = log2 17 ≈ 4.09 here vs 4.000 for K=16 payload), the 23–1305 cells are
confidently wrong — divergence, not graceful degradation. Mechanistic reading (unverified, flag as
interpretation): copy's recall segment demands a *different* symbol every step through a readout
whose held-value path the straight-through gate trains poorly; the recurrence-side send-on-delta
never showed this because the state carries its own dynamics. A stale-readout mechanism that costs
+1.0 bpc on the workload that tolerates output loss WORST (char-LM) and 100% of the margin on the
workload that tolerates it BEST (copy, where LIF output-spiking kept 0.87) means **θ-gated
send-on-delta staleness is not the same degradation axis as LIF output-spiking** — the
datapath-degradation principle's axis ("output degradation") was too coarse for this prediction.

**What this means for the energy-datapath question (pending L=65 confirmation):** the one
pre-registered route left to a real measured pJ win — digital-state + event-readout on
precise-recall — does not survive contact with the data at M=16. If L=65 matches, the honest
verdict is the pre-registered refutation sentence: **"the readout's 75% energy share is stranded on
both workload classes"** — sparsifying the readout by *holding stale values* is quality-fatal
everywhere tested, and the only readout sparsification that works (LIF output-spiking, copy) is one
this energy model prices as events already. n=1 seed; do not fold into paper2 before the L=65
column and (if any signal survives) 3-seed confirmation.

## 2026-08-03 ~09:40Z — eventro-copy L=65 (M=32) column COMPLETE (7/7): pre-registered REFUTATION is now FORMAL at both L, and the energy-datapath phase closes with a NO

**Data (copy L=65, M=32, K=16; ep30/n80k; seed 0; chance acc 0.0625).** Reference = the existing 3-seed noise-regularized digital baseline with no readout gate (`digital_copy_L65_s{0,1,2}_reg_n0.02_ep30`): acc **0.3425 ± 0.0027**, above-chance margin **0.2800**. Gated cells (`_ot<θ>`):

| out_theta | acc | bpc | r_out | margin kept |
|---|---|---|---|---|
| 0.02 | 0.0630 | 20.75 | 0.9964 | 0.002 |
| 0.05 | 0.0629 | 35.09 | 0.9743 | 0.001 |
| 0.10 | 0.0988 | 3.96 | 0.9774 | 0.130 |
| 0.25 | 0.0634 | 4.11 | 0.9774 | 0.003 |
| 0.50 | 0.0768 | 4.02 | 0.9279 | 0.051 |
| 1.00 | 0.0625 | 50.70 | 0.9180 | 0.000 |
| 2.00 | 0.0796 | 3.99 | 0.6315 | 0.061 |

**Criterion application (pre-registered in 94a415f, applied per-L).** CONFIRMATION required some out_theta with r_out ≤ 0.5 keeping ≥ 0.85 of the margin. No L=65 cell even *reaches* r_out ≤ 0.5 (min 0.63), and the best margin kept anywhere is 0.13. REFUTATION required the char-LM step-function shape — margin < 0.5 already at r_out ≈ 1 — at **both** L: at M=16 margin ≤ 0.013 at r_out = 0.9966; at M=32 margin ≤ 0.002 at r_out = 0.9964. Both L satisfy it. **The pre-registered refutation sentence therefore applies: the readout's 75% energy share is stranded on both workload classes.**

**Same collapse shape as M=16, so one mechanism.** bpc is wildly non-monotone in θ (20.7 / 35.1 / 3.96 / 4.11 / 4.02 / 50.7 / 3.99) — cells sit either at learned-uniform (≈4.0 = log2 16) or diverged (20–51), never on a graded quality/sparsity curve. As at M=16, this is a **training failure induced by the gate**, not an inference-quality tax: collapse is complete at θ=0.02 where 99.6% of readout units still emit and essentially nothing is gated. The "most likely" branch pre-registered in 94a415f (cheap at M=16, weaker at M=32) was wrong at both loads, in the same direction.

**ENERGY-DATAPATH PHASE VERDICT (goal step 4): analog does NOT win on pJ, and the 75% readout share is not recoverable by any route tested.**
- Decomposition (c8b596f): `W_out` carries 75% of the analog variant's pJ/token, the recurrence 8%; even a free recurrence caps the analog advantage at 1.74× vs the regularized digital reference, and the published 1.60× is already 92% of that ceiling. Converters/analog storage are 0.00–0.33% across a 4-decade sweep — not the obstacle.
- Send-on-delta staleness on `W_out` is **quality-fatal on both workload classes**: +~1.0 bpc step on char-LM at any threshold (a9ff715); total collapse to ~chance on copy at both M=16 (5df8644) and M=32 (this entry).
- The one readout sparsification that survives copy — LIF output-spiking, margin kept 0.87 / 0.42 — recomputes `W_out` every step, so it saves none of the MAC energy that staleness would have.
- Consequence for the literature framing: every neuromorphic-SSM mechanism tested here (and the cited routes) acts on the recurrence, an 8% term on a language-like datapath; the layer that dominates the energy cannot be event-driven without stepping on quality, by either gating mechanism, on either workload class.

**Caveats that travel with the verdict:** all gated cells are n=1 seed (3-seed confirmation owed before any paper use, though the effect sizes — 0.28 margin → ≤0.036 abs — dwarf the reference's 0.003 seed sd); the reading that send-on-delta staleness and LIF spiking are *different degradation axes* (so "output degradation" in the datapath principle was too coarse a category) remains an interpretation, not an experimentally separated attribution — the clean ablation (hold `uref` for a random unit subset at matched r_out) was pre-registered at a9ff715 and remains unrun.

## 2026-08-03 ~10:20Z — PRE-REGISTERED ABLATION LAUNCHED: is the event-readout collapse STALENESS ITSELF, or the send-on-delta SELECTION RULE?

The energy-datapath phase closed with a formal NO (73c99e6): the readout carries 75% of the
analog SSM's pJ/token and every gating mechanism tested is quality-fatal. The one attribution
in that verdict that was **reasoned from the code rather than measured** is the claim that
send-on-delta *staleness* is a different degradation axis from LIF output-spiking — flagged as
interpretation at a9ff715, with the clean ablation ("hold `uref` for a random unit subset at
matched r_out") pre-registered and unrun. This row runs it.

**Mechanism change (code, verified by diff):** new `--out_prand p` selects the held units by a
per-unit per-step Bernoulli draw (send with probability `p`), *independent of the delta*, instead
of by `|u - uref| > out_theta`. Everything else in the datapath is identical — same incremental
accumulation `acc += W_out·(d·m_o)`, same energy pricing (`ev_out` now also true under the random
gate), same budget. Default `out_prand=0.0`, so **every existing cell is bit-unaffected**;
smoke-tested at p=0.60 → measured `r_out` 0.5997, i.e. the knob does set the staleness rate.

**Row (6 cells, char-LM, seed 0, 6 ep / 1.4M chars — the published budget):** send rates matched
to the theta row's MEASURED r_out. `digital` (reg, `--dig_noise 0.02`) at p ∈ {0.90, 0.59, 0.26,
0.08} against theta 0.02/0.1/0.5/1.0 (r_out 0.901/0.594/0.258/0.084, bpc 4.056/4.067/4.061/4.036);
`analog` (theta 0.15) at p ∈ {0.90, 0.26} against out_theta 0.1/1.0 (r_out 0.895/0.258, bpc
4.127/4.190). No-gate references on disk: digital 3.031, analog 3.185.

**PRE-REGISTERED READINGS (written before any cell lands, so they cannot be fitted after the fact).
The comparison is Δbpc vs the arm's own no-gate reference, at matched r_out:**
- **(A) SELECTION-INDEPENDENT — the flagged interpretation is CONFIRMED.** Random-subset staleness
  costs **≥ 0.7 bpc** at p≈0.90 on both arms (i.e. it lands within ~0.3 bpc of the theta cells' ~1.0).
  Then the harm is *holding a stale readout at all*, not the small-delta selection rule: no smarter
  threshold rescues the readout's 75% energy share, and staleness is confirmed as a degradation axis
  distinct from LIF output-spiking (which keeps 0.87 of the copy margin). Strengthens the phase verdict.
- **(B) SELECTION-DEPENDENT — the verdict needs qualifying.** Random-subset staleness costs
  **≤ 0.2 bpc** at p≈0.90. Then the collapse is a property of *the send-on-delta rule specifically*
  (it preferentially freezes exactly the units whose small increments carry the signal), the
  "staleness is fatal" sentence must be narrowed to "send-on-delta staleness is fatal", and searching
  for a better gating rule becomes a live route back to the readout energy. This would reopen a
  closed phase, so it is the outcome that must not be soft-pedalled if it happens.
- **(C) INTERMEDIATE (0.2–0.7 bpc)** → report as graded: staleness carries part of the cost and
  selection carries part; no reopening, but the axis claim is weakened to "mostly staleness".
- **Also pre-registered: random gating being much WORSE than theta at matched rate (Δbpc ≫ 1.0) is
  consistent with (A), not a failure** — the theta rule at least holds only small deltas, so random
  holding of large deltas should be no better. The discriminating comparison is (B)-vs-(A), i.e.
  whether random staleness is *cheap*, not whether it is worse than theta.
- **Most likely outcome, stated up front: (A).** The theta row's penalty was already flat in r_out
  (a step at the moment holding switches on, +0.14 bpc for a further 10× sparsification), which is
  the signature of a mechanism insensitive to how much is held — and therefore likely insensitive to
  which is held.
- **Caveats owed:** n=1 seed (the phase's convention; effect sizes of interest are ≥0.7 bpc vs a
  seed sd of ~0.003 on the references, so seed noise cannot decide it). char-LM only — the copy arm
  of this ablation is NOT run, so any (A) conclusion about the *copy* collapse stays an inference
  from the shared mechanism, not a measurement.

Driver `run_outrand.sh <gpu>` (lock-dir work queue, idempotent, skips finished cells).
Results land as `ssm3way_runs/{digital_charlm_s0_reg_n0.02,analog_charlm_s0_theta0.15}_pr<p>.json`;
markers in `copy_logs/outrand.log`.

## 2026-08-03 — Readout-staleness attribution ablation: CRITERION (A) — staleness itself is fatal; the send-on-delta selection rule is exonerated

Row from commit 8dac9bb (`--out_prand`, driver `run_outrand.sh`): readout units held by a
per-unit per-step Bernoulli draw at send probability p, independent of the delta; char-LM,
seed 0, 6 ep/1.4M chars; p matched to the theta row's measured r_out. All 6 cells complete.

Δbpc vs each arm's own no-gate reference (digital reg n0.02 = 3.0309; analog θ=0.15 = 3.1848):

| arm     | rule       | r_out  | bpc    | Δbpc vs no-gate |
|---------|------------|--------|--------|-----------------|
| digital | random p=0.90 | 0.900 | 4.0962 | **+1.0653** |
| digital | θ=0.02        | 0.901 | 4.0564 | +1.0255 |
| digital | random p=0.59 | 0.590 | 4.1793 | +1.1484 |
| digital | θ=0.1         | 0.594 | 4.0667 | +1.0358 |
| digital | random p=0.26 | 0.260 | 4.3232 | +1.2923 |
| digital | θ=0.5         | 0.258 | 4.0613 | +1.0304 |
| digital | random p=0.08 | 0.080 | 4.4684 | **+1.4375** |
| digital | θ=1.0         | 0.084 | 4.0364 | +1.0055 |
| analog  | random p=0.90 | 0.900 | 4.1064 | **+0.9216** |
| analog  | θ_out=0.1     | 0.895 | 4.1265 | +0.9417 |
| analog  | random p=0.26 | 0.260 | 4.4239 | +1.2391 |
| analog  | θ_out=1.0     | 0.258 | 4.1895 | +1.0047 |

**Pre-registered verdict (criteria from commit 8dac9bb, applied as written):**
- **(A) CONFIRMED on both arms**: random staleness at p≈0.90 costs +1.065 bpc (digital) and
  +0.922 bpc (analog), both ≥ the 0.7 threshold. Staleness of the readout input — holding the
  last-sent value instead of recomputing — is *itself* the fatal mechanism. The previously
  flagged interpretation ("send-on-delta staleness ≠ the LIF output-spiking degradation axis")
  is now experimentally supported on char-LM, no longer inference-only.
- **(B) is excluded**: the selection rule is NOT the culprit. At every matched r_out the
  delta-triggered gate is *no worse* than random — and at low rates it is clearly better
  (digital r_out≈0.08: θ 4.036 vs random 4.468, a 0.43 bpc gap in θ's favour). So
  send-on-delta is already a *good* selection rule; there is no smarter gating rule to find.
  The readout-energy route stays CLOSED.
- Secondary shape: the random-rule penalty is *monotone* in held fraction (+1.07→+1.44 as
  p 0.90→0.08) while the θ-rule penalty is flat (~+1.01 to +1.04 across a 10.7× rate range).
  Reading: the step function at gate-on is the staleness cost; the θ rule's delta-triggered
  updates then keep the marginal cost of further sparsification near zero, whereas random
  holding keeps paying. Both facts point the same way: the ~+1.0 bpc floor is set by
  staleness, not by which units are held.

**Effect on the phase verdict (73c99e6): STRENGTHENED, unchanged in content.** The readout's
75% energy share is stranded: recompute-every-step spiking (LIF) survives quality but saves
no MAC energy; any hold-based scheme pays ~+1.0 bpc the moment it turns on, under the best
selection rule available.

**Caveats (owed with any use of this row):** n=1 seed (effect sizes ~1.0 bpc vs seed sd ~0.03
on this task/budget); char-LM only — the copy-task collapse's attribution to staleness remains
an inference from the shared mechanism, the copy arm of this ablation was not run; analog arm
has 2 rate points vs digital's 4.

## 2026-08-03 ~11:15Z — 3-SEED CONFIRMATION ROW LAUNCHED (seeds 1,2 of the gated-readout cells) — criteria pre-registered before any cell

The energy-datapath phase verdict (73c99e6) and the staleness attribution (60d627c) rest on
n=1-seed gated cells. This row adds seeds 1,2 at three representative operating points per
curve (`run_seed12.sh`, lock-dir queue; 30 cells, ~10 GPU-h):

- **copy eventro**: digital reg n0.02, L∈{33,65} × out_theta∈{0.02,0.1,1.0} × s∈{1,2} (ep30/n80k)
- **char-LM eventro**: digital reg n0.02 AND analog theta0.15 × out_theta∈{0.02,0.1,1.0} × s∈{1,2} (6ep/1.4M)
- **outrand**: digital pr∈{0.90,0.08}, analog pr0.90 × s∈{1,2} (6ep/1.4M)

The untested thetas {0.05,0.25,0.5,2.0} and outrand rates {0.59,0.26} stay n=1 **by design** —
this row confirms the claims at the operating points that carry them, not the full curves.
References for paired Δbpc/margin: the existing 3-seed no-gate cells
(`digital_charlm_s*_reg_n0.02`, `analog_charlm_s*_theta0.15`, `digital_copy_L{33,65}_s*_reg_n0.02_ep30`).

**PRE-REGISTERED CRITERIA (written before any cell landed):**

1. **CONFIRMED** if, at 3-seed means: (a) the char-LM readout-gate penalty is ≥ +0.7 bpc vs
   each arm's own no-gate reference at EVERY tested out_theta (s0 values: +0.90 to +1.05); (b)
   copy margin kept ≤ 0.15 at every tested theta at BOTH L (s0: ≤0.013 M=16, ≤0.13 M=32); (c)
   outrand: random-hold ≥ +0.7 bpc at p=0.90 on both arms AND delta-triggered ≤ random at the
   matched low rate (s0: theta 0.43 bpc better at r_out≈0.08). Then the phase verdict and the
   attribution both hold at 3 seeds and the n=1 caveat is retired for the cited claims.
2. **WEAKENED** if any tested cell's 3-seed mean effect is below HALF its s0 size — name the
   claim it carries and re-scope it; do not average it away.
3. **Most likely: (1)** — s0 effect sizes (0.9–1.4 bpc; total collapse vs ref sd 0.003–0.009)
   are 30–300× the known seed noise of ungated cells.

Caveat that stays regardless: the copy-collapse attribution to staleness remains an inference
from the char-LM ablation (the copy arm of outrand is not run).


## 2026-08-03 ~13:15Z — 3-seed confirmation row COMPLETE (30/30): pre-registered verdict = CONFIRMED on all three criteria, no claim weakened

The seed-1/2 replication of the gated cells (driver `run_seed12.sh`, pre-registration in commit d84e8f3) finished at 12:51Z; aggregation script `agg_seed12.py` (committed) applies the criteria exactly as pre-registered. All effects below are 3-seed mean±sd; Δbpc is paired per-seed against the same arm's no-gate reference.

**[1] char-LM event-readout — criterion "gate penalty ≥0.7 bpc at every tested ot, both arms": MET.**

| arm (no-gate ref bpc) | ot | bpc | Δbpc vs ref | r_out |
|---|---|---|---|---|
| digital reg n0.02 (3.0338±0.0165) | 0.02 | 4.1233±0.0768 | **+1.0895±0.0650** | 0.899 |
| | 0.1 | 4.1553±0.0867 | **+1.1215±0.0780** | 0.577 |
| | 1.0 | 4.0233±0.0237 | **+0.9895±0.0397** | 0.106 |
| analog θ=0.15 (3.1936±0.0078) | 0.02 | 4.1012±0.0229 | **+0.9075±0.0204** | 0.979 |
| | 0.1 | 4.1334±0.0086 | **+0.9398±0.0081** | 0.895 |
| | 1.0 | 4.2483±0.0527 | **+1.0546±0.0448** | 0.262 |

The step-function shape reproduces at 3 seeds: the penalty is ~+1.0 bpc and FLAT across a r_out range of 0.98→0.11 (digital ot=0.1 is even nominally worse than ot=0.02) — the cost is incurred at gate-on, not per unit of sparsity bought.

**[2] copy event-readout — criterion "margin kept ≤0.15 at both L": MET, with margin to spare (all cells ≤0.05).**

| L (M) | ref acc (3-seed, no gate) | ot | acc | margin kept | r_out |
|---|---|---|---|---|---|
| 33 (16) | 0.6628±0.0090 | 0.02 | 0.0634±0.0009 | **0.0015±0.0015** | 0.998 |
| | | 0.1 | 0.0791±0.0287 | 0.0277±0.0478 | 0.992 |
| | | 1.0 | 0.0703±0.0146 | 0.0130±0.0243 | 0.900 |
| 65 (32) | 0.3425±0.0027 | 0.02 | 0.0737±0.0186 | **0.0400±0.0665** | 0.998 |
| | | 0.1 | 0.0749±0.0207 | 0.0442±0.0741 | 0.991 |
| | | 1.0 | 0.0663±0.0066 | 0.0136±0.0235 | 0.874 |

Total collapse at r_out≈0.998 (nothing meaningfully gated) is now a 3-seed fact at both memory loads, not an s0 anecdote.

**[3] outrand attribution — criterion "random-hold ≥0.7 bpc at p=0.90 both arms AND delta-gate ≤ random at the low rate": MET.**
digital p=0.90 Δbpc **+1.1172±0.0450**; analog p=0.90 Δbpc **+0.9218±0.0215**; digital p=0.08 Δbpc +1.4410±0.0155 vs delta-gate at matched r_out≈0.08 Δbpc +0.9895 → delta-triggered stays **0.45 bpc better** than random at the low rate (branch-A reading intact: staleness itself is fatal; send-on-delta is exonerated as a selection rule and no smarter rule reopens the route).

**WEAKENED check: EMPTY** — every 3-seed mean effect is ≥ half its s0 size (most are within ~10% of s0). Per pre-registration, **the n=1 caveat RETIRES for the cited claims**: the ~+1.0 bpc char-LM readout-staleness step (both arms), the copy total-collapse at both L, and the random-hold attribution at p=0.90.

**Caveats that REMAIN (by design, not omission):** (i) copy-collapse attribution to staleness is still an inference — no copy outrand arm was run; (ii) the untested operating points (ot ∈ {0.05,0.25,0.5,2.0}, outrand rates {0.59,0.26}, analog low-rate pr0.26) stay n=1; the low-rate delta-vs-random comparison is digital-arm only; (iii) energy numbers remain the 45nm Horowitz proxy from simulation.

**Phase state: the energy-datapath phase verdict (73c99e6, strengthened 60d627c) now stands on 3-seed evidence.** Remaining work is WRITEUP only: fold the energy decomposition + stranded-readout negative into paper2 (new section) and memo v3 (one paragraph), with the confirmed numbers above.


## 2026-08-03 ~14:00Z — WRITEUP FOLD-IN: the energy-datapath phase verdict is now in paper2 and memo v3 (zero-GPU tick)

Paper2 gains **sec 8.4 "The energy is not where the mechanisms are"** (`\label{sec:hardware-stranded}`):
the W_out 75% / recurrence 8% decomposition (H·V = 72,448 > H² = 65,536 at V=284), the 1.74×
free-recurrence ceiling (410,040 pJ/token; the published 1.60× = 92% of it), the converter/storage
sweep (ADC/DAC 0.01–1.0 pJ/event, storage 0–0.5 pJ/unit/step → 0.00–0.33% of the analog total at
H=256), the stranded-readout negative at 3 seeds (char-LM +0.99..+1.12 bpc digital / +0.91..+1.05
analog at EVERY threshold, flat across send rates 0.98→0.11; copy margin kept ≤0.05 at both M=16
and M=32 incl. r_out≈0.998; gated copy cells train to uniform or diverge), and the outrand
attribution (staleness itself fatal: +1.12/+0.92 bpc at 0.90 send rate; send-on-delta exonerated,
0.45 bpc better than random at the low rate; LIF output-spiking survives quality but recomputes
every MAC).

Supporting edits: sec 8.3's closing paragraph and sec 9's "Simulation, not silicon" now say the
converter/storage terms ARE priced under the proxy (small at this width) while keeping the
proxy-not-silicon caveat; sec 9 gains a "Scope of the stranded-readout negative" paragraph (copy
attribution = inference, no copy outrand arm; intermediate operating points n=1; low-rate
delta-vs-random digital-arm only; the 1.74× ceiling is a model-shape property — at H ≫ V the
recurrence would dominate). Both section headers' verified-against lists gained
c8b596f/a9ff715, 73c99e6, 60d627c, db7c13b. memo v3 gains one addendum paragraph (before §10)
with the same content and inline caveats.

Numbers verified against ledger entries c8b596f/a9ff715 (decomposition/ceiling/sweep), 73c99e6
(copy event-readout refutation), 60d627c (outrand), db7c13b (3-seed confirmation). Local rebuild
(server has no latex): pdflatex→bibtex→pdflatex×2 all exit 0, ZERO overfull, zero undefined
cites/refs, **14 pp** (was 13 — the new subsection adds a page), 539,859 B, md5
4748b536b2caa4181eb5535c1524af36; new-section text confirmed in the PDF via local pdftotext
("stranded", "1.74×", "410,040", "0.00–0.33%"). PDF re-committed in the same push so the
committed `paper2/paper2.pdf` matches the committed sources.

CLAIMS DISCIPLINE check: no "analog beats digital", no pJ figure presented as silicon, copy
margins only vs the noise-regularized reference, the staleness-attribution inference labelled as
such everywhere.

**THREAD STATE after this tick: the writeup fold-in is DONE. The SSM×neuromorphic thread has NO
open tick-work again — remaining items are USER decisions: (1) send `talk/imam_ssm_memo_v3.md`
to Imam (outward-facing, never auto-send), (2) arXiv-submit paper 2 (now includes the
energy-datapath section), (3) optional paper 1 v2. Deck (`talk/`) does NOT yet carry the
stranded-readout result — a candidate future zero-GPU tick if the user wants the deck current.**

---

## 2026-08-03 ~14:15Z tick — TALK DECK carries the stranded-readout result (zero-GPU)

The last stale Imam-facing artifact is now current. `talk/patch_deck2.py` (new, anchored the same
way as `patch_deck.py`: every target matched by shape id AND asserted text + geometry, so a drifted
deck aborts instead of half-applying) makes two edits:

1. **Results slide (07 · SSM RESULTS) gains a fifth block, "The energy is not where the mechanisms
   are":** the readout carries 75% of analog's pJ/token and the recurrence 8%; even a free
   recurrence costs 1.74×; gating the readout recovers the energy but costs ~+1.0 bpc on char-LM and
   all of copy's margin at 3 seeds; converters, finally priced, are 0.00–0.33%. Numbers copied from
   ledger entries c8b596f/a9ff715, 73c99e6, 60d627c, db7c13b and paper2 sec 8.4 — none re-derived.
2. **Directions slide, direction 01 de-staled:** it claimed "the analog storage element and
   converter are unpriced", which the energy decomposition falsified. Now reads "Finished in
   simulation, including the energy datapath: the readout's 75% share resisted every gating
   mechanism tested. What is unfinished is physical — every pJ figure is a 45nm proxy."

**Layout was verified by rasterizing the built PDF, not by trusting the save — and the check earned
its keep.** The first attempt added a fifth full-height block and shifted the two existing rows up to
make room; the render showed the new block's rectangle overlapping rows 03/04 and its 303-char body
running to the box edge. Restored from the pre-patch backup (`/tmp/deck_pre_patch2.pptx`) and
re-ran with the block as a **slim full-width band** (title and body side by side, 1.00in tall
instead of 1.85in) in the empty strip between row 2 and the footer, with the four existing blocks
left untouched. Second render is clean on both slides: no overlap, no overflow, band clears the
footer. Body trimmed 303 → 226 chars with **no claim, number or caveat dropped**.

Claims discipline held: the block states a NEGATIVE (no pJ win), keeps the 3-seed qualifier, and
prices converters under the proxy rather than calling them measured. pptx→pdf conversion is still
LOCAL-only (server has no LibreOffice); PDF is 10 pages, 169 kB.

**THREAD STATE: unchanged and still NO open tick-work.** Every Imam-facing artifact (memo v3,
paper2 + PDF, deck + PDF, speaker notes) is now post-retraction and carries the energy-datapath
verdict. Remaining items are USER decisions: (1) send `talk/imam_ssm_memo_v3.md` to Imam
(outward-facing, never auto-send), (2) arXiv-submit paper 2, (3) optional paper 1 v2.

---

## 2026-08-04 ~04:40Z — ENERGY-DATAPATH, REOPENED ONE NOTCH: the stranded-readout verdict is a MODEL-SHAPE statement, not a property of the analog route. Zero-GPU (`energy_shape.py`).

All 8 A800s idle (1 MiB, 0%, zero compute apps), zero `ssm3way.py` procs, repo in sync @d05d244.
Nothing launched this tick. New file `energy_shape.py`; it imports `terms()` from
`energy_datapath.py` verbatim, so **no accounting is re-derived or re-priced** — the only
new thing is sweeping the model shape (V, H) and the input event rate over the published
energy model, at the *measured* char-LM operating point.

### 0. An analysis bug found and fixed — display-only, NO published claim affected

`energy_datapath.py`'s `label()` groups cells by variant/θ/dig_reg but **ignores
`out_theta` and `out_prand`**. With the readout-gated cells now on disk (38 carry a
nonzero `out_theta`, 12 an `out_prand`) its PART 1 silently **pools gated with no-gate
cells**: it printed `analog th=0.15` at **n=20, bpc 4.018** and `digital+n0.02` at
**n=24, bpc 4.028** — those bpc values are the *gated collapse*, not the operating point.
Same class of bug as the `agg_copy.py` budget-pooling one (34e1fd0).

`energy_shape.py` filters to the un-gated datapath and **reproduces the published numbers
exactly**, which is how we know only the display was wrong and nothing citable moved:

| quantity | published | recomputed (no-gate filter) |
|---|---|---|
| analog θ=0.15 bpc, 3 seeds | 3.1936 | **3.1936** |
| digital+n0.02 bpc, 3 seeds | 3.0338 | **3.0338** |
| analog quality cost vs regularized ref | +0.160 | **+0.1598** |
| analog θ=0.15 emitted rate | 0.61 | **0.6122 ± 0.0064** |
| analog / ref proxy-energy ratio | 1.60× | **1.597×** |

Anyone re-running `energy_datapath.py` must read its PART 1 `n` column and discount the
pooled rows, or use `energy_shape.py` PART 0 instead.

### 1. THE SHAPE LAW — what the SAME measured event activity is worth at other shapes

Held fixed: E=64, H=256, r_z=0.6122 (measured, 3 seeds), published `terms()` accounting.
Only the readout width V varies. `ceil` = analog with a *free* recurrence at that shape.

| V | V/H | W_out% (ana) | W_mix% (ana) | ref pJ/tok | analog pJ/tok | ratio | ceil |
|---|---|---|---|---|---|---|---|
| 10 | 0.04 | 9.5 | 29.2 | 389,786 | 123,494 | **3.16×** | 4.46× |
| 20 | 0.08 | 17.4 | 26.7 | 401,562 | 135,270 | 2.97× | 4.05× |
| 35 | 0.14 | 27.0 | 23.6 | 419,226 | 152,934 | 2.74× | 3.59× |
| 65 | 0.25 | 40.7 | 19.2 | 454,554 | 188,262 | 2.41× | 2.99× |
| **284** | **1.11** | **75.0** | **8.1** | **712,448** | **446,157** | **1.60×** | **1.74×** |
| 1024 | 4.00 | 91.5 | 2.7 | 1,583,872 | 1,317,581 | 1.20× | 1.24× |
| 32000 | 125.0 | 99.7 | 0.1 | 38,061,210 | 37,794,918 | 1.01× | 1.01× |

**The 75%-readout / 8%-recurrence split and the 1.74× free-recurrence ceiling are
properties of `V/H = 1.11`, not of the analog datapath.** At a streaming-classification
shape (V/H = 0.04) the *same measured activity* is worth **3.16×**, the readout falls to
9.5% of the total, and the recurrence — the term every neuromorphic mechanism in this
literature acts on — becomes the largest single term at 29.2%. At a word-piece vocabulary
it goes the other way and the route is worth **nothing** (1.01×).

Two consequences that must be written into any restatement:
- The phase verdict "analog does not win on pJ" is **correct as a char-LM result and wrong
  as a general one.** Paper2 sec 8.4 / sec 9 already flag the ceiling as "a model-shape
  property — at H≫V the recurrence would dominate"; that flag is now a quantified law and
  should be upgraded from a caveat to a table.
- The published 1.60× sits at **92% of its shape's ceiling** (1.60/1.74) — there was almost
  nothing left to win at char-LM. At V=10 the measured activity reaches only **71%** of the
  4.46× ceiling, so further recurrence sparsification would still buy something there.

### 2. The law in the other variable — the advantage GROWS with state width at small V

V=10, E=64, same measured r_z: H=64 → 1.70×; H=128 → 2.27×; H=256 → 3.16×; H=512 → 4.33×;
H=1024 → **5.58×**. The term the analog datapath event-prices is the one that scales as
H², so at fixed small V a wider state helps it — the exact opposite of the char-LM regime,
and a testable prediction rather than an interpretation.

### 3. Goal item (1) re-scored — the previous deferral was right about char-LM and WRONG in general

Priced **symmetrically**: a genuinely event-like input stream is available to the digital
SSM too, so both arms get W_in at r_in. (This is the lesson the readout row taught — the
digital arm got the same gate for free, which is why none of the readout saving was
attributable to the analog state.)

| V | r_in | ref pJ/tok | analog pJ/tok | ratio | Δ vs dense input |
|---|---|---|---|---|---|
| 10 | 1.00 | 389,786 | 123,494 | 3.16× | — |
| 10 | 0.50 | 321,792 | 55,501 | 5.80× | **+2.64** |
| 10 | 0.20 | 317,368 | 51,077 | **6.21×** | **+3.06** |
| 10 | 0.05 | 315,156 | 48,865 | 6.45× | +3.29 |
| 284 | 1.00 | 712,448 | 446,157 | 1.60× | — |
| 284 | 0.20 | 640,031 | 373,740 | 1.71× | +0.12 |

**Correction to a9ff715's deferral of goal (1).** That entry deferred the sparse-input
streaming task as "low-payoff — W_in is only 11–18% of the total". That share is a
char-LM number, and at char-LM's shape the deferral is right (input pricing is worth
+0.12×). But **W_in's share of the analog total is 61% at V=10**, because removing the
readout blocker promotes the input to being the blocker. So the two goal items are
**multiplicative, not independent**: small V alone is worth 1.60→3.16×, and event input
*on top of* small V is worth a further +3.06×, for **6.21×** at r_in=0.2. Goal item (1) is
only low-payoff at the shape where the route is worthless anyway.

### 4. Converters/storage — checked at the corner where they had the best chance, still small

`energy_datapath.py` PART 4 found the previously-unpriced analog storage + ADC/DAC terms at
0.00–0.33% of the total at V=284, with the standing note that they "only start to matter
once the matrices are cheap". V=10 is that regime, so it was checked rather than assumed:
at V=10 under the current datapath they are **0.00–0.23%** over the full four-decade
converter × 25× storage sweep, and at the genuinely cheapest corner (V=10 **and** event
input at r_in=0.05, E_conv=1.0 pJ/event, E_store=0.5 pJ/unit/step) they reach
**0.58%** of a 49,135 pJ/token total. So the converters do not decide the verdict at any
shape tested — but this is now a *bounded* statement, not an extrapolation, and it still
must not be phrased as "converters are free".

### 5. PRE-REGISTRATION — the streaming run this analysis justifies (NOT yet implemented)

The analysis says exactly one experiment is worth GPU time, and it is the one goal item (1)
asked for, now for the right reason: **the only shape where the analog route has a large
projected pJ win is small-V streaming, and no quality number has ever been measured there.**

Design (fixed here, before any cell, so it cannot be fitted afterwards):
- **Task:** FashionMNIST row-streaming — 28 timesteps × 28 pixels, 10-class readout
  (V=10, V/H=0.039). Data is already local (`neuro_poc.py`'s `DATA_ROOT`, download=False);
  the incomplete 3.1 MB NMNIST partial is NOT usable and is not used. This is a
  *statistical* sequence workload, not precise-recall, which is the class the project
  already recommends the analog route for.
- **Input:** delta-encoded between consecutive rows so W_in is legitimately event-priced,
  and `r_in` is **MEASURED from the data**, never assumed. Priced symmetrically across arms.
- **Variants:** the same param-matched four, plus the `dig_noise 0.02` regularized digital
  reference (the comparator whose absence produced this project's one retraction).
- **Metric:** accuracy (chance 0.10) + margin kept vs the regularized reference; r_z, r_in,
  r_out all measured; pJ/token via `energy_datapath.terms()` at the streaming shape.
- **θ must be re-calibrated on this task.** θ already failed to transfer char-LM→copy
  (a9f36d2), and the ADC LSB floors it at 0.125 (83e110a), so calibration is part of the run.

Criteria, pre-registered:
1. **CONFIRMATION** — some θ keeps **≥0.95** of the regularized reference's accuracy at
   **r_z ≤ 0.65**. Then the projected ≥3× cut is real at measured activity, and the phase
   verdict becomes *"no pJ win at LM shape; a 3–6× win at streaming shape"* — the first
   large measured pJ advantage in the project.
2. **REFUTATION** — at every θ with r_z ≤ 0.8 the accuracy deficit exceeds **5 points**.
   Then the analog state is not cheap on this workload either and the "no pJ win" verdict
   generalizes *beyond* model shape, which is the stronger negative and must be reported
   as such.
3. **PARTIAL** — a graded tradeoff: report the accuracy-per-pJ curve, no headline flip.
4. **MOST LIKELY: (1)**, because a state degradation is the cheap kind on a statistical
   workload (+0.160 bpc on char-LM vs +0.9 on copy) — **but the datapath-degradation
   principle has already made one confidently wrong prediction** (it predicted the event
   readout would be cheap on precise-recall; it eliminated the task entirely, 5df8644 /
   73c99e6), so this is a hypothesis, not a forecast. A distinct failure mode to watch:
   r_z may simply not drop on this task at any usable θ, as happened on copy — in which
   case the calibration, not the criteria, is the result.

Caveats owed regardless of outcome: accuracy not bpc, so "margin kept" is against
chance 0.10; every pJ figure remains a **45 nm Horowitz proxy from simulation**, not
silicon; and the shape law in §1–3 is arithmetic over that proxy at char-LM's *measured*
activity — it establishes where a win is *possible*, and nothing about whether quality
survives there.

**THREAD STATE: one open data question again** (it had none since 13:20Z 2026-08-03), and it
is fully specified above. Paper2 sec 8.4 needs the shape table folded in once the streaming
result lands — until then its "model-shape property" caveat is accurate and no correction is
owed. User decisions unchanged: send memo v3 to Imam; arXiv-submit paper 2.

## 2026-08-04 ~05:15Z — STREAMING SHAPE, ARM 1: `--task stream` implemented, smoke-tested, and the theta-calibration + 4-variant row LAUNCHED (results pending)

This opens the one data question left open by the shape law (`energy_shape.py`, d47511b): the small-output
streaming shape (**V=10, V/H=0.039**) is where the analog/CIM route is *projected* to be worth 3-6x, and
**no quality number has ever been measured there**. Everything below is implementation + pre-registration;
no result is claimed.

### What was implemented (`ssm3way.py`, `--task stream`)

* **FashionMNIST as a 28-step ROW STREAM** — `x[n,28,28]` float rows in [0,1], one class label, 28 steps.
  `data_stream()` reads the LOCAL copy (`/work/zeyuwang/hpc_rebuttal/data`, `download=False`); the local
  NMNIST copy remains an unusable 3.1 MB partial and is not used. Train 20,000 / val 5,000 (`--stream_n`,
  `--stream_nval`), `L` forced to 28.
* **Loss and metrics at the LAST step only** (mask), so `acc` is classification accuracy (**chance 0.10**)
  and `bpc` is the last-step CE in bits (chance log2(10) = 3.32).
* Token embedding is replaced by a real-valued input projection `inp: Linear(28, E)` for this task only.
  All four variants own identical tensors, exactly as on the other two tasks, so the row stays
  parameter-matched (86,858 params at E=64, H=256, V=10).
* **`--in_theta`: optional delta-encoded (event) INPUT stream** — per step, an input dim is *sent* only if
  it moved more than `in_theta` since its last send; the network sees the held (last-sent) row. `r_in` is
  **MEASURED and reported**, never assumed. When `in_theta > 0` the input path (`W_in` + the small `inp`
  projection) is priced event-driven at `r_in`. **Applied to ALL variants**, i.e. priced symmetrically —
  the lesson the readout row taught (an event input is equally available to a digital SSM, so any saving
  it produces does not belong to the analog state).
* **`in_theta` is 0 in THIS row** (dense input). The state-theta calibration is isolated first; the
  event-input arm is a separate follow-up. Per the shape law the two are multiplicative at V=10
  (3.16x -> 6.21x), so the arms must not be conflated.

### Two accounting facts that must travel with any number from this task

1. **The readout is priced PER STEP** (`readout_per_step: true` in every stream JSON): logits are computed
   at all 28 steps even though the loss reads only the last. Real hardware doing last-step-only
   classification would pay `W_out` once per sequence instead of 28 times, which would make `W_out`
   negligible and *help* analog. The per-step price is therefore the **conservative** choice and is kept
   for consistency with `energy_datapath.terms()` / `energy_shape.py`, which assume a per-step readout.
2. `W_in_MAC` for this task includes the extra `28*E` input projection on top of `E*H`.

### Verification done before launch (not assumed)

* **Regression: all 212 existing charlm/copy cells re-priced through the patched `energy()` — zero keyset
  differences, worst relative deviation 4.3e-3**, and that residual is an artifact of feeding the JSON's
  4-decimal-rounded rates back in (it appears only on rate-dependent terms; ungated digital cells match
  exactly). The patch leaves every previously published number intact. `in_theta`/`stream` defaults are
  inert on the other tasks by construction.
* **Smoke test** (digital, 1 epoch, 4,000 train seqs): acc **0.640** vs chance 0.10, runs clean.
* **The shape law's central prediction is now visible in a real run JSON rather than a projection:** at
  V=10 the digital cell's energy splits `W_mix` **75.7%** / `W_in` **21.0%** / `W_out` **3.0%** — the
  recurrence is the dominant term and the readout is nearly free, the exact inversion of char-LM
  (`W_out` 47-75%, recurrence 8%). This is arithmetic over the same proxy, so it is not new evidence for
  the shape law; it confirms the task was built at the intended shape.

### The row (9 cells, seed 0, 10 epochs, 20k train / 5k val, dense input)

`run_stream_calib.sh <gpu>` — same lock-dir idempotent work queue as `run_shrinkH.sh` (atomic
`mkdir stream_locks/<cell>`, skips any cell whose out JSON exists, releases the lock on failure).
Outputs `ssm3way_runs/<name>_ep10.json`, per-cell logs in `stream_logs/`, markers in `stream_logs/calib.log`.

| cell | role |
|---|---|
| `digital_stream_s0_reg_n0.02` | **the reference** — noise-regularized digital, the comparator every claim is made against |
| `digital_stream_s0` | un-regularized digital, for the regularization delta at this shape |
| `analog_stream_s0_th{0.15,0.25,0.5,1.0,2.0}` | theta calibration |
| `spikeout_stream_s0` | continuous state + LIF output (SPikE-SSM-style) |
| `spikestate_stream_s0` | floor control, state carried in spikes |

**theta must exceed the ADC LSB** — `2*rail/2^bits = 2*4/64 = 0.125` — so the sweep starts at 0.15; any
smaller threshold is inoperative (the 83e110a finding). theta is re-calibrated from scratch because it
demonstrably does not transfer across tasks (char-LM's 1.0 collapsed analog to chance on copy).

Launched 05:16Z, 8 workers on GPUs 0-7 (all 8 A800s verified fully idle, zero compute apps, and each
launch behind a per-GPU uuid busy-guard). All 8 claimed distinct cells; `spikestate_stream_s0` is queued
for the first worker to free.

### CRITERIA — pre-registered in d47511b before implementation, restated here before any cell landed

* **CONFIRM** = some theta keeps **>= 0.95 of the reference accuracy at r_z <= 0.65**. Then the verdict
  becomes *"no pJ win at a language-model shape; a measured win at the streaming shape"* — the project's
  first large measured pJ advantage.
* **REFUTE** = accuracy deficit **> 5 points at every theta with r_z <= 0.8**. Then the no-pJ-win verdict
  generalizes **beyond** model shape, which is the stronger negative and supersedes the shape-law
  optimism.
* **PARTIAL** = a graded tradeoff: report the accuracy-per-pJ curve, no headline flip.
* **Most likely CONFIRM** — a *state* degradation is the cheap kind on a statistical workload (+0.160 bpc
  char-LM vs ~+0.9 copy). **But this principle has already made one confidently wrong prediction** (it
  predicted the event readout would be cheap on precise recall; it eliminated the task), so this is a
  hypothesis, not a forecast.
* **Distinct failure mode to watch:** `r_z` may simply fail to drop at any usable theta, as on copy. Then
  **the calibration is the result** — the analog datapath would buy no event sparsity at this shape, and
  the projected 3-6x would be unreachable for a reason unrelated to quality.

**Caveats owed regardless of outcome:** n=1 seed (3 seeds before any paper use); dense input, so the
projected 6.21x event-input corner is untested; energy remains the 45nm Horowitz proxy, not silicon; the
per-step readout price is conservative and must be stated whenever a stream pJ ratio is quoted.

### 2026-08-04 ~05:35Z — the streaming row COMPLETED IN THE SAME TICK (9/9 cells, ~16 s/epoch) and the PRE-REGISTERED **REFUTE** BRANCH IS MET: the projected pJ win is real and measured, but quality does not survive it

Table via `agg_stream.py 10` (new, groups by variant/theta/dig_noise/in_theta/out_theta/epochs so budgets
and gate settings are never pooled). FashionMNIST row-stream, V=10, seed 0, 10 ep, 20k train / 5k val,
dense input, chance acc **0.10**. Reference = noise-regularized digital, **acc 0.7878, 398,029 pJ/token**.

| cell | acc | margin kept | bpc | r_z | pJ/tok | pJ cons. | x vs ref |
|---|---|---|---|---|---|---|---|
| digital (plain) | 0.7990 | 1.016 | 0.801 | 1.000 | 398,029 | — | 1.00 |
| **digital +n0.02 (reference)** | **0.7878** | 1.000 | 0.852 | 1.000 | 398,029 | — | 1.00 |
| analog th=0.15 | **0.6774** | 0.839 | 1.440 | 0.428 | 120,883 | 224,757 | **3.29** (cons. 1.77) |
| analog th=0.25 | 0.5976 | 0.723 | 1.778 | 0.331 | 115,143 | 195,422 | 3.46 (cons. 2.04) |
| analog th=0.5 | 0.5312 | 0.627 | 2.067 | 0.214 | 108,254 | 160,212 | 3.68 (cons. 2.48) |
| analog th=1.0 | 0.3652 | 0.386 | 2.576 | 0.114 | 102,312 | 129,840 | 3.89 (cons. 3.07) |
| analog th=2.0 | 0.2044 | 0.152 | 3.109 | 0.047 | 98,356 | 109,620 | 4.05 (cons. 3.63) |
| spikeout | 0.6486 | 0.798 | 1.361 | 0.386 | 119,345 | — | 3.34 |
| spikestate | 0.5298 | 0.625 | 1.699 | 0.405 | 109,610 | — | 3.63 |

**CRITERIA, applied mechanically by the script (not by eye):** CONFIRM (some theta keeping >= 0.95 of the
reference accuracy at r_z <= 0.65) **NOT MET** — the best cell, theta=0.15, keeps only **0.860** of the
reference accuracy. REFUTE (> 5-point deficit at *every* theta with r_z <= 0.8) **MET** — deficits are
**11.0 / 19.0 / 25.7 / 42.3 / 58.3 points** across the five thetas, all five usable cells. Per the
pre-registered sentence: **the no-pJ-win verdict generalizes beyond model shape.** There is no
quality-matched pJ win at ANY shape we have tested.

**What DID hold: the energy arithmetic.** `energy_shape.py` projected **3.16x** at V=10 for the char-LM
operating point (r_z=0.6122); the measured cell at r_z=0.428 comes in at **3.29x** (optimistic pricing).
So the shape law's pJ side is confirmed out of sample — the 75%/8% split really does inverse at V/H=0.039
(measured digital split at this shape: `W_mix` **75.7%**, `W_in` 21.0%, `W_out` **3.0%**). **What failed is
the quality side, which had never been measured.**

**The failure mode we watched for did NOT occur, which makes the negative cleaner.** `r_z` falls smoothly
0.428 -> 0.047 with theta, unlike copy where the state gate bought no sparsity at any usable threshold. The
analog datapath *does* deliver event sparsity at this shape; it charges 11 accuracy points for the first
2.3x of it. This is a graded tradeoff, not a collapse — so the accuracy-per-pJ curve above is the result.

**Three further readings.**
1. **The ~3.3-4.0x is shape-determined, not mechanism-determined.** All three neuromorphic variants land in
   a narrow band (analog 3.29, spikeout 3.34, spikestate 3.63) because `W_mix` dominates at V=10 and every
   mechanism gets event pricing on it. Same lesson as the readout row in mirror image: the saving belongs
   to the *shape*, not to the analog state.
2. **Analog IS the best non-digital variant here** (0.6774 at 3.29x vs spikeout 0.6486 at 3.34x, spikestate
   0.5298 at 3.63x) — the char-LM ordering reproduces at the streaming shape and the copy inversion does
   not, consistent with the workload dichotomy (a statistical/perception workload tolerates a lossy state).
   But the margin over spikeout is small (2.9 accuracy points), and **spikeout's 3.34x carries no
   graded-event ambiguity** (true 1-bit spikes), whereas analog's 3.29x becomes **1.77x** under conservative
   pricing. Priced conservatively, spikeout is the better engineering point at this shape.
3. **The noise-regularization gift does not appear at this shape:** plain digital 0.7990 > noise-regularized
   0.7878, i.e. state noise is worth **-1.1 points** here versus **-0.309 bpc (a benefit)** on char-LM.
   Using plain digital as the reference makes analog's deficit slightly worse (12.2 points). The char-LM
   retraction's mechanism therefore does not generalize either.

**The datapath-degradation principle has now mispredicted TWICE, both times confidently.** It predicted
the event readout would be cheap on precise recall (it eliminated the task) and it predicted a state
degradation would be cheap on this statistical workload (11-point deficit). Its record is: correct as a
*post-hoc* ordering of already-measured degradations, wrong both times it was used as a forecast. That
must be stated wherever the principle appears — it is a summary, not a predictive law.

**Caveats owed, and one live alternative explanation.** n=1 seed (effect sizes 11-58 points dwarf the seed
noise seen elsewhere, ~0.3-1.5 points, but 3 seeds are owed before paper use). **Budget:** 10 epochs /
20k of 60k train images, reference at 0.788 while a small CNN reaches ~0.91 — so this reference is not at
its ceiling, and the copy row taught us that a budget-limited baseline can mislead. It is possible the
deficit narrows at a longer budget; that is the one cheap check that could move the verdict, and it is
NOT yet run. Dense input (in_theta=0), so the projected multiplicative 6.21x event-input corner is still
untested — but note it can only improve the *energy* axis, and the axis that failed is quality.
Energy remains the 45nm Horowitz proxy; the readout is priced per step (conservative at this shape,
where the readout is only 3% anyway).

**Effect on existing artifacts:** nothing published becomes wrong, but the 2026-08-04 shape-law entry's
optimism must be **narrowed in place** — "3-6x at a streaming shape" is now measured as *3.3x for an
11-point accuracy loss*, not as a free win. Nothing Imam-facing may say the streaming shape rescues the
route. Paper2 sec 8.4 / sec 9 gain this row as the *test* of the shape-law caveat they already carry.

## energy-datapath: stream BUDGET CHECK — pre-registration (written BEFORE any cell ran)

**Why:** the 2026-08-04 stream row's REFUTE verdict (no quality-matched pJ win at the
streaming shape; best analog keeps 0.860 of reference accuracy) has exactly one live
alternative explanation: BUDGET. The row ran at 10 ep / 20k of 60k train images and the
reference sits at 0.788 where a small CNN reaches ~0.91 — and the copy row already taught
us once (31580ac) that a budget-limited baseline misleads. This check is the only test
that could move the verdict.

**Row (driver `run_stream_budget.sh`, lock-dir queue, seed 0, ep30 / stream_n 60000 /
stream_nval 5000, dense input, out suffix `_n60k_ep30`):** 5 cells —
digital+dig_noise0.02 (the ep10 reference arm), plain digital (it BEAT the noise-reg arm
by 1.1 pts at ep10 at this shape, so the proper reference at the long budget must be
measured, not assumed), analog θ=0.15, analog θ=0.25, spikeout (the conservative-pricing
engineering point at ep10).

**Reference definition (fixed now):** the BETTER of {plain digital, noise-reg digital}
at ep30/n60k — the harsher and honest comparator.

**Criteria (must not be refitted):**
- **CLOSE (the arm is dead for good):** both analog θ ∈ {0.15, 0.25} remain >5-point
  accuracy deficits vs the reference at the long budget → the REFUTE verdict is
  budget-robust; the "at our budget" qualifier drops from every artifact and the
  no-pJ-win-at-any-shape sentence becomes unconditional (within tasks/shapes tested).
- **REOPEN:** either analog θ comes within ≤5 points of the reference while keeping
  r_z ≤ 0.65 → the ep10 REFUTE was budget-limited; then a full θ row + 3 seeds are owed
  before any claim, and no Imam-facing artifact changes until those land.
- **Directional note owed either way:** report whether the deficit NARROWS as budget
  grows (ep10→ep30, 20k→60k), even if it stays >5 pts — a shrinking deficit with a
  still-CLOSE verdict must be reported as "closed at this budget, deficit trending down",
  not silently pooled into "closed".
- **Most likely:** CLOSE — the analog deficit is a structural state degradation
  (noise + 6-bit quantization + send-on-delta), not an optimization artifact; but the
  datapath-degradation principle is 0-for-2 as a forecaster, which is why this runs.

**Caveats fixed now:** n=1 seed (this is a budget probe, not a paper cell; 3 seeds owed
at whatever budget survives); dense input (energy-axis-only corner untested); 45nm proxy.

## 2026-08-04 ~06:10Z — STREAM BUDGET CHECK CLOSED: pre-registered CLOSE branch MET — the streaming-shape REFUTE is BUDGET-ROBUST and the arm is dead for good

Row 39a6873 (5 cells, seed 0, ep30 / full 60k train / nval 5000, dense input, suffix `_n60k_ep30`)
completed 05:52–06:00Z; all cells aggregated below. Criteria were pre-registered in 39a6873
BEFORE any cell ran and are applied verbatim, no refitting.

**Reference selection (per pre-registration: the BETTER of the two digital arms at ep30/n60k):**

| cell | acc | bpc | pJ/tok |
|---|---|---|---|
| digital reg n0.02 (REFERENCE) | **0.8442** | 0.6564 | 398,029 |
| digital plain | 0.8382 | 0.6983 | 398,029 |
| analog θ=0.15 | 0.7404 | 1.0912 | 121,158 (cons. 226,162) |
| analog θ=0.25 | 0.6556 | 1.4376 | 115,802 (cons. 198,790) |
| spikeout | 0.6652 | 1.2892 | 123,615 |

Chance acc 0.10. analog r_z (rate_emitted) = 0.433 / 0.342 at θ=0.15/0.25 — both under the 0.65
gate, so the cells are valid tests of the CLOSE/REOPEN question, not calibration failures.

**VERDICT — CLOSE.** Both analog cells still exceed a 5-point deficit vs the reference:
θ=0.15 → **−10.4 pts** (0.7404 vs 0.8442), θ=0.25 → **−18.9 pts**. The pre-registered sentence
applies: the ep10 REFUTE was NOT budget-limited; the "at our budget" qualifier DROPS everywhere.
Combined with c18c724, the project-wide verdict is now unqualified at every shape tested:
**there is no quality-matched pJ win at any model shape tested, at any training budget tested.**

**The trend the pre-registration demanded (deficit ep10→ep30, must not be pooled):** the deficit
is essentially FLAT — θ=0.15: 11.0 → 10.4 pts; θ=0.25: 19.0 → 18.9 pts — while BOTH arms improved
absolutely (ref 0.7878→0.8442, analog θ=0.15 0.6774→0.7404). Tripling epochs and 3× data moves
the gap by ≤0.6 points: the deficit is a property of the lossy analog state at this shape, not of
an under-trained baseline. (Contrast the copy row, where the same check moved digital acc 0.41→0.64
and legitimately reopened the row — this one does not.)

**Secondary observations (n=1, note-grade):**
- The char-LM noise-regularization gift REAPPEARS at the long budget: reg n0.02 beats plain digital
  0.8442 vs 0.8382 at ep30, reversing the ep10 ordering (0.788 vs 0.799). Budget-dependent, small,
  and it is why the pre-registration required measuring BOTH digital arms rather than assuming one.
- Energy side stays exactly on the shape law: analog θ=0.15 = 3.29× optimistic / 1.76× conservative,
  spikeout 3.34× (no graded-event ambiguity) — unchanged from ep10, as expected (pricing depends on
  rates, and r_z is stable 0.428→0.433).
- spikeout again within ~1 pt of analog θ=0.25 at matched-ish emission; the "conservatively priced,
  spikeout is the better engineering point" reading from c18c724 survives the budget change.

**Caveats:** n=1 seed at ep30 (the closing verdict rests on the sign and size of a 10–19-pt effect,
far above the ≤1-pt seed noise seen on this pipeline's other rows, but stream seed sd is unmeasured);
dense input (the event-input corner is untested — it improves only the energy axis, which was never
the failing axis); 45nm proxy, readout priced per step.

**THREAD STATE: the streaming arm is CLOSED (negative, budget-robust). NO open data question.**
Remaining tick-work = the WRITEUP NARROWING (per the 2026-08-04 ~05:40Z entry): paper2 sec 8.4/sec 9
+ memo v3 + deck must state the streaming shape as "energy recoverable (3.3×) but quality-refuted
(10–19-pt deficit, budget-robust)" — nothing Imam-facing may claim the streaming shape rescues the
route, and the "at our budget" hedge is no longer needed or accurate. User decisions unchanged.

## 2026-08-04 ~07:00Z — WRITEUP NARROWING (zero-GPU): shape law + streaming negative folded into paper2 and memo v3

All numbers copied from ledger entries d47511b (shape sweep), 3033631/c18c724 (streaming row, pre-registered REFUTE), 39a6873/0924aad (budget check, pre-registered CLOSE); nothing re-derived.

**paper2 changes:**
- NEW sec 8.5 "The ceiling is a shape property — and the shape that recovers the energy fails on quality" (`sec:hardware-shape`): the V/H sweep (V=10/284/1024/32000 → 3.16/1.60/1.20/1.01×; ceilings 4.46×/1.74×; at V=10 the advantage grows with H: 1.70/3.16/5.58× at H=64/256/1024; symmetric event input +0.12× at LM shape vs +3.06× at V=10, multiplicative 6.21×; converters at the most favourable corner 0.58%); the streaming REFUTE (energy confirmed out of sample 3.29× vs 3.16× projected, digital split inverts to W_mix 75.7%/W_in 21.0%/W_out 3.0%; quality deficits 11.0–58.3 pts at every θ, best keeps 0.86 of the 0.7878 reference; graded tradeoff, not collapse); the budget-check CLOSE (ep30/n60k ref 0.8442 — the noise-reg gift reappears at the long budget — deficits flat at −10.4/−18.9); three readings (the 3.3–4.0× band is shape-determined, not mechanism-determined; conservatively priced the defensible point is spikeout 3.34× at 2.9 pts worse, analog falls to 1.77×; the principle's second failed forecast); the unqualified verdict ("no quality-matched pJ win at any model shape tested, at any training budget tested") + caveats (streaming cells n=1 seed, dense input — the 6.21× corner untested but energy-axis-only, 45nm proxy).
- sec 8.4 closing para: "8% share" now qualified **at a language-model shape** + forward ref to 8.5. sec 8.1: the statistical-workload tolerance is scoped to char-LM (the streaming perception task refutes generality, −10..−19 pts budget-robust).
- **The datapath-degradation principle is re-scoped everywhere as a post-hoc ordering, NOT a predictive law (0-for-2 as a forecast):** sec 6.4 gains a third "what weakens it" paragraph (both failed forecasts named, hindsight causes stated); sec 1 principle paragraph + contribution 3 amended; sec 9 paragraph renamed "Three tasks, and a principle that orders but does not forecast" (the called-for third-workload test was run and weakened the predictive reading); sec 10 "applied before any simulation" rewritten to "organizing rule, not a forecasting one … measure, don't predict" + shape-verdict sentence added to para 1.
- sec 9 "Scope of the stranded-readout negative": the shape caveat upgraded from hypothetical to measured (3.29× real, quality not; streaming limits stated).
- **Abstract updated + one STALE CLAIM FIXED:** "the analog route's storage element and converter are unpriced" had contradicted sec 8.4 since the 2f3d14b fold-in → now "priced only under a supplementary sweep of the same proxy (0.00–0.33%), never on a device". Abstract also gains the stranded-readout/shape/streaming arc and the principle-as-forecast caveat; scope para two→three tasks. main.tex claims-discipline list extended: forbidden now also "converters are unpriced", any "3–6× at a streaming shape" presented as a result (quality side REFUTED), the principle presented as predictive.
- **PDF rebuilt LOCALLY** (server has no latex): exit 0, zero overfull, zero undefined, **15 pp** (was 14); committed in the same push so the committed PDF matches committed sources.

**memo v3:** Addendum 2026-08-04 inserted directly after the 2026-08-03 one — shape law, streaming + budget negative, the conservative-pricing point in spikeout's favour, the principle's 0-for-2 forecast record, and the unqualified bottom line, with caveats inline. Memo remains send-ready; NOTHING Imam-facing claims the streaming shape rescues the route.

**REMAINING:** the talk deck does NOT yet carry the shape-law/streaming narrowing (its "energy is not where the mechanisms are" band is still unqualified language-shape-only) — next tick's candidate via `talk/patch_deck*.py` + LOCAL pptx→pdf. User decisions unchanged: send memo v3 (never auto-send); arXiv-submit paper 2.
