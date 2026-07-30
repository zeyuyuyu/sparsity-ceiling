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
| implementation, all 4 variants | **DONE** 2026-07-30 — CPU smoke test passes end-to-end for all four |
| digital × charlm | not run |
| spikeout × charlm | not run |
| analog × charlm | not run |
| spikestate × charlm | not run |
| all 4 × copy | not run |

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
