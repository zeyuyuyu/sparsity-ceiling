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
