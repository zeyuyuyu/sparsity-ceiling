# Paper 2 — SKELETON (decided this tick; supersedes "restate char-LM headline in paper.tex", which was mis-scoped)

**Working title:** *Degrade What the Task Ignores: Workload-Conditional Datapaths for State-Space Models on Neuromorphic Hardware*
(alt: *No Free Datapath: Where Analog and Spiking SSMs Win, Lose, and Why*)

**Decision recorded:** this is a STANDALONE second paper, not a v2 of the arXiv Sparsity-Ceiling paper
(verified: `sparsity_ceiling.tex` contains zero SSM/analog content). Target: arXiv first
(cs.NE, cross-list cs.LG), same single-author line as paper 1. Length target 6–8 pp.
It cites paper 1 for the firing-ceiling observation and the two-sided target-rate protocol,
and answers the question paper 1 raised: recurrence hits a firing floor — so what datapath
*should* carry recurrent state on neuromorphic hardware?

**Central claim (one sentence):** a neuromorphic SSM datapath is cheap exactly when it degrades
the part of the computation the workload does not depend on — statistical sequence tasks tolerate
a lossy analog state but need an exact graded output; precise-recall tasks tolerate a spiked
output but need an exact state — so there is no universally best route, and the silicon
recommendation splits by workload.

**Evidence inventory (all published in ssm3way_ledger.md; commit hashes inline; NOTHING may be re-derived):**
- E1. char-LM 3-seed θ-sweep + matched-emitted-rate comparisons (a134f05).
- E2. char-LM regularization control → retraction of "analog beats digital"; surviving claim =
  1.60× proxy-energy cut for +0.160 bpc vs a noise-regularized digital baseline (bb79f7f).
- E3. copy rows M=16/32/64 at 3 seeds, ranking INVERSION, margins corrected vs the
  noise-regularized reference: spikeout 0.87/0.42, analog 0.50/0.28, spikestate 0.074/0.00 (48510c8, 8d050cb, 5bb7bcc).
- E4. quantizer contrast (the out-of-sample test of the principle): 6-bit state quantization costs
  −0.002 bpc on char-LM vs +0.369 (M=16) / +0.385 (M=32) bpc on copy. POST-HOC label mandatory (5bb7bcc→48510c8).
- E5. shrink-H bound arm closed as a NEGATIVE at 3 seeds/cell (471f113): 12× floor swing,
  measured ρ anti-correlated (0.496→0.250), margin kept RISES as certified capacity shrinks
  (0.079→0.200) → optimization pathology, not information floor.
- E6. ADC-step-floors-θ co-design constraint (83e110a): θ < quantizer LSB (0.125 at 6 bits/±4 rails)
  is inoperative; event rate is floored by the ADC step, not the threshold. Plus θ non-transfer across tasks.

---

## Section plan

### 1. Introduction
- Hook: SSMs are the obvious sequence engine for neuromorphic hardware (linear recurrence maps to
  physical dynamics), but "neuromorphic SSM" hides a three-way datapath fork: spike the state,
  spike the output, or hold the state in analog.
- Prior work picks one route per paper; no controlled comparison exists at matched parameters.
- Contributions (4): (i) parameter-matched 4-variant protocol (identical tensors, only the
  nonlinearity's position moves); (ii) two-task × matched-communication-rate evidence that the
  variant ranking INVERTS across workloads; (iii) the datapath-degradation principle, with the
  quantizer contrast as its out-of-sample test; (iv) an honest negative on the firing-floor
  bound's empirical reach + the ADC/θ co-design constraint.
- State the retraction discipline up front (footnote): an intermediate "analog beats digital"
  finding was retracted after a regularization control; the paper reports the controlled version.

### 2. Related work
- Lift from imam_ssm_memo_v3 §1–3: SPikE-SSM (continuous state, spiked output);
  LMU/NEF-on-Braindrop (analog-state precedent); CIM/analog SSM (Wei Lu line); S4D.
- Positioning: none of these compare routes under matched capacity; energy claims are
  route-internal. Cite paper 1 for the recurrent firing ceiling that motivates the question.

### 3. Experimental protocol
- The 4 variants (digital / spikeout / analog / spikestate) as one model family: shared emb,
  W_in, diagonal decay, W_mix, W_out; 173,596 params char-LM, 10.5k–87.9k copy.
- Tasks: char-level WikiText (statistical) + synthetic copy with load M∈{16,32,64}, K=16
  (precise recall; chance acc 0.0625 / bpc 4.000).
- Energy proxy: 45nm Horowitz MAC 4.6 pJ / AC 0.9 pJ; analog graded events priced both ways.
  STATE PLAINLY: proxy, not silicon; analog storage element + converter unpriced.
- Controls: noise-regularized digital reference (σ=0.02, training-time) is the baseline for ALL
  margin figures; weight-decay dissociation (worth exactly 0) pins the mechanism to state-level
  stochasticity.
- Metrics: bpc / acc, margin-kept (fraction of regularized baseline's above-chance margin),
  emitted rate vs state activity (distinguish wire traffic from state density), pJ/token proxy.

### 4. Char-LM: the statistical workload (E1 + E2)
- Table 1: 3-seed table, all variants + analog θ-sweep; paired Δbpc vs BOTH references
  (plain digital and noise-regularized digital).
- Matched-rate reading: analog @0.612 emission beats spikestate @0.645 by 0.427 bpc; analog
  @0.380 beats spikeout @0.369 by 1.10 bpc → ordering is not an operating-point artifact.
- §4.1 The control that changed the headline: state noise on the digital baseline is worth
  −0.309 bpc (2.1× the raw analog advantage); quantization contributes ~0; wd exactly 0.
  Decomposition: analog net = −0.309 noise gift + gating cost (+0.160 @ θ=0.15 → +0.326 @ θ=1.0).
- Surviving claim (the honest headline): analog buys a 1.60× proxy-energy cut for a
  +0.160 bpc (5.3%) quality cost. A tradeoff, not a free lunch. NEVER "analog beats digital".

### 5. Copy: the precise-recall workload (E3)
- Table 2: M=16/32(/64) at 3 seeds, margins vs the regularized reference (0.87/0.42, 0.50/0.28,
  0.074/0.00), matched emitted rates ~0.45–0.50.
- The ranking inversion is the paper's pivot: spikeout best→worst flips between tasks;
  spikestate at M=32 is EXACTLY chance while being the cheapest cell (102k pJ/tok) — the
  cleanest demonstration that a pJ number is meaningless without the quality it purchased.
- Gap widens with load for both non-digital routes; analog's emission does not rise to
  compensate (falls 0.501→0.449 at fixed θ) → no single "X% of digital" number exists;
  retention demand sets it.

### 6. The datapath-degradation principle (E4)
- Statement + why state-fidelity-alone fails (spikeout has the most exact state on both tasks,
  best on one, worst on the other).
- Out-of-sample test: the SAME 6-bit state quantizer costs −0.002 bpc (char-LM) vs
  +0.369/+0.385 bpc (copy M=16/32); load-robust; rate_state footprint 1.000→0.976.
- MANDATORY honesty labels: post-hoc-identified (the pre-registered arm turned out to be
  training-time-only and tested baseline tuning, not datapaths); noise benefit trend across
  loads (−0.150→−0.215→−0.309) runs toward char-LM, weakening the strength-based supporting line.

### 7. The firing-floor bound: theory with a tested scope limit (E5)
- Bound restated with the corrected M·log₂K reading. Report the shrink-H row as a designed,
  pre-registered test whose criterion (ii) was REFUTED: across a 12× floor swing (0.042→0.500,
  H=256→64) measured ρ FALLS 0.496→0.250 and margin kept RISES 0.079→0.200 in anti-correlation
  with the capacity certificate → spiking-state failure on precise recall is an optimization
  pathology the bound does not explain. Include the retracted-contrapositive history in one
  sentence (H=64 match was coincidental; H=96 kills it).
- FORBIDDEN SENTENCES: "the bound is confirmed"; "the contrapositive earns its keep"; any "190×" figure (corrected to 12×).
- Why raising M cannot test it either (learnability ceiling ~M=16 sits below the binding load M=64).

### 8. Hardware implications
- Split recommendation: CIM/analog state for statistical/low-retention sequence workloads;
  exact digital state + spiked OUTPUT for precise-recall; spiking the state is a non-option for
  any memory-bearing workload (it eliminates recall, not degrades it).
- Co-design constraint (E6): event sparsity and state precision are NOT independent knobs —
  the ADC step floors the event rate (θ<LSB inoperative); buying a lower event rate costs bits
  (converter/area), not threshold tuning. θ does not transfer across tasks → per-workload calibration required.
- Energy honesty: the only pJ result favoring analog is the char-LM proxy at 1.60×; on copy
  analog loses BOTH axes to spikeout; no silicon measurement anywhere in the paper.

### 9. Limitations
- Simulation, not silicon; 45nm proxy; analog storage + converter unpriced; matching emitted
  rate matches wire traffic, not state density (analog state is dense, 0.994).
- Small scale (≤174k params), short schedules; regularization-mediated effects may shrink at scale.
- Two tasks; the principle is tested out-of-sample once (quantizer) and post-hoc.
- Copy margins are vs a noise-regularized but not exhaustively tuned baseline.

### 10. Conclusion
- One paragraph: the fork answer for neuromorphic SSMs is workload-conditional; the principle
  gives designers a rule ("degrade what the task ignores"); the bound survives as design
  intuition, not validated theory.

---

## Claims ledger (what this paper may and may not say)
MAY: quality-matched-rate ordering inverts across tasks (3 seeds, both tasks); analog = 1.60×
proxy-energy for +0.160 bpc on char-LM; 1-bit state eliminates recall at M=32; quantizer
contrast ±0 vs +0.37/+0.39; ADC floors θ; bound refuted-as-floor-prediction at this scale.
MAY NOT: "analog beats digital" (retracted, bb79f7f); "bound confirmed" / contrapositive claims
(retracted, cef2101); "190×" (corrected to 12×, 471f113); any copy margin without the
regularized-reference correction; any pJ claim presented as a silicon measurement.

## Figures/tables plan
- T1 char-LM master table (from agg_charlm.py). T2 copy master table (agg_copy.py 30).
- F1 variant schematic (nonlinearity position). F2 char-LM Δbpc vs emitted rate (θ-sweep curve
  with spikestate/spikeout points at their own rates). F3 copy margin-kept vs M by variant.
- F4 bound arm: predicted floor vs measured ρ vs H (the anti-correlation picture).
- F5 quantizer contrast bar (char-LM vs copy M=16/32). plot_ssm3way.py (30800c2) is the starting point.

## Next writeup steps after this skeleton
1. Draft §3+§4 (protocol + char-LM) — tables regenerate from agg scripts, prose new.
2. Draft §5+§6 (copy + principle). 3. Draft §7 (bound) — reuse memo v3 §8 language. 4. Intro/related last.
5. LaTeX build happens LOCALLY on /home/zeyu (server has no latex); keep source in repo `paper2/`.
