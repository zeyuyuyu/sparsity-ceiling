# Are neuromorphic chips suitable for SSMs — and is the *non-spiking* route also a fit?

**Response to N. Imam's TODO — Zeyu Wang**
*(v3, 2026-07-31 — v2 was literature-only and ended with a proposed experiment. That experiment has now been **run to completion** on 8×A800: 4 parameter-matched variants × 2 tasks × 3 memory loads × 3 seeds, plus a firing-floor-bound arm and regularization controls. v3 keeps the literature analysis and replaces the hypothesis in §6 with measurements — including the places where my own hypothesis was wrong.)*

---

## TL;DR (v3)

**Yes, and the non-spiking / analog-state route is the right one — but the honest claim is an energy-for-quality *tradeoff*, not a quality win, and it is workload-conditional.**

Three things the data changed relative to v2:

1. **Carrying the recurrent state in spikes is off the table** for any memory-bearing workload. Not "degraded" — *eliminated*: on precise recall at 32 symbols the spiking-state SSM sits at exactly chance (acc 0.0623±0.0012 vs chance 0.0625) while being the *cheapest* variant by the energy proxy. Cheap and useless.
2. **Analog-state (route c) is the best neuromorphic datapath on statistical sequence tasks** — on char-LM it buys a **1.60× proxy-energy cut for a +0.160 bpc (5.3%) quality cost**. An earlier reading of mine said analog *beat* the digital baseline; that is **retracted** (§7) — it was an under-regularized comparator.
3. **The verdict flips by workload**, and one principle predicts both directions: *a neuromorphic variant is cheap exactly when it degrades the part of the datapath the task does not depend on.* Statistical tasks need an exact graded **output** and tolerate a lossy state → analog wins. Precise-recall tasks need an exact **state** and tolerate a spiked output → SPikE-SSM-style output-spiking wins. Neither route is universal.

Also: my own firing-floor bound from *The Sparsity Ceiling* is **not empirically validated by this architecture at this scale, in either direction** (§8). I could not construct a regime where it binds on a net that still learns the task. It stays theory with a *tested* scope limit.

---

## 1. Reading list, decoded  *(unchanged from v2; confidence noted)*

- **Mitrokhin et al., *Sci. Robotics* 2019 (aaw6736)** — hyperdimensional/VSA (Kanerva) for neuromorphic sensorimotor control. *[abstract/title only — paywalled]*
- **Izhikevich, *Spiking Manifesto* (arXiv:2512.11843)** — spikes as look-up tables / polychronization / ~1000× efficiency. *[abstract-level]*
- **ArrowFlow (arXiv:2604.04087)** — computation in permutation space, integer-only, explicitly neuromorphic-aligned (VSA flavor). *[abstract-level]*
- **Kanerva (SDM/VSA) + Eliasmith (NEF)** — the classical frameworks for putting *memory/representation* and *dynamics* on a neural substrate.

**Read as:** the tools to run an SSM (= a dynamical system) on neuromorphic already exist — **NEF for the dynamics, VSA for memory** — so the question is *how*, not *whether*.

## 2. Why SSM ↔ neuromorphic is a structural match  *(LMU, read in full)*

- SSM: `dx/dt = A x + B u`, `y = C x` — a continuous-time **linear** dynamical system = what analog silicon physically *is*.
- **The LMU (Voelker, Kajić, Eliasmith, NeurIPS 2019) IS a linear SSM** (delay line via Padé → Legendre basis; predates/parallels HiPPO/S4). Implemented via NEF on spiking populations and **deployed on both Loihi (digital spiking) and Braindrop (analog mixed-signal)**. psMNIST **97.15%** (> LSTM 89.86%), 10⁵-step memory, Mackey-Glass NRMSE 0.054. *(You are a co-author on the Loihi paper — this is directly in your lineage.)*
- So **"SSM on neuromorphic," including the analog non-spiking form, is a solved existence proof, not a hope.** The open question is which variant costs what — which is what I measured.

## 3. Three ways to put an SSM on neuromorphic  *(taxonomy, unchanged)*

**(a) Digital SSM on digital fabric** — the reference point: the exact recurrence in a digital datapath, optionally quantized. *(Correction 2026-08-01: QS4D (arXiv:2507.06079), previously listed here as the (a) exemplar, was read in full — its deployment target is memristive **analog** in-memory compute (TaOx crossbars co-integrated with 180 nm CMOS), QAT of diagonal S4D, non-spiking, with QAT shown to confer robustness to analog noise. It belongs with the CIM work under (c).)*

**(b) Continuous linear state + spiking/sparse *output*** — the current "spiking SSM" line. **SPikE-SSM** (read in full) keeps the state `h_t` **continuous** and spikes only the output nonlinearity (refractory-LIF on S4D). Its reported firing is *output* sparsity (LRA ~8%, WikiText 24.5%) — **not** a state-carrying firing floor. Cost: recall gap (WikiText ppl 33.2 vs S4's 21.0).

**(c) Native analog SSM — the non-spiking route you asked about** — device physics *is* the dynamics:
- **CIM-SSM (Zhang, …, Wei D. Lu; arXiv:2511.13912, Nat. Commun. 2026):** non-spiking continuous diagonal SSM in a memristor crossbar; the device's native short-term-memory relaxation physically realizes the state decay. *(numbers not verified — paywalled.)*
- Corroborating: **QS4D (2507.06079)** — QAT of diagonal S4D for TaOx memristive crossbars; QAT confers analog-noise robustness (read in full 2026-08-01; moved here from (a)) — plus **IMSSA (2412.20215)** — same group's earlier crossbar deployment of recurrent S4D (read in full 2026-08-01) — **HPD (2508.11935)** for analog-CIM robustness, and **LMU-on-Braindrop**.
- **Mechanism:** the memristor's exponential relaxation *is* `e^{At}`. No discretization mismatch; in-memory MVM for `Ax+Bu`.

## 4. What I actually built and measured

One SSM, **four parameter-matched implementations** (identical tensors — embedding, `W_in`, diagonal decay, `W_mix`, `W_out`; **only the position of the nonlinearity moves**; 173,596 params on char-LM, exactly matched):

| # | variant | recurrent state | output | route |
|---|---------|-----------------|--------|-------|
| 1 | `digital` | exact continuous (S4D-real) | graded | (a) baseline |
| 2 | `spikeout` | exact continuous | LIF spikes | (b) SPikE-SSM-style |
| 3 | `analog` | analog: ±4 rails, σ=0.02 noise, 6-bit, send-on-delta events (θ) | graded events | (c) the non-spiking route |
| 4 | `spikestate` | **carried in spikes** (1-bit) | spikes | the floor control — the only variant inside my bound's scope |

Two tasks: **char-level WikiText** (statistical) and a **synthetic copy** task at memory load M ∈ {16, 32, 64} symbols from a K=16 alphabet (precise recall; gives a tunable, quantifiable memory demand). 3 seeds everywhere. Energy proxy = 45 nm Horowitz (MAC 4.6 pJ / AC 0.9 pJ), with graded analog events priced both optimistically and conservatively. Code + per-run JSON + ledger: `github.com/zeyuyuyu/sparsity-ceiling` (`ssm3way.py`, `ssm3way_ledger.md`).

## 5. Result 1 — char-LM (statistical): analog is the best neuromorphic datapath, at a real cost

3 seeds, paired Δbpc vs the digital baseline (negative = better), matched-communication-rate comparisons:

| variant | bpc | emitted rate | paired Δbpc |
|---|---|---|---|
| `digital` | 3.3429±0.0221 | 1.000 | — |
| `analog` θ=0.15 | 3.1936±0.0078 | 0.612 | **−0.149±0.014** |
| `analog` θ=0.75 | 3.2932±0.0109 | 0.380 | −0.050±0.020 |
| `analog` θ=1.00 | 3.3595±0.0273 | 0.266 | +0.017 |
| `spikestate` | 3.6205±0.0130 | 0.645 | +0.278 |
| `spikeout` | 4.3978±0.0292 | 0.369 | +1.055 (state rate exactly **1.0000**) |

Two things worth noting. First, the comparison is at **matched communication rate**, which was the obvious objection to any earlier version: analog at 0.612 emission beats spiking-state at 0.645 by **0.427 bpc**, and analog at 0.380 beats output-spiking at 0.369 by **1.10 bpc**. So analog's advantage over the two spiking routes is *not* an operating-point artifact. Second, `spikeout` reproduces the SPikE-SSM signature exactly — `rate_state` = 1.0000 at every setting, i.e. **all of its sparsity is in the output, none in the state**. That is the most reproducible single observation in the whole study (it holds at every M on both tasks).

**But see §7 — the comparison against `digital` above is against an under-regularized baseline, and the honest headline is the tradeoff, not the win.**

## 6. Result 2 — copy (precise recall): the ranking INVERTS, and 1-bit state goes to zero

Margin kept = fraction of the digital baseline's above-chance accuracy margin retained (the right metric when absolute accuracy is low). Values below are already corrected against a **noise-regularized** digital reference (§7); 3 seeds; emitted rates matched at ~0.41–0.50 across the three non-digital variants, so the spread is attributable to the datapath, not the rate.

| variant | margin kept, M=16 | margin kept, M=32 |
|---|---|---|
| `spikeout` (exact state, spiked output) | **0.87** | **0.42** |
| `analog` (lossy state, graded events) | 0.50 | 0.28 |
| `spikestate` (1-bit state) | 0.074 | **0.00** (exactly chance) |

Three readings:

1. **The char-LM ordering is exactly reversed.** char-LM best→worst was analog ≪ spikestate ≪ spikeout; copy is spikeout ≪ analog ≪ spikestate. `spikeout` has the *most exact* state on both tasks and is best on one, worst on the other — so "state fidelity" alone is not the axis (§9 gives the axis that is).
2. **A 1-bit recurrent state does not degrade precise recall, it eliminates it** at M=32 — while being the *cheapest* cell by the pJ proxy (102.0k pJ/token). This is the cleanest demonstration I have that **an energy number is meaningless without the quality it purchased**, which I'd rather say myself than have said to me.
3. **Both non-digital routes get worse as retention demand rises** (spikeout 0.87→0.42, analog 0.50→0.28 when M doubles; seed sd ≤0.003). So there is no single "analog is X% of digital quality" number to quote — the workload's retention demand sets it.

## 7. The control that cost me a headline  *(this is the part I'd want a reviewer to see first)*

An earlier version of this result said *analog beats digital by 0.149 bpc on char-LM*. I ran the control a reviewer would demand — give the **digital** baseline the same medicine — and it kills that claim:

- digital + training-time state noise σ=0.02 → **Δbpc −0.309±0.024**, i.e. **2.1× the entire analog advantage**;
- therefore **analog θ=0.15 is +0.160 bpc WORSE than a properly regularized digital baseline** (3 seeds, far outside noise). **The "analog beats digital" claim is retracted.**
- **weight decay is worth exactly zero** (+0.005±0.011) and **6-bit quantization is worth zero on char-LM** (−0.002) — an informative dissociation: the mechanism is specifically **state-level stochasticity in the recurrence**, not generic capacity control. The analog datapath supplies that regularizer *for free and physically*, which is a genuine point in analog's favour.
- **Clean decomposition:** analog's net −0.149 = a **−0.309 noise benefit** + a **+0.160 send-on-delta gating cost**. The datapath gives a real gift but charges more for its sparsity than the gift is worth, at every threshold.

**The claim that survives is energy-for-quality.** The digital noise is training-time only (identical inference cost), so: regularized digital **3.034 bpc @ 712k pJ/token** vs analog θ=0.15 **3.194 bpc @ 446k pJ/token** ⇒ **analog buys a 1.60× proxy-energy cut for a +0.160 bpc (5.3%) quality cost.** A defensible wedge for power-constrained statistical sequence workloads; not a free lunch.

The same control on copy shifts every margin-kept figure by only 3–5 points without changing any ordering (that correction is already applied in §6's table).

**One out-of-sample confirmation from that control, honestly labelled post-hoc:** the noise arm turned out to be train-time-only (`if self.training`), but the 6-bit quantizer is *not* gated — it is a genuine **inference-time state-precision degradation**. The same quantizer costs **−0.002 bpc on char-LM** and **+0.369 / +0.385 bpc on copy at M=16 / M=32**. Same hardware degradation, opposite verdicts by workload, and the copy penalty does not attenuate as retention demand doubles. This was identified post-hoc (my pre-registration mis-assigned its own arms), so I report it as mechanistically motivated, not as a pre-registered win. Useful corollary: quantization *alone* keeps 0.78 of the digital margin at M=32 — far above spikeout's 0.42 and analog's 0.28 — so **state-precision loss explains part but not most of the neuromorphic copy deficit; the event/spike mechanisms add their own cost.**

## 8. Where my firing-floor bound stands  *(negative, and I'd rather state it plainly)*

The bound `ρ ≥ H_b⁻¹(M·log₂K / H)` prices a recurrent net that carries its state in spikes. I tried to make it bind on a net that still learns. Two attempts:

- **Raising M failed structurally:** `spikestate` is at chance by M=32, so the only binding load (M=64) is one where the net retains nothing, and the bound presupposes retention. Its learnability ceiling on copy (~M=16) sits *below* the load where the floor starts to bind.
- **Shrinking H at fixed M=16** (64 bits of demand) is the valid design — the digital reference learns copy at 9–10× chance across H ∈ {64, 96, 128, 256} (10.5k→87.9k params), so every cell is a legitimate test. Predicted floors rise **12×** (0.042 → 0.110 → 0.174 → 0.500), but **measured LIF state activity falls 2×** (0.496 → 0.392 → 0.391 → 0.250) and at H=64 lands *below* its own floor. Worse for the bound: **`spikestate`'s margin kept rises monotonically 0.079 → 0.200 as H shrinks**, i.e. the most over-provisioned net is the worst. An information bottleneck cannot produce that ordering; an optimization pathology can.

**Verdict:** LIF state activity here is **width-determined, not information-determined**. The bound is not validated in either direction at this scale, and spiking-state failure on precise recall is an **optimization failure the bound does not explain**. It stays in the paper as theory with a stated, *tested* scope limit — which is still exactly the reason not to carry a compressed recurrent state in spikes (§6 result 2 makes that case empirically instead).

## 9. The principle, and what it implies for silicon

**One claim predicts both orderings:** *a neuromorphic variant is cheap exactly when it degrades the part of the datapath the task does not depend on.*

- Precise-recall workloads need an exact recurrent **state** and tolerate a spiked output → keep the state in an exact digital datapath, spike the output.
- Statistical workloads need an exact graded **output** distribution and tolerate a lossy state → the analog/CIM route is the fit.
- Evidence: the two matched-rate tables (§5, §6) plus the quantizer contrast at three task/load points (§7).

**Recommendation for the SSM-host question:**
1. **CIM/analog (Wei Lu-style) is the right SSM host for statistical sequence workloads with low retention demand** — perception, sensor streams, language-like statistics. This is the answer in the direction you suspected, and it aligns with the event-vision thesis.
2. **Not a general drop-in for digital SSMs.** The further a workload moves toward precise recall of many items, the worse *both* neuromorphic routes get relative to a digital SSM.
3. **Carrying the recurrent state in spikes is off the table** for anything memory-bearing. The live design fork is only whether the exact state lives in a **digital** datapath (spike the output) or an **analog** one (accept the loss, gain the event sparsity).

**One co-design constraint that fell out, which I think is the most hardware-relevant thing here:** on the analog datapath the send-on-delta threshold θ is **floored by the state quantizer**. At 6-bit precision over ±4 rails the LSB is 0.125, and every θ below that is *bit-identical* on every metric — the **ADC step, not θ, sets the minimum event rate**. So event sparsity and state precision trade off *against each other in hardware*: buying a lower event rate means spending bits, which is a converter/area cost, not a free algorithmic knob. "Just lower the threshold" is not an available answer. Relatedly, θ does **not** transfer across tasks (char-LM's θ=1.0 collapses analog to chance on copy; the usable copy window is 0.125 < θ ≲ 0.3), so any analog SSM deployment needs per-workload threshold calibration.

**Addendum 2026-08-03 — we tried to turn the activity win into a measured pJ win, and the answer is NO, with the blocker precisely located (now at 3 seeds).** Decomposing our own proxy: the readout `W_out` carries **75%** of the analog SSM's pJ/token (47% of digital's), while the recurrence — the only term any mechanism here or in the cited literature sparsifies — carries **8%** (at char-LM's vocabulary, H·V = 72,448 > H² = 65,536, so the readout is the largest matrix in the model). Even a *free* recurrence caps analog at **1.74×** vs the regularized digital reference, so the published 1.60× is already 92% of that ceiling. Pricing the previously-unpriced terms closes the obvious objection rather than the gap: ADC/DAC swept over 0.01–1.0 pJ/event and analog storage over 0–0.5 pJ/unit/step stay at **0.00–0.33%** of the analog total at H=256. The fix that should have worked — an event-driven (send-on-delta) readout — is **quality-fatal on both workload classes**: ~+1.0 bpc on char-LM at *every* threshold (a step at gate-on, flat across send rates 0.98→0.11; the digital arm pays the same, so none of the saving attributes to the analog state), and total collapse on copy (margin kept ≤0.05 at both M=16 and M=32, even with 99.8% of readout units still transmitting). A matched random-hold ablation settles the attribution: **staleness itself is the fatal mechanism** (+1.12/+0.92 bpc at a 0.90 send rate, digital/analog), and send-on-delta is actually *better* than random at low rates — so no smarter gating rule reopens the route. The only readout sparsification that survives quality (LIF output-spiking, the copy winner) recomputes every MAC and saves nothing. Bottom line for your question: **on language-like workloads the pJ ceiling of this SSM datapath family is set by readout width, not by the state mechanism — anyone promising large energy wins from sparsifying an SSM recurrence is optimizing an 8% term, and the 75% term resists both known gating mechanisms.** (Caveats: the copy-collapse attribution to staleness is inferred from the char-LM ablation; intermediate operating points are single-seed; the ceiling is a model-shape property — at H ≫ V the recurrence would dominate instead; all energy is the 45 nm proxy.)

## 10. Caveats I'd want stated before anyone quotes this

- **Scale.** 173k params (char-LM) / 10.5k–88k (copy), 6–30 epochs, small tasks. The char-LM regularization benefit in particular is a small-model/short-schedule effect and may shrink at scale.
- **The energy proxy is a proxy** (45 nm Horowitz MAC/AC), and it does **not** price what analog actually costs: matching on *emitted* rate matches wire traffic, not state activity. Analog's state is dense (rate ~0.994) and needs a storage element + converter per unit. Analog also still pays MAC-priced input/output layers, so on copy it is worse than `spikeout` on **both** quality and pJ. **There is no measured pJ win for analog here — only a proxy one on char-LM.**
- **Simulation, not silicon.** Route (c) is simulated device physics (rails, noise, quantization, send-on-delta), not a memristor array. Device variation, drift, and read/write noise are unmodelled — HPD exists precisely because they matter.
- **Recall weakness inherited from SSMs** (verified on SPikE-SSM) is consistent with everything above: streaming perception/control is the fit; associative recall likely needs a small attention/memory component.
- All numbers are 3 seeds with seed sd reported; per-run JSON and the full running ledger (including every retraction, in order) are in the repo.

## References  *(✓ = read in full)*

- ✓ Voelker, Kajić, Eliasmith. *Legendre Memory Units.* NeurIPS 2019. (+ Braindrop, Neckar…Boahen, Proc. IEEE 2019; Loihi, Davies…Imam, IEEE Micro 2018.)
- ✓ Zhong et al. *SPikE-SSM.* arXiv:2410.17268, 2024.
- ✓/□ Zhang…Wei D. Lu. *Compute-in-Memory Implementation of SSMs for Event Sequence Processing.* arXiv:2511.13912 / Nat. Commun. 2026. *(abstract + secondary sources; paywalled.)*
- ✓ QS4D (2507.06079) — *read in full 2026-08-01; reclassified route (a) → CIM-adjacent (c): deployment target is memristive analog CIM, not digital fabric.*
- ☑ IMSSA (2412.20215) — read in full 2026-08-01: same-group precursor to QS4D; recurrent S4D kernels on a 64×64 memristive crossbar (A/B/C in one array, ternary weights); deployed 81.69% vs 95.06% software on 2-class Heidelberg digits (drop attributed to stuck devices); no energy/area numbers; state analog-vs-digitized between steps NOT explicitly stated.
- ☑ HPD (2508.11935) — read in full 2026-08-01: simulation-only (PyTorch/L20 GPU) weight-perturbation robustness study of Mamba/Mamba2 on analog CIM; final block's output projection is the most noise-sensitive part; fix = hybrid SVD split (UΣ stays on the CIM array, Vᵀ offloaded to digital) — an independent datapath-exactness allocation. Weight noise only (analog state decay not modelled); no energy/area numbers; their "up to 99.57%" is a degradation-removed robustness ratio (their eq. 16), not a perplexity cut.
- □ Mitrokhin et al. *Sci. Robotics* 2019 (aaw6736) — *paywalled.*
- ~ Izhikevich, *Spiking Manifesto* (2512.11843); ArrowFlow (2604.04087) — *abstract-level.*
- Kanerva, *Sparse Distributed Memory*; Eliasmith, *NEF*.
- Wang, *The Sparsity Ceiling* (arXiv, cs.NE) — the firing-floor bound of §8; code and this study's data at `github.com/zeyuyuyu/sparsity-ceiling`.
