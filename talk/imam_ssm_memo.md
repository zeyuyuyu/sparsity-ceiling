> **SUPERSEDED 2026-07-31 — see [imam_ssm_memo_v3.md](imam_ssm_memo_v3.md).** This v2 is literature-only; its §6 states a hypothesis that the completed experiment partly refuted (analog does not match a regularization-controlled digital baseline, and the verdict is workload-conditional). Kept for history.

# Are neuromorphic chips suitable for SSMs — and is the *non-spiking* route also a fit?

**Response to N. Imam's TODO — Zeyu Wang** · *(v2 — claims below checked against full text of the load-bearing papers; reading-status noted at the end)*

---

## TL;DR

**Yes — and the non-spiking route is not speculative: it has a decade-old precedent and an active hardware line.**
An SSM is a continuous-time *linear* dynamical system, which is exactly what analog neuromorphic circuits are. The honest, corrected picture is that the real design axis is **not "spiking vs non-spiking"** but **where the continuous linear state lives**: (a) digital, (b) continuous state with a spiking/sparse *output*, or (c) *native analog device physics*. Route (c) — the non-spiking one you asked about — is the best structural fit, and it already exists (LMU on Braindrop; memristor CIM-SSMs).

## 1. Reading list, decoded  *(confidence noted)*

- **Mitrokhin et al., *Sci. Robotics* 2019 (aaw6736)** — hyperdimensional/VSA (Kanerva) for neuromorphic sensorimotor control. *[abstract/title only — paywalled]*
- **Izhikevich, *Spiking Manifesto* (arXiv:2512.11843)** — spikes as look-up tables / polychronization / ~1000× efficiency. *[abstract-level]*
- **ArrowFlow (arXiv:2604.04087)** — computation in permutation space, integer-only, explicitly neuromorphic-aligned (VSA flavor). *[abstract-level]*
- **Kanerva (SDM/VSA) + Eliasmith (NEF)** — the classical frameworks for putting *memory/representation* and *dynamics* on a neural substrate.

**Read as:** the tools to run an SSM (= a dynamical system) on neuromorphic already exist — **NEF for the dynamics, VSA for memory** — so the question is *how*, not *whether*.

## 2. Why SSM ↔ neuromorphic is a structural match  *(LMU, read in full)*

- SSM: `dx/dt = A x + B u`, `y = C x` — a continuous-time **linear** dynamical system = what analog silicon physically *is*.
- **The LMU (Voelker, Kajić, Eliasmith, NeurIPS 2019) IS a linear SSM** (delay line via Padé → Legendre basis; predates/parallels HiPPO/S4). It is implemented via NEF on spiking neuron populations and **deployed on both Loihi (digital spiking) and Braindrop (analog mixed-signal)**. psMNIST **97.15%** (> LSTM 89.86%), 10⁵-step memory, Mackey-Glass NRMSE 0.054. *(N.B. Imam is a co-author on the Loihi paper — this is directly in your lineage.)*
- So **"SSM on neuromorphic," including the analog non-spiking form, is a solved existence proof, not a hope.**

## 3. Three ways to put an SSM on neuromorphic  *(corrected taxonomy)*

**(a) Digital SSM on digital neuromorphic** — quantized S4D deployed on digital fabric. *QS4D (arXiv:2507.06079)* — quantization-aware S4D for hardware.

**(b) Continuous linear state + spiking/sparse *output*** — the current "spiking SSM" line. **Correction to my earlier memo:** these keep the SSM state `h_t` **continuous** and only spike the *output nonlinearity* (SPikE-SSM, read in full: refractory-LIF on top of S4D; state stays continuous). Reported firing is *output* sparsity (LRA avg **~8%**, WikiText **24.5%**) — **not** a state-carrying firing floor. Cost: a recall gap (WikiText ppl **33.2 vs S4's 21.0**).

**(c) Native analog SSM — the non-spiking route you asked about (best fit)** — let device physics *be* the dynamics:
- **CIM-SSM (Zhang, …, Wei D. Lu; arXiv:2511.13912, Nature Communications 2026):** a **non-spiking, continuous, diagonal SSM** in a memristor crossbar, where the device's **native short-term-memory relaxation physically realizes the state decay** — asynchronous event-based vision & audio, high accuracy + energy efficiency. *(specific accuracy/energy numbers not verified — not quoted here.)*
- Corroborating cluster: **IMSSA (2412.20215)** deploying modern SSMs on memristive in-memory hardware; **HPD (2508.11935)** for robustness of analog-CIM SSMs; and the classic **LMU-on-Braindrop** (NEF analog).
- **Mechanism = physics does the integration:** the memristor's own exponential relaxation *is* the SSM's `e^{At}`. Zero discretization mismatch; in-memory MVM for `Ax+Bu` = no data movement.

## 4. Where my *Sparsity Ceiling* firing floor actually applies  *(honest scope)*

My ~50% firing floor was measured on a **fully-recurrent spiking net whose state is carried in spikes**. It does **not** transfer to route (b) or (c), which keep the linear state **continuous** — so those already embody the "analog-state escape" my bound points to. The floor is the reason **not** to carry a compressed recurrent state in spikes; it is *not* a claim against SSMs on neuromorphic. (This corrects an overstatement in my first draft.)

## 5. Honest caveats

- **Analog non-idealities** dominate route (c): device variation, read/write noise, limited precision, conductance drift, ADC/DAC boundary overhead. (HPD exists precisely to make analog-CIM SSMs robust to these.)
- **Recall weakness inherited from SSMs** (verified on SPikE-SSM): fine for streaming perception/control; weaker for associative-recall / in-context tasks — likely needs a small attention/memory component for those.

## 6. Proposed exploration — one experiment that answers the TODO

One SSM (S4D or LMU), three real implementations, matched capacity; measure **quality / activity-sparsity / energy**:
1. **Digital S4D** (GPU + QS4D-style quantized digital) — baseline;
2. **Continuous-state + spiking-output** (SPikE-SSM-style) — event-driven output;
3. **Native analog / CIM** (simulate device-physics state first, then memristor / Braindrop-style analog).

**Hypothesis:** (3) matches (1)'s quality at the lowest energy by letting device physics integrate the state, with device-noise robustness (à la HPD) the main obstacle — while (2)'s benefit is *output* sparsity, not state sparsity. Directly answers "is the non-spiking SSM also a fit," and fuses your TODO with my paper's proposed direction.

## References  *(✓ = read in full for this memo)*

- ✓ Voelker, Kajić, Eliasmith. *Legendre Memory Units.* NeurIPS 2019. (+ Voelker thesis; Braindrop, Neckar…Boahen, Proc. IEEE 2019; Loihi, Davies…Imam, IEEE Micro 2018.)
- ✓ Zhong et al. *SPikE-SSM.* arXiv:2410.17268, 2024.
- ✓/□ Zhang…Wei D. Lu. *Compute-in-Memory Implementation of SSMs for Event Sequence Processing.* arXiv:2511.13912 / Nat. Commun. 2026. *(abstract + secondary sources; full text not accessed — Nature paywall)*
- □ IMSSA (2412.20215); QS4D (2507.06079); HPD (2508.11935) — *identified via search, not read in full.*
- □ Mitrokhin et al. *Sci. Robotics* 2019 (aaw6736) — *paywalled, title/abstract only.*
- ~ Izhikevich, *Spiking Manifesto* (2512.11843); ArrowFlow (2604.04087) — *abstract-level.*
- Kanerva, *Sparse Distributed Memory*; Eliasmith, *NEF*. Wang, *The Sparsity Ceiling* (arXiv, cs.NE).
