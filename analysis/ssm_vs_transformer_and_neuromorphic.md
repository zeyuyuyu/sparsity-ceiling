# SSM vs Transformer, and how SSMs meet neuromorphic hardware

*Analysis note accompanying the Sparsity Ceiling work + paper 2 (Workload-Conditional Datapaths for SSMs on Neuromorphic Hardware). Grounded in our own measurements where cited; honest about what is not yet shown.*

---

## Part 1 — SSM vs Transformer: two horns of one conservation law

### The essential difference: compression vs. retrieval
- **Transformer = stateless + full storage + random-access retrieval.** It keeps every token's key/value (the KV cache) and attends over all of them, O(L²) compute. Memory is *lossless* — any past token is exactly recoverable — at a cost of **O(L) memory + O(L²) compute**.
- **SSM = stateful + compression + linear recurrence.** It compresses history into a **fixed-size state** (`x_t = A x_{t-1} + B u_t`), **O(Ld) compute** (parallelizable via scan), **O(1) memory**. Memory is a *lossy* compression.

### The three-axis tradeoff

| Axis | Transformer | SSM | Winner |
|---|---|---|---|
| Compute | O(L²) attention | O(L) linear recurrence | **SSM (long context)** |
| Inference memory | KV cache O(L·d·layers), grows with context = memory wall | fixed state O(d) | **SSM (decisive)** |
| Exact recall | yes (induction heads, copy, in-context learning) | lost to compression | **Transformer** |
| Hardware fit | dense matmul — GPU/TPU home turf | continuous-time linear dynamics | split (Part 2) |

### Why the tradeoff is fundamental (not an engineering gap)
Compression is necessarily lossy: you cannot simultaneously have an O(1) fixed state **and** exact recall of an arbitrary distant token — this is information-theoretic (cf. the firing-floor bound: separating M memory contents requires enough state capacity). Mamba's *selectivity* (input-dependent A, B, C) chooses *what* to keep — it mitigates but does not remove the loss.

So: **Transformer trades memory (the KV wall) for exact recall; SSM trades recall (lossy compression) for O(1) memory + cheap compute.** This is the algorithm-level image of the floor-vs-wall conservation law from the Sparsity Ceiling paper. You cannot get exact recall + O(1) memory + cheap compute at once.

### Empirical state (2024–26)
- SSMs (Mamba / Mamba-2) **match** Transformers on many tasks and **win** on throughput / long context.
- SSMs **lose** on copying, associative recall, in-context learning, precise long-context retrieval — the "recall" family.
- The field's answer is **hybrids**: a few attention layers interleaved among many SSM layers (Jamba, Falcon-H1 ~34B, Nemotron-H) = current efficient-model SOTA. A hybrid literally **pays a little memory wall to buy back the recall that compression drops** — the conservation law in practice.
- No pure SSM at true frontier scale (70B+) under controlled comparison; whether pure SSMs scale to the frontier is open.

**One line:** an SSM is not a "better Transformer" — it is a different machine at a different operating point (wins long-context compute & memory, loses exact recall); hybrids interpolate on the conservation law.

---

## Part 2 — SSM × neuromorphic: physically aligned, but the win is one specific cell

### Why an SSM is the natural neuromorphic sequence model
An SSM is a continuous-time linear dynamical system `dx/dt = A x + B u` — which is exactly what an analog neuromorphic circuit *is*. Not a port, a **physical isomorphism**: a memristor's own exponential relaxation realizes the SSM's `e^{At}` (CIM-SSM, Wei Lu group). And the SSM's **O(1) compressed state fits on-chip**, so it also **escapes the memory wall that kills LLMs on neuromorphic silicon**.

### Three ways to put an SSM on neuromorphic (increasing depth)
1. **Digital SSM on digital neuromorphic** (SpiNNaker2; quantized S4D, QS4D) — works, somewhat event-driven, but not the deep win.
2. **Continuous state + spiking *output*** (SPikE-SSM, SpikingSSM) — sparsity lives only in the output; the **state stays dense**, so the firing-floor bound does **not** apply to the state. Modest.
3. **Analog-state / compute-in-memory SSM** (memristor CIM; LMU-on-Braindrop) — the deep win: device physics *is* the state dynamics, escaping **both** the memory wall (O(1) state) **and** the firing floor (analog state does not travel on a spike channel). This is the frontier.

### What our own experiments found (paper 2, honest)
- **Analog-state SSM: ~27% event activity, quality within ~0.5% of the digital baseline.** Spiking-state (the floor control) is stuck at ~65% activity *and* worse quality; spiking-output keeps the state fully dense and is worst on quality.
- **Mechanism:** *gating communication is cheap; gating memory is not.* Analog state gates *communication* (send-on-delta events) while keeping *memory* dense-but-free in an analog variable — which is exactly why the firing-floor bound (premised on a spike-only channel) does not bind it.
- **Red line (honest):** this is an **activity/quality win, NOT yet a measured pJ energy win.** The analog variant still pays MAC-priced input/output layers and only gets event pricing on the recurrence, so the pJ proxy currently ranks it *worse*. The energy advantage is **unproven** — it is a datapath co-design problem, not a rerun.
- **Workload-conditional:** streaming / low-retention wins; precise-recall / reasoning loses (needs exact digital state). The firing-floor bound's memory-load prediction came back **inconclusive** on the copy task — not universal.

### Where this beats a GPU, specifically
- **Datacenter: GPU wins** (Mamba runs great on GPU; no neuromorphic edge).
- **Ultra-low-power edge, mW-scale streaming: this is the real neuromorphic-SSM advantage over GPUs** — analog CIM runs streaming sequence tasks inside a power/latency envelope a GPU cannot enter. Existence proofs: LMU on Loihi + Braindrop; CIM-SSM on event vision/audio.

### Hard open problems (what would make it a real advantage)
1. **Energy datapath (the current blocker):** get the input/output layers off MAC pricing so the activity win becomes a measured pJ win. Co-design, not a rerun. **← highest-leverage next step.**
2. **The recall wall:** SSMs (analog or not) are weak at recall, so a neuromorphic SSM is for perception/streaming, not reasoning. Reasoning would need a hybrid (some attention/memory) — but attention on neuromorphic reintroduces the memory wall. This tension is currently unresolved.
3. **Analog non-idealities:** device variation, noise, drift, ~6-bit precision, ADC/DAC overhead, and θ that does not transfer across tasks (our finding). Needs robustness (HPD-style).
4. **Training:** still BPTT-ish; local-learning (equilibrium-propagation / predictive-coding) analog SSM training is open.
5. **Selectivity:** Mamba's input-dependent dynamics — the thing that makes SSMs competitive — is data-dependent multiplication, the hardest part to make analog/event-driven. The most capable SSM is the hardest to neuromorphic-ize.

---

## Synthesis
- **SSM vs Transformer:** two ends of one conservation law — retrieval (memory wall) vs. compression (recall wall); hybrids interpolate.
- **SSM × neuromorphic:** the one physically-aligned neuromorphic sequence paradigm, but its home is **low-power streaming perception/control**, via **analog compute-in-memory**, and today it is an **activity win with the energy win still unproven**; it is **not** a path to neuromorphic reasoning/LLMs (recall wall).
- **Do first:** close the **energy datapath** so the ~27%-activity win becomes a *measured pJ win* on a real low-power streaming task (event camera / audio / sensor). That single step converts "neuromorphic SSMs have an advantage" from a paper activity number into a hardware energy fact.
