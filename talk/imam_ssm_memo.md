# Are neuromorphic chips suitable for SSMs — and is the *non-spiking* route also a fit?

**Response to N. Imam's TODO — Zeyu Wang**

---

## TL;DR

Yes. SSMs are the sequence-model paradigm **most physically aligned** with neuromorphic hardware, and **non-spiking (analog / compute-in-memory) SSMs are arguably the *stronger* fit than spiking SSMs.** An SSM is a continuous-time linear dynamical system — which is exactly what analog neuromorphic circuits *are* — and its O(1) compressed state avoids the memory wall. The spiking route pays a **firing floor** (from my *Sparsity Ceiling* bound); the analog route **escapes it** but pays device non-idealities. Both routes already have existence proofs.

## 1. The reading list, decoded

- **Mitrokhin et al., *Sci. Robotics* 2019 (aaw6736)** → hyperdimensional computing / VSA (Kanerva) for neuromorphic sensorimotor control.
- **Izhikevich, *Spiking Manifesto* (arXiv:2512.11843)** → spiking as look-up tables / polychronization / ~1000× efficiency.
- **ArrowFlow (arXiv:2604.04087)** → permutation-space, integer-only, explicitly neuromorphic-aligned (VSA flavor).
- **Kanerva (SDM/VSA) + Eliasmith (NEF)** → the two classical frameworks for putting *representation/memory* and *dynamics* onto a neural substrate.

**Read as:** the tools to implement an SSM on neuromorphic already exist — **NEF for the dynamics, VSA for the memory/representation.** The pointer isn't "spike-ify an SSM"; it's "an SSM is a dynamical system — use the frameworks that already run dynamical systems on neurons."

## 2. Why SSM ↔ neuromorphic is a structural match

- An SSM is `dx/dt = A x + B u`, `y = C x` — a **continuous-time linear dynamical system**. Analog neuromorphic hardware *is* continuous-time analog dynamics → a **physical isomorphism, not a port**.
- **O(1) fixed-size compressed state** → no growing KV cache → fits on-chip (**no memory wall**).
- **Precedent:** the Neural Engineering Framework (Eliasmith) implements arbitrary dynamical systems in neural populations; the **Legendre Delay Network / LMU** is a linear SSM implemented this way and **deployed on Loihi** (Voelker & Eliasmith 2019; "LIF-based LMU as a neuromorphic SSM," IEEE 2025).

## 3. Spiking vs. non-spiking SSM — the crux of the TODO

**Route A — Spiking SSM** (SPikE-SSM, SpikySpace, S5-SNN, SiLIF, MIMO-spiking-neurons):
- Event-driven; suits *digital* neuromorphic (Loihi / SpiNNaker).
- *Tension:* staying truly event-driven / multiplication-free costs accuracy; keeping accuracy retains dense float multiplies in the scan. And carrying the recurrent **state in spikes hits the firing floor** — my bound `ρ ≥ H_b⁻¹(log₂ M / H)`.

**Route B — Non-spiking (analog / compute-in-memory) SSM** — *the professor's specific question*:
- Implement the linear recurrence **directly in analog physics** — sub-threshold circuits, or memristor crossbars performing `A x + B u` as an **in-memory matrix-vector product**.
- **Best fit because:** (i) the dynamics *are* analog continuous-time dynamics → zero discretization mismatch; (ii) it **escapes the firing floor** (the bound assumes spike-only state); (iii) in-memory analog MVM is memristor neuromorphic's home turf (no data movement).
- **Existence proof:** *Compute-in-memory implementation of state space models for event sequence processing*, **Nature Communications 2025** — a **non-spiking** SSM in analog CIM hardware.
- *Costs:* analog non-idealities (device variation, noise, drift, precision, ADC/DAC boundary overhead) + the associative-recall weakness inherited from SSMs.

## 4. Framing (from *The Sparsity Ceiling*)

Spiking-vs-non-spiking SSM is a clean instance of the paper's **firing-floor ↔ memory-wall dichotomy**:

| route | escapes | pays |
|---|---|---|
| spiking SSM | memory wall (O(1) state) | **firing floor** (spike-carried state) |
| analog / CIM SSM | **firing floor** (analog state) | device non-idealities (noise, precision) |

So the question isn't which is universally better — it's **which cost is more tractable.** For SSMs, the analog route's costs (noise / precision) look more addressable than the spiking route's firing floor.

## 5. Proposed exploration — one experiment that answers the TODO

Take **one SSM** (S4D or LMU), three implementations at matched capacity; measure **quality / activity-sparsity / energy**:

1. **ANN-SSM on GPU** — baseline;
2. **Spiking SSM** (SPikE-SSM-style) — event-driven;
3. **Analog-state / CIM SSM** — simulate analog state first, then memristor / Loihi.

**Hypothesis:** the analog-state SSM **matches ANN quality while escaping the firing floor**, whereas the spiking SSM either loses accuracy or hits the floor. A clean, decisive test of "non-spiking SSMs are the better neuromorphic fit" — and it fuses the TODO with the direction proposed in my paper.

## Key references

- Voelker, Kajić, Eliasmith. *Legendre Memory Units.* NeurIPS 2019.
- *A LIF-based Legendre Memory Unit as a neuromorphic State Space Model.* IEEE, 2025.
- *Compute-in-memory implementation of state space models for event sequence processing.* Nature Communications, 2025.
- SPikE-SSM (arXiv:2410.17268); SpikySpace (arXiv:2601.02411); "Learning long sequences in SNNs" (Sci. Reports 2024); SiLIF (arXiv:2506.06374).
- Mitrokhin et al., *Sci. Robotics* 2019 (aaw6736); Izhikevich, *Spiking Manifesto* (arXiv:2512.11843); ArrowFlow (arXiv:2604.04087).
- Kanerva, *Sparse Distributed Memory*; Eliasmith, *Neural Engineering Framework*.
- Wang, *The Sparsity Ceiling* (arXiv, cs.NE) — firing-floor bound & dichotomy.
