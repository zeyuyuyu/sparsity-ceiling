# The Sparsity Ceiling

Code for **"The Sparsity Ceiling: Where Spiking Networks Can — and Cannot — Trade Activity for Energy."**

**Thesis.** The energy dividend of spiking sparsity is a property of the *task*, not of SNNs. Holding the
architecture fixed and swapping only the hidden unit (continuous vs. leaky-integrate-and-fire), plus a two-sided
target-firing-rate probe, we measure how far activity can be pushed down before quality breaks:

- **Feed-forward perception** sparsifies to **~5%** firing at no accuracy cost.
- A **recurrent** language model cannot go below **~50%** — the recurrent state must stay active to carry information (a *firing floor*).
- A **spiking Transformer** sparsifies **freely to ~2%** — so the ceiling is a property of *recurrent compression*, not sequence modeling. Attention escapes the floor only by storing the full key–value cache: it trades a **firing floor** for a **memory wall**.

We formalize the ceiling with an information-theoretic bound `ρ ≥ H_b⁻¹(log₂ M / H)` and confirm its predictions
(floor rises with memory load, falls with state width, and — refuting a naive memory-only reading — also rises
with task difficulty).

## Results at a glance

| architecture | min firing preserving quality | regime |
|---|---|---|
| feed-forward perception (CNN) | ~5% | sparsifiable |
| attention (Transformer) | ~2% | sparsifiable (but O(context) KV memory) |
| recurrent (RNN) | ~50% | **firing floor** |

## Scripts → paper claims

| script | what it produces | paper |
|---|---|---|
| `neuro_poc.py` | vision SNN vs matched CNN; firing/energy/input-floor | §4.1, §4.5 |
| `lm_poc.py` | char-LM spiking-RNN vs matched tanh-RNN; the ceiling | §4.2 |
| `mem_poc2.py` | ρ_min vs firing per memory load N (mechanism) | §4.4, Fig. 2 |
| `exp1_2d.py` | 2D bound test: floor ↑ with N, ↓ with width H | §4.5, Fig. 4 |
| `exp2_dissoc.py` | memory-vs-difficulty (honest: difficulty also raises floor) | §4.5, Fig. 5 |
| `spk_transformer.py` | spiking Transformer sparsifies freely (no floor) | §4.6, Fig. 6 |

## Reproduce

```bash
pip install -r requirements.txt

# vision: SNN vs CNN, two-sided sparsity probe to 5%
python neuro_poc.py --gpu 0 --epochs 10 --lam 0.3 --target 0.05

# language: spiking-RNN vs tanh-RNN (the ceiling)
python lm_poc.py --gpu 0 --epochs 5 --lam 1.0 --target 0.10

# mechanism: firing floor vs memory load
python mem_poc2.py --gpu 0

# 2D bound test (floor vs N and hidden width H)
python exp1_2d.py --gpu 0

# memory vs difficulty dissociation
python exp2_dissoc.py --gpu 0

# spiking Transformer (no floor; sparsifies to 2%)
python spk_transformer.py --gpu 0 --epochs 4
```

**Energy model:** 45 nm (Horowitz, ISSCC 2014) — MAC = 4.6 pJ, AC = 0.9 pJ. Energy figures are analytic
proxies from measured firing rates; no custom hardware is used (all runs on a single GPU).

## Data

- **FashionMNIST** (torchvision) as a rate-coded event-stream stand-in for perception. Native N-MNIST/DVS is
  the documented next step.
- **WikiText-103** (HuggingFace `datasets`) for char-level language modeling.

Data paths are set by constants at the top of each script — edit them for your environment.

## Status

Working preprint (v0.3). This is a small-scale, controlled study; scaling the spiking Transformer, native event
data, a tighter bound, and measured neuromorphic-hardware energy are the next steps (see the paper's Limitations).

## Citation

```bibtex
@article{sparsityceiling2026,
  title  = {The Sparsity Ceiling: Where Spiking Networks Can and Cannot Trade Activity for Energy},
  author = {Wang, Zeyu},
  year   = {2026},
  note   = {Preprint}
}
```
