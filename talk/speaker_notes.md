# Beyond Attention — Detailed Speaker Notes

*Per-slide script + facts to have ready + anticipated professor questions. Also embedded in the .pptx notes pane. Full talk ~8-9 min + Q&A.*

---

## Slide 1. Title — Beyond Attention

[~40s · open calm, set the frame]

SAY: Thanks for the time. This is a research direction that grew out of a small empirical result of mine -- a short paper, "The Sparsity Ceiling," now on arXiv under cs.NE. The one-line version: if spiking, brain-inspired hardware cannot ride the Transformer paradigm -- and I have a result showing it structurally can't -- then the real question isn't "how do we port Transformers to neuromorphic," it's "what paradigm of intelligence is neuromorphic actually FOR." Today I'll give you the argument, a map of six non-attention paradigms, and where I'd place the bet -- ending on four concrete projects.

HAVE READY: The paper is my own work -- controlled experiments on an 8xA800 box, plus a small information-theoretic bound. Everything I show is either that result or peer-reviewed 2024-26 literature.

DELIVERY: Don't rush the title. Let "what is neuromorphic FOR" sit for a beat -- that's the whole talk.

---

## Slide 2. The question + empirical hook

[~70s · this is the thesis slide -- slow down, this is the frame everything hangs on]

SAY: Start with what I think intelligence actually is. My working definition: a predictive generative model of the world, held in ongoing internal dynamics, that spends compute only on surprise, and acts to test its own predictions. The operative word is LOOP -- a system continuously interacting with the world -- not a FUNCTION that maps a context window to a token. That distinction is architectural, not philosophical: a Transformer has no persistent internal state between calls; every forward pass is independent, and its "memory" is just the context re-fed each time. A brain never restarts -- it's a continuous dynamical system whose core operation is to predict incoming signals and propagate only the residual error. That's predictive coding -- Rao and Ballard '99, formalized by Friston's free-energy principle.

Transformers reach intelligence the opposite way: stateless, dense retrieval over the entire context, brute-forced with scale.

Now the empirical hook, on the right -- my result. Same network, swap only the neuron. A recurrent spiking net hits a hard firing floor around 50 percent: even with a strong sparsity penalty targeting 10 percent, three seeds, it will not go below ~50. Attention CAN be driven sparse -- I get a spiking Transformer down to 2 percent firing -- but only because it stores the full key-value cache, which is O(context) memory. So on neuromorphic hardware you face a dichotomy: recurrence pays a firing floor, attention pays a memory wall. Either way, the Transformer route is mismatched to the hardware. That mismatch is what motivates the entire talk.

IF ASKED "isn't next-token prediction also prediction?": Yes, but it's prediction implemented as a global retrieval over a stored context in one dense pass -- not prediction as an ongoing generative model with error feedback and persistent latent state. The residual-only, stateful version is what's cheap on neuromorphic; the retrieval version is what's expensive.

---

## Slide 3. Two paradigms of intelligence

[~55s · walk each row briefly, hit the bottom row hard]

SAY: Side by side, the two paradigms are almost point-for-point opposites.
- State: a Transformer is a pure function of its input window; the brain is a stateful dynamical system -- attractors, oscillations, persistent activity.
- Memory: the KV cache grows linearly with context -- that's the memory wall. On-chip SRAM is tiny -- IBM's NorthPole is 224 megabytes; it can't even hold LLM weights, let alone a growing cache. The brain compresses history into a fixed-dimensional state plus synaptic weights.
- Compute: attention is O(L-squared) all-pairs, QK-transpose over everything. Predictive coding computes only on prediction errors -- sparse by construction.
- Learning: backprop needs a global error signal and stored activations for the whole graph; the brain uses local, Hebbian, three-factor rules -- online and continual.
- Source of capability: Transformers get it from scale -- parameters times data times compute; the brain from a good generative model plus embodiment.

The bottom row is the punchline for us: the Transformer column is precisely what neuromorphic hardware is BAD at -- dense MACs and a memory wall -- and the brain column is precisely what it's GOOD at -- event-driven, analog, local. So the paradigm choice and the hardware choice are the same choice. You don't pick the hardware and then port the algorithm; you pick the paradigm the physics wants.

IF ASKED "Transformers work today and brain-like models don't -- so what?": Fair, and I'll be honest about that on slide 7. This slide is about structural alignment with the hardware, not current capability.

---

## Slide 4. Why the brain pays neither cost

[~65s · anticipate the obvious objection -- this is the intellectually strongest slide]

SAY: Here's the fair objection, and I want to meet it head-on: my own paper proves recurrent spiking nets have a firing floor. So how does the brain -- which is recurrent AND spiking -- avoid it?

The answer: the brain violates the assumption behind my bound. My bound says the minimum firing rate rho is at least H-b-inverse of log-2 M over H -- it comes from a counting argument: a binary vector of H units with rho-H ones can only index about 2-to-the-(H times binary-entropy-of-rho) distinct states, so to hold M memories you need enough active bits. But that bound assumes the state channel is BINARY SPIKES. The brain's isn't.

Three escapes:
One -- analog state. Real neurons carry graded information in sub-threshold membrane potential, and hold working memory in short-term synaptic facilitation -- "activity-silent" memory, Mongillo '08, Stokes '15. Memory can persist with near-zero firing. So spikes aren't the memory channel; they signal errors.
Two -- enormous H. Ten-to-the-eleven neurons drives log-2 M over H toward zero, so rho-min goes to zero. Sparse coding, roughly one-to-two percent active, is exactly the observed regime.
Three -- local, multi-timescale learning: fast spikes, slow synaptic eligibility traces, neuromodulation as the third factor. No unrolling the whole history.

The bottom line is the existence proof: the brain runs general intelligence on 20 watts -- about a thousand times more efficient than GPUs per equivalent operation -- using prediction and sparsity, not attention. So the paradigm demonstrably scales to general intelligence. That's the whole reason to take it seriously.

IF ASKED "so just build analog neurons?": Essentially yes -- that's my direction #1, an analog-state SSM. The bet is analog state plus event-driven spikes, not spiking Transformers.

---

## Slide 5. The landscape of non-attention paradigms

[~80s · the survey -- keep brisk, roughly one breath per row; you don't need every citation aloud, have them ready]

SAY: If not attention, what? Six non-attention paradigms, each a real, active line.

- Predictive coding: a hierarchy where each layer predicts the one below and only errors propagate; the learning rule is local yet provably approximates backprop -- Whittington and Bogacz, Millidge et al. 2022, and a stable fast version at ICLR 2024.
- Equilibrium propagation: energy-based; you relax the network to an energy minimum, nudge the output, and the weight update falls out of the local activity difference -- no separate backward pass. It's analog-native; hardware demos on memristors and oscillators claim up to four orders of magnitude less training energy, and there's now ImageNet-scale EP.
- Analog and spiking state-space models: SSMs -- S4, then Mamba -- are the anti-Transformer sequence model: linear recurrence, compressed state, O(L), no KV cache. Spiking versions -- SpikingMamba, SpikySpace -- add event-driven sparsity; if you keep the state analog you sidestep my firing floor.
- World models and JEPA: LeCun's bet -- predict in latent representation space, not pixels or tokens; V-JEPA 2 is 1.2 billion parameters on a million hours of video. Pair it with model-based planning -- Dreamer, MuZero -- and reasoning becomes simulation and search, not one dense pass.
- Modern Hopfield / associative memory: and this one is delightful -- Ramsauer et al. showed the modern Hopfield update is mathematically IDENTICAL to Transformer attention, with exponential storage capacity. So "attention" is just associative retrieval; the brain does the same thing with attractor relaxation dynamics -- which is neuromorphic-native.
- Active inference: minimize expected free energy; perception and action unified in one closed loop; embodied.

The double-checkmarks are where neuromorphic PHYSICS is the enabler, not just an algorithmic fit -- predictive coding, EqProp, analog SSM.

IF ASKED "which is closest to working at scale?": SSMs -- Mamba is already competitive with Transformers and runs on GPU. But predictive coding and EqProp are where neuromorphic hardware adds the most, because their local, error-only structure is what the silicon wants.

---

## Slide 6. The bet

[~55s · commit -- say it like a conviction, not a hedge]

SAY: If I have to place one bet, it's this: a predictive world-model held in analog recurrent dynamics, learned locally and online -- predictive coding or equilibrium propagation -- acting in a closed loop, where spikes carry only surprise.

It's a synthesis, deliberately: SSM-style compressed recurrent state, but in analog dynamics so you escape the floor; predictive-coding error-only propagation so it's event-driven; local learning so training happens on-chip; and a world-model-plus-planning loop so reasoning is search, not a giant forward pass. Each piece has independent evidence; the claim is they compose into a coherent, neuromorphic-native stack.

And I want to stress: this is not settling for less. It's choosing the paradigm where the hardware's physics is a feature, not a bug. The four alignment points map one-to-one onto four hardware advantages: analog state removes the memory wall AND the firing floor; error-only propagation IS event-driven, native to spiking silicon; local plasticity gives you on-chip learning at roughly ten-thousand-fold lower energy; and recurrence plus search gives you test-time reasoning without scaling a single dense pass.

IF ASKED "isn't this just a wish-list of everything brain-like?": No -- it's specifically the subset where the physics is the enabler. I'm explicitly NOT betting on spiking Transformers or datacenter neuromorphic -- my paper shows those lose. The unifying principle is one sentence: spend energy only on surprise, and keep state in physics, not in a cache.

---

## Slide 7. Honest frontier

[~50s · say the hard part plainly -- this slide earns you the most credibility]

SAY: Let me be honest about where this stands, because a professor will ask anyway. None of these paradigms has beaten Transformers yet. Predictive coding, EqProp, SSMs, Hopfield, JEPA, active inference -- all promising, all more brain-like, none has produced GPT-class general capability. Concretely, Transformers still win on three things: the scalability of backprop-on-GPUs for training, in-context learning via massive retrieval, and sheer engineering maturity. This is a research bet, not a finished alternative, and I'll state it as such.

But here's why it's still the right bet. The brain runs general intelligence on 20 watts using exactly this paradigm. So we KNOW the paradigm can reach general intelligence -- it's an existence proof -- we just haven't reproduced it in silicon. That gap is the open frontier, and it's the reason to work on it, not the reason to avoid it.

IF ASKED "why would this ever beat Transformers?": Two forcing functions. The energy wall -- Transformer training and inference cost is becoming unsustainable, and the brain shows a thousand-fold headroom exists. And the memory wall -- attention's KV cache doesn't scale to long-horizon, always-on agents. If either bites hard, the predictive-sparse paradigm is where the escape is.

---

## Slide 8. Proposed directions

[~65s · concrete -- this is what you'd actually do in the lab; sequence it]

SAY: Concretely, four projects.

One -- escape the firing floor with analog state. Build a state-space model whose recurrent state is a leaky ANALOG variable, sub-threshold, with spikes only for output and error. Then measure attainable firing versus the spiking RNN: does it break my bound's assumption and sparsify the way attention did? This is the cleanest test of the core hypothesis, and I can start it in simulation today.

Two -- predictive coding on neuromorphic silicon. Implement an error-propagating hierarchy trained with equilibrium propagation -- local rules -- and MEASURE real energy on Loihi 2 or SpiNNaker2. That turns my 45-nanometer proxy into an actual hardware measurement. This one needs INRC access to Loihi, which the PhD provides.

Three -- formalize the trade-off. Prove the recurrence-versus-attention, firing-floor-versus-memory-wall dichotomy as a conservation law -- a Pareto frontier -- and locate where analog-state models sit on it. This is the theory contribution; it extends the paper.

Four -- closed-loop world-model at the edge. A predictive world-model plus planning loop for a low-power embodied agent, where real-time, event-driven perception-action is decisive. This is the applied payoff.

SEQUENCING: One and three I can start immediately -- simulation and theory. Two needs hardware access. Four is the longer applied arc.

IF ASKED "which first?": Number one -- it directly tests whether analog state is the escape route, and it needs nothing but simulation to get the first signal.

---

## Slide 9. Thesis + Q&A

[~30s · land it, then open the floor]

SAY: To close: intelligence is a loop, not a lookup. If neuromorphic hardware can't ride attention, it shouldn't try -- its physics aligns with the predictive, sparse, local, embodied paradigm that the brain already proves works at 20 watts. That alignment is the thesis, and those four projects are how I'd pursue it.

I'm happy to go deeper on any of the six paradigms, on the bound and its derivation, or on the experiments. Thank you.

HAVE READY FOR Q&A: the arXiv paper; the bound's one-line derivation (counting argument); and the six key citations on the sources slide.

---

## Appendix — Anticipated hard questions (cross-slide)

**"This is all known work — what is YOUR contribution?"**
Three things: (1) the empirical *sparsity-ceiling* result — a controlled, same-architecture measurement that recurrence floors at ~50% firing while attention sparsifies to 2% but pays a KV memory wall; (2) the information-theoretic firing-floor bound and its escape conditions; (3) the *floor-vs-wall dichotomy* as a framing, plus the specific analog-state experiments to test it. The survey is context; the result, the bound, and the experiments are mine.

**"Why not just run Mamba on a GPU?"**
Because the target is the ~1000× energy / 20 W regime — always-on, embodied, at the edge. Mamba-on-GPU is excellent but lives in the datacenter power envelope. The bet is specifically about the hardware where analog state and event-driven compute are the enabler.

**"Predictive coding has existed since 1999 and hasn't scaled — why now?"**
Two things changed: hardware (analog neuromorphic maturing — Loihi 2, memristor crossbars, oscillator networks) and training (equilibrium propagation now at ImageNet scale; stable, fast PCNs at ICLR 2024). The algorithm-hardware pairing is newly viable.

**"Isn't the brain analogy overclaimed?"**
I use the brain as an *existence proof* of the paradigm's energy ceiling, not as a blueprint. Every claim I make is a hardware-physics claim — firing floor, memory wall, analog-state escape — and each is testable in simulation and on Loihi.

**"What would falsify your bet?"**
If an analog-state recurrent model still can't sparsify below the RNN floor at matched quality (direction #1 returns negative), or if EqProp/PC energy on real Loihi doesn't beat the ANN once measured (direction #2). Both are near-term, decisive experiments.
