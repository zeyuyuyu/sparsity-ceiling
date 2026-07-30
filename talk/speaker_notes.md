# Beyond Attention — Speaker Notes

*Per-slide talk script (also embedded in the .pptx notes pane). ~7-8 min total.*

## Slide 1. Title — Beyond Attention

[~30s · open calm, set the frame]
Thanks for the time. I want to walk through a research direction that came out of a small empirical result of mine -- a short paper called "The Sparsity Ceiling."
One-line version: if spiking, brain-inspired hardware cannot ride the Transformer paradigm -- and I'll show you it structurally can't -- then the real question is what paradigm of intelligence it CAN ride. Today I map six candidates and say where I'd place the bet.

## Slide 2. The question + empirical hook

[~60s · this is the thesis slide -- slow down]
Start with what I think intelligence actually is. My working definition: a predictive generative model of the world, held in ongoing dynamics, that spends compute only on surprise and acts to test its own predictions. The key word is LOOP -- model interacting with world -- not a function that maps context to a token.
Transformers reach intelligence the opposite way: stateless, dense retrieval over the entire context, brute-forced with scale.
On the right is my empirical hook. In the sparsity-ceiling work, a recurrent spiking net hits a hard firing floor around 50% -- you cannot make it sparse. Attention CAN be made sparse, but only by storing the full key-value cache -- an O(context) memory wall. Either way, the Transformer route is mismatched to neuromorphic hardware. That mismatch is what motivates the whole talk.

## Slide 3. Two paradigms of intelligence

[~45s · walk the last row hard]
Side by side, the two paradigms are almost point-for-point opposites. Transformer: stateless, external memory that grows with context, dense all-pairs compute, offline global backprop, intelligence from scale. Brain: persistent state, memory compressed into the state and synapses, compute spent only on prediction error, local online learning, intelligence from prediction and action.
The bottom row is the punchline for us: the Transformer column is exactly what neuromorphic hardware is bad at -- dense MACs and a memory wall -- and the brain column is exactly what it is good at -- event-driven, analog, local. So the paradigm choice and the hardware choice are the same choice.

## Slide 4. Why the brain pays neither cost

[~55s · anticipate the obvious objection]
Here's the fair objection: my own paper proves recurrent spiking nets have a firing floor. So how does the brain -- which is recurrent and spiking -- avoid it?
Because the brain violates the assumption behind my bound. My bound assumes information travels ONLY in spikes. The brain doesn't. It holds memory in sub-threshold membrane potential and short-term synaptic state -- activity-silent -- and spikes only to signal errors. That escapes the firing floor AND the memory wall at once. Add an enormous neuron count, which pushes the required firing rate toward zero, and local multi-timescale learning.
The takeaway is the line at the bottom: the brain is an existence proof -- general intelligence at 20 watts, about a thousand times more efficient than GPUs -- using prediction and sparsity, not attention.

## Slide 5. The landscape of non-attention paradigms

[~70s · this is the survey -- keep it brisk, one breath per row]
So if not attention, what? Six non-attention paradigms, each a real, active line of work.
Predictive coding -- hierarchical error propagation, local learning that provably approximates backprop.
Equilibrium propagation -- energy-based, local weight updates, analog-native, with hardware demos claiming four orders of magnitude less training energy.
Analog and spiking state-space models -- linear recurrence, no KV cache; keep the state in analog dynamics and you sidestep the firing floor.
World models and JEPA -- predict in latent space, plan by simulation.
Modern Hopfield / associative memory -- and this one is delightful: a Hopfield read is mathematically identical to attention, so the brain's "attention" is attractor dynamics, not QK-transpose.
Active inference -- minimize expected surprise in a perception-action loop.
The double checkmarks are where neuromorphic physics is an actual advantage, not just an algorithmic fit.

## Slide 6. The bet

[~50s · commit -- say it like a conviction]
If I have to place the bet, it's this: a predictive world-model held in analog recurrent dynamics, learned locally and online -- predictive coding or equilibrium propagation -- acting in a closed loop, where spikes carry only surprise.
I want to stress this is not settling for less. It's choosing the paradigm where the hardware's physics is a feature, not a bug. Analog state escapes both the memory wall and the firing floor. Error-only propagation IS event-driven -- native to spiking silicon. Local plasticity means learning on-chip, no global backprop. And recurrence plus search gives you test-time reasoning without a giant single forward pass.

## Slide 7. Honest frontier

[~45s · say the hard part plainly -- this earns credibility]
Let me be honest about where this stands. None of these paradigms has beaten Transformers yet. Predictive coding, EqProp, SSMs, Hopfield, JEPA, active inference -- all promising, all more brain-like, none has produced GPT-class general capability. This is a research bet, not a finished alternative, and I'll state it as such.
But here's why it's still the right bet: the brain runs general intelligence on 20 watts using exactly this paradigm. So we know the paradigm CAN get there -- we just haven't reproduced it. That gap is the open frontier, and it is the reason to work on it, not the reason to avoid it.

## Slide 8. Proposed directions

[~55s · concrete -- this is what you'd actually do in the lab]
Concretely, four things I'd work on.
One: build a state-space model whose recurrent state lives in analog sub-threshold dynamics rather than spikes, and test directly whether it breaks my bound's assumption and sparsifies the way attention did.
Two: implement predictive coding on neuromorphic silicon with local equilibrium-propagation learning, and measure real energy on Loihi 2 or SpiNNaker2 -- turning my 45-nanometer proxy into an actual measurement.
Three: formalize the recurrence-versus-attention, firing-floor-versus-memory-wall trade-off as a conservation law, and find where analog-state models sit on that frontier.
Four: a closed-loop predictive world-model for a low-power embodied agent, where neuromorphic's real-time, event-driven physics is decisive.
Any of these is a first project; the first two I could start now.

## Slide 9. Thesis

[~25s · land it, then open for questions]
To close: intelligence is a loop, not a lookup. If neuromorphic hardware can't ride attention, it shouldn't try -- its physics aligns with the predictive, sparse, local, embodied paradigm that the brain already proves works at 20 watts. That's the direction I want to pursue.
Happy to go deeper on any of the six paradigms, the bound, or the specific experiments. Thank you.

