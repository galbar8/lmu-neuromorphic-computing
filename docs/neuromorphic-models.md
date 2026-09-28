# Neuromorphic Computing and LMU-Based Spiking Models

Notes derived from §2.3, §4 and §6.2 of the report. Section numbers refer to the [full report](../report/lmu-neuromorphic-computing.pdf).

Related: [LMU](lmu.md) · [HiPPO and SSM](hippo-and-ssm.md) · [Mamba](mamba.md)

## Why neuromorphic computing

The report's starting point is a comparison: the human brain performs language understanding, face recognition, planning, memory and learning on roughly the power of a 20-watt bulb. Neuromorphic computing tries to imitate how it does that, to build artificial systems that are flexible and energy-thrifty for the same structural reasons.

The biological picture the models draw on (§2.3.1): a neuron receives signals through dendrites, each arriving across a synapse. Incoming signals accumulate as electrical potential, and once potential reaches a threshold the neuron fires a spike down its axon to the synapses of the next neurons. The system does not run on a clock — it acts only when something requires a response. That asynchronous operation is where the energy saving comes from. Spikes are not interchangeable either: the timing of each, the interval between them, and which neuron fired all carry meaning.

## Neuromorphic versus classical computation

Three axes of difference (§2.3.3):

**Information encoding.** Classical computation passes continuous values; every neuron receives numbers and computes mathematical functions over them. Neuromorphic computation represents information as a series of spikes — electrical pulses whose timing and rate carry the meaning. The report notes this suits naturally occurring phenomena such as vision, hearing and movement.

**Memory management.** In classical computing, memory is discrete and centralised: records, data structures, or the internal states of a neural network updated each step, requiring explicit read and write mechanisms and often gates to control the flow. In neuromorphic computing, memory is distributed and continuous, an inherent part of the neurons' own dynamics — encoded in their dynamic state and in the synaptic connections between them rather than kept as a separate structure. The system's response depends not only on current input but on accumulated past state. The report's summary of the contrast: classical computing aims at an explicit representation of memory, neuromorphic computing embeds memory in the behaviour of the system itself.

**Hardware.** Classical computing rests on the von Neumann architecture, with a clear separation between processing and memory, requiring CPUs and GPUs to run sequences of operations on externally stored data. Neuromorphic hardware places processing physically adjacent to memory and activates it only when relevant input arrives. Where classical systems run at fixed frequencies with full utilisation, neuromorphic hardware is *sparse* — most neurons and connections are inactive most of the time, waking only in response to significant change. This yields both energy saving and scalability, since the number of units can grow without proportionally increasing total resource consumption.

## Event-based computation

In classical computing, processing is synchronous: work proceeds at a uniform rate regardless of content. In event-based computation, processing happens only when something changes — a change in an image, a change in direction of motion, or another input exceeding a threshold. Nothing runs until an event justifies a response.

Translating a continuous signal into discrete events requires an encoding (§2.3.2):

| Scheme | Mechanism | Trade-off |
|---|---|---|
| **Rate coding** | Input intensity maps to firing rate — a brighter pixel makes a neuron fire more frequently | Simple to implement, stable over time; but needs many spikes to represent information, raising energy use and processing time |
| **Temporal coding** | Information carried in the precise timing of spikes | Precise and fast; sensitive to noise, demands accurate timing |
| **Time-to-first-spike (TTFS)** | A temporal-coding variant: a neuron fires exactly one spike, and *when* it fires encodes input intensity | Energy-efficient and fast; less suited to complex or noisy information |
| **Direct coding** | Input is fed straight into the network, which converts it to spikes itself | The network learns the encoding best suited to the task; requires more computational resources |

Direct coding is the scheme relevant to what follows: there is no prior conversion step, because the model handles the conversion during processing. The report notes this approach was examined in L2MU, where raw motion-sensor signals were converted to spikes by an initial processing layer with no separate encoding stage.

## The LIF neuron

The Leaky Integrate-and-Fire neuron is the most common abstraction for a biological neuron. Neuromorphic computing does not copy the brain one-to-one; it takes the basic principles and implements them in a form a computer can process.

The three phases:

- **Integrate** — each input, usually a spike from another neuron, raises the neuron's membrane potential
- **Leaky** — the neuron continuously loses potential, which prevents it reaching high excitation unless enough input arrives within a short interval
- **Fire** — when potential crosses a threshold the neuron emits a spike and resets its potential to zero

The report notes researchers modelled this with an electrical circuit: input passes through resistors and a capacitor to an amplifier that accumulates voltage until it crosses the excitation threshold, producing a spike at the output.

**Cuba-LIF** is an extended variant that adds a synapse level. Rather than checking only the sum of inputs, it also models synaptic conduction between neurons, through an additional internal state — a synaptic current updated before the voltage computation. Other models exist at varying levels of complexity and abstraction depending on need.

## Adapting LMU dynamics to spikes

The adaptation is not trivial (§4.1.1). The [LMU](lmu.md) operates on continuous-time, real-valued signals; neuromorphic networks generally operate on binary signals, discrete in both time and value — a firing threshold that is either active or not. Translating the LMU's linear differential equations into structures that spiking neurons can execute means preserving the same *global* memory behaviour through different machinery.

**Basic implementation.** The original LMU paper showed in principle how this can be done, using a network of Poisson spiking neurons with weights chosen so that, as neurons fire over time, they integrate the incoming signals. The difficulty is that this is not simple summation: the accumulation must match the Legendre polynomial basis the LMU uses to represent memory. The network effectively learns to decompose the input into components represented by those polynomials, with spikes carrying each component's contribution. The resulting network does not compute the LMU's continuous solution exactly, but approximates it — and the approximation improves as neuron count grows, which the report likens to more pixels giving more detail in an image. The reconstruction error scales as `O(d/√m)` in the number of neurons `m`.

**Why the fit is natural.** Two reasons, from §4.1.2. First, spiking networks struggle to represent long-range context, because information is carried as binary events that lose history quickly unless a dedicated mechanism exists — and the LMU *is* such a mechanism, designed from the outset for precise, compact, controllable retention of input history, with depth governed by `d` and `θ`. Where some neuromorphic models rely on random internal dynamics that must be learned or characterised through long training, the LMU offers a built-in, mathematical, orthogonal memory in which each component is responsible for a distinct temporal range. Second, since spiking neurons can realise differential equations through asynchronous summation of spikes, and the LMU is built on a linear state-space model, the translation follows naturally in both mathematical and computational terms.

The report's conclusion on this point: the LMU complements the weaknesses of typical SNNs, providing precise memory with controllable depth, fully implementable on spiking neurons without sacrificing either energy efficiency or continuity of processing.

## L2MU

**L2MU** (Fra et al., 2024) is a natively neuromorphic LMU implemented with LIF neurons, presented for human activity recognition (HAR) from sensor data. Rather than building an LMU as a conventional network and then trying to adapt it, the model is built from the start to run directly on LIF neurons.

**Architecture (§4.2.1.1).** Three populations of neurons implement the components of the original LMU, each built from LIF or CuBa-LIF neurons, each receiving synaptic input, accumulating membrane voltage and firing when a predefined threshold is crossed:

| Population | Implements | Current |
|---|---|---|
| **U** | Input representation `u` | `u_t,curr = eₓᵀ·x_t,spk + e_yᵀ·h_{t-1},spk + e_mᵀ·m_{t-1},spk` |
| **M** | Memory state `m` — manages long-range relationships | `m_t,curr = Ā·m_{t-1},spk + B̄·u_t,spk` |
| **H** | Hidden state `h` — short-range, relations between consecutive steps | `h_t,curr = Wₓ·x_t,spk + W_h·h_{t-1},spk + W_m·m_t,spk` |

The `spk` subscript denotes the spiking activity of the corresponding population; each fires when its membrane voltage crosses its threshold (`θ_u`, `θ_m`, `θ_h`). Each element of memory `m` is realised by a group of neurons, with the connections between them carrying the values of `Ā` and `B̄` and thereby regulating accumulation and leakage. The spikes from population H feed the output layer, which performs the classification. This is how continuous linear dynamics are represented inside a discontinuous spiking system.

**The encoding module (§4.2.1.2).** The model's main contribution: processing raw sensor data directly, with no preliminary conversion to spike format. In most neuromorphic networks, continuous data must first pass through an external encoding mechanism, which often requires manual tuning and adds delay, complexity, and some loss of fine detail — and is not biological in character. L2MU instead feeds the raw signal to an internal module of three layers:

1. **Per-channel neuron layer** — LIF neurons dedicated to each input channel (the X, Y, Z axes of each sensor), expanding dimensionality and producing a spike pattern representing the input value
2. **Information merging layer** — a further spiking population combining information across channels, producing a richer spatio-temporal representation
3. **Harmonisation layer** — converting the information into an encoding suitable for the L2MU core

The encoding neurons learn for themselves how to produce a spike train representing the data, based on characteristics such as intensity, rate of change and temporal structure. The result is an end-to-end process operating entirely within a neuromorphic setting, from sensor signal to decision, with no intermediate translation layers.

**Experimental setup and results.** Six input channels of raw accelerometer and gyroscope data, segmented into 2-second windows suited to capturing human activity, with 40 measurements per sensor per window. Evaluated on the WISDM dataset, reaching **up to 94.14%** on the test set during training, with prediction under 300 ms.

**Edge deployment (§4.2.1.3).** Compressed models were deployed on three commercial platforms — Raspberry Pi 3B, Raspberry Pi 4B and STM32MP157F — using two optimisation techniques: **pruning** (removing low-impact connections or neurons) and **quantization** (converting weights from floating point to less precise representations). Reported real-time accuracy: **93.91%** with Leaky neurons at a mean response time of at most 0.13 s, and **93.84%** with Synaptic neurons at 0.07–0.22 s. Summarised: above **93%** accuracy, roughly **130 ms** mean response time, and under **200 MB** memory, on conventional hardware with no dedicated neuromorphic chip. The Leaky variant was more accurate than the Synaptic one, and **L2MU outperformed the conventional LMU in all configurations** — moving to a spiking structure improved both accuracy and stability.

## LMUFormer and Spiking LMUFormer

**LMUFormer** (Liu et al., 2024) works in the opposite direction from L2MU: instead of moving the LMU into a spiking network, it places LMU memory inside a modern sequence architecture. The aim is to close the gap between LMU and Transformers, combining local and global processing — memory units that excel at retention, with convolutional depth networks that process information in parallel and find local patterns efficiently.

Three components (§4.2.2.1):

1. **Patch embedding** — the input is divided into short segments (`patches`), each containing several consecutive samples from the sequence. Each patch passes through a convolutional layer that learns to translate it into a feature vector representing its local content
2. **Channel mixer** — further convolutional layers, applied across channels rather than spatially, learning the relationships between different features within each patch
3. **LMU block** — the enriched patches feed the LMU mechanism, which tracks changes over time and preserves relationships between distant segments through its compact representation of history, in linear time

**Spiking LMUFormer (§4.2.2.2)** converts this to an SNN while preserving the LMU principles and the convolutional structure. In the patch-embedding stage the ordinary convolutional layer is replaced by a normalisation layer followed by a layer of spiking neurons, which together produce a spike train representing the input, fed onward to the LMU memory unit.

At the heart of the architecture is the **spiking LMU cell**, which mimics classical LMU dynamics using LIF neurons. Each time step computes membrane potential, checks threshold crossing, generates a spike and resets the potential. The internal formulas update memory and hidden state through spikes alone, using a discretised version of the state matrices `A` and `B` from the original LMU equations. The innovation: instead of using the continuous memory state directly, it is fed to spiking neurons to obtain the memory-state spikes `MS[t]`; memory state and spiking input are then concatenated along the feature dimension and passed through a 1D convolution, then batch normalisation, then a spiking layer, giving the spikes representing the LMU block's output at time `t`.

**Results (§4.2.2.3).** Evaluated on psMNIST, Speech Commands V2 (over 100,000 audio samples of short words) and Long Range Arena.

- **Speech Commands V2** — Spiking LMUFormer reached **96.12%**, against **96.53%** for the non-spiking LMUFormer, with less energy and computation. The report describes this as a new high among neuromorphic models, comparing favourably with SNN competitors such as Spikformer and Spike-Driven ViT
- **LRA** — particularly strong on ListOps (hierarchical structure in a sequence) and Retrieval (identifying an item within a long sequence from a cue), exceeding basic Transformer-based models; average accuracy **0.53% above** the regular LMUFormer, which the report reads as the spiking model exploiting its internal temporal dynamics
- **Efficiency** — roughly **400× fewer operations** than a Transformer on the same task. §4.1.2.1 additionally reports a **27.2× energy reduction** relative to a conventional LMU, attributed to two causes: sparse activation, since very few neurons fire at any given moment, and the use of binary operations in place of expensive multiplications

## SpikMamba

**SpikMamba** (Chen et al., 2024) applies the same principles through [Mamba](mamba.md) rather than the LMU directly, targeting event-camera data for HAR. Event-camera output is unlike ordinary images: it is a sequence of points where light intensity changed, arriving at very high rate, non-uniformly, and sparse — most of the time nothing is happening.

The report presents the motivation as a gap between existing approaches: ANN-based methods compress the event sequence to cope with data volume and lose information in the process; Spiking RNN or GCN methods examine only local relationships and lack a wide view over time. SpikMamba combines local and global vision.

**Architecture (§4.2.3.1–4.2.3.2).** Three stages:

1. **Spiking patch embedding** — data is divided into patches, features extracted with a `Conv3D` layer that learns patterns within each patch. Each patch is converted to a spike-based representation: instead of ordinary values, information is held as a series of zeros and ones indicating whether a neuron fired at that moment. Spatial and temporal position embeddings are added so context is not lost
2. **Spike Linear Attention** — local attention over short spans. Each patch gets `Q` (what is being sought), `K` (what information the patch holds) and `V` (the important content), and query is compared against key to identify relationships between patches, as in a Transformer. The difference is that spiking layers process the query and key — neurons that activate only when the information is strong enough, like a biological neuron firing
3. **Spike Mamba** — global dependency. To understand complete human actions it is not enough to see relationships between adjacent moments; events seconds earlier matter too. A spiking version of Mamba supplies this, with state equations whose parameters `A`, `B` and `Δ` are learned from the input itself, each patch passing through an expansion layer such as `Conv1D`

**Neuromorphic implementation (§4.2.3.3).** The core rests on LIF neurons, chosen for computational simplicity and the ability to retain temporal information; every layer including the attention and transition mechanisms is built from them, keeping the model consistent with biological computation principles. Adapting Mamba is the notable part: its differential equations were converted to a discrete version using **zero-order hold**, which assumes the input is held constant between sampling steps. The report gives the ZOH conversion in its appendix — for continuous `dz(t)/dt = A z(t) + B x(t)`, the discrete coefficients are

```
Ā = exp(ΔA)
B̄ = (ΔA)⁻¹ (exp(ΔA) − I) · ΔB
```

with `Δ` the interval between steps: `exp(ΔA)` describes how much of the previous state carries forward, `B̄` how much of the current input affects the new state.

**Results (§4.2.3.4).** Evaluated on four event-based HAR datasets:

| Dataset | Description |
|---|---|
| **PAF** | 10 action categories, 45 samples each, recorded with a DVS346 event camera |
| **HARDVS** | 300 action categories, over 107,000 samples; considered especially challenging for its diversity |
| **DVSGesture** | 11 hand and arm movement categories at 128×128 pixels |
| **E-Faction** | 128 action categories, 1024 recordings at 260×346 pixels, for complex temporal actions |

SpikMamba achieved the highest accuracy on all four. On efficiency it is also lighter: roughly **91% fewer parameters** than ExACT and **40% fewer** than EvT, while performing better. An ablation removing the SNN layers to make the model ANN-only lost **2.71%** on average, which the report reads as evidence that spiking networks suit event-data analysis specifically, and that combining SNN with Mamba gives the best capability for learning both global and local dependence.

## Hardware challenges

The report is candid about the gap between principle and practice (§6.2).

**Representing linear computation with spikes.** The LMU performs continuous mathematical computation through matrix operations; neuromorphic hardware does not operate continuously and works with spikes rather than streams of numbers. To represent an LMU memory cell, a large population of neurons is used, each contributing a small part of the computation — every value the model wants to retain is represented by a group of neurons, not one. Representing values more precisely requires more neurons. The residual problem is **precision**: faithfully simulating the differential equations requires the hardware to represent numbers to high accuracy, so neurons must be added to compensate for what the hardware lacks, and each addition makes the computation more accurate but heavier.

**Connectivity and timing.** On chips such as Loihi, SpiNNaker and TrueNorth, the LMU wants as many memory units interconnected as possible, but each neuron has a hard limit on its number of connections, forcing implementations to be split across multiple cores. More fundamentally, the LMU's most basic operation is matrix multiplication, which is difficult to realise naturally on spike-based hardware — Loihi and similar chips perform event-based computation in which neurons sum currents, generally without matrix multiplication, and do not natively perform complete linear operations. Compounding this, few neuromorphic chips are readily available and convenient: existing ones remain relatively experimental, complicated to program, and limited in connection count and numeric resolution.

**What is done in practice.** Much work therefore implements neuromorphic models on conventional hardware — building a network of LIF neurons, using spike-based signals, but running it all in simulation on an ordinary computer. The network then does not gain the energy saving the theory promises. The report is explicit that this remains worthwhile: it allows neuromorphic ideas to be tested without expensive and complicated hardware, performance to be compared against ANN-based models, and theoretical and practical research to proceed in parallel. Proof-of-concept work running LMU on a neuromorphic chip does exist. The overall situation is a delicate balance between fidelity to the theoretical model and hardware constraints — a trade-off between accuracy and efficiency.

## Future directions

From §6.3, restricted to what the report actually proposes:

**Attention and hybrid architectures.** Classical LMU cannot focus on important content the way attention does — it retains memory generally, efficiently and compactly, but without knowing what matters. Mamba demonstrates one answer, continuing the line of thinking of models like LMU but adding input-dependent deliberation, so that instead of fixed matrices they change at each step according to what is happening. The cost is simplicity: computation can no longer be done with ordinary convolutions, and the model becomes more complex. A further idea is hybrid models combining LMU-style layers with attention layers applied over short spans, so SSM layers (Mamba, LMU) handle distant context linearly in time and energy while other layers handle local detail. **Samba** is cited as an example of this shape — a layered architecture in which each block combines a Mamba layer with a local sliding-window attention mechanism, compressing the sequence into a compact hidden state while retaining the ability to recall detail in the recent window. For neuromorphic LMU models the challenge is harder, since spike-based hardware is not built for complex attention mechanisms; researchers are therefore attempting to build attention *into the memory itself*, making the memory mechanism behave as though it selects what is important based on spike flow or on varying neuron properties, without introducing heavy computation.

**Advanced spiking variants.** The open question is how to preserve the LMU's exact mathematical structure — the orthogonality of its memory — when implemented with spiking neurons such as LIF, without losing accuracy to quantization, noise or hardware limits. On the applied side, the scalability of spiking LMU-based models needs analysis: building more complex models capable of tasks requiring attention mechanisms, while still running on real hardware such as Loihi. Spiking networks suit real-time event streams particularly well, so for tasks such as sensor signal processing or temporal pattern recognition a spiking model may outperform a conventional network in both accuracy and resource use. Optimal use may also be **partial** — only part of the model, such as the encoding stage or the memory, being spiking while the remaining stages stay deterministic. A central research direction is therefore how to integrate neuromorphic modules, and neuromorphic LMU models in particular, as components of larger systems rather than as standalone solutions: whether they belong at the start of a network for temporal feature extraction, or as an internal memory layer filtering information over time.

**Real-world adaptation.** Most studies evaluate on artificial sets such as psMNIST and ListOps, or on standard NLP tasks. Real-world tasks — autonomous vehicle intelligence, medical event monitoring from sensors (which the report singles out as having enormous potential for LMU-based models and being insufficiently researched) — introduce new challenges. In real-time systems data does not arrive as a closed file but streams at irregular rates, with noise, faults and changing dynamics, requiring noise robustness, fast response and run-time adaptation, none of which the classical LMU has, having been built for a smooth sequence at a fixed rate. Ideas worth investigating: tuning the model to the system's dynamic rate, for instance varying its sampling frequency or memory structure according to how fast the data changes; training the model during use so it improves while running, perhaps through additional adaptive layers or continual learning with a flexible memory mechanism. The open questions the report poses: can a version of LMU preserve the orthogonal structure of its memory while adapting to a changing environment in real time? Can the model be combined with self-correction mechanisms? What further adaptations are needed to make it a usable component in complex real-time systems?
