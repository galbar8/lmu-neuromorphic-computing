# LMU Models and Neuromorphic Computing

A survey of Legendre Memory Units, modern State Space Models, and their integration with neuromorphic computing.

**[Read the full report (PDF)](report/lmu-neuromorphic-computing.pdf)** · 97 pages · Hebrew, with an English abstract and table of contents

**English technical summaries:** [LMU](docs/lmu.md) · [HiPPO & SSM](docs/hippo-and-ssm.md) · [Mamba](docs/mamba.md) · [Neuromorphic models](docs/neuromorphic-models.md)

---

## Overview

This repository holds *A Comprehensive Review of LMU-Based Models and Their Integration into Advanced Neuromorphic Systems*, a survey completed as part of an M.Sc. in Computer Science at the Open University of Israel, Department of Mathematics and Computer Science, under the supervision of Dr. Hadar Cohen Duwek.

The work traces the Legendre Memory Unit from its original formulation through its relationship to HiPPO and State Space Models, then examines how LMU-style memory is being carried into spiking, event-driven hardware. It is a literature survey: it analyses the mathematical structure of the models and compares results reported in the source papers. It contains no new experiments of its own.

Because the report body is written in Hebrew, the `docs/` directory carries English summaries of the four core technical threads.

## Motivation

Sequence modelling has to hold information across time while staying cheap enough to run. Recurrent networks address the first requirement and fail the second: backpropagation through time produces vanishing or exploding gradients, each step depends on the last so training will not parallelise on a GPU, and gating in LSTM and GRU adds matrices per unit without fully solving long-range retention.

Transformers removed the sequential dependency via self-attention, but attention costs `O(T²d)` in time and memory for sequence length `T` and representation width `d`. The survey is explicit that recent work has largely dispelled the claim that Transformers cannot handle long-range dependencies — what remains is their cost, which keeps them awkward for real-time systems and for energy-constrained hardware.

That leaves a gap for models that represent time as a continuous quantity rather than a uniformly sampled discrete index, retain information without a fixed attention window, admit parallel training, and run within a tight power budget. The LMU was designed against that set of requirements, and its linear continuous-time dynamics are what make the neuromorphic direction plausible.

## Scope of the Survey

**Sequential baselines** — RNN, LSTM, GRU, ConvLSTM, Transformers (scaled dot-product and multi-head attention, positional encoding)

**Continuous-time memory** — Legendre Memory Unit, Legendre polynomial basis, Delay Network, LTI memory dynamics

**Theoretical framework** — HiPPO (`HiPPO-LegT`, `HiPPO-LagT`, `HiPPO-LegS`), State Space Models, orthogonal SSMs, S4, SaShiMi

**Scaling and selectivity** — FF-LMU (Feedforward LMU), parallelised LMU training, Mamba and selective state spaces (S4 vs. S6)

**Neuromorphic computing** — biological inspiration, event-based computation, spike encoding (rate, temporal / time-to-first-spike, direct), LIF and CuBa-LIF neurons, spiking neural networks, neuromorphic vs. von Neumann hardware

**Neuromorphic architectures** — Poisson-spiking LMU, L2MU, LMUFormer and Spiking LMUFormer, SpikMamba

**Applications and limits** — biomedical signals, human activity recognition, IoT and edge computing, NLP, plus hardware and training limitations

## Research Context

The relationships in the survey are conceptual, not a clean chronology. LMU (2019) precedes the HiPPO paper (2020) yet is described as resting directly on HiPPO's principles, because HiPPO provides the general theory — projection onto orthogonal polynomial bases under a time-varying measure — of which the LMU's Legendre construction is one instance. Likewise, LMU is characterised as a specific realisation of a State Space Model rather than a predecessor of the SSM literature.

```
        RNN
         │
   LSTM / GRU / ConvLSTM          Transformer (self-attention, O(T²d))
         │                             │
         └──────────────┬──────────────┘
                        │  long-range memory at acceptable cost?
                        ▼
              LMU  (Legendre basis, LTI memory)
                        │
        ┌───────────────┼────────────────────┐
        ▼               ▼                    ▼
   HiPPO / SSM     FF-LMU, parallel      Mamba / selective SSM
   (general theory)   LMU training        (input-dependent A, B, C)
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
        Neuromorphic realisations: L2MU,
        Spiking LMUFormer, SpikMamba
```

## Legendre Memory Unit

The LMU is a recurrent memory cell that represents the recent history of its input rather than an opaque hidden state. Its core is a linear time-invariant system that projects the input `u(t)` over a sliding window of length `θ` onto a Legendre polynomial basis:

```
θ ṁ(t) = A m(t) + B u(t)
```

`m(t)` is a `d`-dimensional memory vector whose components hold the coefficients of that projection; `A` and `B` are fixed matrices derived from the Legendre basis, not learned. Discretised, the cell is:

```
u_t = eₓᵀ x_t + e_hᵀ h_{t-1} + e_mᵀ m_{t-1}     (input encoding)
m_t = Ā m_{t-1} + B̄ u_t                         (linear memory)
h_t = f(Wₓ x_t + W_h h_{t-1} + W_m m_t)          (nonlinear hidden state)
y_t = W_o h_t                                    (output)
```

Two properties follow. First, the memory is linear and gate-free, so information does not have to survive a chain of nonlinearities — which is what removes the vanishing-gradient pressure that LSTM and GRU manage with gates. Second, memory size `d` is decoupled from hidden-state size `n`, so memory depth can be increased without growing the nonlinear computation; in LSTM and GRU the two are bound together. The order `d` sets the fidelity of the reconstruction: at `d = 1` the cell retains a running average of past input, at `d = 2` linear trends, and higher `d` preserves progressively finer temporal detail.

The survey reports LMU capturing continuous dependencies over windows on the order of 100,000 steps, well beyond what it attributes to traditional recurrent models.

→ [`docs/lmu.md`](docs/lmu.md)

## LMU, HiPPO and State Space Models

HiPPO — High-order Polynomial Projection Operators — formalises online function approximation: given a measure `μ(t)` that assigns importance to each past instant, project the history seen so far onto a polynomial basis, keep only the coefficients, and update them through a differential equation instead of recomputing. The choice of measure determines the memory's character: `LegT` weights a fixed window of length `θ` uniformly (the case that matches the LMU's own update rule), `LagT` uses Laguerre polynomials for exponential decay toward recent input, and `LegS` weights all history from 0 to `t` uniformly, making it invariant to time rescaling. The survey reports `HiPPO-LegS` reaching 98.3% on permuted-sequential MNIST against 95.1% for LSTM, 93% for GRU and 97.9% for a basic Transformer, with `O(N)` memory updates.

State Space Models describe a system through a state equation and an observation equation:

```
x′(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

This is the same continuous-time linear foundation the LMU is built on, which is why the survey treats LMU as a particular SSM whose state matrices come from the Legendre basis. Orthogonal SSMs generalise the idea to other bases; `HiPPO-LegS` is notable for *not* being time-invariant, since its matrices carry a `1/t` dependence.

→ [`docs/hippo-and-ssm.md`](docs/hippo-and-ssm.md)

## Mamba and Selective State Spaces

The limitation the survey identifies in LMU and in classical SSMs is that their dynamics are fixed: the matrices do not depend on what arrives, so the model cannot decide that one part of a sequence matters more than another. Mamba makes `A`, `B` and `C` functions of the input, adding a selection gate `g_t = σ(Linear(x_t))` that determines what propagates and what is suppressed.

The cost is that the model is no longer time-invariant, so it cannot be folded into a fixed convolution the way S4 can. Mamba recovers throughput with a hardware-aware selective scan — kernel fusion, parallel scan and recomputation — keeping working state in fast SRAM rather than repeatedly reaching into HBM, and scaling linearly in sequence length. The survey reports generalisation on Induction Heads to sequences up to a million tokens where Hyena and Transformer baselines failed, and 99.8% on Selective Copying.

The survey's own framing is that LMU and Mamba are complementary rather than competing: LMU contributes a principled orthogonal memory basis, Mamba contributes input-dependent dynamics, and a hybrid using a Legendre basis inside a selective SSM is presented as a research direction rather than an existing result.

→ [`docs/mamba.md`](docs/mamba.md)

## Neuromorphic Computing

Neuromorphic computing is relevant here for a structural reason, not just an efficiency one. Classical computation is synchronous: it recomputes on a clock whether or not anything changed, and the von Neumann split between processor and memory means data moves constantly. Event-driven computation does work only when something happens — a change in the scene, a threshold crossing — and neuromorphic hardware places processing next to memory, leaving most units idle most of the time. The survey's reference point is the human brain, which does what it does on roughly the power of a 20-watt bulb.

Spiking neural networks carry information in binary events whose timing and rate are the signal. The Leaky Integrate-and-Fire neuron is the common abstraction: input accumulates as membrane potential, potential leaks away continuously, and crossing a threshold emits a spike and resets. The CuBa-LIF variant adds a synaptic-current state to model conduction between neurons. Converting a real-valued signal into spikes requires an encoding — rate coding, temporal coding such as time-to-first-spike, or direct coding where the network learns the encoding itself.

LMU-style memory is interesting in this setting precisely because SNNs are weak where LMU is strong. Spikes are binary events that lose history quickly unless a mechanism preserves it, whereas the LMU offers an explicit, orthogonal, mathematically characterised memory with depth controlled by `d` and `θ`. And because the LMU's memory is a first-order linear differential system, translating it into spiking dynamics is a natural move rather than a forced one. The original LMU work already demonstrated a Poisson-spiking implementation with reconstruction error scaling as `O(d/√m)` in the number of neurons `m`, so accuracy improves as neuron populations grow.

## Neuromorphic and Modern Architectures

**L2MU** (Fra et al., 2024) is a natively neuromorphic LMU: the input encoding, memory state and hidden state are each realised as a population of LIF or CuBa-LIF neurons, mirroring the original LMU equations with spike-mediated interactions. Its distinguishing feature is being encoding-free — a three-layer encoding module (per-channel, cross-channel merging, harmonisation) learns to turn raw accelerometer and gyroscope signals into spikes, removing the external conversion step most SNN pipelines need. Evaluated on WISDM human activity recognition and deployed after pruning and quantisation to Raspberry Pi 3B, Raspberry Pi 4B and STM32MP157F, it is reported above 93% accuracy at roughly 130 ms response time and under 200 MB of memory, on commodity hardware with no neuromorphic chip. The report notes L2MU outperformed the conventional LMU across the configurations tested.

**LMUFormer** (Liu et al., 2024) goes the other direction, placing LMU memory inside a Transformer-shaped architecture: convolutional patch embedding for local features, a channel mixer to combine them, and an LMU block for long-range temporal structure. Its spiking variant, **Spiking LMUFormer**, replaces the computation with LIF dynamics and a spiking LMU cell. On Speech Commands V2 the spiking version reaches 96.12% against 96.53% for the non-spiking version, and on Long Range Arena it holds a small average advantage over the non-spiking model, with particular strength on ListOps and Retrieval. The energy case rests on sparsity and binary operations: the survey reports roughly 400× fewer operations than a Transformer on the same task, and a 27.2× energy reduction relative to a conventional LMU.

**SpikMamba** (Chen et al., 2024) targets event-camera data for human activity recognition, where input is sparse, asynchronous and high-rate. It pairs a spiking patch embedding (Conv3D with spatial and temporal position embedding) with two mechanisms: Spike Linear Attention for short-range structure between neighbouring patches, and a spiking Spike Mamba block for long-range dependencies, with Mamba's continuous equations discretised via zero-order hold. Across PAF, HARDVS, DVSGesture and E-Faction it is reported as most accurate while using roughly 91% fewer parameters than ExACT and 40% fewer than EvT. An ablation replacing the spiking layers with ANN equivalents loses 2.71% on average, which the authors read as evidence that spiking suits event data specifically.

→ [`docs/neuromorphic-models.md`](docs/neuromorphic-models.md)

## Applications

**Biomedical signal processing** — continuous, fast-changing physiological signals such as EEG suit a model built around a continuous-time memory. Speech is the clearest demonstration: Spiking LMUFormer reaches 96.12% on Speech Commands V2. The survey is careful here, noting that the leading result on that dataset is a QNN at 98.53% with 78k parameters, so the LMU case rests on structural flexibility and neuromorphic compatibility rather than on topping the leaderboard.

**Human activity recognition** — L2MU on WISDM motion-sensor data, reaching up to 94.14% on the test set during training and holding above 93% in compressed form, with sub-300 ms prediction.

**IoT and edge computing** — the same L2MU work deployed to Raspberry Pi and STM32 hardware, running end-to-end on spikes from raw sensor input, which is where the low memory footprint and absence of a separate encoding stage pay off.

**Natural language processing** — FF-LMU reaches 89.1% on IMDb with far fewer parameters than a comparable LSTM, and matches or beats much larger models on QQP and SNLI at under 1% of the compute. The survey states plainly that LMU-based models remain behind Transformer models such as BERT and GPT on most language tasks; the argument is efficiency in constrained settings, not parity.

## Challenges and Future Directions

**Training and time continuity.** The original LMU is recurrent, so it inherits sequential training and BPTT. FF-LMU (Chilkuri & Eliasmith) resolves this by training as a feedforward network — reported at roughly 80× faster training than a comparable LSTM on GPU, 98.49% on psMNIST, and 89.10% on IMDb with over 500× fewer parameters than LSTM — but it gives up the recurrent state, which makes it a better fit for tasks where temporal dependence is local or fixed in advance than for accumulating, context-dependent memory. Parallelised LMU training keeps the original recurrent structure and instead computes all states at once through an impulse-response matrix and the convolution theorem, at the cost of holding every state in memory simultaneously.

**Memory and generalisation.** The LMU's window is fixed and its dynamics are content-independent: it cannot decide what to keep. Mamba addresses this with input-dependent parameters, but is a substantially heavier model with far more parameters, so it does not simply replace the LMU on constrained hardware. Combining adaptive selection with LMU's simplicity is left open.

**Large-scale tasks.** LMU-based models are efficient at small scale — about 98.5% on psMNIST with under 0.2M parameters — but have not displaced Transformers on large tasks. The survey attributes this to full training parallelism and, critically, pre-training; the LMUFormer authors acknowledge their model does not support pre-training and must be retrained per task.

**Hardware.** Representing exact linear computation with spikes costs precision, and recovering it means adding neurons, which raises resource use. Neuromorphic chips such as Loihi, SpiNNaker and TrueNorth impose limited fan-out per neuron, non-uniform timing and limited numeric resolution, and do not natively perform the matrix multiplication the LMU depends on. The survey observes that much current work therefore simulates spiking networks on conventional hardware, which forfeits the energy benefit while still allowing the architecture to be studied — a proof of concept running LMU on neuromorphic silicon does exist.

**Directions the survey identifies.** Adding selective attention to LMU-style memory without losing energy efficiency, including hybrid stacks that pair an SSM layer with local sliding-window attention (Samba is cited as an example of this shape). More capable spiking variants that preserve the orthogonality of LMU memory under quantisation and hardware noise, possibly applying spiking only to parts of a model such as the encoder or the memory. And adaptation to real-world conditions — streaming input at irregular rates, noise and faults, run-time adaptation — as opposed to curated benchmarks like psMNIST and ListOps.

The survey closes with a methodological caveat worth repeating: results across the source papers are not directly comparable, since they use different datasets, pursue different experimental goals, and emphasise different metrics — accuracy, throughput, parameter count, energy. Its conclusion is correspondingly measured: models combining LMU memory with neuromorphic principles show high potential for real-time tasks under tight efficiency constraints.

## Repository Contents

```
report/         Full academic report (PDF, 97 pp., Hebrew; English abstract and contents)
presentation/   Slide deck accompanying the work (PPTX, English, 31 slides)
docs/           English technical summaries of the four core threads
references/     Bibliography of the works cited, as links
```

## Full Report

**[lmu-neuromorphic-computing.pdf](report/lmu-neuromorphic-computing.pdf)** — the complete survey.

The body is in Hebrew. English readers will find the abstract on p. 94, the table of contents on p. 95, and the title page on p. 97; the figures, equations and comparison tables are readable throughout. The [`docs/`](docs/) summaries cover the technical content in English.

The [presentation](presentation/fusion-of-artificial-and-spiking-neural-networks.pptx) is in English and covers the same material in 31 slides. It is a PowerPoint file, so GitHub will download rather than preview it.

## Citation

Bar Lev, G. (2025). *A Comprehensive Review of LMU-Based Models and Their Integration into Advanced Neuromorphic Systems* [M.Sc. final paper]. The Open University of Israel, Department of Mathematics and Computer Science. Supervised by Dr. Hadar Cohen Duwek.

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).
