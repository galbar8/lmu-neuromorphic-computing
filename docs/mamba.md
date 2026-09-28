# Mamba and Selective State Space Models

Notes derived from §3.3.2 and §3.3.3 of the report. Section numbers refer to the [full report](../report/lmu-neuromorphic-computing.pdf).

Related: [LMU](lmu.md) · [HiPPO and SSM](hippo-and-ssm.md) · [Neuromorphic models](neuromorphic-models.md)

## The gap Mamba addresses

The report arrives at Mamba (Gu & Dao, 2023) from the limitations of the [LMU](lmu.md), not from the Transformer literature. The LMU implements SSM principles through a continuous-time linear system projecting input history onto orthogonal polynomials, and three shortcomings follow from that design:

- it cannot dynamically select which information is relevant
- interactions between different states are constrained to be linear
- modern applications need optimisations the original formulation lacks

The first is the substantive one. In an LMU and in a classical SSM, the state matrices are fixed, so the same transformation applies regardless of what arrives. Recurrent networks and basic Transformers share a version of this problem — information is handled identically at every step, with no adaptation to context. Convolutional models compute quickly but their filters cannot select either. The report's framing: the answer is **selectivity**.

## Input-dependent dynamics

Mamba keeps the state equation

```
h′(t) = A h(t) + B x(t)
```

but makes the parameters functions of the input. Where a conventional SSM holds `A`, `B`, `C` constant, Mamba computes them dynamically from the data received, and adds a selection gate:

```
g_t = σ(Linear(x_t))
```

`Linear` is a learned weight matrix identifying which features matter and `σ` a nonlinearity such as sigmoid. The gate determines how much of the previous state propagates and how much is discarded, which filters noise, improves computational efficiency and reduces the need for additional memory.

In the architecture (§3.3.2.2–3.3.2.3) the matrices divide the work:

| Component | Role |
|---|---|
| `A` | Updates internal state — decides whether existing information is strong enough to persist or is replaced |
| `B_t` | The gate admitting external input `x_t` into the internal state |
| `C_t` | Converts internal state into output |
| `Δ_t` | Discretisation step, normalising and updating transitions against available hardware resources |

Because real data arrives at discrete instants, `A` and `B` must be discretised before the model can run on a computer. Once discretised, the model can be evaluated either as a linear recurrence, stepping through inputs one at a time (correct but not parallelisable), or as a convolution computing the whole sequence at once.

## S4 versus S6

The report contrasts two algorithms to make the mechanism concrete. **S4** is the linear, time-invariant version: given input of shape `(B, L, D)` for batch size, sequence length and feature dimension, it defines `A`, `B` and `C`, discretises `A` and `B`, and runs — with all three parameters constant across the entire sequence. **S6** extends S4 with the selection mechanism: `B` and `C` become dependent on the input `X`, `A` receives a correction allowing it to adapt, and `Δ` is computed with a dynamic adjustment. The model adapts itself at every time step, and is therefore **no longer LTI** — it varies in time.

That is the cost. LTI models such as S4 can convert all their computation into fixed convolutions, which makes them very fast. A time-varying model cannot: if the rules change per step, the computation cannot be folded into a static convolution, because each step depends on the results of those before it.

## Selective scan and hardware awareness

Mamba recovers throughput with a **selective scan** built on three techniques:

- **Kernel fusion** — merging computations to reduce memory accesses
- **Parallel scan** — a scan-based computation executed in parallel
- **Recomputation** — recomputing intermediate steps to save memory

The underlying observation is about the GPU memory hierarchy. Traditional SSMs use convolutions, which require heavy memory access, and moving between GPU memory levels (L1, L2, global) slows execution. Instead of convolutions, Mamba uses a scan that keeps working data in fast on-chip memory, avoiding expensive trips to main memory: `A`, `B` and `C` are loaded into SRAM, the heavy operations including discretisation run there, and only the output is written back to the slower HBM. Rather than holding the entire internal state `h` at once, different levels of the hierarchy are used as needed.

The result is **linear scaling** — as input length grows, compute time grows linearly rather than quadratically as in a Transformer.

## Reported results

- **Induction Heads** — retained information and generalised to sequences up to a million tokens, where Hyena and Transformer baselines failed entirely as sequence length grew
- **Selective Copying** — 99.8% accuracy, substantially above Hyena- and S4-based models
- **Parameter efficiency** — a 74K-parameter model achieved strong results across synthetic tasks, where H3 required 153K parameters for comparable or worse performance in some cases
- **Long context** — performance holds or improves as context lengthens, in contrast to Transformers

The report also notes Mamba often surpassing Transformer-based models on very long sequences while using far fewer resources relative to models of comparable capability.

## Relationship to LMU

The report is careful not to present Mamba as superseding the LMU. Both are SSMs: both maintain a state vector to hold information about long sequences and both use linear equations to update it. The LMU's distinguishing feature is that its state does not merely evolve under `A` but passes through the Legendre basis, which the report describes as a more principled form of retention.

The two are characterised as **complementary rather than competing**. The suggested direction is a hybrid: use Legendre polynomials as the memory basis inside a selective SSM architecture, combining the LMU's strong mathematical representation with Mamba's dynamic flexibility. The report presents this as a research direction, not an implemented result.

The reservations are stated plainly in §6.1.2. Mamba is a different model from the LMU — heavier, with many more parameters, and requiring greater computational resources. It is therefore **not as suitable as the LMU for small devices or edge hardware**, and does not fully replace the LMU's simplicity, stability and efficiency. The open challenge is combining adaptive, content-aware memory with computational simplicity.

## Comparison across the three approaches (§3.3.3)

| | HiPPO | FF-LMU | Mamba |
|---|---|---|---|
| Contribution | Optimal retention of history through polynomial projection; no window needs defining in advance; robust to time-scale change | Efficient long-range memory in a feedforward structure; fast parallel training on GPU | Linear-time modelling with selective state; filters irrelevant information |
| Headline result | 98.3% psMNIST (`LegS`) | 98.49% psMNIST, 89.10% IMDb | 99.8% Selective Copying; 1M-token generalisation |
| Efficiency note | 470K time steps/s on one CPU core | ~80× faster training than comparable LSTM on GPU | Linear scaling via selective scan; low memory-access cost |

Mamba's selectivity comes at the price of simplicity — the model can no longer be computed with ordinary convolutions, and both its efficiency profile and its complexity increase. The report treats this as the beginning of a direction worth investigating rather than a settled trade-off, and poses the open question directly: how can such mechanisms be integrated into LMU-based models without sacrificing energy efficiency?

A note on comparing these numbers: the report's own conclusion cautions that results across the source papers are not directly comparable, because they use different datasets, pursue different experimental goals, and emphasise different metrics.
