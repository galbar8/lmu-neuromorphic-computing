# Legendre Memory Unit

Notes derived from §3.1–3.3 of the report. Section numbers refer to the [full report](../report/lmu-neuromorphic-computing.pdf).

Related: [HiPPO and State Space Models](hippo-and-ssm.md) · [Mamba](mamba.md) · [Neuromorphic models](neuromorphic-models.md)

## Motivation

The LMU (Voelker, Kajić & Eliasmith, 2019) was introduced to improve long-range retention in recurrent networks while keeping memory and compute small. The report frames it against two failures of the models that came before it.

Recurrent networks handle vanishing and exploding gradients by adding machinery. LSTM and GRU gate the flow of information explicitly — the network must decide, at every step, how much to keep and how much to forget — and each gate costs weight matrices per unit. The report's point is that this is an indirect solution: the gates manage a symptom of pushing information through repeated nonlinearities.

Separately, many real problems are not discrete sequences at all. Robot motion, EEG, weather forecasting — these are continuous processes, often sampled irregularly. LSTM and Transformers were built for uniformly sampled discrete input and do not represent time as a continuous mathematical quantity. The LMU does, and that is the property the neuromorphic work later depends on.

## Continuous formulation

The LMU's memory is a linear time-invariant (LTI) system that holds a compressed representation of the input `u(t)` over a sliding window of length `θ`:

```
θ ṁ(t) = A m(t) + B u(t)
```

- `m(t)` — memory vector of dimension `d`, holding the projection of the input history onto a Legendre basis
- `θ` — the length of the window of history being represented
- `A`, `B` — fixed state and input matrices derived from the Legendre basis; they are *not* learned

The component that realises this is the Delay Network (DN), the core of the LMU. An ideal delay system takes input `u(t)` and returns it shifted by `θ`:

```
y(t) = D[u(t)] = { 0           for t < θ
                 { u(t − θ)    for t ≥ θ
```

The DN approximates this behaviour with a linear model, and functions as an optimal filter that grades stored information by importance. Because it is separate from the nonlinear part of the network and is itself linear, it preserves history without accumulating error or vanishing gradients, and its computation can be parallelised — the report cites training speedups up to 200× from that property.

## Legendre basis and memory representation

The parameter `d` is the memory size: the number of dimensions used to represent the input's history. The input over the window is written as a weighted sum of the first `d` Legendre polynomials,

```
u(t) ≈ Σ_{i=0}^{d−1} c_i P_i(t)
```

where `P_i` is the Legendre polynomial of order `i` and `c_i` the corresponding coefficient. Each dimension of `m_t` preserves one orthogonal component of the previous inputs over the window. Orthogonality is what makes the representation efficient: the components do not interfere, so each captures information the others do not.

The order `d` sets the approximation error directly:

| `d` | What the memory retains |
|---|---|
| 1 | A running average of past input — behaves like a simple gate-free LSTM cell, retaining information but forgetting gradually at a rate set by `θ` |
| 2 | Linear trends over the window |
| higher | Progressively deeper and more accurate history; higher input frequencies are preserved |

## Discrete architecture

The LMU layer has four parts (§3.2.1). In discrete time:

**Input layer.** Each input `x_t` is combined with the network's own history into the drive for the memory:

```
u_t = eₓᵀ x_t + e_hᵀ h_{t-1} + e_mᵀ m_{t-1}
```

`eₓ`, `e_h`, `e_m` are learned encoding vectors that project the input into the memory space.

**Linear memory layer.** The distinguishing component. No gates — a linear dynamical update over a sliding window:

```
m_t = Ā m_{t-1} + B̄ u_t
```

`Ā` and `B̄` are the discretised, fixed Legendre matrices.

**Hidden state layer.** The nonlinear computation, combining memory with input and previous state:

```
h_t = f(Wₓ x_t + W_h h_{t-1} + W_m m_t)
```

with `f` a nonlinearity such as `tanh` or `ReLU`, and `Wₓ`, `W_h`, `W_m` learned.

**Output layer.** A linear map from hidden state to output, `y_t = W_o h_t`, playing the role a softmax layer plays in a standard sequence classifier.

## Key properties

**Gate-free linear memory.** Retention does not depend on learned gating decisions, and information is not pushed through repeated nonlinearities — which is what relieves the vanishing-gradient pressure rather than merely managing it.

**Memory size decoupled from state size.** `d` (memory) and `n` (hidden state) are independent, so memory capacity can be raised without increasing the nonlinear computation. In LSTM and GRU the two are bound together, so more memory necessarily means more nonlinear work. The report identifies this separation as one of the LMU's central advantages, and as the reason it can offer better performance than traditional recurrent networks at substantially lower computational load on tasks such as NLP and time-series prediction.

**Fixed, principled matrices.** `A` and `B` come from the Legendre basis rather than from training, so the memory's behaviour is analytically characterised in advance.

**Long-range retention.** The report cites continuous dependencies captured over windows on the order of 100,000 steps.

**Natural fit for spiking hardware.** Because the memory is a first-order linear differential system, it maps onto spiking neuron dynamics without distorting its global memory behaviour. The original paper demonstrated an implementation using Poisson spiking neurons with reconstruction error scaling as `O(d/√m)` in the neuron count `m`, and the report notes neuromorphic versions implemented on hardware such as Intel's Loihi.

## Known limitations

These carry into the rest of the survey and motivate the variants:

- **Sequential training.** The original LMU is recurrent, so it trains step by step with BPTT and will not parallelise on GPU. Addressed by [FF-LMU and parallelised LMU training](hippo-and-ssm.md#scaling-lmu-training).
- **Fixed window.** History outside `θ` is gone, and `θ` is set in advance.
- **No content-based selection.** The dynamics do not depend on the input, so the model cannot decide what is worth keeping. This is the gap [Mamba](mamba.md) addresses.
- **No pre-training.** Unlike Transformer-family models, the LMU-based architectures in the survey are trained per task.
