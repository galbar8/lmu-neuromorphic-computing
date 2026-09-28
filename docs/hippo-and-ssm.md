# HiPPO and State Space Models

Notes derived from §3.2.2, §3.2.3 and §3.3.1 of the report. Section numbers refer to the [full report](../report/lmu-neuromorphic-computing.pdf).

Related: [LMU](lmu.md) · [Mamba](mamba.md) · [Neuromorphic models](neuromorphic-models.md)

## HiPPO

HiPPO — **Hi**gh-order **P**olynomial **P**rojection **O**perators (Gu et al., 2020) — is the general theory behind the [LMU](lmu.md)'s memory. The problem it solves: learning from sequential data requires retaining past input, but the full history cannot be stored under real memory and compute limits, so some optimal compression is needed. HiPPO answers this as online function approximation against a chosen measure of importance.

The framework has four stages (§3.2.2.1):

**1. A time-varying measure `μ(t)`.** A probability measure that says which parts of the past matter most. Choosing `μ` is what determines the qualitative character of the resulting memory — weight early instants heavily and the mechanism remembers them longer.

**2. Optimal projection `proj_t`.** At each instant, project the history received so far onto a chosen polynomial basis, minimising the weighted error

```
‖ f_{≤t} − g(t) ‖_{L²(μ(t))}
```

where `f_{≤t}` is the data received up to time `t` and `g(t)` the simpler function approximating it.

**3. Coefficients `coef_t`.** Instead of storing the data, store the coefficients of the projection:

```
f_{≤t} ≈ g(t) = Σ_{n=0}^{N−1} c_n(t) g_n(t)
```

leaving only `c(t) = [c_0(t), …, c_{N−1}(t)]` to carry forward.

**4. Discretisation.** The coefficients are updated through a differential equation rather than recomputed:

```
d/dt c(t) = A(t) c(t) + B(t) f(t)
```

or recursively, `c_t = A_t c_{t-1} + B_t f_t`. Since real data arrives at discrete instants, the discrete form is what actually runs.

## The three measures

| Operator | Measure | Behaviour |
|---|---|---|
| `HiPPO-LegT` | `μ(t)(x) = (1/θ) · 𝕀[t−θ, t](x)` | Legendre polynomials over a fixed window of length `θ`, all instants inside it weighted equally. Suits periodic or temporally stable structure. **This is the case matching the LMU's own update rule.** |
| `HiPPO-LagT` | `μ(t)(x) = e^{−(t−x)} · 𝕀(−∞, t](x)` | Laguerre polynomials, exponential decay — recent input weighted more, older input fading. Suits tasks where dependence naturally weakens with time. |
| `HiPPO-LegS` | `μ(t)(x) = (1/t) · 𝕀[0, t](x)` | Uniform weight over *all* history from 0 to `t`. The window grows with time, so the operator rescales itself and avoids forgetting early data. |

For `LegT` and `LagT` the matrices are given in closed form (§3.2.2.3, eq. 21):

```
LegT:  A_nk = (1/θ) · { (−1)^{n−k}(2n+1)   n ≥ k
                      { (2n+1)              n ≤ k
       B_n  = (1/θ)(2n+1)(−1)ⁿ

LagT:  A_nk = { 1   n ≥ k          B_n = 1
              { 0   n < k
```

`LegS` differs structurally. Its update carries an explicit `1/t`:

```
d/dt c(t) = −(1/t) A c(t) + (1/t) B f(t)
```

with

```
A_nk = { (2n+1)^{1/2}(2k+1)^{1/2}   n > k
       { n + 1                       n = k        B_n = (2n+1)^{1/2}
       { 0                           n < k
```

Discretised, this becomes `c_{k+1} = (1 − A/k) c_k + (1/k) B f_k`. Dividing by `k` means the influence of old data is diluted rather than dropped, and the operator becomes invariant to time rescaling: stretch or compress the input and the approximation stays accurate.

## Relationship to RNNs and to LMU

At the lowest order the framework reduces to something familiar. With `N = 1` and `A = B = 1`, the discretisation becomes

```
c(t + Δt) = (1 − Δt) c(t) + Δt · f(t)
```

— a simple weighted combination of previous state and new input, which is how a GRU update behaves. The report's observation is that HiPPO reaches the same result through polynomial criteria instead of learned gates: at `N = 1` you get GRU-like behaviour without needing the gating machinery.

HiPPO can also be dropped into a standard RNN as a dynamic memory operator, replacing the conventional hidden state or gate mechanism with the linear dynamical system that projects history onto the chosen basis (§3.2.2.2).

The relationship to the LMU is the important one: **LMU is built directly on HiPPO's principles, using Legendre polynomials as its memory basis.** HiPPO is the general theory — any orthogonal polynomial basis, any measure — and the LMU is one instance of it. Note the chronology runs the other way from the logic: the LMU paper is from 2019, the HiPPO paper from 2020. The report treats HiPPO as the theoretical framework that explains and generalises what the LMU does, not as its historical predecessor. The `LegT` matrices above provide the formal link, since they reproduce the LMU's memory update.

**Reported results (§3.2.2.4).** On permuted-sequential MNIST, `HiPPO-LegS` reached **98.3%**, against 97.9% for a basic Transformer, 95.1% for LSTM and 93% for GRU. `LegT` came close but degraded when `θ` was set incorrectly; `LagT` worked but was not optimal across conditions. `HiPPO-LegS` updates memory in `O(N)` rather than `O(N²)` or worse, and the report cites it processing up to **470,000 time steps per second on a single CPU core** — roughly 10× faster than LMU and LSTM.

## State Space Models

State Space Models come from control theory and describe how a system evolves. They rest on an internal state holding everything necessary about the system, so the full past need not be retained. Two equations define the model:

```
State:        x′(t) = A x(t) + B u(t)
Observation:  y(t)  = C x(t) + D u(t)
```

- `x(t)` — internal state, the system's memory
- `A` — how the state evolves on its own, absent external influence
- `B` — how external input enters the state
- `C` — how internal state produces output
- `D` — direct influence of input on output

The report's illustration: for an SSM controlling a drone, `x(t)` is position and velocity, `A x(t)` describes how it moves if untouched, and `B u(t)` how motor commands change that. The state equation predicts evolution; the observation equation connects true state to what sensors can actually measure, which in practice is partial and noisy.

**Applications.** SSMs handle long sequences naturally without the gradient problems of recurrent networks, since they rest on linear differential equations. S4 in particular performs well on real-time prediction and on audio and speech, outperforming CNN- and RNN-based models by preserving internal signal relationships over longer spans. SaShiMi is cited as a worked example: S4 blocks with multi-level pooling, downsampling the sequence (shortening it while widening the representation) and reconstructing through up-pooling, so the model learns patterns at several time scales at once.

## Orthogonal SSMs

Traditional SSMs rely on simple linear equations and struggle with very long memory and numerical stability across thousands of steps. Orthogonal SSMs address this by building the internal state from a projection onto an orthogonal polynomial basis — Legendre, Fourier — rather than updating it linearly in the plain way. Orthogonality keeps distinct features from mixing, which is what allows long-range learning without loss of accuracy.

This is the connection back to LMU: **the LMU is a specific realisation of an SSM whose state matrices come from the Legendre basis**, and both share the same continuous-time LTI foundation.

One structural point worth care: standard SSMs are LTI — `A`, `B`, `C`, `D` do not change with time. `HiPPO-LegS` is *not* LTI, because every matrix entry carries a `t` dependence. That time variation is exactly what lets it rescale its projection to match dynamically changing input.

## Scaling LMU training

The LMU's memory update `m_t = Ā m_{t-1} + B̄ u_t` is recursive, so training is sequential. The report covers two distinct fixes (§3.3.1).

### FF-LMU

The Feedforward LMU (Chilkuri & Eliasmith) changes the architecture. Two adaptations: the nonlinear dynamical system computing internal states is dropped, so every step is computed simultaneously; and the recurrent state update becomes a matrix operation that runs efficiently on GPU. The equations become

```
u_t = g(W_u x_t + b_u)
m_t = A m_{t-1} + B u_t
o_t = f(W_o m_t + b_o)
```

The key change is what is *absent*. In the original LMU, `u_t` depends on `h_{t-1}` and `m_{t-1}`, and `h_t` depends on `h_{t-1}`. In FF-LMU, `u_t` is computed directly from `x_t` alone, and the hidden state is replaced by an output `o_t` computed directly from memory. Removing the dependence on `h_{t-1}` is what permits full parallelism. All nonlinearity collapses into the output computation, leaving the LTI system solely responsible for memory.

The memory can then be expressed in closed form as a convolution rather than a recursion:

```
m(t) = ∫₀ᵗ exp(A(t − τ)) B u(τ) dτ  =  ∫₀ᵗ K(t − τ) u(τ) dτ
```

with convolution kernel `K(t) = e^{A(t−τ)}B`, computable for all time steps at once.

FF-LMU trains as a feedforward network while still able to operate recurrently at inference. Reported results: **98.49% on psMNIST** (exceeding `HiPPO-LegS`); **89.10% on IMDb** with over 500× fewer parameters than LSTM; better than LSTM on Quora Question Pairs with roughly 650× fewer parameters; state-of-the-art on psMNIST and QQP with up to 1000× fewer parameters; and roughly **80× faster training than a similarly sized LSTM on GPU**. The report draws a conclusion from FF-LMU beating the original LMU despite being simpler: the LMU's central advantage is the quality of its temporal memory — the optimal LTI system in the recurrent part — rather than its recurrent structure as such.

### Parallelising LMU training

The alternative (Chilkuri & Eliasmith, 2021) keeps the original LMU intact and parallelises the computation instead. Rather than computing each `m_t` from `m_{t-1}`, all states are obtained at once:

```
m_t = Σ_{j=1}^{t} Ā^{t−j} B u_j
```

This linear convolution is assembled into an impulse-response matrix `H = [Ā⁰B̄, Ā¹B̄, …]`, giving every state through a single product `m_{1:n} = H U`. The convolution theorem then allows the product to be evaluated through the Fourier transform, reducing complexity further — multiplication in the frequency domain replaces convolution in the time domain (see the report's appendix on the Fourier transform and FFT, `O(n log n)` against `O(n²)`).

The trade-off between the two approaches: FF-LMU changes the equations to remove recurrence, and suits settings where a feedforward model is acceptable. Parallelised training preserves the original LMU structure including its linear dynamical memory, and suits cases where the recurrent character must be kept — but it needs more memory, since every state must be held at once rather than updated in place.

## Parameter counts (§3.3.3.2)

| Model | Parameters |
|---|---|
| `HiPPO-LegS` | 512 hidden units (count not reported) |
| LMU | 102K |
| FF-LMU | 165K |
| Mamba | variable, 125M to 6.9B depending on application |

FF-LMU needs more parameters than LMU but delivers higher efficiency. HiPPO keeps a compact memory but offers no clear computational advantage in a complex model. Mamba uses far more parameters yet delivers better performance at relatively lower compute than large Transformers.
