# References

The works cited in the survey, in the numbering used by the report's bibliography (pp. 90–91). Links are those given in the report itself; no papers are redistributed here.

Grouped by theme for navigation; the bracketed number is the report's own citation index.

## Core: LMU and its variants

- **[3]** Voelker, A., Kajić, I., & Eliasmith, C. (2019). *Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks.* NeurIPS. [proceedings.neurips.cc](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf)
  — The original LMU. Basis for §3.2, and for the Poisson-spiking implementation discussed in §4.1.1.1.
- **[2]** Chilkuri, N. R., & Eliasmith, C. *Feedforward Legendre Memory Unit.* OpenReview. [openreview.net](https://openreview.net/forum?id=qkJevgi2Fp3)
  — FF-LMU. Basis for §3.3.1.1.
- **[20]** Chilkuri, N., & Eliasmith, C. (2021). *Parallelizing Legendre Memory Unit training.* arXiv:2102.11417. [arxiv.org](https://arxiv.org/abs/2102.11417)
  — Basis for §3.3.1.2.

## Theoretical framework: HiPPO and State Space Models

- **[5]** Gu, A., Dao, T., Ermon, S., Rudra, A., & Ré, C. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections.* arXiv:2008.07669. [arxiv.org](https://arxiv.org/abs/2008.07669)
  — Basis for §3.2.2, including the LegT / LagT / LegS measures.
- **[10]** Gu, A. (2023). *Modeling Sequences with Structured State Spaces.* Stanford Digital Repository. [stacks.stanford.edu](https://stacks.stanford.edu/file/druid:mb976vf9362/gu_dissertation-augmented.pdf)
  — Basis for §3.2.3 on SSMs, S4, SaShiMi and orthogonal SSMs.
- **[6]** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752. [arxiv.org](https://arxiv.org/abs/2312.00752)
  — Basis for §3.3.2 and for the Spike Mamba component of SpikMamba.
- **[27]** Wikipedia contributors. *State-space representation.* [en.wikipedia.org](https://en.wikipedia.org/wiki/State-space_representation)
  — Source of the SSM block diagram (Figure 15).

## Neuromorphic LMU-based architectures

- **[8]** Fra, V., Leto, B., Pignata, A., Macii, E., & Urgese, G. (2024). *Natively neuromorphic LMU architecture for encoding-free SNN-based HAR on commercial edge devices.* arXiv:2407.04076. [arxiv.org](https://arxiv.org/abs/2407.04076)
  — L2MU. Basis for §4.2.1.
- **[7]** Liu, Z., Datta, G., Li, A., & Beerel, P. A. (2024). *LMUFormer: Low complexity yet powerful spiking model with Legendre memory units.* arXiv:2402.04882. [arxiv.org](https://arxiv.org/abs/2402.04882)
  — LMUFormer and Spiking LMUFormer. Basis for §4.2.2.
- **[9]** Chen, J., Yang, Y., Deng, S., Teng, D., & Pan, L. (2024). *SpikMamba: When SNN meets Mamba in Event-based Human Action Recognition.* arXiv:2410.16746. [arxiv.org](https://arxiv.org/abs/2410.16746)
  — Basis for §4.2.3.
- **[26]** *Legendre Memory Units on Loihi* — NengoExamples. [nengo.ai](https://www.nengo.ai/nengo-examples/loihi/lmu.html)
  — The proof of concept for running LMU on neuromorphic hardware, cited in §6.2.2.

## Neuromorphic computing and spiking neurons

- **[12]** Yang, J., Wang, R., Ren, Y., Mao, J., Wang, Z., Zhou, Y., & Han, S. (2020). *Neuromorphic engineering: From biological to spike-based hardware nervous systems.* Advanced Materials, 32(52). [doi:10.1002/adma.202003610](https://doi.org/10.1002/adma.202003610)
- **[11]** Chen, X., Yajima, T., Inoue, I. H., & Iizuka, T. (2021). *An ultra-compact leaky integrate-and-fire neuron with long and tunable time constant utilizing pseudo resistors for spiking neural networks.* Japanese Journal of Applied Physics, 61(SC), SC1051. [doi:10.35848/1347-4065/ac43e4](https://doi.org/10.35848/1347-4065/ac43e4)
  — Source for the LIF model and circuit in §2.3.1.3.
- **[23]** Bouanane, M. S., Cherifi, D., Chicca, E., & Khacef, L. (2022). *Impact of spiking neurons leakages and network recurrences on event-based spatio-temporal pattern recognition.* arXiv:2211.07761. [arxiv.org](https://arxiv.org/abs/2211.07761)
  — Source for CuBa-LIF.
- **[15]** Ivanov, D., Chezhegov, A., Grunin, A., Kiselev, M., & Larionov, D. (2022). *Neuromorphic Artificial Intelligence Systems.* arXiv:2205.13037. [arxiv.org](https://arxiv.org/abs/2205.13037)
- **[25]** Young, A. R., Dean, M., Plank, J. S., & Rose, G. S. (2019). *A Review of Spiking Neuromorphic Hardware Communication Systems.* IEEE. [ieeexplore.ieee.org](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8843969)
  — Source for the SpiNNaker / TrueNorth constraints in §6.2.
- **[29]** Payong, A. *Neuromorphic computing explained.* Baeldung on Computer Science. [baeldung.com](https://www.baeldung.com/cs/neuromorphic-computing-explained)
- **[16]** *Neurons* — Organismal Biology, Georgia Tech. [organismalbio.biosci.gatech.edu](https://organismalbio.biosci.gatech.edu/chemical-and-electrical-signals/neurons/)

## Spike encoding

- **[18]** Kim, Y., Park, H., Moitra, A., Bhattacharjee, A., Venkatesha, Y., & Panda, P. (2022). *Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?* arXiv:2202.03133. [arxiv.org](https://arxiv.org/abs/2202.03133)
- **[17]** Jiang, Y., Lu, S., & Sengupta, A. (2024). *Stochastic Spiking Neural Networks with First-to-Spike Coding.* arXiv:2404.17719. [arxiv.org](https://arxiv.org/abs/2404.17719)
  — Source for time-to-first-spike coding.

## Sequential baselines

- **[1]** Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). *Empirical evaluation of gated recurrent neural networks on sequence modeling.* arXiv:1412.3555. [arxiv.org](https://arxiv.org/abs/1412.3555)
  — Source for the LSTM and GRU formulations in §2.1.2.
- **[4]** Shi, X., Chen, Z., Wang, H., Yeung, D., Wong, W., & Woo, W. (2015). *Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.* arXiv:1506.04214. [arxiv.org](https://arxiv.org/abs/1506.04214)
  — ConvLSTM.
- **[13]** Pascanu, R., Mikolov, T., & Bengio, Y. (2012). *On the difficulty of training Recurrent Neural Networks.* arXiv:1211.5063. [arxiv.org](https://arxiv.org/abs/1211.5063)
  — Source for the vanishing / exploding gradient discussion.
- **[14]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention is all you need.* arXiv:1706.03762. [arxiv.org](https://arxiv.org/abs/1706.03762)
- **[28]** Chen, Y., You, Z., Zhang, S., Li, H., Li, Y., Wang, Y., & Tan, M. (2024). *Core context aware transformers for long context language modeling.* arXiv:2412.12465. [arxiv.org](https://arxiv.org/abs/2412.12465)
  — Cited in §2.2.2 as evidence that recent Transformers do handle long-range dependencies.

## Datasets and benchmarks

- **[19]** Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). *A simple way to initialize recurrent networks of rectified linear units.* arXiv:1504.00941. [arxiv.org](https://arxiv.org/abs/1504.00941)
  — Source of the psMNIST benchmark used throughout the comparisons.
- **[21]** *IMDB Dataset of 50K Movie Reviews.* Kaggle. [kaggle.com](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- **[22]** Sharma, L., Graesser, L., Nangia, N., & Evci, U. (2019). *Natural Language Understanding with the Quora Question Pairs Dataset.* arXiv:1907.01041. [arxiv.org](https://arxiv.org/abs/1907.01041)
- **[24]** *Keyword spotting on Google Speech Commands V2* — Papers with Code. [paperswithcode.com](https://paperswithcode.com/sota/keyword-spotting-on-google-speech-commands-v2-3)
  — Source for the comparison against the leading result on that dataset in §5.1.
