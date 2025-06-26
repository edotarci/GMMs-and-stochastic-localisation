## Introduction

Diffusion models have emerged as a powerful framework for generative modeling, achieving state-of-the-art results in tasks such as image synthesis and density estimation. These models are typically formulated in a probabilistic setting, where the goal is to learn a generative process that, given i.i.d. samples from an unknown target distribution $\mu$, can generate new samples from the same distribution. Diffusion models achieve this by learning the score function, i.e., the gradient of the log-likelihood.

After briefly comparing diffusion models with other approaches used in generative modeling, we introduce methods for learning the score function, such as score matching [3] and denoising score matching [4]. Once the score function is learned, we explore various sampling algorithms, including *annealed Langevin dynamics* [6], *denoising diffusion probabilistic models (DDPM)* [7], *score-based generative modeling via SDEs* [8], in particular with the OU process, and *stochastic localization* [9].

We then analyze key results from the paper by Montanari [9], which provides further insights into the empirical properties of these methods. In the end, following the approach of Shah et al. [10], we study the convergence properties of stochastic localization when applied to a two-component Gaussian mixture model.

All code used to produce the results and figures in this work is available here.

---

### References

[1] D. P. Kingma and M. Welling. *Auto-Encoding Variational Bayes*, arXiv:1312.6114, 2014.  
[2] M. Arjovsky, S. Chintala, and L. Bottou. *Wasserstein GAN*, ICML, 2017.  
[3] A. Hyvärinen and P. Dayan. *Estimation of non-normalized statistical models by score matching*, JMLR, 2005.  
[4] P. Vincent. *A connection between score matching and denoising autoencoders*, Neural Computation, 2011.  
[5] Y. Song, S. Garg, J. Shi, and S. Ermon. *Sliced Score Matching: A Scalable Approach to Density and Score Estimation*, arXiv:1905.07088, 2019.  
[6] Y. Song and S. Ermon. *Generative modeling by estimating gradients of the data distribution*, NeurIPS, 2019.  
[7] J. Ho, A. Jain, and P. Abbeel. *Denoising diffusion probabilistic models*, NeurIPS, 2020.  
[8] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. *Score-based generative modeling through stochastic differential equations*, ICLR, 2021.  
[9] A. Montanari. *Sampling, Diffusions, and Stochastic Localization*, arXiv:2305.10690, 2023.  
[10] K. Shah, S. Chen, and A. Klivans. *Learning Mixtures of Gaussians Using the DDPM Objective*, arXiv:2307.01178, 2023.

