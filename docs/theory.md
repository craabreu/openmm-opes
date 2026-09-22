# The method as implemented

Notation follows Invernizzi and Parrinello. $\beta = 1/k_BT$, $\gamma$ is the
bias factor, and $\mathbf{s}$ the collective variables.

## The probability estimate

OPES estimates the unbiased distribution by weighted kernel density estimation,

$$P_n(\mathbf{s}) = \frac{\sum_k^n w_k\, G(\mathbf{s}, \mathbf{s}_k)}{\sum_k^n w_k},
\qquad w_k = e^{\beta V_{k-1}(\mathbf{s}_k)}$$

with Gaussian kernels of fixed height $h = \prod_i (\sigma_i\sqrt{2\pi})^{-1}$.
This is `OnlineKDE.getLogPDF()`; the weight is the bias energy read from the
OPES force group *before* the new kernel is deposited, so it really is
$V_{k-1}$.

## Bandwidth

Bandwidths shrink as the effective sample size
$N_{\text{eff}} = (\sum_k w_k)^2 / \sum_k w_k^2$ grows, by Silverman's rule:

$$\sigma_i^{(n)} = \sigma_i^{(0)}\left[N_{\text{eff}}^{(n)}(d+2)/4\right]^{-1/(d+4)}$$

## Normalization

$Z_n$ normalizes over the CV space explored so far, and is approximated by a
sum over the compressed kernel centers — `OnlineKDE.getLogMeanDensity()`.

$$Z_n = \frac{1}{|\Omega_n|}\int_{\Omega_n} P_n(\mathbf{s})\, d\mathbf{s}$$

## The bias

$$V_n(\mathbf{s}) = \left(1 - \tfrac{1}{\gamma}\right)\frac{1}{\beta}
\log\left(\frac{P_n(\mathbf{s})}{Z_n} + \epsilon\right)$$

with $\epsilon = e^{-\beta\Delta E/(1-1/\gamma)}$ limiting the bias to the
barrier $\Delta E$. This is `OPES.getBias()`.

## OPES-explore

The explore variant estimates the *sampled* distribution instead, with uniform
weights,

$$p^{\text{WT}}_n(\mathbf{s}) = \frac{1}{n}\sum_k^n G(\mathbf{s}, \mathbf{s}_k),
\qquad
V_n(\mathbf{s}) = (\gamma-1)\frac{1}{\beta}
\log\left(\frac{p^{\text{WT}}_n(\mathbf{s})}{Z_n} + \epsilon\right)$$

It explores faster and converges more slowly. `getBias()` reads the unweighted
estimate in explore mode and the reweighted one otherwise.

## Free energy

`getFreeEnergy()` always uses the importance-sampling estimate,
$F_n = -\beta^{-1}\log P_n$. In standard OPES the direct and reweighted routes
are equivalent; in explore mode they differ until convergence, and the
reweighted one converges better.

## Kernel compression

Rather than storing a bias grid, kernels closer than `compressionThreshold` in
Mahalanobis distance are merged, preserving total weight, mean and second
moment. The number of compressed kernels is what makes $|\Omega_n|$ estimable.

## Warm-up bandwidth estimation

Both papers prescribe measuring $\sigma^{(0)}$ from a short unbiased run.
Passing `warmupSteps` automates this: no kernels are deposited until that
many steps have elapsed, so the CV variance measured over that span is
genuinely unbiased, and it is frozen (scaled by $\gamma$, since kernels are
built from a sampled-distribution variance by convention) once warm-up ends.
