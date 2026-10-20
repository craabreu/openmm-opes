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
$N_{\text{eff}} = (\sum_{k=0}^n w_k)^2 / \sum_{k=0}^n w_k^2$, with $w_0 = 1$,
grows, by Silverman's rule:

$$\sigma_i^{(n)} = \sigma_i^{(0)}\left[N_{\text{eff}}^{(n)}(d+2)/4\right]^{-1/(d+4)}$$

There is no lower bound by default, as in PLUMED. The bias is tabulated on the
CV grid and interpolated with splines, so kernels narrower than the grid
spacing are not resolved in detail. This costs sampling efficiency, not
correctness. The simulation feels exactly the tabulated bias, and kernels are
weighted with that same bias, so the reweighted estimate stays unbiased. A
floor would instead change the estimate itself, by keeping kernels wider than
Silverman's rule prescribes. `minBandwidth` sets one if needed.

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
estimate in explore mode and the reweighted one otherwise. Because
$p^{\text{WT}}$ is about $\sqrt{\gamma}$ times wider than $P$, explore kernels
start at $\sqrt{\gamma}\,\sigma^{(0)}$, with a fixed `biasWidth` still read as
the unbiased $\sigma^{(0)}$.

## OPES combined with OPES-explore

`CombinedOPES` follows Ray and Rizzi (J. Chem. Theory Comput. 21, 58 (2025)).
Both components deposit a kernel at the same steps and CV values. The explore
component builds its target estimate from the unweighted samples, which feel
the total bias $V^o + V^e$. The OPES component weights its kernels by its own
bias only, $w_k = e^{\beta V^o_{k-1}(\mathbf{s}_k)}$, so its estimate is the
distribution biased by $V^e$, not the unbiased one. The unbiased estimate
reweights every sample by the total bias,

$$P_n(\mathbf{s}) = \frac{\sum_k^n w_k\, G(\mathbf{s}, \mathbf{s}_k)}{\sum_k^n w_k},
\qquad w_k = e^{\beta [V^o_{k-1}(\mathbf{s}_k) + V^e_{k-1}(\mathbf{s}_k)]}$$

which is the paper's Eq. 14 accumulated on the fly, with kernels sized as in
the explore component's reweighted estimate. The two biases share one force,
so the CVs are computed once per step. At each deposition, the explore term is
switched off for one evaluation of the bias force group to read $V^o$ alone.

## Free energy

`getFreeEnergy()` always uses the importance-sampling estimate,
$F_n = -\beta^{-1}\log P_n$. In standard OPES the direct and reweighted routes
are equivalent; in explore mode they differ until convergence, and the
reweighted one converges better.

## Extra biases and c(t)

With other biases $V_x$ acting on the simulation, the kernel weights become
$w_k = e^{\beta [V_{k-1}(\mathbf{s}_k) + V_x(\mathbf{R}_k)]}$, so that $P_n$
estimates the distribution with every bias removed. The OPES bias itself is
built from $P_n$ as before.

For a bias that changes in time, reweighting a sample taken at time $t$ by
$e^{\beta [V_t(\mathbf{s}) - c(t)]}$ is exact, with

$$c(t) = -\frac{1}{\beta}\ln \frac{\int P(\mathbf{s})\,
e^{-\beta V_t(\mathbf{s})}\,d\mathbf{s}}{\int P(\mathbf{s})\,d\mathbf{s}}$$

`getCt()` evaluates this on the grid with $P_n$ in place of $P$. It is
unbiased once $P_n$ is, but wrong while $P_n$ still misses basins.
`getCt("running")` is PLUMED's `rct`,
$\beta^{-1}\ln[(w_0 + \sum_k w_k) / (n + 1)]$, the mean weight of all
samples so far plus one fictitious sample under the initial flat bias,
$w_0 = e^{-\beta\Delta E}$. It lags behind the current bias, since it
averages over past ones, and PLUMED advises against reweighting with it: use
it to check that the bias has converged. On alanine dipeptide, the
current-bias estimate is the more accurate one after about 2 ns.

## Smoothing correction

A kernel estimate is the true density convolved with the kernels, so to
leading order it overshoots by $\tfrac{1}{2}h^2\nabla^2 P$. In free energy
terms, basin floors come out too high and steep walls too low. Silverman's rule
narrows new kernels only slowly, and old kernels keep their width, so this
bias fades slowly.

`getFreeEnergy(correctSmoothing=True)` removes it with the multiplicative
correction of Jones, Linton and Nielsen (Biometrika 82, 327 (1995)). A pilot
estimate $\tilde P$ uses the same kernels broadened by `pilotFactor`. Each
kernel's weight is divided by $\tilde P$ at its center, so the reweighted
samples follow the nearly flat ratio $P/\tilde P$, which the broad kernels
smooth at little cost in bias:

$$\hat P(\mathbf{s}) = \tilde P(\mathbf{s})\,
\frac{\sum_k w_k\, \tilde G_k(\mathbf{s}) / \tilde P(\mathbf{s}_k)}{\sum_k w_k}$$

The bias drops to order $h^4$, while the broad kernels keep the noise of a
single run low. The product does not conserve probability locally, which would
shift free energy differences between basins, so the result is rescaled to
carry, over a few pilot widths, the probability of the standard estimate.

On alanine dipeptide, sixteen 50 ns runs gave these RMS errors against a
reference surface, below 5 kcal/mol, in kcal/mol:

| Estimate | Single run | Average of 16 runs |
|---|---|---|
| Standard | 0.099 | 0.055 |
| Corrected, `pilotFactor=2` | 0.059 | 0.026 |

Basin free energy differences were unchanged. The correction needs the
broadened kernels to be narrow compared with the basins; early in a run, while
kernels are still wide, it can make the surface worse. It changes only the
reported free energy, not the bias.

## Kernel compression

Rather than storing a bias grid, kernels closer than `compressionThreshold` in
Mahalanobis distance are merged, preserving total weight, mean and second
moment. The number of compressed kernels is what makes $|\Omega_n|$ estimable.

This differs on purpose from PLUMED's `OPES_METAD`, which weights the merged
center and variance by kernel *height* $h = w/\prod_i \sigma_i$ and keeps
$h_1 + h_2$ as the merged height. When the two kernels have different
bandwidths, as they do once Silverman's rule has shrunk the newer one, that
rule does not conserve probability mass. Fed the same weighted samples and
bandwidths, both estimates compress to about as many kernels, but against the
uncompressed estimate the mass-preserving merge is five to eight times more
accurate (mean absolute log-density error 0.03–0.05 against 0.17–0.25), while
the height-weighted one inflates the total mass by 17–27%. The script is
`benchmarks/plumed/merge_accuracy.py`. Biases built from the same samples
therefore differ slightly from PLUMED's, by about 1 kJ/mol RMS in the tests
there.

## Adaptive bandwidth

With a `varianceFrequency`, $\sigma^{(0)}$ follows the CV variance measured
on the fly with an exponentially decaying mean of window
$\tau$ = `statsWindowSize` deposition strides (Invernizzi & Parrinello, JCTC
2022, SI Eqs. S1–S4). The variance is that of the *sampled* distribution, so
OPES divides it by $\gamma$ to size its reweighted kernels, while OPES-explore
uses it as is.

No kernel is deposited during the first $\tau$ strides, so those samples are
unbiased, and their variance is $\gamma$ times narrower than a sampled one.
When the wait ends, the squared deviations accumulated so far are multiplied
by $\gamma$ once; later samples come from the biased run and enter unscaled.
Without this, the first kernels would be $\sqrt{\gamma}$ times too narrow and
the early bias spiky. The published equations omit this step, but PLUMED's
`OPES_METAD` and `OPES_METAD_EXPLORE` apply it at their first deposition.

## Warm-up bandwidth estimation

Both papers prescribe measuring $\sigma^{(0)}$ from a short unbiased run.
Passing `warmupSteps` automates this: no kernels are deposited until that
many steps have elapsed, so the CV variance measured over that span is
genuinely unbiased, and it is frozen (scaled by $\gamma$, since kernels are
built from a sampled-distribution variance by convention) once warm-up ends.
