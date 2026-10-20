# API reference

The samplers:

- [`OPES`](opes.md): OPES and OPES-explore, biasing a simulation through one
  OpenMM force.
- [`CombinedOPES`](combined-opes.md): OPES and OPES-explore acting together
  on the same collective variables (OPES_COM).

The kernel density estimation behind them, for direct use or inspection:

- [`OnlineKDE`](online-kde.md): a weighted kernel density estimate on a grid,
  with on-the-fly compression and the smoothing correction.
- [`CVSpace`](cv-space.md): the grid, periodicity and reflective boundaries of
  the collective variables.
- [`Kernel`](kernel.md): one multivariate kernel with diagonal covariance.
